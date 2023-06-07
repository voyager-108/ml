import os
import uuid
import numpy as np
from ...utility.split_video import split_video_by_frames
from ...nn.yolov8.score import YOLOv8Objects, analyze_video
from ...nn.room_segmentation.embedding import RoomEmbedderPipeline
from ...nn.room_segmentation.predictor import RoomClassifier
from ...nn.room_segmentation.osg_predictions import predict
from ..stats import derive_statistics
from typing import Annotated
from fastapi import Body
import apps.nn as nn

from ultralytics import YOLO
from fastapi import APIRouter, File, UploadFile

import tempfile
import torch 
import shutil
import logging
import gc


# Number of frames to skip when splitting the video
skip = 5
USE_CUDA_EMBEDDINGS = 'EMBEDDINGS_CUDA' in os.environ and os.environ['EMBEDDINGS_CUDA'] == 'TRUE'
USE_CUDA_YOLO = 'YOLO_CUDA' in os.environ and os.environ['YOLO_CUDA'] == 'TRUE'

# Device used for calculating embeddings
embedding_device = ('cuda:0' if torch.cuda.is_available() and USE_CUDA_EMBEDDINGS else 'cpu')

# Device used for running YOLO
yolo_device = (0 if torch.cuda.is_available() and USE_CUDA_YOLO else 'cpu')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s :: %(message)s",
)


# Server logger
logger = logging.getLogger("ServerApplication")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter("%(asctime)s ::  %(message)s"))
logger.addHandler(streamHandler)

# Score card router
score_card_router = APIRouter(
    prefix='/score-card'
)

# Textual representation of room classes
room_classes = ['bathroom', 'corridor', 'kitchen', 'livingroom', 'common_area', 'null']

# YOLO model
model = YOLO(os.environ['YOLO_PT_PATH'], task='detect')




def calculate_iou(range1: tuple[int, int], range2: tuple[int, int]) -> float:
    """Given two ranges, calculate their intersection over union.

    Args:
        range1 (tuple[int, int]): The first range.
        range2 (tuple[int, int]): The second range.

    Returns:  
        float: The intersection over union of the two ranges.
    """   
    x, y = range1
    p, q = range2
    intersection = max(0, min(y, q) - max(x, p))
    union = max(y, q) - min(x, p)
    iou = intersection / union
    return iou


def assign_to_rooms(rooms: list[int], outputs: list[YOLOv8Objects]) -> list[YOLOv8Objects]:
    """Given a list of room ids and a list of YOLOv8Objects, assign each object to a room.

    Args:
        rooms (list[int]): A list of room ids.
        outputs (list[YOLOv8Objects]): A list of YOLOv8Objects.

    Returns:
        list[YOLOv8Objects]: A list of YOLOv8Objects with room ids assigned.
    """
    
    spans = []
    l, prev, room = 0, None, 0
    for i, room in enumerate(rooms):
        if prev is None:
            prev = room
            l = i
            continue
        if room != prev or len(rooms) - 1 == i:
            spans.append(
                (
                    prev,
                    (l, i - 1 + (len(rooms) - 1 == i)),
                ),
            )
            l = i
            prev = room
    
    for obj in outputs.values():
        overlap, room = 0, 0
        for room_class, span in spans:
            # Giving two ranges [x, y] and [p, q] calculate their
            # intersection
            new_overlap = calculate_iou(obj.framesSpan, (span[0] * skip, span[1] * skip))
            if new_overlap > overlap:
                overlap = calculate_iou(obj.framesSpan, (span[0] * skip, span[1] * skip))
                room = room_class
        obj.roomClass = room_classes[int(room)]
    
    return outputs


def _process_video_file_for_score_card(video_path: str) -> dict:
    """(Internal function) Process a video file for score card.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing the statistics and the YOLOv8Objects.
              in the following format:

                ```python
                {
                        'stats': ScoreCardReport
                        'output': list[YOLOv8Objects]
                }
                ```

    """
    frames = split_video_by_frames(video_path, skip, return_arrays=True)
    logger.info(f"{video_path}, total frames: {len(frames)}")
    # run embeddings in parallel
    # embeddings = pool.submit(worker, 'embeddings', frames)
    with torch.no_grad(): 
        embeddings = embedder(frames)

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"{video_path}, embeddings: {embeddings.shape}")
    yolo_output, vectors = analyze_video(
        model, 
        video_path, 
        vid_stride=skip, 
        verbose=False, 
        stream=True,
        device=yolo_device
    )
    
    logger.info(f"{video_path}, yolo finished")
    inputs = torch.Tensor(
        np.hstack((embeddings, np.array(vectors),))
    )
    logger.info(f"{video_path}, yolo data prepared")
    with torch.no_grad():
        logits = classifier(inputs).cpu()
        logger.info(f"{video_path}, logits: {logits.shape}")
    classification = predict(embeddings, logits, logits=True)
    logger.info(f"{video_path}, classfied")
    yolo_output = assign_to_rooms(classification, yolo_output)
    logger.info(f"{video_path}, finished")

    return {
        'stats': derive_statistics(list(yolo_output.values())),
        'output': yolo_output,
    }


@score_card_router.post('/video')
def process_video_for_score_card(video: UploadFile = File(...)):
    video_format = video.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(prefix=video.filename, suffix=f'.{video_format.lower()}') as video_temp_file:
        shutil.copyfileobj(video.file, video_temp_file)
        return _process_video_file_for_score_card(video_temp_file.name)
    




@score_card_router.post('/v2/video')
def process_video_for_score_card_v2(
    video: UploadFile = File(...), 
    embeddings: Annotated[str, Body(embed=True)] = None,
    yolo_results: Annotated[str, Body(embed=True)] = None,
):
    video_format = video.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(prefix=video.filename, suffix=f'.{video_format.lower()}') as video_temp_file:
        shutil.copyfileobj(video.file, video_temp_file)
        video_path = video_temp_file.name
        logger.info(f"{video_path}, total frames: {len(frames)}")
        frames = split_video_by_frames(video_path, skip, return_arrays=True)
    
        embeddings_ready = False
        if embeddings:
            try:
                embeddings_id = embeddings
                embeddings_vector = nn.vector_store.get(embeddings)     
                embeddings_ready = True
            except: 
                embeddings_vector = None
    
        if not embeddings or not embeddings_vector:
            embeddings_vector = nn.run_embedder(frames)
            embeddings_id = uuid.uuid4().hex
            embeddings_vector.add_done_callback(lambda f: nn.vector_store.put(embeddings_id, f.result()))

        yolo_ready = False

        if yolo_results:
            try:
                yolo_id = yolo_results
                yolo_outputs = nn.vector_store.get(yolo_results)
                yolo_ready = True
            except:
                yolo_outputs = None

        if not yolo_results or not yolo_outputs:
            yolo_outputs = nn.run_yolo(video_path)
            yolo_id = uuid.uuid4().hex
            yolo_outputs.add_done_callback(lambda f: nn.vector_store.put(yolo_id, f.result()))


        logits = nn.run_classifier(
            embeddings_vector.result() if not embeddings_ready else embeddings_vector, 
            yolo_outputs.result() if not yolo_ready else yolo_outputs,
        )

        embeddings_vector = embeddings_vector.result() if not embeddings_ready else embeddings_vector
        classification = nn.run_classifier(embeddings_vector, logits.result(), logits=True)
        yolo_outputs = assign_to_rooms(classification, yolo_outputs.result() if not yolo_ready else yolo_outputs)

    return {
        'stats': derive_statistics(list(yolo_outputs.values())),
        'output': yolo_outputs,
        'embeddings': embeddings_id,
        'yolo': yolo_id,
    }
    




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
    args = parser.parse_args()
    video = args.video
    
    _process_video_file_for_score_card(video)
        
        
