import os
import uuid
import numpy as np
from ...utility.split_video import split_video_by_frames
from ...nn.yolov8.score import YOLOv8Objects, analyze_video
from ...nn.room_segmentation.embedding import RoomEmbedderPipeline
from ...nn.room_segmentation.predictor import RoomClassifier
from ...nn.room_segmentation.osg_predictions import predict
from ..stats import derive_statistics
from typing import Annotated, Any
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


def assign_to_rooms(rooms: list[int], outputs: dict[Any, list[YOLOv8Objects]]) -> list[YOLOv8Objects]:
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
    embeddings: Annotated[str | list[str], Body(embed=True)] = None,
    yolo_results: Annotated[str | list[str], Body(embed=True)] = None,
    isLast: Annotated[bool, Body(embed=True)] = False,
):
    video_format = video.filename.split('.')[-1]
    with tempfile.NamedTemporaryFile(prefix=video.filename, suffix=f'.{video_format.lower()}') as video_temp_file:
        shutil.copyfileobj(video.file, video_temp_file)
        video_path = video_temp_file.name
        frames = split_video_by_frames(video_path, skip, return_arrays=True)
        logger.info(f"{video_path}, total frames: {len(frames)}")
    
        if isinstance(embeddings, list):
            embeddings = list(map(nn.vector_store.get, embeddings))
            embeddings = np.hstack(embeddings)
            logger.info(f"stacked embeddings: {embeddings.shape}")

        if isinstance(yolo_results, list):
            yolo_results = list(map(nn.vector_store.get, yolo_results))
            _yolo_results_outputs = map(lambda x: x[0], yolo_results)
            _yolo_results_vectors = map(lambda x: x[1], yolo_results)
            yolo_results_outputs = sum(_yolo_results_outputs, [])
            yolo_results_vectors = np.hstack(_yolo_results_vectors)
            logger.info(f"stacked yolo results: {yolo_results_vectors.shape}")
            
            yolo_results = (yolo_results_outputs, yolo_results_vectors)
        
        embeddings_vector = embeddings or nn.run_embedder(frames)     
        if not embeddings:
            embeddings_id = uuid.uuid4().hex
            nn.vector_store.put(embeddings_id, embeddings_vector)

        yolo_outputs, yolo_vectors = yolo_results or nn.run_yolo(video_path)
        if not yolo_results:
            yolo_id = uuid.uuid4().hex
            nn.vector_store.put(yolo_id, (yolo_outputs, yolo_vectors))
        

        if isLast:
            logits = nn.run_classifier(embeddings_vector, yolo_vectors)
            classification = nn.predict(embeddings_vector, logits, logits=True)
            yolo_outputs = assign_to_rooms(classification, yolo_outputs)

            return {
                'stats': derive_statistics(list(yolo_outputs.values())),
                'output': yolo_outputs,
                'embeddings': embeddings_id,
                'yolo': yolo_id,
            }
        
        else:
            return {
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
        
        
