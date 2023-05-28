import os
import numpy as np
from ...utility.split_video import split_video_by_frames
from ...nn.yolov8.score import YOLOv8Objects, analyze_video
from ...nn.room_segmentation.embedding import RoomEmbedderPipeline
from ...nn.room_segmentation.predictor import RoomClassifier
from ...nn.room_segmentation.osg_predictions import predict
from ..stats import derive_statistics

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

embedder = RoomEmbedderPipeline(device=embedding_device)


classifier = RoomClassifier.from_pretrained(
    'ummagumm-a/samolet-room-classifier', 
    use_auth_token=os.environ['HF_AUTH_TOKEN'],
).to_device(embedding_device)


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

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"{video_path}, yolo finished")
    inputs = torch.Tensor(
        np.hstack((embeddings, np.array(vectors),))
    )
    logger.info(f"{video_path}, yolo data prepared")
    logits = classifier(inputs)
    logger.info(f"{video_path}, logits: {logits.shape}")
    classification = predict(embeddings, logits.detach().cpu(), logits=True)
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
                                          


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
    args = parser.parse_args()
    video = args.video
    
    _process_video_file_for_score_card(video)
        
        
