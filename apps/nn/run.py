import os
import torch
import multiprocessing as mp
import numpy as np

from ultralytics import YOLO

from .room_segmentation.embedding import RoomEmbedderPipeline
from .room_segmentation.predictor import RoomClassifier
from .room_segmentation.osg_predictions import predict
from .yolov8.score import analyze_video, predict_video, YOLOv8Objects
from ..object.store import storage

USE_CUDA_EMBEDDINGS = 'EMBEDDINGS_CUDA' in os.environ and os.environ['EMBEDDINGS_CUDA'] == 'TRUE'
USE_CUDA_YOLO = 'YOLO_CUDA' in os.environ and os.environ['YOLO_CUDA'] == 'TRUE'
YOLO_DEVICE = 0 if torch.cuda.is_available() and USE_CUDA_YOLO else 'cpu'
EMBEDDING_DEVICE = 'cuda:0' if torch.cuda.is_available() and USE_CUDA_EMBEDDINGS else 'cpu'
NUM_CPU = (1 + (mp.cpu_count() - 1) // 2) if 'NUM_CPU' not in os.environ else int(os.environ['NUM_CPU'])

assert os.path.exists(os.environ['YOLO_PT_PATH']), f'YOLO weights not found at {os.environ["YOLO_PT_PATH"]}'

yolo = YOLO(os.environ['YOLO_PT_PATH'], task='detect')
embedder = RoomEmbedderPipeline(device=EMBEDDING_DEVICE).eval()
classifier = RoomClassifier.from_pretrained(
    'ummagumm-a/samolet-room-classifier',
    use_auth_token=os.environ['HF_AUTH_TOKEN'],
    device=EMBEDDING_DEVICE,
).to_device(EMBEDDING_DEVICE).eval()


def _run_yolo(source: str, raw=False) -> list[YOLOv8Objects] | list:
    """Run YOLO on a video source.
    
    Args:
        source (str): Path to video source.
        raw (bool, optional): Whether to return raw YOLO results. Defaults to False.
    
    Returns:
        list[YOLOv8Objects] or list: YOLO results.
    """

    if not raw:
        # Returns postprocessed results
        return analyze_video(  
            yolo,               
            source,
            vid_stride=5,
            verbose=False,
            stream=True,
            device=YOLO_DEVICE
        )
    else:
        # Returns collected raw results
        return [*predict_video(  
            yolo,
            source,
            vid_stride=5,
            verbose=False,
            stream=True,
            device=YOLO_DEVICE        
        )]

def run_yolo(source: str, raw=False):
    """(Wrapper) Run YOLO on a video source.
    
    Args:
        source (str): Path to video source.
        raw (bool, optional): Whether to return raw YOLO results. Defaults to False.

    Returns:
        list[YOLOv8Objects] or list: YOLO results.

    """
    return _run_yolo(source, raw=raw)


def _run_embedder(frames: list[np.ndarray]) -> np.ndarray:
    """Run the room segmentation embedding pipeline on a list of frames.

    Args:
        frames (list[np.ndarray]): List of frames to run the pipeline on.

    Returns:
        np.ndarray: Embeddings of the frames.
    """

    global embedder
    return embedder(frames)


def run_embedder(frames: list[np.ndarray]) -> np.ndarray:
    """(Wrapper) Run the room segmentation embedding pipeline on a list of frames.

    Args:
        frames (list[np.ndarray]): List of frames to run the pipeline on.

    Returns:
        np.ndarray: Embeddings of the frames.
    """
    return _run_embedder(frames)


def _run_classifier(embeddings: np.ndarray, yolo_vectors: list[np.ndarray]):
    """Run the room segmentation classifier on a list of embeddings and YOLO results.

    Args:
        embeddings (torch.Tensor): Embeddings of the frames.
        yolo_vectors (list[np.ndarray]): List of YOLO results.  

    Returns:
        torch.Tensor: Predictions of the classifier.
    """
    inputs = torch.Tensor(
        np.hstack((embeddings, np.array(yolo_vectors),))
    )
    global classifier
    with torch.no_grad():
        return classifier(inputs)


def run_classifier(embeddings: torch.Tensor, yolo_vectors: list[np.ndarray]):
    """(Wrapper) Run the room segmentation classifier on a list of embeddings and YOLO results.

    Args:
        embeddings (torch.Tensor): Embeddings of the frames.
        yolo_vectors (list[np.ndarray]): List of YOLO results.

    Returns:
        torch.Tensor: Predictions of the classifier.
    """
    with torch.no_grad():
        return _run_classifier(embeddings, yolo_vectors)


def run_predict(embeddings, classifier_output, logits) -> list[np.int32]:
    """Run OSG postprocessing on a list of embeddings, classifier outputs, and logits.

    Args:
        embeddings (torch.Tensor): Embeddings of the frames.
        classifier_output (torch.Tensor): Predictions of the classifier.
        logits (torch.Tensor): Logits of the classifier.

    Returns:
        list: List of OSG predictions.
    """

    with torch.no_grad():
        return predict(embeddings, classifier_output, logits)

