import logging
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
from ultralytics import YOLO
from .room_segmentation.embedding import RoomEmbedderPipeline
from .room_segmentation.predictor import RoomClassifier
from .room_segmentation.osg_predictions import predict
from .yolov8.score import analyze_video
import signal
from ..vector.yandex import YandexS3VectorStore

USE_CUDA_EMBEDDINGS = 'EMBEDDINGS_CUDA' in os.environ and os.environ['EMBEDDINGS_CUDA'] == 'TRUE'
USE_CUDA_YOLO = 'YOLO_CUDA' in os.environ and os.environ['YOLO_CUDA'] == 'TRUE'
YOLO_DEVICE = 0 if torch.cuda.is_available() and USE_CUDA_YOLO else 'cpu'
EMBEDDING_DEVICE = 'cuda:0' if torch.cuda.is_available() and USE_CUDA_EMBEDDINGS else 'cpu'
NUM_CPU = (1 + (mp.cpu_count() - 1) // 2) if 'NUM_CPU' not in os.environ else int(os.environ['NUM_CPU'])

assert os.path.exists(os.environ['YOLO_PT_PATH']), f'YOLO weights not found at {os.environ["YOLO_PT_PATH"]}'

vector_store = YandexS3VectorStore(os.environ['VECTOR_STORE_BUCKET'])

yolo = YOLO(os.environ['YOLO_PT_PATH'], task='detect')
embedder = RoomEmbedderPipeline(device=EMBEDDING_DEVICE).eval()
classifier = RoomClassifier.from_pretrained(
    'ummagumm-a/samolet-room-classifier',
    use_auth_token=os.environ['HF_AUTH_TOKEN'],
    device=EMBEDDING_DEVICE,
).to_device(EMBEDDING_DEVICE).eval()


def _run_yolo(source: str):
    global yolo, logger
    return analyze_video(
        yolo,
        source,
        vid_stride=5,
        verbose=False,
        stream=True,
        device=YOLO_DEVICE
    )


def run_yolo(source: str):
    return _run_yolo(source)


def _run_embedder(frames: list[np.ndarray]):
    global embedder, logger
    return embedder(frames)


def run_embedder(frames: list[np.ndarray]):
    return _run_embedder(frames)


def _run_classifier(embeddings: torch.Tensor, yolo_vectors: list[np.ndarray]):
    inputs = torch.Tensor(
        np.hstack((embeddings, np.array(yolo_vectors),))
    )
    global classifier, logger
    return classifier(embeddings, inputs)


def run_classifier(embeddings: torch.Tensor, yolo_vectors: list[np.ndarray]):
    return _run_classifier(embeddings, yolo_vectors)


def run_predict(embeddings, classifier_output, logits):
    return predict(embeddings, classifier_output, logits)

