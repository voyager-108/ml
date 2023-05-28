import os
import numpy as np
from ...utility.split_video import split_video_by_frames
from ...nn.yolov8.score import YOLOv8Objects, analyze_video
from ...nn.room_segmentation.embedding import RoomEmbedderPipeline
from ...nn.room_segmentation.predictor import RoomClassifier
from ...nn.room_segmentation.osg_predictions import predict
from ..score_card import derive_statistics

from ultralytics import YOLO
from fastapi import APIRouter, File, UploadFile
from concurrent import futures
import tempfile
import torch 
import shutil
import logging
import gc

skip = 5
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("ServerApplication")

score_card_router = APIRouter(
    prefix='/score-card'
)

room_classes = ['bathroom', 'corridor', 'kitchen', 'livingroom', 'common_area', 'null']
model = YOLO('models/best.pt', task='detect')
embedder = RoomEmbedderPipeline(device=device)
classifier = RoomClassifier.from_pretrained(
    'ummagumm-a/samolet-room-classifier', 
    use_auth_token=os.environ['HF_AUTH_TOKEN'],
).to_device(device)
# if torch.cuda.is_available():
#     print("Transfering models to GPU...")
#     classifier = classifier.cuda()


    
def calculate_iou(range1, range2):
    x, y = range1
    p, q = range2
    intersection = max(0, min(y, q) - max(x, p))
    union = max(y, q) - min(x, p)
    iou = intersection / union
    return iou

def assign_to_rooms(rooms: list[int], outputs: list[YOLOv8Objects]):
    """Given a list of room ids and a list of YOLOv8Objects, assign each object to a room.
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


pool = futures.ProcessPoolExecutor(max_workers=3)

def _process_video_file_for_score_card(video_path):
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
    )

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"{video_path}, yolo finished")
    # vectors = vectors[::skip]
    # if len(frames) % skip != 0:
    #     vectors = vectors[:-1]
    inputs = torch.Tensor(
        np.hstack((embeddings, np.array(vectors),))
    )
    
    logits = classifier(inputs)
    # print("Embeddings:", embeddings.shape)
    # print("Logits:", logits.shape)
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
    with tempfile.NamedTemporaryFile(prefix=video.filename, suffix=f'.{video_format}') as video_temp_file:
        shutil.copyfileobj(video.file, video_temp_file)
        return _process_video_file_for_score_card(video_temp_file.name)
                                          


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str)
    args = parser.parse_args()
    video = args.video
    
    _process_video_file_for_score_card(video)
        
        
