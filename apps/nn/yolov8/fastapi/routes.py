import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File

from ..model import YOLOv8Wrapper
from ...serve import ServedModel
from .models import Prediction, BBox
from ....utility.split_video import split_video_by_frames
router = APIRouter(
    prefix="/yolo",
    tags=["yolo"],
)

_ServedYOLO = ServedModel(
    YOLOv8Wrapper,
    pt_path="./models/1200.pt",
    cpu=1,
    enable_printing=True
)

BATCH = 8

@router.post("/video")
async def detect_video(video: UploadFile = File(...)) -> list[Prediction]:
    with tempfile.TemporaryDirectory() as tempdir:
        shutil.copyfileobj(video.file, open(f"{tempdir}/video.mp4", "wb"))
        split_video_by_frames(f"{tempdir}/video.mp4", take_each_n=1, output_collection=f"{tempdir}/frames", verbose=False)
        frames_path = [
            f"{tempdir}/frames/{frame}" for frame in os.listdir(f"{tempdir}/frames")
        ]

        # Split on frames_path on batches
        batch_frames_path = []
        for i in range(0, len(frames_path), BATCH):
            if i+BATCH > len(frames_path):
                batch_frames_path.append((frames_path[i:],))
            else:
                batch_frames_path.append((frames_path[i:i+BATCH],))

        results_ = _ServedYOLO.run("run", batch_frames_path, [{}] * len(batch_frames_path))

        # Merge batched results
        results = []
        for result in results_:
            results.extend(result)
            

        predictions = []
        for i, result in enumerate(results):
            result = result[0]
            predictions.append(
                Prediction(
                    frameNumber=i,
                    bboxes=[
                        BBox(
                            topLeft=(bbox.xyxy[0][0], bbox.xyxy[0][1]),
                            bottomRight=(bbox.xyxy[0][2], bbox.xyxy[0][3]),
                            className=result.names[int(bbox.cls.item())]
                        ) for bbox in result.boxes
                    ]
                )
            )

    return predictions
        
