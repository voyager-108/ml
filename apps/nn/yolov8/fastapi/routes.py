import os
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File
import hydra
from omegaconf import OmegaConf
from ..model import YOLOv8Wrapper
from ...serve import ServedModel
from .models import Prediction, BBox
from ....utility.split_video import split_video_by_frames


router = APIRouter(
    prefix="/yolo",
    tags=["yolo"],
)


def serve_yolo(cfg_path):
    global _ServedYOLO
    
    # Load OmegaConf config
    cfg = OmegaConf.load(cfg_path)
    cfg['pt_path'] = hydra.utils.to_absolute_path(cfg['pt_path'])

    _ServedYOLO = ServedModel(
        YOLOv8Wrapper,
        **cfg,
        enable_printing=True
    )



@router.post("/video")
async def detect_video(video: UploadFile = File(...)) -> list[Prediction]:
    global _ServedYOLO

    # with tempfile.TemporaryDirectory() as tempdir:
    #     shutil.copyfileobj(video.file, open(f"{tempdir}/video.mp4", "wb"))
    #     split_video_by_frames(f"{tempdir}/video.mp4", take_each_n=1, output_collection=f"{tempdir}/frames", verbose=False)
    #     frames_path = [
    #         (f"{tempdir}/frames/{frame}",) for frame in os.listdir(f"{tempdir}/frames")
    #     ]
    #     results = _ServedYOLO.run("run", frames_path, [{}] * len(frames_path))
    #     predictions = []
    #     for i, result in enumerate(results):
    #         result = result[0]
    #         predictions.append(
    #             Prediction(
    #                 frameNumber=i,
    #                 bboxes=[
    #                     BBox(
    #                         topLeft=(bbox.xyxy[0][0], bbox.xyxy[0][1]),
    #                         bottomRight=(bbox.xyxy[0][2], bbox.xyxy[0][3]),
    #                         className=result.names[int(bbox.cls.item())]
    #                     ) for bbox in result.boxes
    #                 ]
    #             )
    #         )

    # Get uploaded file suffix
    suffix = video.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}") as temp:
        shutil.copyfileobj(video.file, temp)
        frame_results = _ServedYOLO.run("run", [(temp.name,)], [{}])
        predictions = []
        for results in frame_results:
            for i, result in enumerate(results):
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

        
