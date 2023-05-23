import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File

from ..model import YOLOv8Wrapper
from ...serve import ServedModel

router = APIRouter(
    prefix="/yolo",
    tags=["yolo"],
)

_ServedYOLO = ServedModel(
    YOLOv8Wrapper,
    pt_path="./models/1200.pt",
    gpu=1,
    enable_printing=True
)

@router.post("/video")
async def detect_video(video: UploadFile = File(...)):
    result = []
    # Get file format
    fmt = video.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as temp:
        shutil.copyfileobj(video.file, temp)
        video = temp.name
        result = _ServedYOLO("run", video)
    return result
        
