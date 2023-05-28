from pydantic import BaseModel

class BBox(BaseModel):
    topLeft: list[float, float]
    bottomRight: tuple[float, float]
    className: str

class Prediction(BaseModel):
    frameNumber: int
    bboxes: list[BBox]
