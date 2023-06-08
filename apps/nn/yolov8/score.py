from collections import defaultdict
import os
import numpy as np
import ultralytics
import tqdm
from pydantic import BaseModel


class YOLOv8Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class YOLOv8Objects(BaseModel):
    id: int | None = None
    className: str | None = None
    framesObserved: list[int] | None = []
    boxes: list | None = []
    framesSpan: tuple[int, int] | None = (np.inf, -np.inf)
    roomClass: str | None = None



def analyze_video(
        model: ultralytics.YOLO,
        input_stream: str,
        *args,
        **kwargs
) -> tuple[dict[int, YOLOv8Objects], list[np.ndarray]]:
    results = track_video(model, input_stream, *args, **kwargs)
    return postprocess_results(results)


def postprocess_results(results):
    objects = defaultdict(lambda: YOLOv8Objects())
    frames = []

    for frame, result in enumerate(results):
        classes = result.names
        vector = np.zeros(len(classes))
        for box in result.boxes:
            if box.id:
                objects[int(box.id)].id = int(box.id)
                objects[int(box.id)].className = classes[int(box.cls)]
                objects[int(box.id)].framesObserved.append(frame)
                objects[int(box.id)].boxes.append(
                    YOLOv8Box(
                        x1=box.xyxy[0][0],
                        y1=box.xyxy[0][1],
                        x2=box.xyxy[0][2],
                        y2=box.xyxy[0][3],
                        confidence=box.conf,
                    )
                )
                objects[int(box.id)].framesSpan = (
                    min(objects[int(box.id)].framesSpan[0], frame),
                    max(objects[int(box.id)].framesSpan[1], frame),
                )
            vector[int(box.cls)] += 1
        frames.append(vector)

    return dict(objects), frames


def predict_video(model, input_stream, *args, **kwargs):
    return model.predict(
        input_stream,
        *args,
        **kwargs,
    )


def track_video(model, input_stream, *args, **kwargs):
    return model.track(
        input_stream,
        *args,
        tracker=os.path.join(os.path.dirname(__file__), 'trackers', 'tracker.yaml'),
        **kwargs,
    )

