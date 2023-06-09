from collections import defaultdict
import os
from typing import Any
import numpy as np
import ultralytics
from pydantic import BaseModel

from ...permanent import ROOM_CLASSES


class YOLOv8Box(BaseModel):
    """YOLO box representation"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class YOLOv8Objects(BaseModel):
    """Objects detected by YOLOv8"""

    id: int | None = None  # The id of the object (according to tracker)
    className: str | None = None # The class name of the object
    framesObserved: list[int] | None = [] # The frames in which the object was observed
    boxes: list | None = [] # The boxes corresponding to each frame
    framesSpan: tuple[int, int] | None = (np.inf, -np.inf) # The first and last frame in which the object was observed
    roomClass: str | None = None # The room class of the object (might not be present)



def analyze_video(
        model: ultralytics.YOLO,
        input_stream: str,
        *args,
        **kwargs
) -> tuple[dict[int, YOLOv8Objects], list[np.ndarray]]:
    """Analyze a video using YOLOv8.

    Runs a YOLOv8 tracker and post-processes the results, creating
    a dictionary of YOLOv8Object instances and a list of frames.

    (Outdated). The function is not used since tracking in batches
    is not supported by the tracker. So, we use the predict function
    instead, and then use tracker to track the objects (outside YOLO).

    Args:
        model (ultralytics.YOLO): The YOLOv8 model.
        input_stream (str): The path to the input video.

    Returns:
        tuple[dict[int, YOLOv8Objects], list[np.ndarray]]: A tuple
            containing a dictionary (id -> object) of YOLOv8Objects and a list of
            frames.
    """

    results = track_video(model, input_stream, *args, **kwargs)
    return postprocess_results(results)


def postprocess_results(results):
    """Postprocess the results of YOLOv8.

    Args:
        results: YOLO.predict outputs (https://docs.ultralytics.com/modes/predict/#working-with-results)

    Returns:
        tuple[dict[int, YOLOv8Objects], list[np.ndarray]]: A tuple
            containing a dictionary (id -> object) of YOLOv8Objects and a list of
            frames.
    """

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
    """Predict a video using YOLOv8.predict function."""
    return model.predict(
        input_stream,
        *args,
        **kwargs,
    )


def track_video(model, input_stream, *args, **kwargs):
    """Track a video using YOLOv8.track function."""
    return model.track(
        input_stream,
        *args,
        tracker=os.path.join(os.path.dirname(__file__), 'trackers', 'tracker.yaml'),
        **kwargs,
    )


def filter_objects(yolo_results_dict: dict[YOLOv8Objects]):
    """Apply postprocessing filters to the YOLOv8Objects dictionary."""
    # Filter objects that appears in less than 5 frames (1 second)

    yolo_results_dict = {
        k: v for k, v in yolo_results_dict.items() if len(v.framesObserved) > 5
    }

    return yolo_results_dict


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


def assign_to_rooms(rooms: list[int], outputs: dict[Any, list[YOLOv8Objects]], skip=5) -> list[YOLOv8Objects]:
    """Given a list of room ids and a list of YOLOv8Objects, assign each object to a room.

    Args:
        rooms (list[int]): A list of room ids.
        outputs (list[YOLOv8Objects]): A list of YOLOv8Objects.
        skip (int, optional): The number of frames skipped between each room. Defaults to 5.

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
        obj.roomClass = ROOM_CLASSES[int(room)]
    
    return outputs