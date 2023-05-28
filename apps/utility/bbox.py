import numpy as np
import cv2

def iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) metric for two bounding boxes.
    
    Arguments:
    box1 -- Tuple of four numbers representing the coordinates of the first bounding box (X, Y, H, W).
    box2 -- Tuple of four numbers representing the coordinates of the second bounding box (X, Y, H, W).
    
    Returns:
    iou -- IoU metric value between the two bounding boxes.
    """
    # Extract the coordinates of the boxes
    x1, y1, h1, w1 = box1
    x2, y2, h2, w2 = box2

    # Calculate the coordinates of the intersection rectangle
    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + h1, x2 + h2)
    ymax = min(y1 + w1, y2 + w2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)

    # Calculate the area of each bounding box
    box1_area = (h1 + 1) * (w1 + 1)
    box2_area = (h2 + 1) * (w2 + 1)

    # Calculate the Union area by subtracting the intersection area
    # from the sum of the areas of both bounding boxes
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU metric
    iou = intersection_area / union_area

    return iou

def merge(box1, box2):
    """
    Merges two bounding boxes into a single bounding box.

    Arguments:
    box1 -- Tuple of four numbers representing the coordinates of the first bounding box (X, Y, H, W).
    box2 -- Tuple of four numbers representing the coordinates of the second bounding box (X, Y, H, W).

    Returns:
    merged_box -- Tuple of four numbers representing the merged bounding box (X, Y, H, W).
    """
    # Extract the coordinates of the boxes
    x1, y1, h1, w1 = box1
    x2, y2, h2, w2 = box2

    # Calculate the coordinates of the merged bounding box
    xmin = min(x1, x2)
    ymin = min(y1, y2)
    xmax = max(x1 + h1, x2 + h2)
    ymax = max(y1 + w1, y2 + w2)

    # Calculate the width and height of the merged bounding box
    merged_h = xmax - xmin
    merged_w = ymax - ymin

    # Create the merged bounding box
    merged_box = (xmin, ymin, merged_h, merged_w)

    return merged_box

def drop_overlaps(bboxes: list[tuple[int, int, int, int]], drop_thresh: float):
    bboxes_ = bboxes.copy()
    while True:
        to_merge = None
        for i, ib in enumerate(bboxes_):
            for j, jb in enumerate(bboxes_[(i + 1):], start=i + 1):
                if iou(ib, jb) > drop_thresh:
                   to_merge = (i, j)
                   break
      
        if to_merge:
            bboxes_.append(merge(bboxes_[to_merge[0]], bboxes_[to_merge[1]]))
            bboxes_.pop(to_merge[0])
            bboxes_.pop(to_merge[1])
        else:
            break

    return bboxes_
                