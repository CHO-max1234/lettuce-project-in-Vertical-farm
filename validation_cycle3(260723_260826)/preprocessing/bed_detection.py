"""Bed detection calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
VERSION = 'season4_bedseg_pilot_v1'
OUTPUT_VERSION = 1
BATCH_SIZE = 4
CHUNK_SIZE = 8
IMAGE_SIZE = 1024
PREDICT_CONF = 0.25
NMS_IOU = 0.7
DEVICE = 0
import argparse
import csv
import gc
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while (chunk := stream.read(1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'  : {path}')
    return image

def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (ok, encoded) = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(path)
    encoded.tofile(str(path))

def baseline_reason(detections, model_name, width, height):
    if not detections:
        return 'no_detection'
    first = detections[0]
    if first['confidence'] < 0.45:
        return 'low_confidence'
    if len(detections) > 1 and first['confidence'] - detections[1]['confidence'] < 0.1:
        return 'ambiguous_multiple_detections'
    (x1, y1, x2, y2) = first['crop_bbox']
    (wr, hr) = ((x2 - x1) / width, (y2 - y1) / height)
    area = wr * hr
    if model_name == 'top_model':
        if not 0.1 <= area <= 0.8:
            return 'crop_area_out_of_range'
        if wr < 0.4 or hr < 0.1:
            return 'crop_shape_out_of_range'
    else:
        if not 0.08 <= area <= 0.4:
            return 'crop_area_out_of_range'
        if wr < 0.85:
            return 'crop_width_too_small'
        if not 0.1 <= hr <= 0.35:
            return 'crop_height_out_of_range'
        if not 0.42 <= (y1 + y2) / 2 / height <= 0.88:
            return 'crop_vertical_position_out_of_range'
    return ''

def predictions(result, image):
    (height, width) = image.shape[:2]
    if result.boxes is None or result.masks is None:
        return []
    if len(result.boxes) != len(result.masks.xy):
        raise ValueError('Invalid input or execution contract')
    detections = []
    for (index, polygon) in enumerate(result.masks.xy):
        polygon = np.asarray(polygon, dtype=float)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError('invalid_polygon')
        (x1, y1) = (max(0, math.floor(polygon[:, 0].min())), max(0, math.floor(polygon[:, 1].min())))
        (x2, y2) = (min(width, math.ceil(polygon[:, 0].max())), min(height, math.ceil(polygon[:, 1].max())))
        detections.append(dict(confidence=float(result.boxes.conf[index].item()), class_id=int(result.boxes.cls[index].item()), crop_bbox=[x1, y1, x2, y2], polygon=polygon.round(2).tolist()))
    return sorted(detections, key=lambda d: -d['confidence'])
