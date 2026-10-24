"""Top bed policy calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
import csv
import argparse
import json
import os
import runpy
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
os.environ['YOLO_AUTOINSTALL'] = 'false'
base = run_source(str(CODE_ROOT / 'bed_detection.py'))
YOLO = base['YOLO']

def transform(image, kind):
    if kind == 'clahe':
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if kind.startswith('gamma'):
        power = 1.4 if kind == 'gamma14' else 0.7
        lut = np.asarray([255 * (i / 255) ** power for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, lut)
    if kind == 'flip':
        return cv2.flip(image, 1)
    if kind == 'no_floor':
        return image[:int(image.shape[0] * 0.82)].copy()
    if kind.startswith('window'):
        (a, b) = {'window1075': (0.1, 0.75), 'window2085': (0.2, 0.85), 'window0070': (0, 0.7)}[kind]
        return image[int(image.shape[0] * a):int(image.shape[0] * b)].copy()
    if kind == 'compress_y':
        return cv2.resize(image, (image.shape[1], int(image.shape[0] * 0.8)))
    if kind.startswith('rotate'):
        angle = float(kind[6:])
        (h, w) = image.shape[:2]
        return cv2.warpAffine(image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return image

def detections(result, original, kind):
    out = base['predictions'](result, original)
    if kind == 'flip':
        width = original.shape[1]
        for d in out:
            (x1, y1, x2, y2) = d['crop_bbox']
            d['crop_bbox'] = [width - x2, y1, width - x1, y2]
            d['polygon'] = [[width - x, y] for (x, y) in d['polygon']]
    if kind.startswith('window') or kind == 'compress_y' or kind.startswith('rotate'):
        (h, w) = original.shape[:2]
        for d in out:
            p = np.asarray(d['polygon'], dtype=float)
            if kind.startswith('window'):
                a = {'window1075': 0.1, 'window2085': 0.2, 'window0070': 0}[kind]
                p[:, 1] += int(h * a)
            elif kind == 'compress_y':
                p[:, 1] *= h / int(h * 0.8)
            else:
                m = cv2.invertAffineTransform(cv2.getRotationMatrix2D((w / 2, h / 2), float(kind[6:]), 1))
                p = cv2.transform(p[None, :, :], m)[0]
            p[:, 0] = np.clip(p[:, 0], 0, w)
            p[:, 1] = np.clip(p[:, 1], 0, h)
            d['polygon'] = p.round(2).tolist()
            d['crop_bbox'] = [int(np.floor(p[:, 0].min())), int(np.floor(p[:, 1].min())), int(np.ceil(p[:, 0].max())), int(np.ceil(p[:, 1].max()))]
    return out

def acceptable(d, width, height):
    (x1, y1, x2, y2) = d['crop_bbox']
    wr = (x2 - x1) / width
    hr = (y2 - y1) / height
    return d['confidence'] >= 0.45 and 0.1 <= wr * hr <= 0.8 and (wr >= 0.4) and (hr >= 0.1)
