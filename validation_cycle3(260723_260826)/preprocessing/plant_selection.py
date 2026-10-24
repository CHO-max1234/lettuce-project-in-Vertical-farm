"""Plant selection calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
LAYOUT_CODE_ROOT = CYCLE.PROJECT_ROOT / 'preprocessing'
REFERENCE_ROOT = CYCLE.THIRD_ROOT / 'preprocessing'
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT / '4. 자리배치와 bbox'
TEST_ROOT = CYCLE.SOURCE_ROOT/'4. 자리배치와 bbox/testimages'
SAMPLE_MANIFEST = CYCLE.SOURCE_ROOT/'4. 자리배치와 bbox/자리배치_ver1/ver1/sample_manifest.json'
DEFAULT_OUTPUT = DATA_ROOT / 'bbox_ver2'
SHARED_INFERENCE_CACHE = DATA_ROOT / 'bbox_ver1' / '_inference_cache'
PYTHON_ENVIRONMENT = str(CYCLE.PYTHON)
(WHOLE_SIZE, LOCAL_SIZE, LOCAL_SCALE) = (1280, 512, 1.45)
VERSION = 'S4_BBOX_VER2_S3_FROZEN_2_2_HOLE_MASK_REVIEW_GUARD'
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)
def module(name,path):
    spec=resolved_spec(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

READER = module('s4_bbox_current_layout', LAYOUT_CODE_ROOT / 'layout_reader.py')
SUPPORT = module('s4_bbox_frozen_prediction_support', REFERENCE_ROOT / '261011_bbox_후보와본체기록.py')
SELECTOR = module('s4_bbox_frozen_selector_2_2', REFERENCE_ROOT / '261001_bbox_seg_selector_ver2.2.py')

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def true(value):
    return str(value).lower() in ('true', '1', 'yes')

def direct_hole(slot):
    if not true(slot.get('eval_hole_visible')):
        return None
    try:
        (x, y) = (float(slot['eval_hole_x']), float(slot['eval_hole_y']))
        (ax, ay) = (float(slot['anchor_x']), float(slot['anchor_y']))
        radius = max(6.0, 0.12 * min(float(slot['cell_width']), float(slot['cell_height'])))
        if math.hypot(x - ax, y - ay) <= radius:
            return (x, y)
    except (KeyError, ValueError, TypeError):
        pass
    return None

def adapt_anchors(frame, slots):
    supported_rows = {}
    for name in ('t', 'b'):
        group = [s for s in slots if s['row'] == name and READER.geometry_candidate(frame, s)]
        observed = [(float(s['anchor_x']), float(s['anchor_y']), direct_hole(s)) for s in group if direct_hole(s)]
        pitch = np.median([float(s['cell_width']) for s in group]) if group else 1.0
        if len(observed) >= 3:
            xs = [v[0] for v in observed]
            if max(xs) - min(xs) >= 1.5 * pitch:
                supported_rows[name] = (min(xs), max(xs), 0.5 * pitch, len(observed))
    (rows, unavailable) = ([], {})
    for original in slots:
        key = original['record_id']
        if not READER.geometry_candidate(frame, original):
            status = 'LAYOUT_HOLD_EXCLUDED' if frame['geometry_status'] == 'HOLD' or original.get('geometry_qc_status') == 'HOLD' else 'INVALID_NATIVE_GEOMETRY_EXCLUDED'
            unavailable[key] = dict(mask=None, status=status, reasons=[status], route='')
            continue
        s = dict(original)
        (s['eval_anchor_x'], s['eval_anchor_y']) = (float(s['anchor_x']), float(s['anchor_y']))
        s['physical_slot'] = s['slot_name']
        hole = direct_hole(original)
        existing = str(original.get('anchor_registration_status', '')).startswith('REGISTERED') and (not true(original.get('anchor_registration_review')))
        row = supported_rows.get(s['row'])
        row_supported = bool(row and row[0] - row[2] <= s['eval_anchor_x'] <= row[1] + row[2])
        if hole:
            (status, review, evidence) = ('REGISTERED_S4_OBSERVED_HOLE_FIXED_ANCHOR', False, 'INDEPENDENT_OBSERVED_HOLE_NEAR_APPROVED_ANCHOR')
        elif existing:
            (status, review, evidence) = (original['anchor_registration_status'], False, 'PRESERVED_UPSTREAM_INDEPENDENT_REGISTRATION')
        elif row_supported:
            (status, review, evidence) = ('REGISTERED_S4_ROW_SUPPORTED_FIXED_ANCHOR', False, 'APPROVED_ROW_WITH_3_OR_MORE_OBSERVED_HOLES')
        else:
            (status, review, evidence) = ('APPROVED_LAYOUT_REVIEW_PROPOSAL', True, 'USER_APPROVED_LAYOUT_REVIEW_FIXED_ANCHOR_NO_AUTOMATIC_OWNERSHIP_REMOVAL')
        s.update(anchor_registration_status=status, anchor_registration_review=review, bbox_anchor_status=status, bbox_anchor_review=review, bbox_anchor_evidence=evidence, bbox_anchor_hole_support_count=row[3] if row else int(hole is not None), eval_hole_visible=hole is not None)
        rows.append(s)
    return (rows, unavailable)

def reject_hole_like_masks(rows, source, decisions):
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 70)
    dark = hsv[:, :, 2] < 110
    for row in rows:
        d = decisions[row['record_id']]
        mask = d.get('mask')
        if mask is None or not mask.any():
            continue
        (green_fraction, dark_fraction) = (float(green[mask].mean()), float(dark[mask].mean()))
        d.update(selected_mask_bright_green_fraction=green_fraction, selected_mask_dark_fraction=dark_fraction)
        if row.get('eval_hole_visible') and green_fraction < 0.03 and (dark_fraction >= 0.8):
            d.update(mask=None, status='REVIEW_HOLE_LIKE_NONVEGETATIVE_MASK', rejected_mask_pixels=int(mask.sum()), review_required=True, reasons=d.get('reasons', []) + ['OBSERVED_HOLE_DARK_NONVEGETATIVE_SEGMENT_NOT_ACCEPTED_AS_PLANT'])

def palette(slot):
    ix = (int(slot[1:]) * 2 + (slot[0] == 'b')) % 24
    hsv = np.uint8([[[round(ix * 179 / 24), 220, 255]]])
    return tuple((int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))

def text(image, label, point, color, size=0.55, thickness=1):
    cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

def dashed_box(image, box, color, thickness=1):
    (x1, y1, x2, y2) = [round(v) for v in box]
    for x in range(x1, x2, 18):
        cv2.line(image, (x, y1), (min(x + 9, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + 9, x2), y2), color, thickness)
    for y in range(y1, y2, 18):
        cv2.line(image, (x1, y), (x1, min(y + 9, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + 9, y2)), color, thickness)

def thumbnail(image, width, height):
    ratio = min(width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
