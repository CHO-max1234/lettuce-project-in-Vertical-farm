"""Processing utilities for bbox artifacts."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from pathlib import Path
import os
import random
import cv2
import numpy as np
from pipeline_paths import identity

def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    (ok, data) = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError('Invalid input or state in save (rejected condition: not ok)')
    temp = path.with_suffix('.writing')
    data.tofile(str(temp))
    os.replace(temp, path)

def choose_overlays(paths, limit=30, seed=260911):
    choices = sorted({str(Path(p)) for p in paths})
    return set(random.Random(seed).sample(choices, min(limit, len(choices))))

def write(source, rows, root, save_overlay=True, image=None, save_crops=True):
    (source, root) = (Path(source), Path(root))
    if image is None:
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Invalid input or state in write (rejected condition: image is None)')
    info = identity(source)
    overlay_path = root / 'overlays' / info['camera'] / info['bed_id'] / (source.stem + '.jpg')
    panel = image.copy() if save_overlay else None
    outputs = []
    (height, width) = image.shape[:2]
    for row in rows:
        row['bbox_overlay_path'] = str(overlay_path) if save_overlay else ''
        row['bbox_crop_path'] = ''
        row['bbox_crop_status'] = 'OUT_OF_FOV'
        if str(row.get('visibility', '')).upper() == 'OUT_OF_FOV':
            continue
        (x1, y1, x2, y2) = [int(row['search_' + axis]) for axis in ('x1', 'y1', 'x2', 'y2')]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Invalid input or state in write (rejected condition: not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height))')
        if save_crops:
            target = root / 'crops' / info['camera'] / info['bed_id'] / f"{source.stem}__{row['physical_slot']}_{row['record_id'][:8]}.png"
            save(target, image[y1:y2, x1:x2])
            outputs.append(target)
            row.update(bbox_crop_path=str(target), bbox_crop_status='SAVED')
        else:
            row['bbox_crop_status'] = 'NOT_SAVED_BY_POLICY'
        color = (0, 190, 255) if row.get('upstream_qc') else (0, 255, 0)
        if save_overlay:
            cv2.rectangle(panel, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            cv2.putText(panel, str(row['physical_slot']), (x1, max(15, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if save_overlay and (not rows):
        cv2.putText(panel, 'NO MAPPED SLOTS - REVIEW', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if save_overlay:
        save(overlay_path, panel)
        outputs.append(overlay_path)
    return outputs
