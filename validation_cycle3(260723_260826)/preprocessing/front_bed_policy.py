"""Front bed policy calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
STAGE_ROOT = CYCLE.PREPROCESS_OUTPUT / '1. ROI_BOX'
MODEL_ROOT = CYCLE.LOCAL_ARTIFACT_ROOT
RAW_FRONT = CYCLE.RAW_FRONT_ROOT
RAW_TOP = CYCLE.RAW_TOP_ROOT
OUTPUT_ROOT = STAGE_ROOT / 'ROI_BOX'
GPU_PYTHON = CYCLE.PYTHON
TRAIN_COMPLETE = CYCLE.LOCAL_ARTIFACT_ROOT / 'training_completion.json'
POLICY_PATH = CYCLE.LOCAL_ARTIFACT_ROOT / 'front_inference_policy.json'
POLICIES = [dict(name='rect896', imgsz=896, rect=True)]
SHAPE_GATE = dict(min_width=0.85, min_height=0.12, max_height=0.44, min_cy=0.32, max_cy=0.86)
DEFAULT_CROP_PROFILES = {'early': dict(up=0.0, down=0.0), 'middle': dict(up=0.0, down=0.0), 'tall': dict(up=0.0, down=0.0)}
import argparse, csv, hashlib, json, math, os, random, re, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
cv2.setNumThreads(1)

def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        keys = list(dict.fromkeys((k for row in rows for k in row)))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for (k, v) in r.items()} for r in rows])

def regime(row):
    return 'early_CAM2' if row['date'] <= '20260820' else 'late_source_CAM0_as_CAM2'

def crop_profile_key(date):
    return 'early' if date < '20260801' else 'middle' if date < '20260811' else 'tall'

def discover():
    rows = []
    seen = set()
    for (root, source_cam, lo, hi) in [(RAW_FRONT, 'cam2', '20260724', '20260820'), (RAW_TOP, 'cam0', '20260821', '20260826')]:
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not re.fullmatch('\\d{6}', folder.name):
                continue
            date = '20' + folder.name
            if not lo <= date <= hi:
                continue
            for p in sorted(folder.glob('*.jpg')):
                m = re.fullmatch('(bed\\d+)_(20\\d{6})_(\\d{6})_(cam\\d)\\.jpg', p.name, re.I)
                if not m or m[4].lower() != source_cam:
                    continue
                assert m[2] == date
                corrected = f'{m[1]}_{date}_{m[3]}_cam2.jpg'
                target = OUTPUT_ROOT / 'cam2' / folder.name / corrected
                key = str(target).lower()
                assert key not in seen, 'Input contract failed: key not in seen'
                seen.add(key)
                stat = p.stat()
                rows.append(dict(source_path=str(p), source_filename=p.name, source_camera=source_cam.upper(), date=date, camera='CAM2', corrected_camera='CAM2', camera_corrected=source_cam != 'cam2', filename=corrected, target_path=str(target), source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns))
    assert len(rows) == 16843 and Counter((regime(r) for r in rows)) == {'early_CAM2': 13969, 'late_source_CAM0_as_CAM2': 2874}
    return rows

def read_source(row):
    out = dict(row)
    try:
        p = Path(row.get('sample_image_path', row.get('image_path', row['source_path'])))
        data = p.read_bytes()
        im = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError('Invalid input or execution contract')
        out.update(source_sha256=hashlib.sha256(data).hexdigest(), image_width=im.shape[1], image_height=im.shape[0])
        if min(im.shape[:2]) < 100:
            out.update(status='INVALID_LOW_RESOLUTION', reason='저해상도')
            return (out, None)
        thumb = cv2.resize(im, (64, 48))
        out['image_mean'] = float(thumb.mean())
        out['image_std'] = float(thumb.std())
        if thumb.mean() < 3 or thumb.std() < 1:
            out.update(status='INVALID_BLACK_OR_CONSTANT', reason='검정/상수 영상')
            return (out, None)
        return (out, im)
    except Exception as e:
        out.update(status='ERROR_READ', reason=f'{type(e).__name__}: {e}')
        return (out, None)

def extract_detections(result):
    found = []
    if result.boxes is None:
        return found
    for (i, box) in enumerate(result.boxes):
        if int(box.cls.item()) != 0:
            continue
        bbox = [float(x) for x in box.xyxy[0].detach().cpu().tolist()]
        polygon = result.masks.xy[i].tolist() if result.masks is not None and i < len(result.masks.xy) else None
        found.append(dict(confidence=float(box.conf.item()), bbox=bbox, polygon=polygon))
    return found

def box_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0

def select_target(detections, w, h, conf_accept=0.45):
    eligible = []
    for d in detections:
        (x1, y1, x2, y2) = d['bbox']
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cy = (y1 + y2) / 2 / h
        if d['confidence'] >= conf_accept and SHAPE_GATE['min_width'] <= bw <= 1.01 and (SHAPE_GATE['min_height'] <= bh <= SHAPE_GATE['max_height']) and (SHAPE_GATE['min_cy'] <= cy <= SHAPE_GATE['max_cy']):
            eligible.append(d)
    eligible.sort(key=lambda d: d['confidence'], reverse=True)
    if len(eligible) > 1 and all((box_iou(eligible[0]['bbox'], d['bbox']) >= 0.8 for d in eligible[1:])):
        eligible = eligible[:1]
    if len(eligible) == 1:
        return (eligible[0], 'TARGET_CANDIDATE')
    return (None, 'REVIEW_MULTIPLE_TARGETS' if len(eligible) > 1 else 'NO_VALID_TARGET')

def predict_rows(model, rows, policy, batch_size=4, desc='추론'):
    out = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in tqdm(range(0, len(rows), 16), desc=desc, unit='chunk'):
            loaded = list(pool.map(read_source, rows[start:start + 16]))
            groups = defaultdict(list)
            for (row, im) in loaded:
                if im is None:
                    out.append({**row, 'detections': []})
                else:
                    groups[im.shape].append((row, im))
            for (shape, items) in groups.items():
                for j in range(0, len(items), batch_size):
                    batch = items[j:j + batch_size]
                    results = model.predict([im for (_, im) in batch], imgsz=policy['imgsz'], rect=policy['rect'], conf=0.2, iou=0.7, max_det=5, device=0, half=True, verbose=False)
                    assert len(results) == len(batch)
                    for ((row, im), r) in zip(batch, results):
                        assert r.orig_img.shape == im.shape and np.array_equal(r.orig_img, im)
                        out.append({**row, 'detections': extract_detections(r), 'prediction_policy': policy['name']})
    assert len(out) == len(rows)
    return sorted(out, key=lambda r: (r['date'], r.get('filename', r.get('source_filename', ''))))

def expanded_box(detected, w, h, date, profiles):
    p = profiles[crop_profile_key(date)]
    if p.get('up', 0) != 0 or p.get('down', 0) != 0:
        raise RuntimeError('Invalid input or execution contract')
    (x1, y1, x2, y2) = detected
    box = [max(0, round(x1)), max(0, round(y1)), min(w, round(x2)), min(h, round(y2))]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('Invalid input or execution contract')
    return box

def load_model():
    from ultralytics import YOLO
    complete = json.loads(TRAIN_COMPLETE.read_text(encoding='utf-8'))
    assert complete['status'] == 'COMPLETE'
    assert sha(CYCLE.BED_FRONT_MODEL) == complete['best_sha256']
    return (YOLO(str(CYCLE.BED_FRONT_MODEL)), dict(complete, best_model=str(CYCLE.BED_FRONT_MODEL)))
