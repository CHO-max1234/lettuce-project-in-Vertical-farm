"""Segment CAM0/CAM1 bed regions over the recorded cycle-3 observation dates. Paths and shared models are configured in settings.py."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from runtime import yolo_device
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
RAW_ROOT = CYCLE.RAW_TOP_ROOT
BED_ROOT = CYCLE.BED_OUTPUT
MODEL_PATH = CYCLE.BED_TOP_MODEL
GPU_PYTHON = CYCLE.PYTHON
LOG_ROOT = CYCLE.OUTPUT_ROOT / 'preprocessing_logs' / 'bed_top'
TRIAL_DATES = frozenset([f'2605{day:02d}' for day in range(8, 28)] + [f'2606{day:02d}' for day in range(12, 21)])
CAMERAS = ('cam0', 'cam1')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
NAME_RE = re.compile('^(?:bed\\d+_)?(?P<date>20\\d{6})_\\d{6}_(?P<camera>cam[01])(?:_[a-z0-9]+)*$', re.IGNORECASE)
IMGSZ = 1024
BATCH_SIZE = 8
CHUNK_SIZE = 128
PREDICT_CONF = 0.25
NMS_IOU = 0.7
MIN_ACCEPT_CONF = 0.45
AMBIGUOUS_CONF_GAP = 0.1
MIN_CROP_AREA_RATIO = 0.1
MAX_CROP_AREA_RATIO = 0.8
MIN_CROP_WIDTH_RATIO = 0.4
MIN_CROP_HEIGHT_RATIO = 0.1
JPEG_QUALITY = 95
LOG_FIELDS = ['source_path', 'relative_path', 'camera', 'date_folder', 'target_path', 'status', 'reason', 'detection_count', 'confidence', 'second_confidence', 'polygon_area_ratio', 'crop_area_ratio', 'x1', 'y1', 'x2', 'y2', 'image_width', 'image_height']
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda : stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def camera_and_date(path: Path) -> tuple[str, str]:
    relative = path.relative_to(RAW_ROOT)
    if len(relative.parts) < 2 or relative.parts[0] not in TRIAL_DATES:
        raise ValueError(f'Invalid input or state in camera_and_date (len(relative.parts) < 2 or relative.parts[0] not in TRIAL_DATES): {path} ')
    match = NAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f'Invalid input or state in camera_and_date (match is None): {path} ')
    if match.group('date') != '20' + relative.parts[0]:
        raise ValueError(f"Invalid input or state in camera_and_date (match.group('date') != '20' + relative.parts[0]): {path} ")
    return (match.group('camera').lower(), relative.parts[0])

def target_for(path: Path) -> Path:
    (camera, _) = camera_and_date(path)
    return BED_ROOT / camera / path.relative_to(RAW_ROOT)

def collect_sources() -> list[Path]:
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f'Invalid input or state in collect_sources (not RAW_ROOT.is_dir()): {RAW_ROOT} ')
    selected: list[Path] = []
    invalid: list[str] = []
    for date in sorted(TRIAL_DATES):
        folder = RAW_ROOT / date
        if not folder.is_dir():
            continue
        for path in folder.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                (camera, _) = camera_and_date(path)
            except ValueError as error:
                invalid.append(str(error))
                continue
            if camera in CAMERAS:
                selected.append(path)
    if invalid:
        raise ValueError(f'Invalid input or state in collect_sources (invalid): {len(invalid)} ')
    selected.sort(key=lambda p: str(p.relative_to(RAW_ROOT)).lower())
    if not selected:
        raise RuntimeError('Invalid input or state in collect_sources (not selected)')
    targets = [str(target_for(path)).casefold() for path in selected]
    if len(set(targets)) != len(targets):
        raise RuntimeError('Invalid input or state in collect_sources (len(set(targets)) != len(targets))')
    return selected

def preflight(paths: list[Path]) -> dict[str, object]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Invalid input or state in preflight (not MODEL_PATH.is_file()): {MODEL_PATH} ')
    if RAW_ROOT.resolve() == BED_ROOT.resolve() or RAW_ROOT.resolve() in BED_ROOT.resolve().parents:
        raise RuntimeError('Invalid input or state in preflight (RAW_ROOT.resolve() == BED_ROOT.resolve() or RAW_ROOT.resolve() in BED_ROOT.resolve().parents)')
    camera_counts: Counter[str] = Counter()
    date_counts: Counter[str] = Counter()
    for path in paths:
        (camera, date) = camera_and_date(path)
        camera_counts[camera] += 1
        date_counts[date] += 1
    info = {'status': 'READ_ONLY_PREFLIGHT_PASS', 'python': sys.executable, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'source_root': str(RAW_ROOT), 'output_root': str(BED_ROOT), 'model': str(MODEL_PATH), 'source_count': len(paths), 'camera_counts': dict(camera_counts), 'date_counts': dict(sorted(date_counts.items())), 'first_target': str(target_for(paths[0])), 'last_target': str(target_for(paths[-1]))}
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return info

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid input or state in read_image (image is None): {path} ')
    return image

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = '.png' if path.suffix.lower() == '.png' else '.jpg'
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3] if ext == '.png' else [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    (success, encoded) = cv2.imencode(ext, image, params)
    if not success:
        raise ValueError(f'Invalid input or state in write_image (not success): {path} ')
    partial = path.with_name(path.name + '.partial')
    encoded.tofile(str(partial))
    os.replace(partial, path)

def base_row(source: Path) -> dict[str, object]:
    (camera, date) = camera_and_date(source)
    row: dict[str, object] = {field: '' for field in LOG_FIELDS}
    row.update(source_path=str(source), relative_path=str(source.relative_to(RAW_ROOT)), camera=camera, date_folder=date, target_path=str(target_for(source)), detection_count=0)
    return row

def evaluate_prediction(result, image: np.ndarray, source: Path) -> tuple[dict[str, object], np.ndarray | None]:
    row = base_row(source)
    (height, width) = image.shape[:2]
    row.update(image_width=width, image_height=height)
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        row.update(status='failed', reason='no_detection')
        return (row, None)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    polygons = result.masks.xy
    if len(confidences) != len(polygons):
        row.update(status='failed', reason='box_mask_count_mismatch', detection_count=len(confidences))
        return (row, None)
    order = np.argsort(-confidences)
    best = int(order[0])
    confidence = float(confidences[best])
    second = float(confidences[order[1]]) if len(order) > 1 else None
    polygon = np.asarray(polygons[best], dtype=np.float32)
    row.update(detection_count=len(confidences), confidence=round(confidence, 6), second_confidence='' if second is None else round(second, 6))
    if confidence < MIN_ACCEPT_CONF:
        row.update(status='failed', reason='low_confidence')
        return (row, None)
    if second is not None and confidence - second < AMBIGUOUS_CONF_GAP:
        row.update(status='failed', reason='ambiguous_multiple_detections')
        return (row, None)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        row.update(status='failed', reason='invalid_polygon')
        return (row, None)
    x1 = max(0, math.floor(float(np.min(polygon[:, 0]))))
    y1 = max(0, math.floor(float(np.min(polygon[:, 1]))))
    x2 = min(width, math.ceil(float(np.max(polygon[:, 0]))))
    y2 = min(height, math.ceil(float(np.max(polygon[:, 1]))))
    (crop_width, crop_height) = (x2 - x1, y2 - y1)
    area_ratio = crop_width * crop_height / float(width * height)
    polygon_ratio = abs(cv2.contourArea(polygon)) / float(width * height)
    row.update(x1=x1, y1=y1, x2=x2, y2=y2, crop_area_ratio=round(area_ratio, 6), polygon_area_ratio=round(polygon_ratio, 6))
    if not MIN_CROP_AREA_RATIO <= area_ratio <= MAX_CROP_AREA_RATIO:
        row.update(status='failed', reason='crop_area_out_of_range')
        return (row, None)
    if crop_width / float(width) < MIN_CROP_WIDTH_RATIO or crop_height / float(height) < MIN_CROP_HEIGHT_RATIO:
        row.update(status='failed', reason='crop_shape_out_of_range')
        return (row, None)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        row.update(status='failed', reason='empty_crop')
        return (row, None)
    row.update(status='saved', reason='')
    return (row, crop)

def process_one(source: Path, result) -> dict[str, object]:
    row = base_row(source)
    try:
        image = read_image(source)
        if result.orig_img.shape != image.shape or not np.array_equal(result.orig_img, image):
            raise RuntimeError('Invalid input or state in process_one (result.orig_img.shape != image.shape or not np.array_equal(result.orig_img, image))')
        (row, crop) = evaluate_prediction(result, image, source)
        if crop is not None:
            write_image(target_for(source), crop)
        return row
    except Exception as error:
        row.update(status='error', reason=f'{type(error).__name__}: {error}'[:500])
        return row

def predict_chunk(model: YOLO, sources: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pending: list[Path] = []
    for source in sources:
        target = target_for(source)
        if target.is_file() and target.stat().st_size > 0:
            row = base_row(source)
            row.update(status='skipped_existing', reason='')
            rows.append(row)
        else:
            pending.append(source)
    if not pending:
        return rows
    predict_options = dict(imgsz=IMGSZ, conf=PREDICT_CONF, iou=NMS_IOU, max_det=3, device=yolo_device(), half=yolo_device() != 'cpu', stream=True, verbose=False)
    processed = 0
    batch_error = ''
    results = None
    try:
        results = iter(model.predict(source=[str(p) for p in pending], batch=BATCH_SIZE, **predict_options))
        for source in pending:
            try:
                result = next(results)
            except StopIteration:
                batch_error = 'Fewer model results than input frames'
                break
            rows.append(process_one(source, result))
            processed += 1
    except Exception as error:
        batch_error = f'{type(error).__name__}: {error}'
    if processed < len(pending):
        print(f'predict chunk: {batch_error} {len(pending) - processed} ', flush=True)
        results = None
        gc.collect()
        torch.cuda.empty_cache()
        for source in pending[processed:]:
            try:
                result = next(iter(model.predict(source=[str(source)], batch=1, **predict_options)))
                rows.append(process_one(source, result))
            except Exception as error:
                row = base_row(source)
                row.update(status='error', reason=f'{type(error).__name__}: {error}'[:500])
                rows.append(row)
    if len(rows) != len(sources):
        raise RuntimeError(f'Invalid input or state in predict_chunk (len(rows) != len(sources)): {len(sources)} {len(rows)} ')
    return rows

def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    encoding = 'utf-8' if path.exists() else 'utf-8-sig'
    with path.open('a', newline='', encoding=encoding) as stream:
        writer = csv.DictWriter(stream, fieldnames=LOG_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)

def run(paths: list[Path], preflight_info: dict[str, object]) -> None:
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = LOG_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f'Invalid input or state in run (run_dir.exists()): {run_dir} ')
    run_dir.mkdir(parents=True)
    manifest = run_dir / 'crop_manifest.csv'
    model_hash = sha256_file(MODEL_PATH)
    code_hash = sha256_file(Path(__file__))
    model = YOLO(str(MODEL_PATH))
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    date_camera: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    progress = tqdm(total=len(paths), desc='run', unit='image')
    try:
        for start in range(0, len(paths), CHUNK_SIZE):
            rows = predict_chunk(model, paths[start:start + CHUNK_SIZE])
            append_rows(manifest, rows)
            for row in rows:
                status = str(row['status'])
                statuses[status] += 1
                date_camera[str(row['date_folder']), str(row['camera'])][status] += 1
                if row['reason']:
                    reasons[str(row['reason'])] += 1
            progress.update(len(rows))
            progress.set_postfix(saved=statuses['saved'], existing=statuses['skipped_existing'], failed=statuses['failed'], error=statuses['error'])
    finally:
        progress.close()
    with (run_dir / 'date_camera_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['date_folder', 'camera', 'selected', 'saved', 'skipped_existing', 'failed', 'error'])
        for ((date, camera), counts) in sorted(date_camera.items()):
            writer.writerow([date, camera, sum(counts.values()), counts['saved'], counts['skipped_existing'], counts['failed'], counts['error']])
    if sum(statuses.values()) != len(paths):
        raise RuntimeError(f'Invalid input or state in run (sum(statuses.values()) != len(paths)): {len(paths)} {sum(statuses.values())} ')
    summary = {'status': 'COMPLETE_WITH_ROW_ERRORS' if statuses['error'] else 'COMPLETE_WITH_SEG_FAILURES' if statuses['failed'] else 'COMPLETE', 'run_id': run_id, 'source_root': str(RAW_ROOT), 'output_root': str(BED_ROOT), 'model': str(MODEL_PATH), 'model_sha256': model_hash, 'code_sha256': code_hash, 'python': sys.executable, 'gpu': preflight_info['gpu'], 'source_count': len(paths), 'camera_counts': preflight_info['camera_counts'], 'date_counts': preflight_info['date_counts'], 'status_counts': dict(statuses), 'reason_counts': dict(reasons), 'manifest': str(manifest), 'date_camera_summary': str(run_dir / 'date_camera_summary.csv'), 'exclusion': '원본 없는 날짜와 260602 및 260602~260611 구간 전체는 이 코드의 대상 밖', 'black_frame_policy': 'Retain source frames and report model/QC status after processing'}
    (run_dir / 'completion.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"run: {BED_ROOT / 'cam0'} {BED_ROOT / 'cam1'} ", flush=True)
    print(f"run: {run_dir / 'date_camera_summary.csv'} ", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='main')
    args = parser.parse_args()
    paths = collect_sources()
    info = preflight(paths)
    if args.check:
        return
    run(paths, info)
if __name__ == '__main__':
    main()
