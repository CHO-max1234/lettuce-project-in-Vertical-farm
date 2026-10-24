"""Segment CAM2 beds in isolated chunks using the shared final front-view model."""
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
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
CODE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = CYCLE.RAW_FRONT_ROOT
OUTPUT_ROOT = CYCLE.BED_OUTPUT / 'cam2'
MODEL_PATH = CYCLE.BED_FRONT_MODEL
LOG_ROOT = CYCLE.OUTPUT_ROOT / 'preprocessing_logs' / 'bed_front'
PREVIEW_CROP_ROOT = LOG_ROOT / 'preview' / 'crops'
PREVIEW_OVERLAY_ROOT = LOG_ROOT / 'preview' / 'overlays'
EXPECTED_SOURCE_COUNT = 7944
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
MARGIN_RATIO = 0.0
MIN_ACCEPT_CONF = 0.45
AMBIGUOUS_CONF_GAP = 0.1
MIN_CROP_AREA_RATIO = 0.08
MAX_CROP_AREA_RATIO = 0.4
MIN_CROP_WIDTH_RATIO = 0.85
MIN_CROP_HEIGHT_RATIO = 0.1
MAX_CROP_HEIGHT_RATIO = 0.35
MIN_CENTER_Y_RATIO = 0.42
MAX_CENTER_Y_RATIO = 0.88
LOG_FIELDS = ['source_path', 'relative_path', 'target_path', 'status', 'reason', 'detection_count', 'confidence', 'second_confidence', 'polygon_area_ratio', 'crop_area_ratio', 'crop_width_ratio', 'crop_height_ratio', 'center_y_ratio', 'x1', 'y1', 'x2', 'y2', 'image_width', 'image_height']

def discover_images() -> list[Path]:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f'Invalid input or state in discover_images (not SOURCE_ROOT.is_dir()): {SOURCE_ROOT} ')
    paths = [path for path in SOURCE_ROOT.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and ('_cam2' in path.stem.lower())]
    return sorted(paths, key=lambda path: str(path.relative_to(SOURCE_ROOT)).lower())

def select_spread_preview(paths: list[Path], count: int) -> list[Path]:
    by_date: dict[str, list[Path]] = {}
    for path in paths:
        by_date.setdefault(path.parent.name, []).append(path)
    per_date = max(1, math.ceil(count / len(by_date)))
    selected: list[Path] = []
    for day_paths in by_date.values():
        indices = np.linspace(0, len(day_paths) - 1, num=min(per_date, len(day_paths)), dtype=int)
        selected.extend((day_paths[int(index)] for index in indices))
    if len(selected) <= count:
        return selected
    indices = np.linspace(0, len(selected) - 1, num=count, dtype=int)
    return [selected[int(index)] for index in indices]

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid input or state in read_image (image is None): {path} ')
    return image

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = '.png' if path.suffix.lower() == '.png' else '.jpg'
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == '.png' else [cv2.IMWRITE_JPEG_QUALITY, 95]
    (success, encoded) = cv2.imencode(extension, image, parameters)
    if not success:
        raise ValueError(f'Invalid input or state in write_image (not success): {path} ')
    encoded.tofile(str(path))

def target_path_for(source_path: Path, mode: str) -> Path:
    relative_path = source_path.relative_to(SOURCE_ROOT)
    root = OUTPUT_ROOT if mode == 'run' else PREVIEW_CROP_ROOT
    return root / relative_path

def is_valid_existing_image(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0

def polygon_bbox(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    (x_min, y_min) = np.min(polygon, axis=0).astype(float)
    (x_max, y_max) = np.max(polygon, axis=0).astype(float)
    margin_x = (x_max - x_min) * MARGIN_RATIO
    margin_y = (y_max - y_min) * MARGIN_RATIO
    return (max(0, math.floor(x_min - margin_x)), max(0, math.floor(y_min - margin_y)), min(width, math.ceil(x_max + margin_x)), min(height, math.ceil(y_max + margin_y)))

def empty_log_row(source_path: Path, target_path: Path) -> dict[str, object]:
    row = {field: '' for field in LOG_FIELDS}
    row.update(source_path=str(source_path), relative_path=str(source_path.relative_to(SOURCE_ROOT)), target_path=str(target_path))
    return row

def evaluate_prediction(result, image: np.ndarray, source_path: Path, target_path: Path) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
    row = empty_log_row(source_path, target_path)
    (height, width) = image.shape[:2]
    row.update(image_width=width, image_height=height)
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        row.update(status='failed', reason='no_detection', detection_count=0)
        return (row, None, None)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    polygons = result.masks.xy
    if len(confidences) != len(polygons):
        row.update(status='failed', reason='box_mask_count_mismatch', detection_count=len(confidences))
        return (row, None, None)
    order = np.argsort(-confidences)
    best_index = int(order[0])
    best_confidence = float(confidences[best_index])
    second_confidence = float(confidences[order[1]]) if len(order) > 1 else None
    polygon = np.asarray(polygons[best_index], dtype=np.float32)
    row.update(detection_count=len(confidences), confidence=round(best_confidence, 6), second_confidence='' if second_confidence is None else round(second_confidence, 6))
    if best_confidence < MIN_ACCEPT_CONF:
        row.update(status='failed', reason='low_confidence')
        return (row, None, polygon)
    if second_confidence is not None and best_confidence - second_confidence < AMBIGUOUS_CONF_GAP:
        row.update(status='failed', reason='ambiguous_multiple_detections')
        return (row, None, polygon)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        row.update(status='failed', reason='invalid_polygon')
        return (row, None, polygon)
    (x1, y1, x2, y2) = polygon_bbox(polygon, width, height)
    crop_width = x2 - x1
    crop_height = y2 - y1
    crop_width_ratio = crop_width / float(width)
    crop_height_ratio = crop_height / float(height)
    crop_area_ratio = crop_width_ratio * crop_height_ratio
    polygon_area_ratio = abs(cv2.contourArea(polygon)) / float(width * height)
    center_y_ratio = (y1 + y2) / 2.0 / float(height)
    row.update(polygon_area_ratio=round(polygon_area_ratio, 6), crop_area_ratio=round(crop_area_ratio, 6), crop_width_ratio=round(crop_width_ratio, 6), crop_height_ratio=round(crop_height_ratio, 6), center_y_ratio=round(center_y_ratio, 6), x1=x1, y1=y1, x2=x2, y2=y2)
    if not MIN_CROP_AREA_RATIO <= crop_area_ratio <= MAX_CROP_AREA_RATIO:
        row.update(status='failed', reason='crop_area_out_of_range')
        return (row, None, polygon)
    if crop_width_ratio < MIN_CROP_WIDTH_RATIO:
        row.update(status='failed', reason='crop_width_too_small')
        return (row, None, polygon)
    if not MIN_CROP_HEIGHT_RATIO <= crop_height_ratio <= MAX_CROP_HEIGHT_RATIO:
        row.update(status='failed', reason='crop_height_out_of_range')
        return (row, None, polygon)
    if not MIN_CENTER_Y_RATIO <= center_y_ratio <= MAX_CENTER_Y_RATIO:
        row.update(status='failed', reason='crop_vertical_position_out_of_range')
        return (row, None, polygon)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        row.update(status='failed', reason='empty_crop')
        return (row, None, polygon)
    row.update(status='saved', reason='')
    return (row, crop, polygon)

def draw_overlay(image: np.ndarray, row: dict[str, object], polygon: np.ndarray | None) -> np.ndarray:
    overlay = image.copy()
    if polygon is not None and polygon.ndim == 2 and (polygon.shape[0] >= 3):
        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], True, (0, 0, 255), 5, cv2.LINE_AA)
    if all((row[key] != '' for key in ('x1', 'y1', 'x2', 'y2'))):
        cv2.rectangle(overlay, (int(row['x1']), int(row['y1'])), (int(row['x2']), int(row['y2'])), (0, 255, 255), 3)
    text = f"{row['status']} {row['reason']} conf={row['confidence']}"
    cv2.putText(overlay, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return overlay

def append_log_rows(log_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open('a', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def save_failure_overlay(source_path: Path, image: np.ndarray, row: dict[str, object], polygon: np.ndarray | None, run_id: str) -> None:
    if row['reason'] == 'no_detection':
        return
    relative_path = source_path.relative_to(SOURCE_ROOT)
    write_image(LOG_ROOT / run_id / 'failure_overlays' / relative_path, draw_overlay(image, row, polygon))

def process_chunk(model: YOLO, paths: list[Path], args: argparse.Namespace, run_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pending: list[Path] = []
    for source_path in paths:
        target_path = target_path_for(source_path, args.mode)
        if not args.overwrite and is_valid_existing_image(target_path):
            row = empty_log_row(source_path, target_path)
            row.update(status='skipped_existing', reason='')
            rows.append(row)
        else:
            pending.append(source_path)
    if not pending:
        return rows
    results = model.predict(source=[str(path) for path in pending], imgsz=args.imgsz, conf=args.predict_conf, iou=args.nms_iou, max_det=5, batch=args.batch_size, device=args.device, half=args.half, stream=True, verbose=False)
    for (source_path, result) in zip(pending, results):
        target_path = target_path_for(source_path, args.mode)
        try:
            image = read_image(source_path)
            (row, crop, polygon) = evaluate_prediction(result, image, source_path, target_path)
            if crop is not None:
                write_image(target_path, crop)
            if args.mode == 'preview':
                relative_path = source_path.relative_to(SOURCE_ROOT)
                write_image(PREVIEW_OVERLAY_ROOT / relative_path, draw_overlay(image, row, polygon))
            elif row['status'] == 'failed' and args.save_failure_overlays:
                save_failure_overlay(source_path, image, row, polygon, run_id)
        except Exception as error:
            row = empty_log_row(source_path, target_path)
            row.update(status='error', reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    return rows

def run_all_isolated(args: argparse.Namespace, all_paths: list[Path]) -> None:
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_run_all'
    master_dir = LOG_ROOT / run_id
    master_manifest = master_dir / 'crop_manifest.csv'
    counts: Counter[str] = Counter()
    print(f'run all isolated: {len(all_paths)} {args.chunk_size} ', flush=True)
    progress = tqdm(total=len(all_paths), desc='cam2 isolated chunks', unit='image')
    for start in range(0, len(all_paths), args.chunk_size):
        stop = min(start + args.chunk_size, len(all_paths))
        part_id = f'{run_id}/part_{start:05d}_{stop:05d}'
        command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'run', '--start-index', str(start), '--stop-index', str(stop), '--run-id', part_id, '--imgsz', str(args.imgsz), '--batch-size', str(args.batch_size), '--chunk-size', str(args.chunk_size), '--device', args.device, '--predict-conf', str(args.predict_conf), '--nms-iou', str(args.nms_iou), '--half' if args.half else '--no-half', '--save-failure-overlays' if args.save_failure_overlays else '--no-save-failure-overlays']
        if args.overwrite:
            command.append('--overwrite')
        part_log = master_dir / f'part_{start:05d}_{stop:05d}.log'
        part_log.parent.mkdir(parents=True, exist_ok=True)
        with part_log.open('w', encoding='utf-8') as file:
            completed = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'Invalid input or state in run_all_isolated (completed.returncode != 0): {start} {stop} {part_log} ')
        part_dir = LOG_ROOT / part_id
        with (part_dir / 'crop_manifest.csv').open('r', newline='', encoding='utf-8-sig') as file:
            rows = list(csv.DictReader(file))
        if len(rows) != stop - start:
            raise RuntimeError(f'Invalid input or state in run_all_isolated (len(rows) != stop - start): {start} {stop} {len(rows)} ')
        append_log_rows(master_manifest, rows)
        counts.update((str(row['status']) for row in rows))
        progress.update(len(rows))
        progress.set_postfix(dict(counts))
    progress.close()
    saved_files = sum((1 for path in OUTPUT_ROOT.rglob('*.jpg') if path.is_file() and path.stat().st_size > 0))
    summary = {'run_id': run_id, 'mode': 'run', 'model': str(MODEL_PATH), 'source_root': str(SOURCE_ROOT), 'output_root': str(OUTPUT_ROOT), 'source_count': len(all_paths), 'selected_count': len(all_paths), 'status_counts': dict(counts), 'saved_files': saved_files, 'manifest': str(master_manifest), 'isolated_chunk_size': args.chunk_size}
    (master_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('[DONE]', flush=True)

def run_inference(args: argparse.Namespace) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Invalid input or state in run_inference (not MODEL_PATH.is_file()): {MODEL_PATH} ')
    all_paths = discover_images()
    if len(all_paths) != EXPECTED_SOURCE_COUNT:
        raise RuntimeError(f'Invalid input or state in run_inference (len(all_paths) != EXPECTED_SOURCE_COUNT): {EXPECTED_SOURCE_COUNT} {len(all_paths)} ')
    if args.mode == 'run' and args.start_index is None:
        run_all_isolated(args, all_paths)
        return
    if args.mode == 'run':
        if args.stop_index is None or not 0 <= args.start_index < args.stop_index <= len(all_paths):
            raise ValueError('Invalid input or state in run_inference (args.stop_index is None or not 0 <= args.start_index < args.stop_index <= len(all_paths))')
        selected = all_paths[args.start_index:args.stop_index]
    else:
        selected = select_spread_preview(all_paths, args.preview_count)
    if not selected:
        raise RuntimeError('Invalid input or state in run_inference (not selected)')
    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{args.mode}'
    log_path = LOG_ROOT / run_id / 'crop_manifest.csv'
    summary_path = LOG_ROOT / run_id / 'run_summary.json'
    print(f'python={os.sys.executable}')
    print(f'model={MODEL_PATH}')
    print(f'source={SOURCE_ROOT}')
    print(f'mode={args.mode} selected={len(selected)} total_source={len(all_paths)}')
    print(f"output={(OUTPUT_ROOT if args.mode == 'run' else PREVIEW_CROP_ROOT)}")
    print(f'log={log_path}')
    model = YOLO(str(MODEL_PATH))
    status_counts: Counter[str] = Counter()
    progress = tqdm(total=len(selected), desc='cam2 bed crop', unit='image')
    for start in range(0, len(selected), args.chunk_size):
        chunk = selected[start:start + args.chunk_size]
        rows = process_chunk(model, chunk, args, run_id)
        if len(rows) != len(chunk):
            raise RuntimeError(f'Invalid input or state in run_inference (len(rows) != len(chunk)): {len(chunk)} {len(rows)} ')
        append_log_rows(log_path, rows)
        status_counts.update((str(row['status']) for row in rows))
        progress.update(len(chunk))
        progress.set_postfix(dict(status_counts))
    progress.close()
    if sum(status_counts.values()) != len(selected):
        raise RuntimeError('Invalid input or state in run_inference (sum(status_counts.values()) != len(selected))')
    summary = {'run_id': run_id, 'mode': args.mode, 'model': str(MODEL_PATH), 'source_root': str(SOURCE_ROOT), 'output_root': str(OUTPUT_ROOT), 'preserve_relative_folders': True, 'margin_ratio': MARGIN_RATIO, 'source_count': len(all_paths), 'selected_count': len(selected), 'status_counts': dict(status_counts), 'arguments': vars(args), 'manifest': str(log_path)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('[DONE]')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preview', 'run'), default='run')
    parser.add_argument('--preview-count', type=int, default=60)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--device', default=yolo_device())
    parser.add_argument('--predict-conf', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--half', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-failure-overlays', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--start-index', type=int)
    parser.add_argument('--stop-index', type=int)
    parser.add_argument('--run-id')
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    if args.preview_count <= 0 or args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError('Invalid input or state in main (args.preview_count <= 0 or args.batch_size <= 0 or args.chunk_size <= 0)')
    run_inference(args)
if __name__ == '__main__':
    main()
