"""Segment bed regions using the configured checkpoint and source paths."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from pipeline_paths import INPUT_ROOT, BED_ROOT, BED_IMAGES, BED_MODELS, DEVICE
SOURCE_ROOT = INPUT_ROOT
OUTPUT_ROOTS = {camera: BED_IMAGES / camera for camera in ('cam0', 'cam1')}
MODEL_PATH = BED_MODELS['cam01']
LOG_ROOT = BED_ROOT / 'logs' / 'cam01'
PREVIEW_CROP_ROOT = BED_ROOT / 'preview' / 'crops' / 'cam01'
PREVIEW_OVERLAY_ROOT = BED_ROOT / 'preview' / 'overlays' / 'cam01'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
CAMERA_PATTERN = re.compile('_(cam0|cam1)(?:_|$)', re.IGNORECASE)
MARGIN_RATIO = 0.0
MIN_ACCEPT_CONF = 0.45
AMBIGUOUS_CONF_GAP = 0.1
MIN_CROP_AREA_RATIO = 0.1
MAX_CROP_AREA_RATIO = 0.8
MIN_CROP_WIDTH_RATIO = 0.4
MIN_CROP_HEIGHT_RATIO = 0.1
LOG_FIELDS = ['source_path', 'relative_path', 'camera', 'target_path', 'status', 'reason', 'detection_count', 'confidence', 'second_confidence', 'polygon_area_ratio', 'crop_area_ratio', 'x1', 'y1', 'x2', 'y2', 'image_width', 'image_height']

def detect_camera(path: Path) -> str | None:
    match = CAMERA_PATTERN.search(path.stem)
    return match.group(1).lower() if match else None

def discover_images() -> list[Path]:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f'Invalid input or state in discover_images (rejected condition: not SOURCE_ROOT.is_dir()): {SOURCE_ROOT} ')
    paths = [path for path in SOURCE_ROOT.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and (detect_camera(path) in OUTPUT_ROOTS)]
    return sorted(paths, key=lambda path: str(path.relative_to(SOURCE_ROOT)).lower())

def select_spread_preview(paths: list[Path], per_camera: int) -> list[Path]:
    selected: list[Path] = []
    for camera in OUTPUT_ROOTS:
        camera_paths = [path for path in paths if detect_camera(path) == camera]
        if not camera_paths:
            continue
        count = min(per_camera, len(camera_paths))
        indices = np.linspace(0, len(camera_paths) - 1, num=count, dtype=int)
        selected.extend((camera_paths[index] for index in indices))
    return selected

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid input or state in read_image (rejected condition: image is None): {path} ')
    return image

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    extension = '.png' if suffix == '.png' else '.jpg'
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == '.png' else [cv2.IMWRITE_JPEG_QUALITY, 95]
    (success, encoded) = cv2.imencode(extension, image, parameters)
    if not success:
        raise ValueError(f'Invalid input or state in write_image (rejected condition: not success): {path} ')
    encoded.tofile(str(path))

def target_path_for(source_path: Path, mode: str) -> Path:
    camera = detect_camera(source_path)
    if camera is None:
        raise ValueError(f'Invalid input or state in target_path_for (rejected condition: camera is None): {source_path} ')
    relative_path = source_path.relative_to(SOURCE_ROOT)
    root = OUTPUT_ROOTS[camera] if mode == 'run' else PREVIEW_CROP_ROOT / camera
    return root / relative_path

def is_valid_existing_image(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0

def polygon_bbox(polygon: np.ndarray, width: int, height: int, margin_ratio: float) -> tuple[int, int, int, int]:
    x_min = float(np.min(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    x_max = float(np.max(polygon[:, 0]))
    y_max = float(np.max(polygon[:, 1]))
    margin_x = (x_max - x_min) * margin_ratio
    margin_y = (y_max - y_min) * margin_ratio
    x1 = max(0, math.floor(x_min - margin_x))
    y1 = max(0, math.floor(y_min - margin_y))
    x2 = min(width, math.ceil(x_max + margin_x))
    y2 = min(height, math.ceil(y_max + margin_y))
    return (x1, y1, x2, y2)

def empty_log_row(source_path: Path, target_path: Path) -> dict[str, object]:
    camera = detect_camera(source_path) or 'unknown'
    return {'source_path': str(source_path), 'relative_path': str(source_path.relative_to(SOURCE_ROOT)), 'camera': camera, 'target_path': str(target_path), 'status': '', 'reason': '', 'detection_count': 0, 'confidence': '', 'second_confidence': '', 'polygon_area_ratio': '', 'crop_area_ratio': '', 'x1': '', 'y1': '', 'x2': '', 'y2': '', 'image_width': '', 'image_height': ''}

def evaluate_prediction(result, image: np.ndarray, source_path: Path, target_path: Path) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
    row = empty_log_row(source_path, target_path)
    (height, width) = image.shape[:2]
    row['image_width'] = width
    row['image_height'] = height
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        row.update(status='failed', reason='no_detection')
        return (row, None, None)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
    polygons = result.masks.xy
    if len(polygons) != len(confidences):
        row.update(status='failed', reason='box_mask_count_mismatch', detection_count=len(confidences))
        return (row, None, None)
    order = np.argsort(-confidences)
    best_index = int(order[0])
    best_confidence = float(confidences[best_index])
    second_confidence = float(confidences[order[1]]) if len(order) > 1 else None
    polygon = np.asarray(polygons[best_index], dtype=np.float32)
    row['detection_count'] = len(confidences)
    row['confidence'] = round(best_confidence, 6)
    row['second_confidence'] = '' if second_confidence is None else round(second_confidence, 6)
    if best_confidence < MIN_ACCEPT_CONF:
        row.update(status='failed', reason='low_confidence')
        return (row, None, polygon)
    if second_confidence is not None and best_confidence - second_confidence < AMBIGUOUS_CONF_GAP:
        row.update(status='failed', reason='ambiguous_multiple_detections')
        return (row, None, polygon)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        row.update(status='failed', reason='invalid_polygon')
        return (row, None, polygon)
    (x1, y1, x2, y2) = polygon_bbox(polygon, width, height, MARGIN_RATIO)
    crop_width = x2 - x1
    crop_height = y2 - y1
    crop_area_ratio = crop_width * crop_height / float(width * height)
    polygon_area_ratio = abs(cv2.contourArea(polygon)) / float(width * height)
    width_ratio = crop_width / float(width)
    height_ratio = crop_height / float(height)
    row.update(polygon_area_ratio=round(polygon_area_ratio, 6), crop_area_ratio=round(crop_area_ratio, 6), x1=x1, y1=y1, x2=x2, y2=y2)
    if not MIN_CROP_AREA_RATIO <= crop_area_ratio <= MAX_CROP_AREA_RATIO:
        row.update(status='failed', reason='crop_area_out_of_range')
        return (row, None, polygon)
    if width_ratio < MIN_CROP_WIDTH_RATIO or height_ratio < MIN_CROP_HEIGHT_RATIO:
        row.update(status='failed', reason='crop_shape_out_of_range')
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
    cv2.putText(overlay, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
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
    camera = detect_camera(source_path) or 'unknown'
    relative_path = source_path.relative_to(SOURCE_ROOT)
    overlay_path = LOG_ROOT / run_id / 'failure_overlays' / camera / relative_path
    write_image(overlay_path, draw_overlay(image, row, polygon))

def process_chunk(model: YOLO, paths: list[Path], args: argparse.Namespace, run_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pending_paths: list[Path] = []
    for source_path in paths:
        target_path = target_path_for(source_path, args.mode)
        if not args.overwrite and is_valid_existing_image(target_path):
            row = empty_log_row(source_path, target_path)
            row.update(status='skipped_existing', reason='')
            rows.append(row)
        else:
            pending_paths.append(source_path)
    if not pending_paths:
        return rows
    results = model.predict(source=[str(path) for path in pending_paths], imgsz=args.imgsz, conf=args.predict_conf, iou=args.nms_iou, max_det=3, batch=args.batch_size, device=args.device, half=args.half, stream=True, verbose=False)
    for (source_path, result) in zip(pending_paths, results):
        target_path = target_path_for(source_path, args.mode)
        try:
            image = read_image(source_path)
            (row, crop, polygon) = evaluate_prediction(result, image, source_path, target_path)
            if crop is not None:
                write_image(target_path, crop)
            if args.mode == 'preview':
                camera = detect_camera(source_path) or 'unknown'
                relative_path = source_path.relative_to(SOURCE_ROOT)
                overlay_path = PREVIEW_OVERLAY_ROOT / camera / relative_path
                write_image(overlay_path, draw_overlay(image, row, polygon))
            elif row['status'] == 'failed' and args.save_failure_overlays:
                save_failure_overlay(source_path, image, row, polygon, run_id)
        except Exception as error:
            row = empty_log_row(source_path, target_path)
            row.update(status='error', reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    return rows

def run_inference(args: argparse.Namespace) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Invalid input or state in run_inference (rejected condition: not MODEL_PATH.is_file()): {MODEL_PATH} ')
    all_paths = discover_images()
    selected_paths = select_spread_preview(all_paths, args.preview_per_camera) if args.mode == 'preview' else all_paths
    if not selected_paths:
        raise RuntimeError('Invalid input or state in run_inference (rejected condition: not selected_paths)')
    camera_counts = Counter((detect_camera(path) for path in selected_paths))
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_{args.mode}'
    log_path = LOG_ROOT / run_id / 'crop_manifest.csv'
    summary_path = LOG_ROOT / run_id / 'run_summary.json'
    print(f'model={MODEL_PATH}')
    print(f'source={SOURCE_ROOT}')
    print(f'mode={args.mode}')
    print(f"selected={len(selected_paths)} cam0={camera_counts['cam0']} cam1={camera_counts['cam1']}")
    if args.mode == 'run':
        print(f"cam0_output={OUTPUT_ROOTS['cam0']}")
        print(f"cam1_output={OUTPUT_ROOTS['cam1']}")
    else:
        print(f'preview_crops={PREVIEW_CROP_ROOT}')
        print(f'preview_overlays={PREVIEW_OVERLAY_ROOT}')
    print(f'log={log_path}')
    model = YOLO(str(MODEL_PATH))
    status_counts: Counter[str] = Counter()
    progress = tqdm(total=len(selected_paths), desc='bed crop', unit='image')
    for start in range(0, len(selected_paths), args.chunk_size):
        chunk_paths = selected_paths[start:start + args.chunk_size]
        rows = process_chunk(model, chunk_paths, args, run_id)
        if len(rows) != len(chunk_paths):
            raise RuntimeError(f'Invalid input or state in run_inference (rejected condition: len(rows) != len(chunk_paths)): {len(chunk_paths)} {len(rows)} ')
        append_log_rows(log_path, rows)
        status_counts.update((str(row['status']) for row in rows))
        progress.update(len(chunk_paths))
        progress.set_postfix(dict(status_counts))
    progress.close()
    if sum(status_counts.values()) != len(selected_paths):
        raise RuntimeError(f'Invalid input or state in run_inference (rejected condition: sum(status_counts.values()) != len(selected_paths)): {len(selected_paths)} {sum(status_counts.values())} ')
    summary = {'run_id': run_id, 'mode': args.mode, 'model': str(MODEL_PATH), 'source_root': str(SOURCE_ROOT), 'output_roots': {key: str(value) for (key, value) in OUTPUT_ROOTS.items()}, 'preserve_relative_folders': True, 'margin_ratio': MARGIN_RATIO, 'selected_count': len(selected_paths), 'camera_counts': dict(camera_counts), 'status_counts': dict(status_counts), 'arguments': vars(args), 'manifest': str(log_path)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('[DONE]')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preview', 'run'), default='run')
    parser.add_argument('--preview-per-camera', type=int, default=5)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=500)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--predict-conf', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.7)
    parser.add_argument('--half', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save-failure-overlays', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    if args.preview_per_camera <= 0:
        raise ValueError('Invalid input or state in main (rejected condition: args.preview_per_camera <= 0)')
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError('batch-size and chunk-size must be positive.')
    run_inference(args)
if __name__ == '__main__':
    main()
