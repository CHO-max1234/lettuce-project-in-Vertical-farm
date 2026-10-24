"""Processing utilities for 6. Build bboxes top."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import argparse
import csv
import hashlib
import importlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
SCRIPT_DIR = Path(__file__).resolve().parent
from pipeline_paths import FROZEN_CAM01, BRIGHT_IMAGES, BBOX_ROOT
FROZEN_VER4_DIR = FROZEN_CAM01
SOURCE_ROOTS = {camera: BRIGHT_IMAGES / camera for camera in ('cam0', 'cam1')}
OUTPUT_ROOT = BBOX_ROOT / 'cam01'
MANIFEST_DIR = OUTPUT_ROOT / '01_manifest'
MANIFEST_PATH = MANIFEST_DIR / 'cam01_bbox_manifest_VER4.csv'
PARTIAL_MANIFEST_PATH = MANIFEST_DIR / 'cam01_bbox_manifest_VER4.partial.csv'
PROGRESS_PATH = MANIFEST_DIR / 'cam01_bbox_manifest_VER4.progress.json'
SAMPLE_PATH = MANIFEST_DIR / 'cam01_train3000_manifest_VER4.csv'
SAMPLE_AUDIT_PATH = MANIFEST_DIR / 'cam01_train3000_sampling_audit_VER4.json'
MODEL_EXPECTED_SHA256 = 'A2DED62BBA4D577F4EAD18907B824C587E179357C8089B72DABDE41B755D6FE8'
CAMERA_MAPPING_VERSION = {'cam0': 'VER4_final (frozen CAM0 logic = verified VER3.3)', 'cam1': 'VER4_final (frozen CAM1 logic = verified VER3.4)'}
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FILENAME_RE = re.compile('^(?P<bed>bed\\d+)_(?P<date>\\d{8})_(?P<time>\\d{6})_(?P<camera>cam[01])(?:_[a-z0-9]+)*$', re.IGNORECASE)
MANIFEST_COLUMNS = ['record_id', 'crop_name', 'source_root_key', 'bed_source_path', 'source_image_name', 'bed_id', 'camera', 'capture_date', 'capture_time', 'canonical_id', 'row', 'col', 'local_col', 'physical_col', 'mapping_version', 'map_status', 'qc_flags', 'visibility', 'reg_status', 'evidence_kind', 'evidence_score', 'seg_status', 'seg_id', 'seg_conf', 'source_width', 'source_height', 'bed_x1', 'bed_y1', 'bed_x2', 'bed_y2', 'cell_x1', 'cell_y1', 'cell_x2', 'cell_y2', 'cell_width', 'cell_height', 'cell_area', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'crop_width', 'crop_height', 'crop_area', 'crop_aspect_ratio', 'crop_center_x', 'crop_center_y', 'is_crop_clipped', 'slot_center_x', 'slot_center_y', 'slot_prior_x', 'slot_prior_y', 'seg_x1', 'seg_y1', 'seg_x2', 'seg_y2', 'seg_width', 'seg_height', 'seg_bbox_area', 'seg_center_x', 'seg_center_y', 'seg_area', 'residual_dx', 'residual_dy', 'observed_column_count', 'physical_visible_column_count', 'is_train_candidate', 'legacy_slot_id', 'slot_namespace_version']

def _load_frozen_modules() -> tuple[Any, Any, Any, Any]:
    if not FROZEN_VER4_DIR.is_dir():
        raise FileNotFoundError(f'VER4_final code directory not found: {FROZEN_VER4_DIR}')
    frozen_path = str(FROZEN_VER4_DIR)
    if frozen_path not in sys.path:
        sys.path.insert(0, frozen_path)
    core = importlib.import_module('slot_geometry')
    lettuce_seg = importlib.import_module('plant_segmentation')
    cam0_adapter = importlib.import_module('cam0_geometry')
    cam1_adapter = importlib.import_module('cam1_geometry')
    return (core, lettuce_seg, cam0_adapter, cam1_adapter)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest().upper()

def _verify_model(lettuce_seg: Any) -> None:
    model_path = Path(lettuce_seg.LETTUCE_SEG_MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f'lettuce-seg model not found: {model_path}')
    actual = _sha256(model_path)
    if actual != MODEL_EXPECTED_SHA256:
        raise RuntimeError(f'lettuce-seg model hash differs from the frozen VER4 contract: expected={MODEL_EXPECTED_SHA256}, actual={actual}')
    print(f'[verified] lettuce-seg model: {model_path}')
    print(f'[verified] SHA256: {actual}')

def _collect_images(camera: str) -> list[Path]:
    root = SOURCE_ROOTS[camera]
    if not root.is_dir():
        raise FileNotFoundError(f'source root not found: {root}')
    images: list[Path] = []
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        matched = FILENAME_RE.match(path.stem)
        if matched and matched.group('camera').lower() == camera:
            images.append(path)
    if not images:
        raise RuntimeError(f'no {camera} images found under: {root}')
    return sorted(images)

def _tuple4(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        return None
    return tuple((float(item) for item in value))

def _ints_or_blank(box: tuple[float, float, float, float] | None) -> tuple[Any, Any, Any, Any]:
    if box is None:
        return ('', '', '', '')
    return tuple((int(round(value)) for value in box))

def _dimensions(box: tuple[float, float, float, float] | None) -> tuple[Any, Any, Any, Any, Any]:
    if box is None:
        return ('', '', '', '', '')
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return (int(round(width)), int(round(height)), int(round(width * height)), round(width / height, 6) if height else '', (box[0] + box[2]) / 2.0)

def _parse_source(path: Path, camera: str) -> dict[str, str]:
    matched = FILENAME_RE.match(path.stem)
    if not matched:
        raise ValueError(f'unparseable image name: {path.name}')
    info = matched.groupdict()
    if info['camera'].lower() != camera:
        raise ValueError(f'camera/path mismatch: {camera} vs {path.name}')
    return info

def _map_one(image: Any, camera: str, model: Any, lettuce_seg: Any, cam0_adapter: Any, cam1_adapter: Any) -> tuple[list[Any], dict[str, Any]]:
    raw_segments = lettuce_seg.predict_lettuce_segments(model, image)
    if camera == 'cam0':
        segments = raw_segments
        (slots, meta) = cam0_adapter.map_cam0_v3(image, segments)
        observed_columns = meta.get('n_visible', '')
        physical_visible_columns = ''
        refine_kwargs = {'enforce_cell_corridor': True}
    else:
        (segments, rejected_segments) = cam1_adapter.filter_cam1_segments(image, raw_segments)
        (slots, meta) = cam1_adapter.map_cam1_v3(image, segments)
        meta['raw_segment_count'] = len(raw_segments)
        meta['rejected_segment_count'] = len(rejected_segments)
        observed_columns = meta.get('n_visible_evidence', '')
        physical_visible_columns = meta.get('n_visible_physical', '')
        refine_kwargs = {'enforce_cell_corridor': True, 'conservative_prior_gate': False, 'prune_single_curve_outlier': not bool(meta.get('towel_present'))}
    if slots:
        segment_meta = lettuce_seg.refine_slots_with_segments(slots, segments, image.shape[1], image.shape[0], bed_bbox=meta.get('bed_bbox'), **refine_kwargs)
        meta.update({key: value for (key, value) in segment_meta.items() if key != 'assigned_by_slot_object_id'})
        if camera == 'cam1' and meta.get('curve_rejected_segment_count', 0):
            meta.setdefault('qc_flags', []).append('AMBIGUOUS_SEG_PRIOR_KEPT')
    meta['manifest_observed_column_count'] = observed_columns
    meta['manifest_physical_visible_column_count'] = physical_visible_columns
    return (slots, meta)

def _slot_to_row(slot: Any, meta: dict[str, Any], path: Path, camera: str, image_w: int, image_h: int) -> dict[str, Any] | None:
    if slot.local_col <= 0 or not slot.canonical_id:
        return None
    crop_box = _tuple4(slot.bbox)
    cell_box = _tuple4(slot.cell_bbox)
    if crop_box is None or crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return None
    source = _parse_source(path, camera)
    bed_box = _tuple4(meta.get('bed_bbox'))
    seg_box = _tuple4(slot.seg_bbox)
    (cell_x1, cell_y1, cell_x2, cell_y2) = _ints_or_blank(cell_box)
    (crop_x1, crop_y1, crop_x2, crop_y2) = _ints_or_blank(crop_box)
    (seg_x1, seg_y1, seg_x2, seg_y2) = _ints_or_blank(seg_box)
    (cell_w, cell_h, cell_area, _, _) = _dimensions(cell_box)
    (crop_w, crop_h, crop_area, crop_ratio, crop_center_x) = _dimensions(crop_box)
    crop_center_y = (crop_box[1] + crop_box[3]) / 2.0
    (seg_w, seg_h, seg_box_area, _, _) = _dimensions(seg_box)
    (bed_x1, bed_y1, bed_x2, bed_y2) = _ints_or_blank(bed_box)
    is_clipped = int(crop_box[0] <= 0 or crop_box[1] <= 0 or crop_box[2] >= image_w or (crop_box[3] >= image_h))
    crop_name = f'{path.stem}__{slot.canonical_id}.jpg'
    status = str(getattr(slot, 'seg_status', ''))
    visibility = str(getattr(slot, 'visibility', ''))
    train_candidate = int(visibility != 'Visibility.OCCLUDED' and crop_w >= 32 and (crop_h >= 32))
    row = {'record_id': f'{camera}|{path.name}|{slot.canonical_id}', 'crop_name': crop_name, 'source_root_key': f'roi_crop_ver2_{camera}', 'bed_source_path': (Path(SOURCE_ROOTS[camera].name) / path.relative_to(SOURCE_ROOTS[camera])).as_posix(), 'source_image_name': path.name, 'bed_id': source['bed'], 'camera': camera, 'capture_date': source['date'], 'capture_time': source['time'], 'canonical_id': slot.canonical_id, 'row': slot.row, 'col': slot.col, 'local_col': slot.local_col, 'physical_col': slot.physical_col if slot.physical_col is not None else '', 'mapping_version': CAMERA_MAPPING_VERSION[camera], 'map_status': meta.get('status', ''), 'qc_flags': '|'.join(meta.get('qc_flags', [])), 'visibility': visibility.replace('Visibility.', ''), 'reg_status': str(getattr(slot, 'reg_status', '')).replace('RegistrationStatus.', ''), 'evidence_kind': getattr(slot, 'evidence_kind', ''), 'evidence_score': getattr(slot, 'evidence_score', ''), 'seg_status': status.replace('SegStatus.', ''), 'seg_id': slot.seg_id if slot.seg_id is not None else '', 'seg_conf': slot.seg_conf if slot.seg_conf is not None else '', 'source_width': image_w, 'source_height': image_h, 'bed_x1': bed_x1, 'bed_y1': bed_y1, 'bed_x2': bed_x2, 'bed_y2': bed_y2, 'cell_x1': cell_x1, 'cell_y1': cell_y1, 'cell_x2': cell_x2, 'cell_y2': cell_y2, 'cell_width': cell_w, 'cell_height': cell_h, 'cell_area': cell_area, 'crop_x1': crop_x1, 'crop_y1': crop_y1, 'crop_x2': crop_x2, 'crop_y2': crop_y2, 'crop_width': crop_w, 'crop_height': crop_h, 'crop_area': crop_area, 'crop_aspect_ratio': crop_ratio, 'crop_center_x': round(crop_center_x, 3), 'crop_center_y': round(crop_center_y, 3), 'is_crop_clipped': is_clipped, 'slot_center_x': round(float(slot.x), 3), 'slot_center_y': round(float(slot.y), 3), 'slot_prior_x': round(float(slot.x_prior), 3), 'slot_prior_y': round(float(slot.y_prior), 3), 'seg_x1': seg_x1, 'seg_y1': seg_y1, 'seg_x2': seg_x2, 'seg_y2': seg_y2, 'seg_width': seg_w, 'seg_height': seg_h, 'seg_bbox_area': seg_box_area, 'seg_center_x': getattr(slot, 'seg_center_x', '') or '', 'seg_center_y': getattr(slot, 'seg_center_y', '') or '', 'seg_area': getattr(slot, 'seg_area', '') or '', 'residual_dx': getattr(slot, 'residual_dx', '') or '', 'residual_dy': getattr(slot, 'residual_dy', '') or '', 'observed_column_count': meta.get('manifest_observed_column_count', ''), 'physical_visible_column_count': meta.get('manifest_physical_visible_column_count', ''), 'is_train_candidate': train_candidate}
    from pipeline_contracts import remap_cam1
    return remap_cam1(row)

def _existing_processed(partial_path: Path) -> set[tuple[str, str]]:
    if not partial_path.exists():
        return set()
    processed: set[tuple[str, str]] = set()
    with partial_path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            processed.add((row['camera'], row['source_image_name']))
    return processed

def _write_progress(payload: dict[str, Any]) -> None:
    PROGRESS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def run_manifest(cameras: Iterable[str], checkpoint_every: int) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        raise FileExistsError(f'final manifest already exists; refusing overwrite: {MANIFEST_PATH}')
    (core, lettuce_seg, cam0_adapter, cam1_adapter) = _load_frozen_modules()
    _verify_model(lettuce_seg)
    model = lettuce_seg.load_lettuce_seg_model()
    processed = _existing_processed(PARTIAL_MANIFEST_PATH)
    all_images = [(camera, path) for camera in cameras for path in _collect_images(camera)]
    todo = [(camera, path) for (camera, path) in all_images if (camera, path.name) not in processed]
    print(f'[input] total={len(all_images):,}, already_done={len(all_images) - len(todo):,}, todo={len(todo):,}')
    write_header = not PARTIAL_MANIFEST_PATH.exists()
    error_count = 0
    with PARTIAL_MANIFEST_PATH.open('a', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = lambda x, **_: x
        for (index, (camera, path)) in enumerate(tqdm(todo, desc='VER4 BBox manifest', unit='bed'), start=1):
            try:
                image = core.imread_unicode(path)
                if image is None:
                    raise RuntimeError('image read failed')
                (slots, meta) = _map_one(image, camera, model, lettuce_seg, cam0_adapter, cam1_adapter)
                for slot in slots:
                    row = _slot_to_row(slot, meta, path, camera, image.shape[1], image.shape[0])
                    if row is not None:
                        writer.writerow(row)
                handle.flush()
            except Exception as error:
                error_count += 1
                print(f'[warn] {camera}/{path.name}: {type(error).__name__}: {error}')
            if index % checkpoint_every == 0:
                _write_progress({'state': 'running', 'last_camera': camera, 'last_image': path.name, 'processed_rows_source_images': len(processed) + index, 'total_source_images': len(all_images), 'error_count': error_count, 'partial_manifest': str(PARTIAL_MANIFEST_PATH)})
    if error_count:
        _write_progress({'state': 'needs_review', 'error_count': error_count, 'partial_manifest': str(PARTIAL_MANIFEST_PATH)})
        raise RuntimeError(f'manifest run had {error_count} image errors; partial CSV retained')
    PARTIAL_MANIFEST_PATH.replace(MANIFEST_PATH)
    _write_progress({'state': 'complete', 'manifest': str(MANIFEST_PATH), 'total_source_images': len(all_images)})
    print(f'[complete] manifest: {MANIFEST_PATH}')

def _read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f'generate manifest first: {MANIFEST_PATH}')
    with MANIFEST_PATH.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def _select_from_date(records: list[dict[str, str]], quota: int, rng: random.Random) -> list[dict[str, str]]:
    by_bed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_bed[record['bed_id']].append(record)
    for choices in by_bed.values():
        rng.shuffle(choices)
    bed_order = list(by_bed)
    rng.shuffle(bed_order)
    selected: list[dict[str, str]] = []
    while len(selected) < quota:
        moved = False
        for bed_id in bed_order:
            if not by_bed[bed_id] or len(selected) >= quota:
                continue
            selected.append(by_bed[bed_id].pop())
            moved = True
        if not moved:
            break
    return selected

def _date_quotas(records: list[dict[str, str]], target: int) -> dict[str, int]:
    dates = sorted({record['capture_date'] for record in records})
    (base, remainder) = divmod(target, len(dates))
    return {date: base + int(index < remainder) for (index, date) in enumerate(dates)}

def run_sample(seed: int, target_per_camera: int) -> None:
    if SAMPLE_PATH.exists():
        raise FileExistsError(f'sample manifest already exists; refusing overwrite: {SAMPLE_PATH}')
    rows = _read_manifest()
    candidates = [row for row in rows if row['is_train_candidate'] == '1' and row['visibility'] != 'OCCLUDED']
    selected: list[dict[str, str]] = []
    audit: dict[str, Any] = {'seed': seed, 'target_per_camera': target_per_camera, 'cameras': {}}
    for camera in ('cam0', 'cam1'):
        camera_rows = [row for row in candidates if row['camera'] == camera]
        if len(camera_rows) < target_per_camera:
            raise RuntimeError(f'not enough {camera} candidates: {len(camera_rows)}')
        quota_by_date = _date_quotas(camera_rows, target_per_camera)
        selected_camera: list[dict[str, str]] = []
        per_date_audit = []
        for (date, quota) in quota_by_date.items():
            date_rows = [row for row in camera_rows if row['capture_date'] == date]
            date_rng = random.Random(f'{seed}|{camera}|{date}')
            chosen = _select_from_date(date_rows, quota, date_rng)
            if len(chosen) != quota:
                raise RuntimeError(f'{camera}/{date}: requested {quota}, selected {len(chosen)}')
            selected_camera.extend(chosen)
            per_date_audit.append({'date': date, 'quota': quota, 'selected': len(chosen), 'unique_beds': len({row['bed_id'] for row in chosen}), 'unique_source_images': len({row['source_image_name'] for row in chosen})})
        if len({row['record_id'] for row in selected_camera}) != len(selected_camera):
            raise RuntimeError(f'duplicate record selected in {camera}')
        for row in selected_camera:
            row['sample_seed'] = str(seed)
            row['sample_strategy'] = 'camera_balanced__date_balanced__bed_round_robin__slot_random'
            row['split'] = 'UNASSIGNED'
        selected.extend(selected_camera)
        audit['cameras'][camera] = {'available_candidates': len(camera_rows), 'selected': len(selected_camera), 'per_date': per_date_audit}
    if len(selected) != 2 * target_per_camera:
        raise RuntimeError(f'sample count mismatch: {len(selected)}')
    sample_columns = MANIFEST_COLUMNS + ['sample_seed', 'sample_strategy', 'split']
    with SAMPLE_PATH.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: (row['camera'], row['capture_date'], row['bed_id'], row['record_id'])))
    audit['total_selected'] = len(selected)
    SAMPLE_AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[complete] selected {len(selected):,} crops: {SAMPLE_PATH}')

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    manifest = subparsers.add_parser('manifest', help='full BBox manifest generation')
    manifest.add_argument('--camera', choices=('cam0', 'cam1', 'all'), default='all')
    manifest.add_argument('--checkpoint-every', type=int, default=50)
    sample = subparsers.add_parser('sample', help='deterministic 3,000-crop selection')
    sample.add_argument('--seed', type=int, default=20260902)
    sample.add_argument('--target-per-camera', type=int, default=1500)
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    if args.command == 'manifest':
        cameras = ('cam0', 'cam1') if args.camera == 'all' else (args.camera,)
        run_manifest(cameras, args.checkpoint_every)
    else:
        run_sample(args.seed, args.target_per_camera)
if __name__ == '__main__':
    main()
