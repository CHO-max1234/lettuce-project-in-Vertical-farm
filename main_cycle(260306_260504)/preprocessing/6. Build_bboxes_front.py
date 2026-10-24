from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
'CAM2 Ver1 full manifest and SEG-centred crop materialization.\n\nInput is every image below ver2_cam2.  The frozen CAM2 Ver1 rail=t1-b1 board\nis reused without changing its phase or names.  Every visible slot is written\nto the manifest.  Only a deterministic, clean 2,000-slot subset is written as\nRGB crops under crop_bbox.\n\nExecution is image-chunked.  Completed chunk CSVs are durable checkpoints;\nrerun with --resume after an interruption.\n'
import argparse
import csv
import hashlib
import heapq
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import cv2
import numpy as np
from tqdm import tqdm
SCRIPT_DIR = Path(__file__).resolve().parent
from pipeline_paths import CODE_ROOT, BRIGHT_IMAGES, BBOX_ROOT
NAME_MATCHING_SCRIPT = CODE_ROOT / '5. Assign_slots_cam2.py'
INPUT_ROOT = BRIGHT_IMAGES / 'cam2'
OUTPUT_ROOT = BBOX_ROOT / 'cam2'
RUN_NAME = 'CAM2_VER1_BBOX'
EXPECTED_SOURCE_IMAGES = None
CHUNK_SIZE = 200
TARGET_CROP_COUNT = 0
SAMPLING_SEED = 'CAM2_VER1_SEG_CENTERED_CROP_20260904'
SEG_MARGIN_RATIO = 0.1
MIN_MARGIN_PX = 12
MIN_CELL_SCALE = 1.1
MAX_CELL_SCALE = 1.45
MIN_CLEAN_CELL_INSIDE_RATIO = 0.9
JPEG_QUALITY = 95

def load_name_matching_module():
    spec = importlib.util.spec_from_file_location('cam2_name_matching_ver1', NAME_MATCHING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Invalid input or state in load_name_matching_module (rejected condition: spec is None or spec.loader is None): {NAME_MATCHING_SCRIPT} ')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
VER1 = load_name_matching_module()
RAIL = VER1.RAIL
BASE = VER1.BASE
MANIFEST_FIELDS = ['record_id', 'source_rel_path', 'source_path', 'date', 'image_name', 'bed_id', 'camera', 'source_width', 'source_height', 'image_qc', 'front_rail_y', 'front_rail_mode', 'rail_a', 'rail_b', 'rail_front_x', 'visible_cols', 'slot_name', 'row', 'local_col', 'x_prior', 'y_prior', 'cell_polygon', 'cell_x1', 'cell_y1', 'cell_x2', 'cell_y2', 'cell_width', 'cell_height', 'cell_inside_image_ratio', 'seg_status', 'seg_id', 'seg_conf', 'seg_center_x', 'seg_center_y', 'seg_x1', 'seg_y1', 'seg_x2', 'seg_y2', 'seg_width', 'seg_height', 'seg_area', 'mask_coverage', 'cell_coverage', 'runner_up_coverage', 'center_distance', 'is_source_edge_contact', 'neighbor_center_count_in_crop', 'neighbor_bbox_overlap_count', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'crop_width', 'crop_height', 'is_crop_clipped', 'target_bbox_retention', 'target_bbox_fully_in_crop', 'oversize_seg_limited', 'clean_candidate', 'neighbor_tier']
IMAGE_SUMMARY_FIELDS = ['source_rel_path', 'image_name', 'date', 'source_width', 'source_height', 'front_rail_y', 'front_rail_mode', 'rail_a', 'rail_b', 'rail_front_x', 'rail_candidate_count', 'visible_cols', 'seg_count', 'assigned', 'unassigned', 'merge_review', 'image_qc', 'error']

@dataclass
class ImageSummary:
    source_rel_path: str
    image_name: str
    date: str
    source_width: int
    source_height: int
    front_rail_y: float | str
    front_rail_mode: str
    rail_a: float | str
    rail_b: float | str
    rail_front_x: float | str
    rail_candidate_count: int
    visible_cols: int
    seg_count: int
    assigned: int
    unassigned: int
    merge_review: int
    image_qc: str
    error: str

def as_int(value: str) -> int:
    return int(float(value))

def truth(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}

def imread_unicode(path: Path) -> np.ndarray:
    image = BASE.imread_unicode(path)
    if image is None:
        raise RuntimeError(f'Invalid input or state in imread_unicode (rejected condition: image is None): {path} ')
    return image

def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    (ok, encoded) = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f'JPEG encoding failed: {path}')
    encoded.tofile(str(path))

def serialize_polygon(points: np.ndarray) -> str:
    return ';'.join((f'{float(x):.3f},{float(y):.3f}' for (x, y) in points))

def finite(value: float) -> str:
    return '' if not np.isfinite(value) else f'{float(value):.5f}'

def bbox_from_polygon(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, int(math.floor(float(np.min(polygon[:, 0])))))
    y1 = max(0, int(math.floor(float(np.min(polygon[:, 1])))))
    x2 = min(width, int(math.ceil(float(np.max(polygon[:, 0])))))
    y2 = min(height, int(math.ceil(float(np.max(polygon[:, 1])))))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError('Invalid input or state in bbox_from_polygon (rejected condition: x2 <= x1 or y2 <= y1)')
    return (x1, y1, x2, y2)

def centered_bounds(center: float, size: int, limit: int) -> tuple[int, int]:
    size = min(max(1, size), limit)
    start = int(round(center - size / 2.0))
    end = start + size
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    return (max(0, start), min(limit, end))

def rect_intersection_area(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def crop_from_seg(*, source_width: int, source_height: int, cell_bbox: tuple[int, int, int, int], seg_bbox: tuple[float, float, float, float], seg_center_x: float, seg_center_y: float) -> tuple[dict, bool]:
    """Return a SEG-centred crop contract and its oversize/merge safeguard."""
    cell_width = cell_bbox[2] - cell_bbox[0]
    cell_height = cell_bbox[3] - cell_bbox[1]
    seg_width = max(1, int(round(seg_bbox[2] - seg_bbox[0])))
    seg_height = max(1, int(round(seg_bbox[3] - seg_bbox[1])))
    margin_x = max(MIN_MARGIN_PX, round(seg_width * SEG_MARGIN_RATIO))
    margin_y = max(MIN_MARGIN_PX, round(seg_height * SEG_MARGIN_RATIO))
    wanted_width = max(round(cell_width * MIN_CELL_SCALE), seg_width + 2 * margin_x)
    wanted_height = max(round(cell_height * MIN_CELL_SCALE), seg_height + 2 * margin_y)
    max_width = max(1, round(cell_width * MAX_CELL_SCALE))
    max_height = max(1, round(cell_height * MAX_CELL_SCALE))
    oversize = wanted_width > max_width or wanted_height > max_height
    crop_width = min(wanted_width, max_width)
    crop_height = min(wanted_height, max_height)
    (x1, x2) = centered_bounds(seg_center_x, crop_width, source_width)
    (y1, y2) = centered_bounds(seg_center_y, crop_height, source_height)
    crop_bbox = (x1, y1, x2, y2)
    seg_area = max(1.0, (seg_bbox[2] - seg_bbox[0]) * (seg_bbox[3] - seg_bbox[1]))
    retention = rect_intersection_area(crop_bbox, seg_bbox) / seg_area
    return ({'crop_x1': x1, 'crop_y1': y1, 'crop_x2': x2, 'crop_y2': y2, 'crop_width': x2 - x1, 'crop_height': y2 - y1, 'is_crop_clipped': int(x1 == 0 or y1 == 0 or x2 == source_width or (y2 == source_height)), 'target_bbox_retention': retention, 'target_bbox_fully_in_crop': int(retention >= 0.999999), 'oversize_seg_limited': int(oversize)}, oversize)

def parse_identity(relative_path: Path) -> tuple[str, str, str]:
    from pipeline_paths import identity
    info = identity(relative_path)
    return (info['capture_date'], info['bed_id'], info['camera'])

def process_image(model, image_path: Path, input_root: Path, image=None) -> tuple[ImageSummary, list[dict]]:
    relative_path = image_path.relative_to(input_root)
    (date, bed_id, camera) = parse_identity(relative_path)
    if image is None:
        image = imread_unicode(image_path)
    (height, width) = image.shape[:2]
    try:
        (front_y, front_score, front_mode) = VER1.detect_front_rail_y(image)
        (rail, candidate_count) = VER1.PREVIEW.choose_left_rail(image, front_y)
        (top, bottom) = VER1.transfer_board(rail, front_y, width / VER1.PREVIEW.REF_WIDTH)
        slots = VER1.make_slots(image, top, bottom)
        segments = BASE.predict_segments(model, image)
        top_pitch = float(np.median(np.linalg.norm(np.diff(top, axis=0), axis=1)))
        bottom_pitch = float(np.median(np.linalg.norm(np.diff(bottom, axis=0), axis=1)))
        pitch = 0.5 * (top_pitch + bottom_pitch)
        row_gap = float(np.median(np.linalg.norm(bottom - top, axis=1)))
        assignment = RAIL.assign_segments(image, slots, segments, pitch, row_gap)
        image_qc: list[str] = []
        if abs(width - VER1.PREVIEW.REF_WIDTH) > VER1.WIDTH_WARN_PX:
            image_qc.append('WIDTH_SCALE_REVIEW')
        rail_front_x = float(rail.x_at(front_y))
        if abs(rail_front_x) > 0.035 * width:
            image_qc.append('RAIL_FRONT_CROSS_REVIEW')
        if front_score < 20.0:
            image_qc.append('FRONT_RAIL_WEAK')
        if front_mode == 'SOBEL_CONTINUITY_NEAR_BOTTOM':
            image_qc.append('FRONT_RAIL_NEAR_BOTTOM_REVIEW')
        image_qc_text = 'OK' if not image_qc else '+'.join(image_qc)
        segment_lookup = {int(segment['seg_id']): segment for segment in segments}
        visible_cols = max((slot.local_col for slot in slots), default=0)
        rows: list[dict] = []
        for slot in slots:
            cell_bbox = bbox_from_polygon(slot.cell_polygon, width, height)
            base_row = {'record_id': hashlib.sha1(f'{relative_path.as_posix()}|{slot.slot_name}'.encode('utf-8')).hexdigest()[:20], 'source_rel_path': relative_path.as_posix(), 'source_path': str(image_path), 'date': date, 'image_name': image_path.name, 'bed_id': bed_id, 'camera': camera, 'source_width': width, 'source_height': height, 'image_qc': image_qc_text, 'front_rail_y': f'{front_y:.3f}', 'front_rail_mode': front_mode, 'rail_a': f'{rail.a:.6f}', 'rail_b': f'{rail.b:.3f}', 'rail_front_x': f'{rail_front_x:.3f}', 'visible_cols': visible_cols, 'slot_name': slot.slot_name, 'row': slot.row, 'local_col': slot.local_col, 'x_prior': f'{slot.x_prior:.3f}', 'y_prior': f'{slot.y_prior:.3f}', 'cell_polygon': serialize_polygon(slot.cell_polygon), 'cell_x1': cell_bbox[0], 'cell_y1': cell_bbox[1], 'cell_x2': cell_bbox[2], 'cell_y2': cell_bbox[3], 'cell_width': cell_bbox[2] - cell_bbox[0], 'cell_height': cell_bbox[3] - cell_bbox[1], 'cell_inside_image_ratio': f'{slot.cell_inside_image_ratio:.5f}', 'seg_status': 'NO_SEG_PRIOR', 'seg_id': '', 'seg_conf': '', 'seg_center_x': '', 'seg_center_y': '', 'seg_x1': '', 'seg_y1': '', 'seg_x2': '', 'seg_y2': '', 'seg_width': '', 'seg_height': '', 'seg_area': '', 'mask_coverage': finite(slot.mask_coverage), 'cell_coverage': finite(slot.cell_coverage), 'runner_up_coverage': finite(slot.runner_up_coverage), 'center_distance': finite(slot.center_distance), 'is_source_edge_contact': 0, 'neighbor_center_count_in_crop': '', 'neighbor_bbox_overlap_count': '', 'crop_x1': '', 'crop_y1': '', 'crop_x2': '', 'crop_y2': '', 'crop_width': '', 'crop_height': '', 'is_crop_clipped': '', 'target_bbox_retention': '', 'target_bbox_fully_in_crop': '', 'oversize_seg_limited': '', 'clean_candidate': 0, 'neighbor_tier': ''}
            if slot.seg_id is None:
                rows.append(base_row)
                continue
            segment = segment_lookup[int(slot.seg_id)]
            seg_bbox = tuple((float(value) for value in segment['bbox']))
            seg_status = 'SEG_MERGE_REVIEW' if slot.status == 'SEG_MERGE_REVIEW' else 'SEG_ASSIGNED'
            (crop_values, oversize) = crop_from_seg(source_width=width, source_height=height, cell_bbox=cell_bbox, seg_bbox=seg_bbox, seg_center_x=float(segment['cx']), seg_center_y=float(segment['cy']))
            source_edge = int(seg_bbox[0] <= 1.0 or seg_bbox[1] <= 1.0 or seg_bbox[2] >= width - 1.0 or (seg_bbox[3] >= height - 1.0))
            crop_bbox = (crop_values['crop_x1'], crop_values['crop_y1'], crop_values['crop_x2'], crop_values['crop_y2'])
            other_segments = [candidate for candidate in segments if int(candidate['seg_id']) != int(slot.seg_id)]
            neighbor_center_count = sum((crop_bbox[0] <= float(candidate['cx']) < crop_bbox[2] and crop_bbox[1] <= float(candidate['cy']) < crop_bbox[3] for candidate in other_segments))
            neighbor_overlap_count = sum((rect_intersection_area(crop_bbox, tuple((float(value) for value in candidate['bbox']))) > 0.0 for candidate in other_segments))
            clean = seg_status == 'SEG_ASSIGNED' and source_edge == 0 and (not oversize) and (crop_values['target_bbox_fully_in_crop'] == 1) and (slot.cell_inside_image_ratio >= MIN_CLEAN_CELL_INSIDE_RATIO)
            base_row.update({'seg_status': seg_status, 'seg_id': int(slot.seg_id), 'seg_conf': f"{float(segment['conf']):.5f}", 'seg_center_x': f"{float(segment['cx']):.3f}", 'seg_center_y': f"{float(segment['cy']):.3f}", 'seg_x1': f'{seg_bbox[0]:.3f}', 'seg_y1': f'{seg_bbox[1]:.3f}', 'seg_x2': f'{seg_bbox[2]:.3f}', 'seg_y2': f'{seg_bbox[3]:.3f}', 'seg_width': f'{seg_bbox[2] - seg_bbox[0]:.3f}', 'seg_height': f'{seg_bbox[3] - seg_bbox[1]:.3f}', 'seg_area': f"{float(segment['area']):.3f}", 'is_source_edge_contact': source_edge, 'neighbor_center_count_in_crop': neighbor_center_count, 'neighbor_bbox_overlap_count': neighbor_overlap_count, 'target_bbox_retention': f"{crop_values['target_bbox_retention']:.6f}", 'clean_candidate': int(clean), 'neighbor_tier': 0 if neighbor_center_count == 0 else 1, **crop_values})
            rows.append(base_row)
        summary = ImageSummary(source_rel_path=relative_path.as_posix(), image_name=image_path.name, date=date, source_width=width, source_height=height, front_rail_y=round(float(front_y), 3), front_rail_mode=front_mode, rail_a=round(float(rail.a), 6), rail_b=round(float(rail.b), 3), rail_front_x=round(rail_front_x, 3), rail_candidate_count=candidate_count, visible_cols=visible_cols, seg_count=len(segments), assigned=int(assignment['assigned']), unassigned=int(assignment['unassigned']), merge_review=int(assignment['merge_review']), image_qc=image_qc_text, error='')
        return (summary, rows)
    except Exception as error:
        if not (type(error) is RuntimeError and str(error) in {'LEFT_CONTINUOUS_RAIL_NOT_FOUND', 'FRONT_RAIL_SEARCH_RANGE_EMPTY'}):
            raise
        return (ImageSummary(source_rel_path=relative_path.as_posix(), image_name=image_path.name, date=date, source_width=width, source_height=height, front_rail_y='', front_rail_mode='', rail_a='', rail_b='', rail_front_x='', rail_candidate_count=0, visible_cols=0, seg_count=0, assigned=0, unassigned=0, merge_review=0, image_qc='PROCESS_REVIEW', error=f'{type(error).__name__}: {error}'), [])

def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    tmp_path.replace(path)
    return count

def source_paths(input_root: Path) -> list[Path]:
    return sorted((path for path in input_root.rglob('*') if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and ('_cam2' in path.stem.lower())))

def chunk_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f'cam2_manifest_chunk_{chunk_index:04d}.csv'

def image_chunk_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f'cam2_image_chunk_{chunk_index:04d}.csv'

def process_chunks(model, paths: Sequence[Path], args: argparse.Namespace, chunks_dir: Path) -> None:
    chunk_count = math.ceil(len(paths) / args.chunk_size)
    progress = tqdm(total=len(paths), desc=f'{RUN_NAME} manifest', unit='image')
    for chunk_index in range(chunk_count):
        start = chunk_index * args.chunk_size
        end = min(len(paths), start + args.chunk_size)
        manifest_chunk = chunk_path(chunks_dir, chunk_index)
        image_chunk = image_chunk_path(chunks_dir, chunk_index)
        if args.resume and manifest_chunk.is_file() and image_chunk.is_file():
            progress.update(end - start)
            continue
        if manifest_chunk.exists() or image_chunk.exists():
            raise RuntimeError(f'Invalid input or state in process_chunks (rejected condition: manifest_chunk.exists() or image_chunk.exists()): {chunk_index} ')
        manifest_rows: list[dict] = []
        image_rows: list[dict] = []
        for image_path in paths[start:end]:
            (summary, rows) = process_image(model, image_path, args.input_root)
            image_rows.append(asdict(summary))
            manifest_rows.extend(rows)
            progress.update(1)
        write_csv(manifest_chunk, MANIFEST_FIELDS, manifest_rows)
        write_csv(image_chunk, IMAGE_SUMMARY_FIELDS, image_rows)
    progress.close()

def consolidate_chunks(paths: Sequence[Path], args: argparse.Namespace, chunks_dir: Path, manifest_dir: Path) -> tuple[Path, Path, int, int]:
    full_manifest = manifest_dir / 'cam2_bbox_manifest_VER1.csv'
    image_summary = manifest_dir / 'cam2_image_summary_VER1.csv'
    chunk_count = math.ceil(len(paths) / args.chunk_size)

    def manifest_rows() -> Iterable[dict]:
        for index in range(chunk_count):
            path = chunk_path(chunks_dir, index)
            if not path.is_file():
                raise FileNotFoundError(f'Invalid input or state in manifest_rows (rejected condition: not path.is_file()): {path} ')
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                yield from csv.DictReader(handle)

    def image_rows() -> Iterable[dict]:
        for index in range(chunk_count):
            path = image_chunk_path(chunks_dir, index)
            if not path.is_file():
                raise FileNotFoundError(f'Invalid input or state in image_rows (rejected condition: not path.is_file()): {path} ')
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                yield from csv.DictReader(handle)
    manifest_count = write_csv(full_manifest, MANIFEST_FIELDS, manifest_rows())
    image_count = write_csv(image_summary, IMAGE_SUMMARY_FIELDS, image_rows())
    return (full_manifest, image_summary, manifest_count, image_count)

def priority_hash(record_id: str) -> int:
    return int(hashlib.sha256(f'{SAMPLING_SEED}|{record_id}'.encode('utf-8')).hexdigest()[:16], 16)

def candidate_key(row: dict[str, str]) -> tuple[str, str]:
    return (row['date'], row['row'])

def quota_by_stratum(counts: dict[tuple[str, str], int], target_count: int) -> dict[tuple[str, str], int]:
    if sum(counts.values()) < target_count:
        raise RuntimeError(f'Invalid input or state in quota_by_stratum (rejected condition: sum(counts.values()) < target_count): {sum(counts.values())} {target_count} ')
    quota = {key: 0 for key in sorted(counts)}
    for _ in range(target_count):
        candidates = [key for key in quota if quota[key] < counts[key]]
        min_quota = min((quota[key] for key in candidates))
        key = min((key for key in candidates if quota[key] == min_quota))
        quota[key] += 1
    return quota

def select_records(full_manifest: Path, target_count: int) -> tuple[set[str], dict[tuple[str, str], int], int]:
    counts: dict[tuple[str, str], int] = {}
    with full_manifest.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if truth(row['clean_candidate']):
                key = candidate_key(row)
                counts[key] = counts.get(key, 0) + 1
    quota = quota_by_stratum(counts, target_count)
    heaps: dict[tuple[str, str], list[tuple[tuple[int, int, str], str]]] = {key: [] for (key, value) in quota.items() if value > 0}
    with full_manifest.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if not truth(row['clean_candidate']):
                continue
            key = candidate_key(row)
            limit = quota[key]
            tier = as_int(row['neighbor_tier'])
            rank = (-tier, -priority_hash(row['record_id']), row['record_id'])
            heap = heaps[key]
            if len(heap) < limit:
                heapq.heappush(heap, (rank, row['record_id']))
            elif rank > heap[0][0]:
                heapq.heapreplace(heap, (rank, row['record_id']))
    selected = {record_id for heap in heaps.values() for (_, record_id) in heap}
    if len(selected) != target_count:
        raise RuntimeError(f'Invalid input or state in select_records (rejected condition: len(selected) != target_count): {len(selected)} {target_count} ')
    return (selected, quota, sum(counts.values()))

def materialize_selected(full_manifest: Path, selected_ids: set[str], args: argparse.Namespace, crop_root: Path, manifest_dir: Path) -> tuple[Path, int]:
    crop_root.mkdir(parents=True, exist_ok=True)
    train_manifest = manifest_dir / 'cam2_train2000_manifest_VER1.csv'
    train_fields = [*MANIFEST_FIELDS, 'selection_stratum', 'crop_name']

    def selected_rows() -> Iterable[dict]:
        written = 0
        with full_manifest.open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['record_id'] not in selected_ids:
                    continue
                source = args.input_root / Path(row['source_rel_path'])
                image = imread_unicode(source)
                (x1, y1, x2, y2) = (as_int(row[key]) for key in ('crop_x1', 'crop_y1', 'crop_x2', 'crop_y2'))
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    raise RuntimeError(f"Invalid input or state in selected_rows (rejected condition: crop.size == 0): {row['record_id']} ")
                crop_name = f"{row['date']}__{Path(row['image_name']).stem}__{row['slot_name']}__{row['record_id']}.jpg"
                imwrite_unicode(crop_root / crop_name, crop)
                row['selection_stratum'] = f"{row['date']}|{row['row']}"
                row['crop_name'] = crop_name
                written += 1
                yield row
        if written != len(selected_ids):
            raise RuntimeError(f'Invalid input or state in selected_rows (rejected condition: written != len(selected_ids)): {written} {len(selected_ids)} ')
    count = write_csv(train_manifest, train_fields, selected_rows())
    return (train_manifest, count)

def write_run_report(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp_path.replace(path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CAM2 Ver1 full manifest and 2,000 SEG-centred crops')
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--target-crops', type=int, default=TARGET_CROP_COUNT)
    parser.add_argument('--resume', action='store_true', help='parse args: chunk')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.chunk_size < 1 or args.target_crops < 0:
        raise ValueError('Invalid input or state in main (rejected condition: args.chunk_size < 1 or args.target_crops < 0)')
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    paths = source_paths(args.input_root)
    if EXPECTED_SOURCE_IMAGES is not None and len(paths) != EXPECTED_SOURCE_IMAGES:
        raise RuntimeError(f'Invalid input or state in main (rejected condition: EXPECTED_SOURCE_IMAGES is not None and len(paths) != EXPECTED_SOURCE_IMAGES): {EXPECTED_SOURCE_IMAGES} {len(paths)} ')
    if BASE.sha256_file(BASE.MODEL_PATH) != BASE.MODEL_SHA256:
        raise RuntimeError('Invalid input or state in main (rejected condition: BASE.sha256_file(BASE.MODEL_PATH) != BASE.MODEL_SHA256)')
    manifest_dir = args.output_root / '01_manifest'
    chunks_dir = manifest_dir / 'chunks'
    crop_root = args.output_root / 'crop_bbox'
    if not args.resume and (chunks_dir.exists() or crop_root.exists()):
        existing = list(chunks_dir.glob('*.csv')) if chunks_dir.exists() else []
        existing += list(crop_root.glob('*.jpg')) if crop_root.exists() else []
        if existing:
            raise RuntimeError('Invalid input or state in main (rejected condition: existing)')
    manifest_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {'run_name': RUN_NAME, 'input_root': str(args.input_root), 'source_images': len(paths), 'source_dates': len({parse_identity(path.relative_to(args.input_root))[0] for path in paths}), 'target_crops': args.target_crops, 'chunk_size': args.chunk_size, 'model_path': str(BASE.MODEL_PATH), 'model_sha256': BASE.MODEL_SHA256, 'geometry_contract': 'CAM2_VER1__rail_is_t1_b1__fixed_empty_bed_board', 'crop_contract': 'seg_center__seg_bbox_plus_10pct_margin__per_slot_cell_scale_1.10_to_1.45', 'sampling_contract': 'SEG_ASSIGNED__not_source_edge__not_oversize__full_target_bbox__cell_inside_ge_0.90__date_x_row_stratified'}
    write_run_report(manifest_dir / 'cam2_ver1_run_contract.json', run_contract)
    model = BASE.YOLO(str(BASE.MODEL_PATH))
    process_chunks(model, paths, args, chunks_dir)
    (full_manifest, image_summary, manifest_rows, image_rows) = consolidate_chunks(paths, args, chunks_dir, manifest_dir)
    if args.target_crops:
        (selected_ids, quota, clean_candidates) = select_records(full_manifest, args.target_crops)
        (train_manifest, crops_written) = materialize_selected(full_manifest, selected_ids, args, crop_root, manifest_dir)
    else:
        (quota, clean_candidates, train_manifest, crops_written) = ({}, 0, '', 0)
    report = {**run_contract, 'status': 'complete', 'manifest_path': str(full_manifest), 'image_summary_path': str(image_summary), 'train_manifest_path': str(train_manifest), 'crop_root': str(crop_root), 'manifest_rows': manifest_rows, 'image_rows': image_rows, 'clean_candidates': clean_candidates, 'selected_crops': crops_written, 'strata': len(quota), 'quota_min': min(quota.values(), default=0), 'quota_max': max(quota.values(), default=0)}
    write_run_report(manifest_dir / 'cam2_ver1_run_report.json', report)
    print(json.dumps(report, ensure_ascii=False))
if __name__ == '__main__':
    main()
