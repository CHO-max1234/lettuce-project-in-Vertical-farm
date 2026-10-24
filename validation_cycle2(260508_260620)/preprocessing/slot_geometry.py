"""Cycle-specific physical-slot geometry and provenance mapping used by whole-cycle processing."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT.parent))
from runtime import moved_path
REFERENCE_ROOT = CYCLE.MAIN_PREPROCESS
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
INPUT_ROOT = DATA_ROOT / '3. 밝기보정과 scale'
OUTPUT_ROOT = DATA_ROOT / '4. 자리배치'
TEST_ROOT = OUTPUT_ROOT / 'testimages'
RESULT_ROOT = OUTPUT_ROOT / 'test_results'
OCR_MANIFEST = DATA_ROOT / '2. OCR 기록' / 'manifest.csv'
BRIGHT_MANIFEST = INPUT_ROOT / '_실행기록' / 'manifest.json'
SCALE_MANIFEST = INPUT_ROOT / '_실행기록' / 'scale' / 'scale_map.csv'
NAMESPACE = 'image_left_to_right_v1'
CAM1_NAMESPACE = 'physical_12col_v1'
CAM1_RIGHTMOST_COL = 12
CAM1_ALLOW_LEGACY_TOWEL_INFERENCE = False
SEED = 260930
COPY_WORKERS = 4
SAMPLE_DATES = {'cam0': ['260508', '260511', '260514', '260517', '260520', '260612', '260614', '260616', '260618', '260620'], 'cam1': ['260508', '260512', '260516', '260522', '260527', '260612', '260614', '260616', '260618', '260620'], 'cam2': ['260508', '260510', '260512', '260514', '260516', '260518', '260521', '260523', '260525', '260527']}
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
cv2.setNumThreads(1)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def keypath(path):
    return os.path.normcase(os.path.abspath(os.fspath(moved_path(path))))

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def encode(value):
    return json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def write_csv(path, rows):
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    if not fields:
        fields = ['status']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encode(v) if isinstance(v, (dict, list, tuple, np.ndarray)) else clean(v) for (k, v) in row.items()})

def iter_json_array(path):
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8-sig') as f:
        buffer = f.read(1024 * 1024).lstrip()
        if not buffer.startswith('['):
            raise ValueError('brightness manifest must be a JSON array')
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip(' \t\r\n,')
            if buffer.startswith(']'):
                return
            try:
                (row, end) = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                block = f.read(1024 * 1024)
                if not block:
                    raise ValueError('brightness manifest truncated')
                buffer += block
                continue
            yield row
            buffer = buffer[end:]

def choose_samples():
    with OCR_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        ocr = {keypath(r['current_path']): r for r in csv.DictReader(f)}
    rng = random.Random(SEED)
    (chosen, used_beds) = ([], {cam: set() for cam in SAMPLE_DATES})
    for (cam, dates) in SAMPLE_DATES.items():
        for (index, date) in enumerate(dates):
            candidates = sorted((INPUT_ROOT / cam / date).glob('*.png'))
            if not candidates:
                raise FileNotFoundError(f'Invalid input or state in choose_samples (not candidates): {cam} {date} ')
            records = []
            for p in candidates:
                old = DATA_ROOT / '1. ROI_BOX 원본' / cam / date / (p.stem + '.jpg')
                record = ocr.get(keypath(old))
                if record is None:
                    raise ValueError(f'Invalid input or state in choose_samples (record is None): {p} ')
                records.append((p, record))
            want_review = index in (2, 8)
            pool = [(p, r) for (p, r) in records if (r['status'] == 'REVIEW') == want_review]
            pool = pool or records
            diverse = [(p, r) for (p, r) in pool if p.stem.split('_')[0] not in used_beds[cam]]
            (p, r) = rng.choice(diverse or pool)
            used_beds[cam].add(p.stem.split('_')[0])
            chosen.append(dict(camera=cam, sample_index=index + 1, date_folder=date, input_path=str(p), test_path=str(TEST_ROOT / cam / p.name), image_name=p.name, selection_seed=SEED, selection_reason='date_stratified_May_June' if cam != 'cam2' else 'date_stratified_May', ocr_current_path=r['current_path'], ocr_action=r['status'], ocr_status=r['ocr_status'], old_key=r['old_key'], new_key=r['new_key'], ocr_source_sha256=r['sha256'], old_filename=r['old_filename'], ocr_bed_id=r['new_bed_id'], bed_identity_status='REVIEW' if r['status'] == 'REVIEW' else 'CONFIRMED'))
    return chosen

def attach_lineage(samples):
    selected = {keypath(r['input_path']): r for r in samples}
    for bright in iter_json_array(BRIGHT_MANIFEST):
        target = selected.get(keypath(bright['output_path']))
        if target is not None:
            if keypath(bright['source_path']) != keypath(target['ocr_current_path']):
                raise ValueError("Invalid input or state in attach_lineage (keypath(bright['source_path']) != keypath(target['ocr_current_path']))")
            if bright['source_sha256'] != target['ocr_source_sha256']:
                raise ValueError("Invalid input or state in attach_lineage (bright['source_sha256'] != target['ocr_source_sha256'])")
            target.update(source_sha256=bright['output_sha256'], source_width=bright['shape'][1], source_height=bright['shape'][0], brightness_status=bright['status'])
    with SCALE_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        for scale in csv.DictReader(f):
            target = selected.get(keypath(scale['source_path']))
            if target is not None:
                if scale['source_sha256'] != target['source_sha256']:
                    raise ValueError("Invalid input or state in attach_lineage (scale['source_sha256'] != target['source_sha256'])")
                for k in ['mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_qc', 'scale_version', 'geometry_group']:
                    target[k] = scale[k]
                target['scale_image_key'] = scale['image_key']
    for r in samples:
        if not r.get('source_sha256') or not r.get('scale_image_key'):
            raise ValueError(f"Invalid input or state in attach_lineage (not r.get('source_sha256') or not r.get('scale_image_key')): {r['image_name']} ")

def copy_sample(row):
    (source, target) = (Path(row['input_path']), Path(row['test_path']))
    if digest(source) != row['source_sha256']:
        raise ValueError(f"Invalid input or state in copy_sample (digest(source) != row['source_sha256']): {source} ")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != row['source_sha256']:
            raise FileExistsError(f"Invalid input or state in copy_sample (digest(target) != row['source_sha256']): {target} ")
    else:
        shutil.copy2(source, target)
    if digest(target) != row['source_sha256']:
        raise ValueError(f"Invalid input or state in copy_sample (digest(target) != row['source_sha256']): {target} ")
    return row

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def load_references():
    sys.path.insert(0, str(REFERENCE_ROOT))
    sys.path.insert(0, str(REFERENCE_ROOT / '_support' / 'cam01'))
    core = importlib.import_module('slot_geometry')
    lettuce = importlib.import_module('plant_segmentation')
    cam0 = importlib.import_module('cam0_geometry')
    cam1 = importlib.import_module('cam1_geometry')
    cam2 = load_module('season3_cam2_geometry_reference', REFERENCE_ROOT / '5. Assign_slots_cam2.py')
    if digest(lettuce.LETTUCE_SEG_MODEL_PATH).lower() != lettuce.LETTUCE_SEG_SHA256.lower():
        raise ValueError('lettuce SEG frozen weight hash mismatch')
    return (core, lettuce, cam0, cam1, cam2)

def slot_dict(slot, cam, width, height):
    if cam != 'cam2':
        r = slot.to_dict()
        polygon = np.array([[r['cell_bbox'][0], r['cell_bbox'][1]], [r['cell_bbox'][2], r['cell_bbox'][1]], [r['cell_bbox'][2], r['cell_bbox'][3]], [r['cell_bbox'][0], r['cell_bbox'][3]]], float)
        return dict(reference_slot_name=r['canonical_id'], reference_local_col=r['local_col'], row=r['row'], x_prior=r['x_prior'], y_prior=r['y_prior'], slot_center_x=r['x'], slot_center_y=r['y'], cell_bbox=r['cell_bbox'], cell_polygon=polygon, bbox=r['bbox'], seg_id=r['seg_id'], seg_conf=r['seg_conf'], seg_center_x=r['seg_center_x'], seg_center_y=r['seg_center_y'], seg_bbox=r['seg_bbox'], seg_area=r['seg_area'], seg_status=r['seg_status'], visibility=r['visibility'], reference_reg_status=r['reg_status'], evidence_kind=r['evidence_kind'], evidence_score=r['evidence_score'])
    p = np.asarray(slot.cell_polygon, float)
    box = (max(0, math.floor(p[:, 0].min())), max(0, math.floor(p[:, 1].min())), min(width, math.ceil(p[:, 0].max())), min(height, math.ceil(p[:, 1].max())))
    return dict(reference_slot_name=slot.slot_name, reference_local_col=slot.local_col, row=slot.row, x_prior=slot.x_prior, y_prior=slot.y_prior, slot_center_x=slot.seg_center_x if np.isfinite(slot.seg_center_x) else slot.x_prior, slot_center_y=slot.seg_center_y if np.isfinite(slot.seg_center_y) else slot.y_prior, cell_bbox=box, cell_polygon=p, bbox=slot.bbox or box, seg_id=slot.seg_id, seg_conf=slot.seg_conf, seg_center_x=slot.seg_center_x, seg_center_y=slot.seg_center_y, seg_bbox=slot.seg_bbox, seg_area=slot.seg_area, seg_status=slot.status, visibility=slot.visibility, cell_inside_image_ratio=slot.cell_inside_image_ratio, mask_coverage=slot.mask_coverage, runner_up_coverage=slot.runner_up_coverage)

def map_image(image, cam, model, references):
    (core, lettuce, cam0, cam1, cam2) = references
    raw = lettuce.predict_lettuce_segments(model, image)
    segments = raw
    if cam == 'cam2':
        (h, w) = image.shape[:2]
        (front_y, front_score, front_mode) = cam2.detect_front_rail_y(image)
        (rail, count) = cam2.PREVIEW.choose_left_rail(image, front_y)
        (top, bottom) = cam2.transfer_board(rail, front_y, w / cam2.PREVIEW.REF_WIDTH)
        slots = cam2.make_slots(image, top, bottom)
        pitch = float(np.median(np.linalg.norm(np.diff(top, axis=0), axis=1)) + np.median(np.linalg.norm(np.diff(bottom, axis=0), axis=1))) / 2
        gap = float(np.median(np.linalg.norm(bottom - top, axis=1)))
        assignment = cam2.RAIL.assign_segments(image, slots, segments, pitch, gap)
        flags = []
        if front_score < 20:
            flags.append('FRONT_RAIL_WEAK')
        if abs(rail.x_at(front_y)) > 0.035 * w:
            flags.append('RAIL_FRONT_CROSS_REVIEW')
        if front_mode == 'SOBEL_CONTINUITY_NEAR_BOTTOM':
            flags.append('FRONT_RAIL_NEAR_BOTTOM_REVIEW')
        if assignment['unassigned']:
            flags.append('UNASSIGNED_SEG')
        if assignment['merge_review']:
            flags.append('SEG_MERGE_REVIEW')
        meta = dict(status='CAM2_GEOMETRY_OK' if not flags else 'CAM2_GEOMETRY_REVIEW', qc_flags=flags, pitch=pitch, front_rail_y=front_y, front_rail_mode=front_mode, rail_a=rail.a, rail_b=rail.b, rail_candidates=count, board_top=top, board_bottom=bottom)
    else:
        if cam == 'cam0':
            (slots, meta) = cam0.map_cam0_v3(image, segments)
            options = dict(enforce_cell_corridor=True)
        else:
            (segments, rejected) = cam1.filter_cam1_segments(image, raw)
            occlusion_candidate = cam1.detect_right_occlusion(image)
            original_occlusion_detector = cam1.detect_right_occlusion
            try:
                if not CAM1_ALLOW_LEGACY_TOWEL_INFERENCE:
                    cam1.detect_right_occlusion = lambda _: dict(present=False, start_x=None, end_x=None, run_ratio=0.0)
                (slots, meta) = cam1.map_cam1_v3(image, segments)
            finally:
                cam1.detect_right_occlusion = original_occlusion_detector
            meta['legacy_occlusion_candidate'] = occlusion_candidate
            meta['numbering_anchor'] = 'RIGHTMOST_T12_B12'
            if occlusion_candidate['present'] and (not CAM1_ALLOW_LEGACY_TOWEL_INFERENCE):
                meta.setdefault('qc_flags', []).append('LEAF_AS_TOWEL_CANDIDATE_REJECTED')
            meta['rejected_segment_count'] = len(rejected)
            options = dict(enforce_cell_corridor=True, conservative_prior_gate=False, prune_single_curve_outlier=not bool(meta.get('towel_present')))
        if slots:
            refined = lettuce.refine_slots_with_segments(slots, segments, image.shape[1], image.shape[0], bed_bbox=meta.get('bed_bbox'), **options)
            meta.update({k: v for (k, v) in refined.items() if k != 'assigned_by_slot_object_id'})
    (h, w) = image.shape[:2]
    valid = [s for s in slots if s.local_col > 0 and (cam == 'cam2' or s.cell_bbox is not None)]
    mapped = [slot_dict(s, cam, w, h) for s in valid]
    ordered_cols = sorted({r['reference_local_col'] for r in mapped}, key=lambda c: np.mean([r['x_prior'] for r in mapped if r['reference_local_col'] == c]))
    crosswalk = {c: i + 1 for (i, c) in enumerate(ordered_cols)}
    for r in mapped:
        r['local_col'] = crosswalk[r['reference_local_col']]
        r['col'] = CAM1_RIGHTMOST_COL - len(ordered_cols) + r['local_col'] if cam == 'cam1' else r['local_col']
        r['slot_name'] = r['canonical_id'] = r['physical_slot'] = f"{r['row']}{r['col']}"
        r['slot_namespace_version'] = CAM1_NAMESPACE if cam == 'cam1' else NAMESPACE
        r['numbering_status'] = 'RIGHT_ANCHOR_12_DESCENDING' if cam == 'cam1' else 'LEFT_TO_RIGHT_GRID'
        if cam == 'cam1':
            r['legacy_slot_id'] = r['reference_slot_name']
    return (mapped, meta, raw)

def color(row, col):
    index = (col - 1) * 2 + (row == 'b')
    hsv = np.uint8([[[int(index * 47 % 180), 235, 255]]])
    return tuple((int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))

def label(canvas, text, center, ink, scale, header_height, occupied):
    (font, thickness) = (cv2.FONT_HERSHEY_SIMPLEX, max(1, round(3 * scale)))
    fs = 0.95 * scale
    ((tw, th), baseline) = cv2.getTextSize(text, font, fs, thickness)
    candidates = []
    for dy in (-12 * scale, th + 16 * scale, -th - 24 * scale, 2 * th + 24 * scale):
        for dx in (8 * scale, -tw - 8 * scale, -tw / 2):
            x = int(np.clip(center[0] + dx, 3, max(3, canvas.shape[1] - tw - 5)))
            y = int(np.clip(center[1] + dy, header_height + th + 4, canvas.shape[0] - baseline - 3))
            rect = (x - 3, y - th - 4, x + tw + 3, y + baseline + 3)
            overlap = sum((max(0, min(rect[2], r[2]) - max(rect[0], r[0])) * max(0, min(rect[3], r[3]) - max(rect[1], r[1])) for r in occupied))
            candidates.append((overlap, x, y, rect))
    (_, x, y, rect) = min(candidates, key=lambda c: c[0])
    occupied.append(rect)
    cv2.rectangle(canvas, (x - 3, y - th - 4), (x + tw + 3, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, fs, ink, thickness, cv2.LINE_AA)

def draw_overlay(image, mapped, segments, summary, meta):
    (h, w) = image.shape[:2]
    scale = max(0.55, min(1.1, w / 2200))
    head = max(54, round(65 * scale))
    canvas = cv2.copyMakeBorder(image, head, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    offset = np.array([0, head])
    thick = max(2, round(4 * scale))
    seg_lookup = {s['seg_id']: s for s in segments}
    occupied = []
    for (row, ink) in [('t', (255, 255, 0)), ('b', (255, 0, 255))]:
        group = sorted([s for s in mapped if s['row'] == row], key=lambda s: s['col'])
        priors = np.array([[s['x_prior'], s['y_prior'] + head] for s in group])
        if len(priors) > 1:
            cv2.polylines(canvas, [np.rint(priors).astype(np.int32)], False, ink, max(2, thick - 1), cv2.LINE_AA)
    for r in mapped:
        ink = color(r['row'], r['col'])
        cv2.polylines(canvas, [np.rint(np.asarray(r['cell_polygon']) + offset).astype(np.int32)], True, ink, thick, cv2.LINE_AA)
        prior = tuple(np.rint([r['x_prior'], r['y_prior'] + head]).astype(int))
        center = tuple(np.rint([r['slot_center_x'], r['slot_center_y'] + head]).astype(int))
        cv2.circle(canvas, prior, max(4, round(8 * scale)), (255, 255, 0) if r['row'] == 't' else (255, 0, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(canvas, prior, center, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA, tipLength=0.18)
        cv2.circle(canvas, center, max(4, round(8 * scale)), ink, -1, cv2.LINE_AA)
        segment = seg_lookup.get(r['seg_id'])
        if segment is not None:
            cv2.polylines(canvas, [np.rint(segment['polygon'] + offset).astype(np.int32)], True, ink, thick, cv2.LINE_AA)
            evidence = tuple(np.rint([segment['cx'], segment['cy'] + head]).astype(int))
            cv2.drawMarker(canvas, evidence, ink, cv2.MARKER_TILTED_CROSS, max(8, round(18 * scale)), 2)
            if r['bbox']:
                (x1, y1, x2, y2) = (round(v) for v in r['bbox'])
                cv2.rectangle(canvas, (x1, y1 + head), (x2, y2 + head), ink, thick, cv2.LINE_AA)
        suffix = 'SEG' if segment else 'PRIOR'
        if r.get('visibility') == 'OCCLUDED':
            suffix = 'OCCLUDED'
        label(canvas, f"{r['slot_name']} {suffix}", center, ink, scale, head, occupied)
    for evidence_list in meta.get('hole_evidence', {}).values():
        for e in evidence_list:
            cv2.circle(canvas, (round(e['x']), round(e['y']) + head), max(3, round(7 * scale)), (0, 180, 255), 2, cv2.LINE_AA)
    rule = 'RIGHT_ANCHOR_T12_B12' if summary['camera'] == 'cam1' else 'LEFT_TO_RIGHT'
    lines = [f"{summary['camera'].upper()} S3 | {rule} | cols={summary['column_count']} | slots={len(mapped)} | seg={len(segments)} | assigned={summary['assigned_segment_count']} | {summary['processing_status']}", f"{summary['image_name']} | OCR={summary['ocr_action']} | QC={summary['qc_flags'] or 'none'}"]
    canvas[:head] = 0
    for (index, line) in enumerate(lines):
        fs = 0.7 * scale if index == 0 else 0.5 * scale
        max_width = w - 16
        ((tw, _), _) = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if tw > max_width:
            fs *= max_width / tw
        cv2.putText(canvas, line, (8, round(head * (0.4 if index == 0 else 0.83))), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)
    return canvas

def process_sample(sample, model, refs):
    path = Path(sample['test_path'])
    image = refs[0].imread_unicode(path)
    if image is None:
        raise ValueError(f'Invalid input or state in process_sample (image is None): {path} ')
    (h, w) = image.shape[:2]
    if (w, h) != (sample['source_width'], sample['source_height']):
        raise ValueError('brightness shape mismatch')
    (mapped, meta, segments) = map_image(image, sample['camera'], model, refs)
    flags = list(meta.get('qc_flags', []))
    assigned_ids = {r['seg_id'] for r in mapped if r['seg_id'] is not None}
    unassigned_count = sum((s['seg_id'] not in assigned_ids for s in segments))
    if unassigned_count:
        flags.append('UNASSIGNED_SEG_FOR_BBOX_REVIEW')
    if sample['ocr_action'] == 'REVIEW':
        flags.append('OCR_IDENTITY_REVIEW')
    if not mapped:
        flags.append('NO_SLOT_GEOMETRY')
    for r in mapped:
        if not (0 <= r['x_prior'] < w and 0 <= r['y_prior'] < h):
            flags.append('PARTIAL_SLOT_PRIOR_OUTSIDE_IMAGE')
    flags = list(dict.fromkeys(flags))
    overlay = RESULT_ROOT / sample['camera'] / (path.stem + '_overlay.jpg')
    summary = dict(sample, source_path=sample['input_path'], overlay_path=str(overlay), processing_status='REVIEW' if flags else 'OK', map_status=meta.get('status', 'UNKNOWN'), column_count=len({r['col'] for r in mapped}), slot_count=len(mapped), segment_count=len(segments), assigned_segment_count=sum((r['seg_id'] is not None for r in mapped)), unassigned_segment_count=unassigned_count, qc_flags='|'.join(flags), slot_namespace_version=CAM1_NAMESPACE if sample['camera'] == 'cam1' else NAMESPACE, visual_review='PENDING', geometry_details={k: v for (k, v) in meta.items() if k not in ('hole_evidence', 'lettuce_evidence', 'green_mask')})
    lookup = {s['seg_id']: s for s in segments}
    for r in mapped:
        s = lookup.get(r['seg_id'])
        r.update({k: sample[k] for k in ('camera', 'source_sha256', 'source_width', 'source_height', 'ocr_current_path', 'ocr_action', 'ocr_status', 'old_key', 'new_key', 'bed_identity_status', 'mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_qc', 'scale_version', 'scale_image_key')})
        r.update(source_path=sample['input_path'], test_path=str(path), source_image_name=path.name, source_coordinate_space='bed_pixels', season_id='3작기', bed_id=path.stem.split('_')[0], capture_date=path.stem.split('_')[1], capture_time=path.stem.split('_')[2], mapping_version='S3_cam1_right12_no_leaf_towel' if sample['camera'] == 'cam1' else 'S3_left_to_right__reference_camera_geometry', record_id=hashlib.sha256(f"{sample['input_path']}|{r['slot_namespace_version']}|{r['slot_name']}".encode()).hexdigest()[:24], overlay_path=str(overlay), qc_flags=summary['qc_flags'], map_status=summary['map_status'], seg_polygon=None if s is None else s['polygon'], anchor_x=r['slot_center_x'], anchor_y=r['slot_center_y'])
        (x1, y1, x2, y2) = r['cell_bbox']
        r.update(cell_x1=x1, cell_y1=y1, cell_x2=x2, cell_y2=y2, cell_width=x2 - x1, cell_height=y2 - y1, cell_geometry_ready=bool(x2 > x1 and y2 > y1), bbox_handoff_status='GEOMETRY_REVIEW' if not mapped or x2 <= x1 or y2 <= y1 else 'SLOT_CSV_READY')
    overlay.parent.mkdir(parents=True, exist_ok=True)
    canvas = draw_overlay(image, mapped, segments, summary, meta)
    if not refs[0].imwrite_unicode(overlay, canvas):
        raise OSError(f'Cannot write overlay: {overlay}')
    assigned = {r['seg_id']: r['slot_name'] for r in mapped if r['seg_id'] is not None}
    seg_rows = [dict(source_path=sample['input_path'], test_path=str(path), camera=sample['camera'], image_name=path.name, seg_id=s['seg_id'], slot_name=assigned.get(s['seg_id'], ''), assignment_status='ASSIGNED' if s['seg_id'] in assigned else 'UNASSIGNED', bbox=s['bbox'], center_x=s['cx'], center_y=s['cy'], area=s['area'], confidence=s['conf'], polygon=s['polygon']) for s in segments]
    return (summary, mapped, seg_rows)
