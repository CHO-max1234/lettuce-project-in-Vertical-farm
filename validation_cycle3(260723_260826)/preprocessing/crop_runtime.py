"""Crop runtime calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
SEASON_ROOT = CODE_ROOT.parent
LAYOUT_PROCESS = SEASON_ROOT / '4. 자리배치와 bbox'
CORE_PATH = CYCLE.PROJECT_ROOT / 'preprocessing/plant_selection.py'
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
OUTPUT_ROOT = DATA_ROOT / '5. bbox to crop, feature'
APPROVED_BBOX_ROOT = CYCLE.SOURCE_ROOT/'4. 자리배치와 bbox/bbox_ver2_final'
CROP_ROOT = OUTPUT_ROOT / 'crops'
SAMPLE_ROOT = OUTPUT_ROOT / 'samples'
STATE_ROOT = OUTPUT_ROOT / '_whole_run'
STATE_DATABASE = STATE_ROOT / 'state.sqlite3'
MASK_HELPER_PATH = CYCLE.MAIN_ROOT / 'preprocessing/mask_record.py'
(EXPECTED_FRAMES, EXPECTED_SLOTS) = (39401, 740473)
VERSION = 'S4_FULL_MASKED_CROP_FROZEN_BBOX_VER2_V1'
CORE_SHA256 = '451dae909ef111e7d3c2b31c6fcff305c2aedcb1471a14c2f50998f802e27bc4'
SYSTEM_RAM_LIMIT_GIB = 50.0
(GPU_WORKERS, POST_WORKERS, READ_WORKERS, SAVE_WORKERS) = (4, 8, 12, 16)
(PREFETCH_FRAMES, MAX_POST_FRAMES) = (48, 64)
LOCAL_BATCH = 1
(SAMPLE_SEED, SAMPLE_PER_CAMERA) = (20261011, 30)
(PNG_COMPRESSION, CHECKPOINT_FRAMES) = (3, 64)
import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
import traceback
import zlib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import numpy as np
import psutil
from tqdm import tqdm
cv2.setNumThreads(1)
csv.field_size_limit(64 * 1024 * 1024)

def load_module(name, path):
    spec = resolved_spec(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj
CORE = load_module('s4_production_frozen_ver2', CORE_PATH)
MASK = load_module('s4_feature_mask_record', MASK_HELPER_PATH)
(sha, save_json) = (CORE.sha, CORE.save_json)
(READ_LOCAL, GPU_LOCAL) = (threading.local(), threading.local())
MODEL_INIT_LOCK = threading.Lock()

def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')

def encoded(value):
    return json.dumps(CORE.clean(value), ensure_ascii=False, separators=(',', ':'))

def pack(value):
    return zlib.compress(encoded(value).encode('utf-8'), 3)

def unpack(blob):
    return json.loads(zlib.decompress(blob))

def source_db():
    p = Path(CORE.READER.resolve_layout()['database'])
    db = sqlite3.connect(p.as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-65536')
    return db

def memory_used_gib():
    v = psutil.virtual_memory()
    return (v.total - v.available) / 1024 ** 3

def inventory():
    assert sha(CORE_PATH) == CORE_SHA256, 'Input contract failed: sha(CORE_PATH) == CORE_SHA256'
    assert sha(CORE.SUPPORT.MODEL_PATH) == CORE.SUPPORT.MODEL_SHA256
    check = CORE.READER.check()
    with source_db() as db:
        items = [dict(frame_id=f, sequence=n, camera=c, geometry_status=q, slot_count=s) for (f, n, c, q, s) in db.execute('SELECT frame_id,sequence,camera,geometry_status,slot_count FROM frames ORDER BY sequence')]
        assert len(items) == EXPECTED_FRAMES and sum((i['slot_count'] for i in items)) == EXPECTED_SLOTS
    rng = random.Random(SAMPLE_SEED)
    sampled = []
    for cam in ('cam0', 'cam1', 'cam2'):
        sampled.extend(rng.sample([r for r in items if r['camera'] == cam], SAMPLE_PER_CAMERA))
    return (items, sampled, check)

def decode(item):
    if not hasattr(READ_LOCAL, 'db'):
        READ_LOCAL.db = source_db()
    payload = unpack(READ_LOCAL.db.execute('SELECT payload FROM frames WHERE frame_id=?', (item['frame_id'],)).fetchone()[0])
    f = payload['frame']
    path = Path(f['source_path'])
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == f['source_sha256'], 'SOURCE_SHA_MISMATCH ' + str(path)
    source = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    assert source is not None and source.shape[:2] == (f['source_height'], f['source_width']), 'SOURCE_SHAPE_MISMATCH'
    return (item, payload, source)

def gpu_model():
    if not hasattr(GPU_LOCAL, 'model'):
        with MODEL_INIT_LOCK:
            from ultralytics import YOLO
            GPU_LOCAL.model = YOLO(str(CORE.SUPPORT.MODEL_PATH))
            GPU_LOCAL.model.predict(np.zeros((128, 128, 3), np.uint8), imgsz=512, conf=0.2, iou=0.7, retina_masks=True, rect=False, device=yolo_device(), verbose=False)
    return GPU_LOCAL.model

def infer_inputs(decoded, local_batch):
    (item, payload, source) = decoded
    (rows, excluded) = CORE.adapt_anchors(payload['frame'], payload['slot'])
    (whole_candidates, local) = ([], {})
    (calls, local_images) = (0, 0)
    if rows:
        model = gpu_model()
        pred = model.predict(source, imgsz=CORE.WHOLE_SIZE, conf=0.2, iou=0.7, retina_masks=True, rect=True, device=yolo_device(), verbose=False)[0]
        calls += 1
        whole_candidates = CORE.SUPPORT.candidates(pred, source, [0, 0, source.shape[1], source.shape[0]], 'whole')
        requests = []
        for row in rows:
            if not CORE.SELECTOR.needs_local(row, source, whole_candidates):
                local[row['record_id']] = []
                continue
            box = CORE.SUPPORT.centered_box(row, source, CORE.LOCAL_SCALE)
            requests.append((row['record_id'], box))
        for start in range(0, len(requests), local_batch):
            group = requests[start:start + local_batch]
            images = [source[b[1]:b[3], b[0]:b[2]] for (_, b) in group]
            predictions = model.predict(images, imgsz=CORE.LOCAL_SIZE, conf=0.2, iou=0.7, retina_masks=True, rect=False, device=yolo_device(), verbose=False)
            calls += 1
            local_images += len(group)
            for ((key, box), p) in zip(group, predictions):
                local[key] = CORE.SUPPORT.candidates(p, source, box, 'local')
    return (item, payload, source, rows, excluded, whole_candidates, local, calls, local_images)

def select(inferred):
    (item, payload, source, rows, excluded, whole, local, calls, local_images) = inferred
    decisions = dict(excluded)
    if rows:
        selected = CORE.SELECTOR.select_frame(rows, source, whole, local)
        CORE.reject_hole_like_masks(rows, source, selected)
        decisions.update(selected)
    return (item, payload, source, rows, decisions, calls, local_images)

def target_box(mask, slot, shape):
    (ys, xs) = np.where(mask)
    margin = max(6, round(0.04 * min(float(slot['cell_width']), float(slot['cell_height']))))
    tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    (h, w) = shape
    box = [max(0, tight[0] - margin), max(0, tight[1] - margin), min(w, tight[2] + margin), min(h, tight[3] + margin)]
    return (tight, box)

def verify_ver2(items, local_batch, gpu_workers):
    samples = json.loads(CORE.SAMPLE_MANIFEST.read_text(encoding='utf-8'))
    chosen = {s['frame_id'] for s in samples}
    expected = {}
    for sample in samples:
        p = APPROVED_BBOX_ROOT / sample['camera'] / 'records' / (Path(sample['image_name']).stem + '.json')
        for r in json.loads(p.read_text(encoding='utf-8'))['targets']:
            expected[r['record_id']] = r
    (report, mismatches) = (Counter(), [])
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as readers, ThreadPoolExecutor(max_workers=gpu_workers) as gpu:
        tasks = [i for i in items if i['frame_id'] in chosen]
        decoded = list(readers.map(decode, tasks))
        futures = [gpu.submit(infer_inputs, d, local_batch) for d in decoded]
        for future in tqdm(futures, desc='Frozen ver2 live parity', unit='frame'):
            (item, payload, source, rows, decisions, calls, local_images) = select(future.result())
            report['frames'] += 1
            for slot in payload['slot']:
                (d, e) = (decisions[slot['record_id']], expected[slot['record_id']])
                report['slots'] += 1
                available = d.get('mask') is not None and d['mask'].any()
                if d['status'] != e['selection_status'] or bool(available) != e['target_available']:
                    mismatches.append(dict(record_id=slot['record_id'], image=payload['frame']['image_name'], slot=slot['slot_name'], reason='STATUS_OR_AVAILABILITY'))
                    continue
                if available:
                    report['targets'] += 1
                    (_, box) = target_box(d['mask'], slot, source.shape[:2])
                    (x1, y1, x2, y2) = box
                    digest = hashlib.sha256(d['mask'][y1:y2, x1:x2].tobytes()).hexdigest()
                    if box != e['final_bbox'] or digest != e['mask_pixels_sha256']:
                        mismatches.append(dict(record_id=slot['record_id'], image=payload['frame']['image_name'], slot=slot['slot_name'], reason='MASK_PIXELS_OR_BBOX'))
    return dict(status='PASS' if not mismatches else 'FAIL', counts=dict(report), mismatches=mismatches, local_batch=local_batch, gpu_workers=gpu_workers, model_sha256=CORE.SUPPORT.MODEL_SHA256, algorithm='APPROVED_VER2_NO_VER3_REFINEMENT', actual_model_inference=True, checked_at=now())

def save_crop(path, crop):
    assert crop.size and crop.dtype == np.uint8
    (ok, data) = cv2.imencode('.png', crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    assert ok
    assert np.array_equal(cv2.imdecode(data, cv2.IMREAD_COLOR), crop)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing')
    data.tofile(temporary)
    os.replace(temporary, path)
    return (hashlib.sha256(data.tobytes()).hexdigest(), int(data.size))

def build_records(payload, source, rows, decisions, sampled, saves):
    (f, slots) = (payload['frame'], payload['slot'])
    (h, w) = source.shape[:2]
    by = {r['record_id']: r for r in rows}
    (records, blobs, pending, sample_crops) = ([], {}, [], {})
    for slot in slots:
        d = decisions[slot['record_id']]
        (a, mask) = (by.get(slot['record_id']), d.get('mask'))
        available = bool(mask is not None and mask.any())
        r = dict(slot)
        r.update(bbox_selection_version=CORE.VERSION, crop_stage_version=VERSION, input_layout_geometry_status=f['geometry_status'], upstream_bbox_target_selection_executed=slot.get('bbox_target_selection_executed'), upstream_bbox_stage=slot.get('bbox_stage'), upstream_target_confirmation=slot.get('target_confirmation'), bbox_target_selection_executed=True, bbox_stage='FULL_BODY_SELECTION_EXECUTED', target_confirmation='BODY_SUPPORTED_REVIEW_PENDING' if available else d['status'], target_available=available, crop_saved=available, has_crop=available, selection_status=d['status'], status=d['status'], selection_reasons=d.get('reasons', []), selected_route=d.get('route', ''), target_anchor_x=float(slot['anchor_x']), target_anchor_y=float(slot['anchor_y']), bbox_anchor_status=a['bbox_anchor_status'] if a else 'EXCLUDED', bbox_anchor_review=a['bbox_anchor_review'] if a else True, bbox_anchor_evidence=a['bbox_anchor_evidence'] if a else '', full_batch_bbox_inference=True, full_batch_crop_materialized=True, crop_path='', masked_crop_path='', crop_sha256='', crop_pixels_sha256='', mask_path='', final_bbox=None, target_bbox=None, mask_pixels=None, target_mask_pixels_sha256='', mask_encoding='', mask_height='', mask_width='', mask_data='', mask_sha256='', target_mask_database=str(STATE_DATABASE), target_mask_record_id=slot['record_id'], target_mask_encoding='ZLIB_PACKBITS_LITTLE_CROP_LOCAL_V1', target_mask_shape=None, source_coordinate_space='bed_pixels', crop_coordinate_space='bed_pixels', mask_coordinate_space='crop_local_pixels', crop_image_policy='MASKED_RGB_BACKGROUND_ZERO', photometric_policy='UPSTREAM_BRIGHT_RGB_NO_ADDITIONAL_CORRECTION', crop_stage_status='TARGET_UNAVAILABLE_PRESERVED', training_eligible=False, production_verified=False, feature_extraction_executed=False, human_review_status='NOT_REVIEWED')
        for (k, v) in d.items():
            if k not in ('mask', 'status', 'reasons', 'route'):
                r['decision_' + k] = v
        flags = []
        if f['geometry_status'] == 'REVIEW' or slot.get('geometry_qc_status') == 'REVIEW':
            flags.append('UPSTREAM_LAYOUT_REVIEW_USED')
        if r['bbox_anchor_review']:
            flags.append('CONDITIONAL_ANCHOR_REVIEW')
        if f.get('bed_identity_status') != 'CONFIRMED':
            flags.append('UPSTREAM_OCR_IDENTITY_REVIEW')
        if d.get('review_required'):
            flags.append('MASK_OWNERSHIP_REVIEW')
        border = False
        if available:
            assert CORE.READER.geometry_candidate(f, slot)
            (tight, box) = target_box(mask, slot, source.shape[:2])
            (x1, y1, x2, y2) = box
            binary = mask[y1:y2, x1:x2]
            crop = source[y1:y2, x1:x2].copy()
            crop[~binary] = 0
            inline = MASK.pack(binary)
            assert inline['mask_sha256'] == hashlib.sha256(binary.tobytes()).hexdigest()
            blob = zlib.compress(np.packbits(binary, bitorder='little').tobytes(), 3)
            blobs[slot['record_id']] = blob
            path = CROP_ROOT / f['camera'] / f['date_folder'] / (Path(f['image_name']).stem + '__' + slot['slot_name'] + '.png')
            pending.append((r, saves.submit(save_crop, path, crop)))
            if sampled:
                sample_crops[slot['record_id']] = crop
            neighbors = [v['slot_name'] for v in slots if v['record_id'] != slot['record_id'] and x1 <= float(v['anchor_x']) < x2 and (y1 <= float(v['anchor_y']) < y2)]
            border = tight[0] == 0 or tight[1] == 0 or tight[2] == w or (tight[3] == h)
            if border:
                flags.append('SELECTED_MASK_TOUCHES_SOURCE_BORDER')
            if neighbors:
                flags.append('RGB_CONTEXT_CONTAINS_NEIGHBOR_ANCHOR')
            if x2 - x1 > 1.8 * float(slot['cell_width']) or y2 - y1 > 1.8 * float(slot['cell_height']):
                flags.append('LARGE_BODY_OR_FRAGMENT_EXTENT_REVIEW')
            r.update(**inline, final_bbox=box, target_bbox=tight, mask_pixels=int(mask.sum()), target_mask_pixels_sha256=inline['mask_sha256'], target_mask_shape=list(binary.shape), crop_path=str(path), masked_crop_path=str(path), crop_source_box=box, crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2, crop_width=x2 - x1, crop_height=y2 - y1, crop_pixels_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), crop_anchor_x=r['target_anchor_x'] - x1, crop_anchor_y=r['target_anchor_y'] - y1, crop_stage_status='SAVED_FROZEN_VER2_MASKED_CROP', source_boundary_contact=border, is_crop_clipped=x1 == 0 or y1 == 0 or x2 == w or (y2 == h), neighbor_anchors_in_rgb_crop=neighbors, raw_final_bbox=[x1 + f['raw_crop_bbox'][0], y1 + f['raw_crop_bbox'][1], x2 + f['raw_crop_bbox'][0], y2 + f['raw_crop_bbox'][1]])
        r.update(bbox_qc_flags=flags, bbox_review_required=bool(flags or slot.get('qc_flags') or d.get('reasons')), target_body_review_required=bool(r['bbox_anchor_review'] or d.get('review_required') or border) if available else True)
        records.append(r)
    for (r, future) in pending:
        (digest, size) = future.result()
        r.update(crop_sha256=digest, crop_bytes=size)
    return (records, blobs, sample_crops)

def save_sample(payload, source, records, decisions, crops):
    f = payload['frame']
    (h, w) = source.shape[:2]
    canvas = source.copy()
    stem = Path(f['image_name']).stem
    columns = sorted({int(r['slot_name'][1:]) for r in records})
    sheet = np.full((70 + 2 * 240, max(1, len(columns)) * 190, 3), 24, np.uint8)
    CORE.text(sheet, stem + ' | FROZEN VER2 | masked crops only', (8, 28), (255, 255, 255), 0.55)
    for r in records:
        color = CORE.palette(r['slot_name']) if r['target_available'] else (140, 140, 140)
        mask = decisions[r['record_id']].get('mask')
        if r['target_available']:
            (contours, _) = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, max(2, round(w / 1200)))
            (x1, y1, x2, y2) = r['final_bbox']
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), color, max(2, round(w / 1200)))
        elif r.get('cell_bbox'):
            CORE.dashed_box(canvas, r['cell_bbox'], color)
        (ax, ay) = (r['target_anchor_x'], r['target_anchor_y'])
        suffix = 'BODY-R' if r['target_available'] and r['target_body_review_required'] else 'BODY' if r['target_available'] else 'EMPTY' if r['selection_status'] == 'VISIBLE_EMPTY_CORE' else 'HOLD' if 'EXCLUDED' in r['selection_status'] else 'NO-BODY'
        if 0 <= ax < w and 0 <= ay < h:
            point = (round(ax), round(ay))
            cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, max(10, round(w / 160)), max(1, round(w / 1500)))
            CORE.text(canvas, r['slot_name'] + ' ' + suffix, (max(1, min(w - 120, point[0] + 5)), max(18, point[1] - 12)), color, max(0.5, w / 2400))
        ix = columns.index(int(r['slot_name'][1:]))
        iy = int(r['row'] == 'b')
        (x, y) = (ix * 190, 65 + iy * 240)
        CORE.text(sheet, r['slot_name'] + ' ' + suffix, (x + 5, y + 20), color, 0.45)
        if r['target_available']:
            thumb = CORE.thumbnail(crops[r['record_id']], 180, 180)
            xx = x + (190 - thumb.shape[1]) // 2
            yy = y + 30 + (180 - thumb.shape[0]) // 2
            sheet[yy:yy + thumb.shape[0], xx:xx + thumb.shape[1]] = thumb
        else:
            CORE.text(sheet, r['selection_status'][:25], (x + 4, y + 60), (160, 160, 160), 0.3)
    header = np.zeros((max(80, round(w * 0.04)), w, 3), np.uint8)
    CORE.text(header, f"{f['camera'].upper()} FULL VER2 | layout={f['geometry_status']} | bodies={sum((r['target_available'] for r in records))}/{len(records)}", (8, round(header.shape[0] * 0.38)), (255, 255, 255), max(0.55, w / 2800))
    CORE.text(header, stem + ' | PASS/REVIEW used; HOLD excluded', (8, round(header.shape[0] * 0.78)), (220, 220, 220), max(0.45, w / 3200))
    overlay = SAMPLE_ROOT / f['camera'] / 'overlays' / (stem + '_overlay.jpg')
    sheet_path = SAMPLE_ROOT / f['camera'] / 'sheets' / (stem + '_masked_crops.jpg')
    CORE.SUPPORT.save_image(overlay, np.vstack((header, canvas)))
    CORE.SUPPORT.save_image(sheet_path, sheet)
    return (str(overlay), str(sheet_path))

def postprocess(inferred, sample_ids, saves):
    (item, payload, source, rows, decisions, calls, local_images) = select(inferred)
    sampled = item['frame_id'] in sample_ids
    (records, masks, crops) = build_records(payload, source, rows, decisions, sampled, saves)
    (overlay, sheet) = save_sample(payload, source, records, decisions, crops) if sampled else ('', '')
    frame = dict(payload['frame'])
    frame.update(sequence=item['sequence'], crop_stage_version=VERSION, frame_status='PROCESSED', slot_rows=len(records), target_count=sum((r['target_available'] for r in records)), selection_statuses=dict(Counter((r['selection_status'] for r in records))), sample_selected=sampled, overlay_path=overlay, sheet_path=sheet, inference_calls=calls, local_input_images=local_images)
    for (r, s) in zip(records, payload['slot']):
        for k in ('record_id', 'frame_id', 'slot_name', 'anchor_x', 'anchor_y', 'qc_flags', 'geometry_qc_status', 'anchor_registration_status', 'anchor_registration_review'):
            assert r.get(k) == s.get(k), (k, r['record_id'])
        assert not (frame['geometry_status'] == 'HOLD' and r['target_available'])
    return (item, frame, records, masks)

def database():
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE_DATABASE, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072')
    db.execute('CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS frames(frame_id TEXT PRIMARY KEY,sequence INTEGER UNIQUE,camera TEXT,status TEXT,slot_rows INTEGER,targets INTEGER,payload BLOB)')
    db.execute('CREATE TABLE IF NOT EXISTS slots(record_id TEXT PRIMARY KEY,frame_id TEXT,camera TEXT,slot_order INTEGER,target_available INTEGER,selection_status TEXT,crop_path TEXT,payload BLOB,mask BLOB)')
    db.execute('CREATE INDEX IF NOT EXISTS slots_frame ON slots(frame_id)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS crop_unique ON slots(crop_path) WHERE target_available=1')
    db.commit()
    return db

def save_result(db, result):
    (item, frame, records, masks) = result
    assert len(records) == item['slot_count']
    with db:
        db.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?)', (item['frame_id'], item['sequence'], item['camera'], 'PROCESSED', len(records), frame['target_count'], pack(frame)))
        db.executemany('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?)', [(r['record_id'], item['frame_id'], r['camera'], i, int(r['target_available']), r['selection_status'], r['crop_path'], pack(r), masks.get(r['record_id'])) for (i, r) in enumerate(records)])

def read_target_mask(database_path, record_id):
    with sqlite3.connect(Path(database_path).as_uri() + '?mode=ro', uri=True) as db:
        value = db.execute('SELECT payload,mask FROM slots WHERE record_id=?', (record_id,)).fetchone()
    if value is None:
        raise KeyError(record_id)
    record = unpack(value[0])
    if value[1] is None:
        return (record, None)
    shape = record['target_mask_shape']
    mask = np.unpackbits(np.frombuffer(zlib.decompress(value[1]), np.uint8), bitorder='little', count=math.prod(shape)).reshape(shape).astype(bool)
    assert hashlib.sha256(mask.tobytes()).hexdigest() == record['target_mask_pixels_sha256']
    return (record, mask)

def runtime_contract(local_batch, gpu_workers, sampled):
    pointer = CORE.READER.resolve_layout()
    paths = [Path(__file__), CORE_PATH, Path(CORE.SELECTOR.__file__), CORE.SELECTOR.BASE_SELECTOR_PATH, Path(CORE.SUPPORT.__file__), Path(CORE.READER.__file__), MASK_HELPER_PATH]
    return dict(version=VERSION, bbox_version=CORE.VERSION, model_path=str(CORE.SUPPORT.MODEL_PATH), model_sha256=CORE.SUPPORT.MODEL_SHA256, layout_database=pointer['database'], layout_manifest=str(pointer['slot_manifest']), layout_rows=EXPECTED_SLOTS, input_frames=EXPECTED_FRAMES, dependencies={str(p): sha(p) for p in paths}, gpu_workers=gpu_workers, local_batch=local_batch, sample_seed=SAMPLE_SEED, sampled_frame_ids=[s['frame_id'] for s in sampled], system_ram_limit_gib=SYSTEM_RAM_LIMIT_GIB, image_policy='ONLY_FULL_MASKED_PNG;90_FRAME_OVERLAY_SHEET', layout_policy='PASS_REVIEW_USED_HOLD_EXCLUDED', feature_extraction_executed=False)

def status(db, started, counters, state='RUNNING'):
    (frames, targets) = db.execute('SELECT count(*),coalesce(sum(targets),0) FROM frames').fetchone()
    rows = db.execute('SELECT count(*) FROM slots').fetchone()[0]
    elapsed = time.monotonic() - started
    report = dict(status=state, updated_at=now(), pid=os.getpid(), processed=frames, total=EXPECTED_FRAMES, remaining=EXPECTED_FRAMES - frames, slot_rows=rows, targets=targets, cameras=dict(db.execute('SELECT camera,count(*) FROM frames GROUP BY camera')), system_ram_used_gib=round(memory_used_gib(), 2), system_ram_limit_gib=SYSTEM_RAM_LIMIT_GIB, elapsed_seconds=round(elapsed, 1), session_completed=counters['completed'], session_fps=round(counters['completed'] / max(elapsed, 1), 3), inference_calls=counters['inference_calls'], local_input_images=counters['local_images'], gpu_workers=counters['gpu_workers'], local_batch=counters['local_batch'])
    save_json(STATE_ROOT / 'status.json', report)
    save_json(OUTPUT_ROOT / 'progress.json', report)
    return report

def run_full(items, sampled, args, contract, signature):
    import msvcrt
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with (STATE_ROOT / '.run.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        db = database()
        previous = db.execute("SELECT value FROM config WHERE key='signature'").fetchone()
        if previous and previous[0] != signature:
            raise RuntimeError('Invalid input or execution contract')
        with db:
            db.execute('INSERT OR IGNORE INTO config VALUES(?,?)', ('signature', signature))
        done = {r[0] for r in db.execute('SELECT frame_id FROM frames')}
        pending = [i for i in items if i['frame_id'] not in done]
        if args.max_newframes is not None:
            pending = pending[:args.max_newframes]
        sample_ids = {s['frame_id'] for s in sampled}
        save_json(STATE_ROOT / 'execution_contract.json', dict(contract, signature=signature))
        save_json(SAMPLE_ROOT / 'sample_manifest.json', sampled)
        started = time.monotonic()
        counters = Counter(completed=0, inference_calls=0, local_images=0, gpu_workers=args.gpu_workers, local_batch=args.local_batch)
        status(db, started, counters)
        bar = tqdm(total=len(pending), desc='S4 FULL bbox ver2 → masked crop', unit='frame', mininterval=5)
        try:
            with ThreadPoolExecutor(max_workers=READ_WORKERS) as readers, ThreadPoolExecutor(max_workers=args.gpu_workers) as gpu, ThreadPoolExecutor(max_workers=POST_WORKERS) as posts, ThreadPoolExecutor(max_workers=SAVE_WORKERS) as saves:
                iterator = iter(pending)
                read_queue = deque()
                gpu_futures = set()
                post_futures = set()
                exhausted = False
                last_report = time.monotonic()

                def replenish():
                    nonlocal exhausted
                    while not exhausted and len(read_queue) < PREFETCH_FRAMES and (memory_used_gib() < SYSTEM_RAM_LIMIT_GIB - 2):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        read_queue.append(readers.submit(decode, item))
                replenish()
                while read_queue or gpu_futures or post_futures or (not exhausted):
                    while read_queue and len(gpu_futures) < args.gpu_workers * 2 and (len(post_futures) < MAX_POST_FRAMES) and (memory_used_gib() < SYSTEM_RAM_LIMIT_GIB - 1):
                        decoded = read_queue[0].result()
                        read_queue.popleft()
                        gpu_futures.add(gpu.submit(infer_inputs, decoded, args.local_batch))
                        replenish()
                    finished_gpu = {f for f in gpu_futures if f.done()}
                    for future in finished_gpu:
                        gpu_futures.remove(future)
                        post_futures.add(posts.submit(postprocess, future.result(), sample_ids, saves))
                    finished_post = {f for f in post_futures if f.done()}
                    for future in finished_post:
                        post_futures.remove(future)
                        result = future.result()
                        save_result(db, result)
                        frame = result[1]
                        counters['completed'] += 1
                        counters['inference_calls'] += frame['inference_calls']
                        counters['local_images'] += frame['local_input_images']
                        bar.update(1)
                    if finished_post and counters['completed'] and (counters['completed'] % CHECKPOINT_FRAMES == 0) or time.monotonic() - last_report > 20:
                        status(db, started, counters)
                        last_report = time.monotonic()
                        db.execute('PRAGMA wal_checkpoint(PASSIVE)')
                    replenish()
                    if not finished_gpu and (not finished_post):
                        wait(gpu_futures | post_futures, timeout=0.2, return_when=FIRST_COMPLETED) if gpu_futures or post_futures else time.sleep(0.2)
            bar.close()
            if db.execute('SELECT count(*) FROM frames').fetchone()[0] < EXPECTED_FRAMES:
                report = status(db, started, counters, 'CHUNK_COMPLETE_RESTARTABLE')
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return
            status(db, started, counters, 'EXPORTING')
            finalize(db, items, sampled, signature)
        except BaseException as e:
            save_json(STATE_ROOT / 'failure.json', dict(status='FAILED_RESTARTABLE', at=now(), error=str(e), traceback=traceback.format_exc()))
            status(db, started, counters, 'FAILED_RESTARTABLE')
            raise
        finally:
            db.close()

def export_rows(db, path, sql, params=()):
    fields = []
    seen = set()
    count = 0
    for (blob,) in db.execute(sql, params):
        for key in unpack(blob):
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temp = path.with_suffix('.partial.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        for (blob,) in db.execute(sql, params):
            r = unpack(blob)
            writer.writerow({k: encoded(v) if isinstance(v, (dict, list, tuple)) else v for (k, v) in r.items()})
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return count

def finalize(db, items, sampled, signature):
    assert db.execute('SELECT count(*) FROM frames').fetchone()[0] == EXPECTED_FRAMES
    assert db.execute('SELECT count(*) FROM slots').fetchone()[0] == EXPECTED_SLOTS
    with source_db() as source:
        upstream_ids = {r[0] for r in source.execute('SELECT record_id FROM slot_index')}
    current_ids = {r[0] for r in db.execute('SELECT record_id FROM slots')}
    assert upstream_ids == current_ids
    del upstream_ids, current_ids
    sql = 'SELECT s.payload FROM slots s JOIN frames f USING(frame_id) ORDER BY f.sequence,s.slot_order'
    assert export_rows(db, CROP_ROOT / 'crop_manifest.csv', sql) == EXPECTED_SLOTS
    feature_sql = 'SELECT s.payload FROM slots s JOIN frames f USING(frame_id) WHERE s.target_available=1 ORDER BY f.sequence,s.slot_order'
    selected = db.execute('SELECT count(*) FROM slots WHERE target_available=1').fetchone()[0]
    assert export_rows(db, CROP_ROOT / 'feature_input_manifest.csv', feature_sql) == selected
    assert export_rows(db, OUTPUT_ROOT / 'frame_manifest.csv', 'SELECT payload FROM frames ORDER BY sequence') == EXPECTED_FRAMES
    sample_frames = []
    for item in sampled:
        sample_frames.append(unpack(db.execute('SELECT payload FROM frames WHERE frame_id=?', (item['frame_id'],)).fetchone()[0]))
    assert len(sample_frames) == 90
    for cam in ('cam0', 'cam1', 'cam2'):
        assert len(list((SAMPLE_ROOT / cam / 'overlays').glob('*.jpg'))) == 30
        assert len(list((SAMPLE_ROOT / cam / 'sheets').glob('*.jpg'))) == 30
    verify_count = 0

    def verify_one(blob):
        r = unpack(blob)
        assert sha(r['crop_path']) == r['crop_sha256']
        mask = MASK.unpack(r) > 0
        assert int(mask.sum()) == r['mask_pixels'] and list(mask.shape) == r['target_mask_shape']
        return 1
    with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as pool:
        group = []
        for (blob,) in db.execute(feature_sql):
            group.append(blob)
            if len(group) >= 256:
                verify_count += sum(pool.map(verify_one, group))
                group.clear()
        if group:
            verify_count += sum(pool.map(verify_one, group))
    assert verify_count == selected
    counts = {cam: dict(frames=db.execute('SELECT count(*) FROM frames WHERE camera=?', (cam,)).fetchone()[0], slots=db.execute('SELECT count(*) FROM slots WHERE camera=?', (cam,)).fetchone()[0], crops=db.execute('SELECT count(*) FROM slots WHERE camera=? AND target_available=1', (cam,)).fetchone()[0], statuses=dict(db.execute('SELECT selection_status,count(*) FROM slots WHERE camera=? GROUP BY selection_status', (cam,)))) for cam in ('cam0', 'cam1', 'cam2')}
    page = ['<!doctype html><meta charset="utf-8"><title>4작기 전수 ver2 crop 표본</title><style>body{background:#181818;color:white;font:16px sans-serif;margin:24px}img{width:100%;max-width:1900px}a{color:#7bcfff}article{margin:25px 0}</style>', '<h1>4작기 전수 bbox ver2 — 무작위 90장 검토</h1><p>crops에는 전체 masked PNG만 저장. PASS/REVIEW 사용, HOLD 제외. feature/CNN은 아직 실행하지 않았습니다.</p>']
    for f in sorted(sample_frames, key=lambda r: (r['camera'], r['sequence'])):
        overlay = Path(f['overlay_path']).relative_to(SAMPLE_ROOT).as_posix()
        sheet = Path(f['sheet_path']).relative_to(SAMPLE_ROOT).as_posix()
        page.append(f'''<article><h3>{html.escape(f['image_name'])} — layout {f['geometry_status']}, bodies {f['target_count']}</h3><a href="{overlay}"><img loading="lazy" src="{overlay}"></a><a href="{sheet}"><img loading="lazy" src="{sheet}"></a></article>''')
    (SAMPLE_ROOT / '검토목록.html').write_text('\n'.join(page), encoding='utf-8')
    save_json(SAMPLE_ROOT / 'completion.json', dict(status='COMPLETE_REVIEW_PENDING', frames=90, camera_frames=30, masked_crops_are_in=str(CROP_ROOT), seed=SAMPLE_SEED))
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    completed = dict(status='COMPLETE_WITH_REVIEW', verification='PASS', completed_at=now(), version=VERSION, bbox_fixed_version=CORE.VERSION, signature=signature, frames=EXPECTED_FRAMES, slot_rows=EXPECTED_SLOTS, masked_crops=selected, errors=0, remaining=0, camera_summary=counts, verified_crop_files=verify_count, sample_frames=90, overlays=90, sheets=90, upstream_layout_modified=False, feature_extraction_executed=False, cnn_inference_executed=False, target_mask_database=str(STATE_DATABASE), crop_manifest=str(CROP_ROOT / 'crop_manifest.csv'), feature_input_manifest=str(CROP_ROOT / 'feature_input_manifest.csv'), frame_manifest=str(OUTPUT_ROOT / 'frame_manifest.csv'), sample_gallery=str(SAMPLE_ROOT / '검토목록.html'), source_coordinate_space='bed_pixels', review_policy='PASS_REVIEW_USED_HOLD_EXCLUDED; UPSTREAM_QC_PRESERVED')
    save_json(OUTPUT_ROOT / 'completion.json', completed)
    save_json(OUTPUT_ROOT / 'CROP_CURRENT.json', completed)
    save_json(OUTPUT_ROOT / 'progress.json', completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-ver2', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--local-batch', type=int, default=LOCAL_BATCH)
    parser.add_argument('--gpu-workers', type=int, default=GPU_WORKERS)
    parser.add_argument('--max-newframes', type=int, help='Execution option')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    (items, sampled, check) = inventory()
    if args.check:
        print(json.dumps(dict(status='PREFLIGHT_PASS', layout=check, frames=len(items), slots=EXPECTED_SLOTS, sample_frames=len(sampled), system_ram_used_gib=memory_used_gib(), system_ram_limit_gib=SYSTEM_RAM_LIMIT_GIB), ensure_ascii=False, indent=2))
        return
    if args.verify_ver2:
        report = verify_ver2(items, args.local_batch, args.gpu_workers)
        save_json(STATE_ROOT / f'ver2_parity_batch{args.local_batch}_gpu{args.gpu_workers}.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report['status'] != 'PASS':
            raise SystemExit(2)
        return
    proof = STATE_ROOT / f'ver2_parity_batch{args.local_batch}_gpu{args.gpu_workers}.json'
    if not proof.exists() or json.loads(proof.read_text(encoding='utf-8'))['status'] != 'PASS':
        raise RuntimeError('Invalid input or execution contract')
    contract = runtime_contract(args.local_batch, args.gpu_workers, sampled)
    signature = hashlib.sha256(encoded(contract).encode()).hexdigest()
    completed = OUTPUT_ROOT / 'completion.json'
    if completed.exists():
        data = json.loads(completed.read_text(encoding='utf-8'))
        if data.get('signature') == signature and data.get('verification') == 'PASS':
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return
    run_full(items, sampled, args, contract, signature)
if __name__ == '__main__':
    main()
