"""Materialize RGB plant crops from the saved final mask database. No segmentation or slot reassignment is performed."""
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
from runtime import CROP_FOLDER, moved_tree
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
UPSTREAM_ROOT = DATA_ROOT / '4. 자리배치'
BRIGHT_ROOT = DATA_ROOT / '3. 밝기보정과 scale'
OUTPUT_ROOT = DATA_ROOT / CROP_FOLDER
CROP_ROOT = OUTPUT_ROOT / 'crops'
STATE_ROOT = OUTPUT_ROOT / '_whole_run'
SOURCE_DATABASE = UPSTREAM_ROOT / '_whole_run' / 'state.sqlite3'
SAMPLE_ROOT = OUTPUT_ROOT / 'samples'
SAMPLE_CODE = CODE_ROOT / 'crop_masks.py'
MASK_HELPER = CYCLE.MAIN_PREPROCESS / 'mask_record.py'
WORKERS = 20
PREFETCH_FRAMES = 80
MEMORY_RESERVE_GIB = 12
SQLITE_CACHE_MIB_PER_WORKER = 128
PNG_COMPRESSION = 3
(EXPECTED_FRAMES, EXPECTED_ROWS, EXPECTED_TARGETS) = (43198, 717121, 244456)
VERSION = 'S3_FROZEN_MASK_TO_CROP_FULL_V1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import threading
import time
import traceback
import zlib
sys.dont_write_bytecode = True
import cv2
import numpy as np
import psutil
from tqdm import tqdm
cv2.setNumThreads(1)
csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
S = load('s3_full_crop_sample_helpers', SAMPLE_CODE)
M = load('s3_full_crop_mask_pack', MASK_HELPER)
(require, yes, digest, token, encoded) = (S.require, S.yes, S.digest, S.token, S.encoded)
(read_json, csv_rows, now) = (S.read_json, S.csv_rows, S.now)
LOCAL = threading.local()

def source_db():
    db = sqlite3.connect(SOURCE_DATABASE.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute(f'PRAGMA cache_size=-{SQLITE_CACHE_MIB_PER_WORKER * 1024}')
    db.execute(f'PRAGMA mmap_size={SOURCE_DATABASE.stat().st_size}')
    return db

def worker_db():
    if not hasattr(LOCAL, 'db'):
        LOCAL.db = source_db()
    return LOCAL.db

def native_box(row):
    box = row['final_bbox']
    require(isinstance(box, list) and len(box) == 4 and all((not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and (v == int(v)) for v in box)), 'Validation condition failed in native_box: isinstance(box, list) and len(box) == 4 and all((not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and (v == int(v)) for v in box))')
    (x1, y1, x2, y2) = box = list(map(int, box))
    require(box == [row['crop_' + k] for k in ('x1', 'y1', 'x2', 'y2')], "Validation condition failed in native_box: box == [row['crop_' + k] for k in ('x1', 'y1', 'x2', 'y2')]")
    require(0 <= x1 < x2 <= row['source_width'] and 0 <= y1 < y2 <= row['source_height'], "Validation condition failed in native_box: 0 <= x1 < x2 <= row['source_width'] and 0 <= y1 < y2 <= row['source_height']")
    require(row['target_mask_shape'] == [y2 - y1, x2 - x1] and [row['crop_width'], row['crop_height']] == [x2 - x1, y2 - y1], "Validation condition failed in native_box: row['target_mask_shape'] == [y2 - y1, x2 - x1] and [row['crop_width'], row['crop_height']] == [x2 - x1, y2 - y1]")
    require(row['source_coordinate_space'] == 'bed_pixels' and row['crop_coordinate_space'] == 'bed_pixels' and (row['mask_coordinate_space'] == 'crop_local_pixels'), "Validation condition failed in native_box: row['source_coordinate_space'] == 'bed_pixels' and row['crop_coordinate_space'] == 'bed_pixels' and (row['mask_coordinate_space'] == 'crop_local_pixels')")
    return box

def decode_mask(row, blob):
    require(blob is not None and row['target_available'], "Validation condition failed in decode_mask: blob is not None and row['target_available']")
    require(row['target_mask_encoding'] == 'ZLIB_PACKBITS_LITTLE_CROP_LOCAL_V1', "Validation condition failed in decode_mask: row['target_mask_encoding'] == 'ZLIB_PACKBITS_LITTLE_CROP_LOCAL_V1'")
    shape = row['target_mask_shape']
    count = math.prod(shape)
    packed = zlib.decompress(blob)
    require(len(packed) == (count + 7) // 8, 'Validation condition failed in decode_mask: len(packed) == (count + 7) // 8')
    mask = np.unpackbits(np.frombuffer(packed, np.uint8), bitorder='little', count=count).reshape(shape).astype(bool)
    require(hashlib.sha256(mask.tobytes()).hexdigest() == row['target_mask_pixels_sha256'], "Validation condition failed in decode_mask: hashlib.sha256(mask.tobytes()).hexdigest() == row['target_mask_pixels_sha256']")
    require(mask.any() and int(mask.sum()) == row['mask_pixels'], "Validation condition failed in decode_mask: mask.any() and int(mask.sum()) == row['mask_pixels']")
    return mask

def prepare_crop(image, row, blob):
    (x1, y1, x2, y2) = box = native_box(row)
    mask = decode_mask(row, blob)
    crop = image[y1:y2, x1:x2].copy()
    require(crop.shape[:2] == mask.shape, 'Validation condition failed in prepare_crop: crop.shape[:2] == mask.shape')
    crop[~mask] = 0
    inline = M.pack(mask)
    require(inline['mask_sha256'] == row['target_mask_pixels_sha256'], "Validation condition failed in prepare_crop: inline['mask_sha256'] == row['target_mask_pixels_sha256']")
    require(np.array_equal(M.unpack(inline) > 0, mask), 'Validation condition failed in prepare_crop: np.array_equal(M.unpack(inline) > 0, mask)')
    return (crop, box, inline)

def protect_paths():
    return [UPSTREAM_ROOT / n for n in ('completion.json', 'frame_manifest.csv', 'bbox_manifest.csv', 'slot_manifest.csv')] + [SOURCE_DATABASE, UPSTREAM_ROOT / '_whole_run/preflight.json', SAMPLE_CODE, MASK_HELPER]

def preflight():
    started = time.perf_counter()
    snapshot = S.file_snapshot(protect_paths())
    completion = read_json(UPSTREAM_ROOT / 'completion.json')
    require(completion['status'] == 'COMPLETE_WITH_REVIEW' and completion['frame_errors'] == 0, "Validation condition failed in preflight: completion['status'] == 'COMPLETE_WITH_REVIEW' and completion['frame_errors'] == 0")
    frames = []
    lookup = {}
    statuses = Counter()
    camera_counts = Counter()
    paths = set()
    keep = ('frame_id', 'camera', 'status', 'source_path', 'source_sha256', 'source_width', 'source_height', 'slot_count', 'target_count', 'image_name', 'ocr_action', 'scale_verified')
    for row in csv_rows(UPSTREAM_ROOT / 'frame_manifest.csv'):
        frame = {k: row[k] for k in keep}
        require(frame['frame_id'] not in lookup and frame['source_path'] not in paths, "Validation condition failed in preflight: frame['frame_id'] not in lookup and frame['source_path'] not in paths")
        require(Path(frame['source_path']).resolve().is_relative_to(BRIGHT_ROOT.resolve()), "Validation condition failed in preflight: Path(frame['source_path']).resolve().is_relative_to(BRIGHT_ROOT.resolve())")
        for k in ('source_width', 'source_height', 'slot_count', 'target_count'):
            frame[k] = int(frame[k])
        if frame['status'] in S.HELD_STATUSES:
            require(frame['target_count'] == 0, "Validation condition failed in preflight: frame['target_count'] == 0")
        frames.append(frame)
        lookup[frame['frame_id']] = frame
        paths.add(frame['source_path'])
        statuses[frame['status']] += 1
        camera_counts[frame['camera']] += 1
    require(len(frames) == completion['input_images'] == EXPECTED_FRAMES, "Validation condition failed in preflight: len(frames) == completion['input_images'] == EXPECTED_FRAMES")
    require(sum((f['slot_count'] for f in frames)) == completion['slot_rows'] == EXPECTED_ROWS and sum((f['target_count'] for f in frames)) == completion['target_rows'] == EXPECTED_TARGETS, "Validation condition failed in preflight: sum((f['slot_count'] for f in frames)) == completion['slot_rows'] == EXPECTED_ROWS and sum((f['target_count'] for f in frames)) == completion['target_rows'] == EXPECTED_TARGETS")
    require(statuses['NO_SLOT_GEOMETRY_REVIEW'] == 128 and statuses['INVALID_CELL_GEOMETRY_REVIEW'] == 6, "Validation condition failed in preflight: statuses['NO_SLOT_GEOMETRY_REVIEW'] == 128 and statuses['INVALID_CELL_GEOMETRY_REVIEW'] == 6")
    dependencies = read_json(UPSTREAM_ROOT / '_whole_run/preflight.json')['dependency_hashes']
    for (path, sha) in dependencies.items():
        require(digest(path) == sha, 'Validation condition failed in preflight: digest(path) == sha')
    require(shutil_free() > 100 * 2 ** 30, 'Validation condition failed in preflight: shutil_free() > 100 * 2 ** 30')
    counts = defaultdict(Counter)
    order = hashlib.sha256()
    n = targets = 0
    output_paths = set()
    with source_db() as database:
        require(database.execute('SELECT value FROM config WHERE key=?', ('fingerprint',)).fetchone()[0] == completion['fingerprint'], "Validation condition failed in preflight: database.execute('SELECT value FROM config WHERE key=?', ('fingerprint',)).fetchone()[0] == completion['fingerprint']")
        require(database.execute('SELECT COUNT(*) FROM slots').fetchone()[0] == EXPECTED_ROWS, "Validation condition failed in preflight: database.execute('SELECT COUNT(*) FROM slots').fetchone()[0] == EXPECTED_ROWS")
        require(dict(database.execute('SELECT status,COUNT(*) FROM frames GROUP BY status')) == dict(statuses), "Validation condition failed in preflight: dict(database.execute('SELECT status,COUNT(*) FROM frames GROUP BY status')) == dict(statuses)")
        cursor = database.execute('SELECT record_id,frame_id,bbox_json,mask IS NOT NULL FROM slots ORDER BY rowid')
        important = ('record_id', 'frame_id', 'camera', 'physical_slot', 'selection_status', 'source_path', 'source_sha256', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'target_mask_record_id', 'target_mask_database', 'target_mask_encoding', 'target_mask_pixels_sha256')
        for row in tqdm(csv_rows(UPSTREAM_ROOT / 'bbox_manifest.csv'), total=EXPECTED_ROWS, desc='preflight', unit='row', mininterval=3):
            item = cursor.fetchone()
            require(item is not None, 'Validation condition failed in preflight: item is not None')
            (rid, fid, data, has_mask) = item
            record = moved_tree(json.loads(data))
            require(rid == row['record_id'] and fid == row['frame_id'], "Validation condition failed in preflight: rid == row['record_id'] and fid == row['frame_id']")
            for k in important:
                require(row.get(k, '') == str(record.get(k, '') if record.get(k) is not None else ''), "Validation condition failed in preflight: row.get(k, '') == str(record.get(k, '') if record.get(k) is not None else '')")
            require(yes(row['target_available']) == record['target_available'] == bool(has_mask), "Validation condition failed in preflight: yes(row['target_available']) == record['target_available'] == bool(has_mask)")
            frame = lookup[fid]
            for k in ('camera', 'source_path', 'source_sha256'):
                require(row[k] == frame[k], 'Validation condition failed in preflight: row[k] == frame[k]')
            counts[fid]['rows'] += 1
            if record['target_available']:
                require(frame['status'] not in S.HELD_STATUSES and record['selection_status'] == 'SELECTED', "Validation condition failed in preflight: frame['status'] not in S.HELD_STATUSES and record['selection_status'] == 'SELECTED'")
                require(record['target_mask_database'] == str(SOURCE_DATABASE) and record['target_mask_record_id'] == rid, "Validation condition failed in preflight: record['target_mask_database'] == str(SOURCE_DATABASE) and record['target_mask_record_id'] == rid")
                native_box(record)
                targets += 1
                counts[fid]['targets'] += 1
                path = str(crop_path(record)).casefold()
                require(path not in output_paths, 'Validation condition failed in preflight: path not in output_paths')
                output_paths.add(path)
            else:
                require(not has_mask and row.get('mask_pixels', '') == '' and (row.get('area_if_unavailable', '') == ''), "Validation condition failed in preflight: not has_mask and row.get('mask_pixels', '') == '' and (row.get('area_if_unavailable', '') == '')")
            order.update((rid + '\n').encode())
            n += 1
        require(cursor.fetchone() is None and n == EXPECTED_ROWS and (targets == EXPECTED_TARGETS), 'Validation condition failed in preflight: cursor.fetchone() is None and n == EXPECTED_ROWS and (targets == EXPECTED_TARGETS)')
        parity = 0
    for frame in frames:
        require(counts[frame['frame_id']]['rows'] == frame['slot_count'] and counts[frame['frame_id']]['targets'] == frame['target_count'], "Validation condition failed in preflight: counts[frame['frame_id']]['rows'] == frame['slot_count'] and counts[frame['frame_id']]['targets'] == frame['target_count']")
    slot_order = hashlib.sha256()
    slots = 0
    with (UPSTREAM_ROOT / 'slot_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        field = next(reader).index('record_id')
        for row in reader:
            slot_order.update((row[field] + '\n').encode())
            slots += 1
    require(slots == EXPECTED_ROWS and slot_order.hexdigest() == order.hexdigest(), 'Validation condition failed in preflight: slots == EXPECTED_ROWS and slot_order.hexdigest() == order.hexdigest()')
    require(S.file_snapshot(protect_paths()) == snapshot, 'Validation condition failed in preflight: S.file_snapshot(protect_paths()) == snapshot')
    resources = resource_state()
    report = dict(status='FULL_CROP_PREFLIGHT_PASS', version=VERSION, checked_at=now(), input_images=len(frames), slot_rows=n, target_rows=targets, camera_counts=dict(camera_counts), frame_statuses=dict(statuses), held_frames=134, sample_parity_crops=parity, dependencies=dependencies, protected_snapshot=snapshot, source_fingerprint=completion['fingerprint'], source_row_order_sha256=order.hexdigest(), code_sha256=digest(__file__), sample_helper_sha256=digest(SAMPLE_CODE), mask_helper_sha256=digest(MASK_HELPER), resources=resources, inference_calls=0, seconds=round(time.perf_counter() - started, 2))
    return (frames, report)

def shutil_free():
    import shutil
    return shutil.disk_usage(OUTPUT_ROOT).free

def resource_state():
    memory = psutil.virtual_memory()
    return dict(total_memory_gib=round(memory.total / 2 ** 30, 2), available_memory_gib=round(memory.available / 2 ** 30, 2), logical_cpus=os.cpu_count(), reserve_memory_gib=MEMORY_RESERVE_GIB)

def crop_path(row):
    require(S.re.fullmatch('[tb]\\d+', row['physical_slot']) is not None, "Validation condition failed in crop_path: S.re.fullmatch('[tb]\\\\d+', row['physical_slot']) is not None")
    relative = Path(row['source_path']).relative_to(BRIGHT_ROOT)
    require(relative.parts[0] == row['camera'], "Validation condition failed in crop_path: relative.parts[0] == row['camera']")
    require('..' not in relative.parts, "Validation condition failed in crop_path: '..' not in relative.parts")
    path = CROP_ROOT / relative.parent / (relative.stem + '__' + row['physical_slot'] + '.png')
    return path

def save_crop(path, crop):
    (ok, data) = cv2.imencode('.png', crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    require(ok, 'Validation condition failed in save_crop: ok')
    S.atomic_bytes(path, data.tobytes())
    (observed, sha) = S.read_image(path)
    require(np.array_equal(observed, crop), 'Validation condition failed in save_crop: np.array_equal(observed, crop)')
    return sha

def frame_task(frame):
    database = worker_db()
    (image, _) = S.read_image(frame['source_path'], frame['source_sha256'])
    require(image.ndim == 3 and image.shape[2] == 3 and (image.shape[:2] == (frame['source_height'], frame['source_width'])), "Validation condition failed in frame_task: image.ndim == 3 and image.shape[2] == 3 and (image.shape[:2] == (frame['source_height'], frame['source_width']))")
    outputs = []
    count = 0
    source_signature = hashlib.sha256()
    for (rid, data, blob) in database.execute('SELECT record_id,bbox_json,mask FROM slots WHERE frame_id=? ORDER BY rowid', (frame['frame_id'],)):
        source_signature.update(data.encode())
        row = moved_tree(json.loads(data))
        count += 1
        if not row['target_available']:
            require(blob is None and (not crop_path(row).exists()), 'Validation condition failed in frame_task: blob is None and (not crop_path(row).exists())')
            continue
        require(frame['status'] not in S.HELD_STATUSES, "Validation condition failed in frame_task: frame['status'] not in S.HELD_STATUSES")
        (crop, box, inline) = prepare_crop(image, row, blob)
        path = crop_path(row)
        sha = save_crop(path, crop)
        result = dict(record_id=rid, frame_id=frame['frame_id'], has_crop=True, crop_saved=True, crop_path=str(path), crop_sha256=sha, crop_source_box=box, crop_stage_status='SAVED_FROM_FROZEN_MASK', crop_image_policy='MASKED_RGB_BACKGROUND_ZERO', auto_qc_pass=False, training_eligible=False, production_verified=False, crop_stage_version=VERSION, **inline)
        outputs.append(result)
    require(count == frame['slot_count'] and len(outputs) == frame['target_count'], "Validation condition failed in frame_task: count == frame['slot_count'] and len(outputs) == frame['target_count']")
    return dict(frame_id=frame['frame_id'], source_signature=source_signature.hexdigest(), slot_rows=count, saved_crops=len(outputs), frame_status=frame['status'], camera=frame['camera'], outputs=outputs)

def output_database():
    database = sqlite3.connect(STATE_ROOT / 'state.sqlite3')
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('PRAGMA synchronous=NORMAL')
    database.execute('CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    database.execute('CREATE TABLE IF NOT EXISTS completed(frame_id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    database.execute('CREATE TABLE IF NOT EXISTS crops(record_id TEXT PRIMARY KEY,frame_id TEXT NOT NULL,path TEXT UNIQUE NOT NULL,payload TEXT NOT NULL)')
    database.execute('CREATE INDEX IF NOT EXISTS crop_frame ON crops(frame_id)')
    return database

def verify_reuse(database, frame, payload):
    require(payload['saved_crops'] == frame['target_count'] and payload['slot_rows'] == frame['slot_count'], "Validation condition failed in verify_reuse: payload['saved_crops'] == frame['target_count'] and payload['slot_rows'] == frame['slot_count']")
    for (data,) in database.execute('SELECT payload FROM crops WHERE frame_id=?', (frame['frame_id'],)):
        row = moved_tree(json.loads(data))
        require(digest(row['crop_path']) == row['crop_sha256'], "Validation condition failed in verify_reuse: digest(row['crop_path']) == row['crop_sha256']")
    require(digest(frame['source_path']) == frame['source_sha256'], "Validation condition failed in verify_reuse: digest(frame['source_path']) == frame['source_sha256']")

def save_status(status, **values):
    S.save_json(STATE_ROOT / 'status.json', dict(status=status, updated_at=now(), pid=os.getpid(), **values))

def export_manifests(database):
    destination = OUTPUT_ROOT / 'crop_manifest.csv'
    temporary = destination.with_suffix('.partial.csv')
    totals = Counter()
    cameras = Counter()
    order = hashlib.sha256()
    extra = ('crop_stage_upstream_has_crop', 'crop_stage_upstream_crop_saved', 'crop_stage_upstream_mask_sha256', 'crop_stage_upstream_masked_crop_path', 'crop_stage_status', 'crop_stage_version', 'crop_image_policy', 'crop_path', 'crop_sha256', 'crop_source_box', 'mask_path', 'mask_encoding', 'mask_height', 'mask_width', 'mask_data', 'status', 'auto_qc_pass', 'inference_calls', 'camera_selection_applied', 'photometric_policy')
    with (UPSTREAM_ROOT / 'bbox_manifest.csv').open(encoding='utf-8-sig', newline='') as source, temporary.open('w', encoding='utf-8-sig', newline='') as target:
        reader = csv.DictReader(source)
        fields = list(dict.fromkeys(reader.fieldnames + list(extra)))
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in tqdm(reader, total=EXPECTED_ROWS, desc='export manifests', unit='row', mininterval=3):
            row = moved_tree(row)
            rid = row['record_id']
            order.update((rid + '\n').encode())
            totals['rows'] += 1
            for field in ('has_crop', 'crop_saved', 'mask_sha256', 'masked_crop_path'):
                row['crop_stage_upstream_' + field] = row.get(field, '')
            row.update(has_crop=False, crop_saved=False, crop_path='', crop_sha256='', crop_source_box='', mask_path='', mask_sha256='', masked_crop_path='', mask_encoding='', mask_height='', mask_width='', mask_data='', status=row['selection_status'], auto_qc_pass=False, training_eligible=False, production_verified=False, crop_stage_version=VERSION, crop_stage_status='TARGET_UNAVAILABLE_PRESERVED', inference_calls=0, camera_selection_applied=False, photometric_policy='UPSTREAM_BRIGHT_RGB_NO_ADDITIONAL_CORRECTION', crop_image_policy='MASKED_RGB_BACKGROUND_ZERO')
            item = database.execute('SELECT payload FROM crops WHERE record_id=?', (rid,)).fetchone()
            if yes(row['target_available']):
                require(item is not None, 'Validation condition failed in export_manifests: item is not None')
                saved = moved_tree(json.loads(item[0]))
                row.update(saved)
                row['masked_crop_path'] = row['crop_path']
                totals['crops'] += 1
                cameras[row['camera']] += 1
                totals['ocr_review_crops'] += row['ocr_action'] == 'REVIEW'
            else:
                require(item is None, 'Validation condition failed in export_manifests: item is None')
                totals['unavailable_rows'] += 1
            writer.writerow({k: encoded(v) if isinstance(v, (list, dict)) else v for (k, v) in row.items()})
        target.flush()
        os.fsync(target.fileno())
    require(totals['rows'] == EXPECTED_ROWS and totals['crops'] == EXPECTED_TARGETS, "Validation condition failed in export_manifests: totals['rows'] == EXPECTED_ROWS and totals['crops'] == EXPECTED_TARGETS")
    temporary.replace(destination)
    destination = OUTPUT_ROOT / 'frame_manifest.csv'
    temporary = destination.with_suffix('.partial.csv')
    with (UPSTREAM_ROOT / 'frame_manifest.csv').open(encoding='utf-8-sig', newline='') as source, temporary.open('w', encoding='utf-8-sig', newline='') as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames + ['crop_stage_status', 'saved_crops', 'inference_calls'])
        writer.writeheader()
        n = 0
        for frame in reader:
            frame = moved_tree(frame)
            payload = json.loads(database.execute('SELECT payload FROM completed WHERE frame_id=?', (frame['frame_id'],)).fetchone()[0])
            frame.update(crop_stage_status='HELD_FRAME_PRESERVED' if frame['status'] in S.HELD_STATUSES else 'SAVED', saved_crops=payload['saved_crops'], inference_calls=0)
            writer.writerow(frame)
            n += 1
        target.flush()
        os.fsync(target.fileno())
    require(n == EXPECTED_FRAMES, 'Validation condition failed in export_manifests: n == EXPECTED_FRAMES')
    temporary.replace(destination)
    return (dict(totals), dict(cameras), order.hexdigest())

def final_verify(database, report, args):
    checked = 0

    def verify_one(data):
        row = moved_tree(json.loads(data))
        (image, sha) = S.read_image(row['crop_path'], row['crop_sha256'])
        mask = M.unpack(row) > 0
        require(image.shape[:2] == mask.shape and np.all(image[~mask] == 0), 'Validation condition failed in verify_one: image.shape[:2] == mask.shape and np.all(image[~mask] == 0)')
        require(hashlib.sha256(mask.astype(bool).tobytes()).hexdigest() == row['mask_sha256'], "Validation condition failed in verify_one: hashlib.sha256(mask.astype(bool).tobytes()).hexdigest() == row['mask_sha256']")
        return 1
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=EXPECTED_TARGETS, desc='final verify', unit='crop', mininterval=3) as progress:
        cursor = database.execute('SELECT payload FROM crops ORDER BY rowid')
        while True:
            batch = cursor.fetchmany(512)
            if not batch:
                break
            for value in pool.map(verify_one, (v[0] for v in batch)):
                checked += value
                progress.update(value)
    require(checked == EXPECTED_TARGETS, 'Validation condition failed in final_verify: checked == EXPECTED_TARGETS')
    physical_files = sum((1 for p in CROP_ROOT.rglob('*.png')))
    require(physical_files == EXPECTED_TARGETS, 'Validation condition failed in final_verify: physical_files == EXPECTED_TARGETS')
    require(S.file_snapshot(protect_paths()) == report['protected_snapshot'], "Validation condition failed in final_verify: S.file_snapshot(protect_paths()) == report['protected_snapshot']")
    return dict(crops_verified=checked, inline_masks_verified=checked, png_files=physical_files)

def run(frames, report, args):
    signature = token({k: v for (k, v) in report.items() if k not in ('checked_at', 'resources', 'seconds')})
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (STATE_ROOT / '.run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        with output_database() as database:
            item = database.execute('SELECT value FROM config WHERE key=?', ('signature',)).fetchone()
            require(item is None or item[0] == signature, 'Validation condition failed in run: item is None or item[0] == signature')
            if item is None:
                require(not CROP_ROOT.exists() or not any(CROP_ROOT.rglob('*.png')), "Validation condition failed in run: not CROP_ROOT.exists() or not any(CROP_ROOT.rglob('*.png'))")
            database.execute('INSERT OR REPLACE INTO config VALUES(?,?)', ('signature', signature))
            database.commit()
            S.save_json(STATE_ROOT / 'preflight.json', report)
            started = time.perf_counter()
            done = saved = reused = 0
            pending = {}
            iterator = iter(frames)
            finished = False
            S.save_json(OUTPUT_ROOT / 'completion.json', dict(status='FULL_CROP_RUNNING', started_at=now(), pid=os.getpid(), input_images=EXPECTED_FRAMES, target_rows=EXPECTED_TARGETS, workers=args.workers, signature=signature, inference_calls=0))
            try:
                try:
                    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except (AttributeError, psutil.Error):
                    pass
                with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(total=len(frames), desc='run', unit='bed', mininterval=3) as progress:
                    last = 0
                    while pending or not finished:
                        available = psutil.virtual_memory().available
                        limit = args.prefetch if available >= MEMORY_RESERVE_GIB * 2 ** 30 else args.workers
                        while not finished and len(pending) < limit:
                            frame = next(iterator, None)
                            if frame is None:
                                finished = True
                                break
                            old = database.execute('SELECT payload FROM completed WHERE frame_id=?', (frame['frame_id'],)).fetchone()
                            if old:
                                payload = json.loads(old[0])
                                verify_reuse(database, frame, payload)
                                done += 1
                                reused += 1
                                saved += payload['saved_crops']
                                progress.update(1)
                            else:
                                pending[pool.submit(frame_task, frame)] = frame
                        if pending:
                            (ready, _) = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                            for future in ready:
                                frame = pending.pop(future)
                                payload = future.result()
                                outputs = payload.pop('outputs')
                                with database:
                                    database.executemany('INSERT INTO crops VALUES(?,?,?,?)', [(r['record_id'], frame['frame_id'], r['crop_path'], encoded(r)) for r in outputs])
                                    database.execute('INSERT INTO completed VALUES(?,?)', (frame['frame_id'], encoded(payload)))
                                done += 1
                                saved += payload['saved_crops']
                                progress.update(1)
                        if time.perf_counter() - last > 3:
                            elapsed = time.perf_counter() - started
                            rate = done / max(elapsed, 1)
                            save_status('FULL_CROP_RUNNING', completed_frames=done, input_images=EXPECTED_FRAMES, saved_crops=saved, target_rows=EXPECTED_TARGETS, reused_frames=reused, workers=args.workers, seconds=round(elapsed, 1), eta_seconds=round((len(frames) - done) / max(rate, 0.001)), resources=resource_state())
                            last = time.perf_counter()
                require(done == EXPECTED_FRAMES and saved == EXPECTED_TARGETS, 'Validation condition failed in run: done == EXPECTED_FRAMES and saved == EXPECTED_TARGETS')
                save_status('EXPORTING_MANIFESTS', completed_frames=done, saved_crops=saved)
                (totals, cameras, order) = export_manifests(database)
                require(order == report['source_row_order_sha256'], "Validation condition failed in run: order == report['source_row_order_sha256']")
                save_status('FINAL_VERIFICATION', completed_frames=done, saved_crops=saved)
                verification = final_verify(database, report, args)
                completed = dict(status='FULL_CROP_COMPLETE_WITH_REVIEW', verification_status='PASS', finished_at=now(), input_images=done, completed_frames=done, slot_rows=totals['rows'], crop_count=saved, crop_counts=cameras, unavailable_rows=totals['unavailable_rows'], held_frames_preserved=134, ocr_review_crops=totals['ocr_review_crops'], frame_errors=0, workers=args.workers, memory_reserve_gib=MEMORY_RESERVE_GIB, reused_frames=reused, inference_calls=0, source_and_samples_unchanged=True, signature=signature, verification=verification, crop_manifest=str(OUTPUT_ROOT / 'crop_manifest.csv'), frame_manifest=str(OUTPUT_ROOT / 'frame_manifest.csv'), mask_policy='EXPLICIT_INLINE_BINARY_MASK_AND_FROZEN_SOURCE_DB', scale_verified=False, production_verified=False, quality_review='UPSTREAM_REVIEW_PRESERVED', full_batch_crop=True, seconds=round(time.perf_counter() - started, 1))
                S.save_json(OUTPUT_ROOT / 'completion.json', completed)
                save_status('COMPLETE', **{k: v for (k, v) in completed.items() if k != 'status'})
                database.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                print(json.dumps(completed, ensure_ascii=False, indent=2))
            except BaseException as error:
                failure = dict(status='FULL_CROP_FAILED_OR_INTERRUPTED', finished_at=now(), error=repr(error), traceback=traceback.format_exc(), completed_frames=done, saved_crops=saved, signature=signature, inference_calls=0, resume_supported=True)
                S.save_json(OUTPUT_ROOT / 'completion.json', failure)
                save_status('FAILED', **{k: v for (k, v) in failure.items() if k != 'status'})
                raise

def relocated_completed():
    path = OUTPUT_ROOT / 'completion.json'
    if not path.is_file():
        return None
    prior = read_json(path)
    if prior.get('status') != 'FULL_CROP_COMPLETE_WITH_REVIEW':
        return None
    require(prior.get('verification_status') == 'PASS' and prior.get('input_images') == EXPECTED_FRAMES and (prior.get('slot_rows') == EXPECTED_ROWS) and (prior.get('crop_count') == EXPECTED_TARGETS) and (prior.get('held_frames_preserved') == 134), "Validation condition failed in relocated_completed: prior.get('verification_status') == 'PASS' and prior.get('input_images') == EXPECTED_FRAMES and (prior.get('slot_rows') == EXPECTED_ROWS) and (prior.get('crop_count') == EXPECTED_TARGETS) and (prior.get('held_frames_preserved') == 134)")
    for name in ('crop_manifest.csv', 'frame_manifest.csv'):
        require((OUTPUT_ROOT / name).is_file(), 'Validation condition failed in relocated_completed: (OUTPUT_ROOT / name).is_file()')
    require(SOURCE_DATABASE.is_file(), 'Validation condition failed in relocated_completed: SOURCE_DATABASE.is_file()')
    try:
        with SOURCE_DATABASE.open('rb') as source:
            require(source.read(16) == b'SQLite format 3\x00', "Validation condition failed in relocated_completed: source.read(16) == b'SQLite format 3\\x00'")
    except PermissionError as exc:
        raise PermissionError('Saved mask database is locked by another process: ' + str(SOURCE_DATABASE)) from exc
    with sqlite3.connect((STATE_ROOT / 'state.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        require(db.execute('SELECT count(*) FROM completed').fetchone()[0] == EXPECTED_FRAMES and db.execute('SELECT count(*) FROM crops').fetchone()[0] == EXPECTED_TARGETS, "Validation condition failed in relocated_completed: db.execute('SELECT count(*) FROM completed').fetchone()[0] == EXPECTED_FRAMES and db.execute('SELECT count(*) FROM crops').fetchone()[0] == EXPECTED_TARGETS")
    require(sum((1 for _ in CROP_ROOT.rglob('*.png'))) == EXPECTED_TARGETS, "Validation condition failed in relocated_completed: sum((1 for _ in CROP_ROOT.rglob('*.png'))) == EXPECTED_TARGETS")
    return dict(status='EXISTING_FULL_CROP_RELOCATED_PASS', output_root=str(OUTPUT_ROOT), input_images=EXPECTED_FRAMES, crops=EXPECTED_TARGETS, held_frames_preserved=134, inference_calls=0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true', help='main')
    parser.add_argument('--workers', type=int, default=min(WORKERS, max(1, (os.cpu_count() or 4) - 4)))
    parser.add_argument('--prefetch', type=int, default=PREFETCH_FRAMES)
    args = parser.parse_args()
    require(args.workers > 0 and args.prefetch >= args.workers, 'Validation condition failed in main: args.workers > 0 and args.prefetch >= args.workers')
    completed = relocated_completed()
    if completed:
        print(encoded(completed), flush=True)
        return 0
    print(encoded(dict(stage='READ_ONLY_PREFLIGHT', resources=resource_state(), workers=args.workers)), flush=True)
    (frames, report) = preflight()
    print(encoded({k: v for (k, v) in report.items() if k not in ('dependencies', 'protected_snapshot')}), flush=True)
    if args.check:
        return 0
    run(frames, report, args)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
