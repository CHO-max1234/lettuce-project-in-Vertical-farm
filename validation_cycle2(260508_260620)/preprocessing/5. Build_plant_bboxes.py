"""Assign physical slots, select target masks and save crop-local masks in SQLite. Unknown or unavailable plants are never represented as zero area."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from runtime import yolo_device
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT.parent))
from runtime import moved_tree, legacy_identity_path
UPSTREAM_ROOT = CYCLE.PREPROCESS_OUTPUT
OUTPUT_ROOT = CYCLE.BBOX_OUTPUT
SAMPLE_ROOT = OUTPUT_ROOT / 'whole_samples'
STATE_ROOT = OUTPUT_ROOT / '_whole_run'
MODEL_PATH = CYCLE.PLANT_MODEL
SEED = 261001
SAMPLES_PER_CAMERA = 30
(WHOLE_SIZE, LOCAL_SIZE, DEVICE) = (1280, 512, yolo_device())
(DECODE_WORKERS, PREFETCH) = (3, 6)
CSV_CHECKPOINT_FRAMES = 2000
VERSION = 'ver2.2_final_whole'
RUNNER_REVISION = '1.1_INVALID_CELL_FRAME_REVIEW_GUARD'
import argparse
import ast
import csv
import hashlib
import json
import os
import random
import sqlite3
import time
import traceback
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.dont_write_bytecode = True
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj
MAPPER = load_module('s3_whole_mapper', CODE_ROOT / 'slot_geometry.py')
RUNNER = load_module('s3_whole_sample_runner', CODE_ROOT / 'segmentation_candidates.py')
BBOX = load_module('s3_whole_bbox', CODE_ROOT / 'bbox_records.py')
REGISTRATION = load_module('s3_whole_registration', CODE_ROOT / 'anchor_registration.py')
SELECTOR = load_module('s3_whole_selector', CODE_ROOT / 'mask_selection.py')
(clean, digest, keypath) = (MAPPER.clean, MAPPER.digest, MAPPER.keypath)

def encode(value):
    return json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def csv_values(row):
    return {key: encode(val) if isinstance(val, (dict, list, tuple, np.ndarray)) else '' if clean(val) is None else clean(val) for (key, val) in row.items()}

def clone_record_functions():
    tree = ast.parse((CODE_ROOT / 'slot_geometry.py').read_text(encoding='utf-8-sig'))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_sample'))
    fn.body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and (ast.unparse(n.value.func) == 'overlay.parent.mkdir')) and (not (isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'canvas' for t in n.targets)))) and (not (isinstance(n, ast.If) and 'imwrite_unicode' in ast.unparse(n.test)))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<frozen_layout_without_image_write>', 'exec'), MAPPER.__dict__)
    tree = ast.parse((CODE_ROOT / 'segmentation_candidates.py').read_text(encoding='utf-8-sig'))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export_frame'))
    fn.name = 'records_without_images'
    loop_index = next((i for (i, n) in enumerate(fn.body) if isinstance(n, ast.For)))
    fn.body = fn.body[:loop_index + 1] + [ast.Return(value=ast.Name(id='records', ctx=ast.Load()))]

    class RemoveVisuals(ast.NodeTransformer):

        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in ('save_image', 'cv2.drawContours', 'cv2.rectangle', 'cv2.drawMarker', 'cv2.putText'):
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any((isinstance(t, ast.Name) and t.id == 'canvas' for t in node.targets)):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == '0 <= ax < w and 0 <= ay < h':
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'digest':
                return ast.Constant(value='')
            return self.generic_visit(node)
    fn = RemoveVisuals().visit(fn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<frozen_bbox_record_serializer>', 'exec'), RUNNER.__dict__)

def inventory():
    with MAPPER.OCR_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        ocr_rows = [moved_tree(r) for r in csv.DictReader(f)]
    with MAPPER.SCALE_MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        scale_rows = [moved_tree(r) for r in csv.DictReader(f)]
    ocr = {keypath(r['current_path']): r for r in ocr_rows}
    scales = {keypath(r['source_path']): r for r in scale_rows}
    if len(ocr) != len(ocr_rows) or len(scales) != len(scale_rows):
        raise ValueError('Invalid input or state in inventory (len(ocr) != len(ocr_rows) or len(scales) != len(scale_rows))')
    (samples, seen) = ([], set())
    for b in MAPPER.iter_json_array(MAPPER.BRIGHT_MANIFEST):
        b = moved_tree(b)
        path = Path(b['output_path'])
        k = keypath(path)
        if k in seen:
            raise ValueError('Invalid input or state in inventory (k in seen)')
        seen.add(k)
        (o, sc) = (ocr.get(keypath(b['source_path'])), scales.get(k))
        if not o or not sc or o['sha256'] != b['source_sha256'] or (sc['source_sha256'] != b['output_sha256']):
            raise ValueError(f"Invalid input or state in inventory (not o or not sc or o['sha256'] != b['source_sha256'] or (sc['source_sha256'] != b['output_sha256'])): {path} ")
        if not path.is_file() or b['camera'] != sc['camera'] or b['camera'] != o['camera']:
            raise ValueError(f"Invalid input or state in inventory (not path.is_file() or b['camera'] != sc['camera'] or b['camera'] != o['camera']): {path} ")
        row = dict(camera=b['camera'], input_path=str(path), test_path=str(path), image_name=path.name, source_sha256=b['output_sha256'], source_width=b['shape'][1], source_height=b['shape'][0], brightness_status=b['status'], ocr_current_path=o['current_path'], ocr_action=o['status'], ocr_status=o['ocr_status'], old_key=o['old_key'], new_key=o['new_key'], ocr_source_sha256=o['sha256'], bed_identity_status='REVIEW' if o['status'] == 'REVIEW' else 'CONFIRMED', date_folder=path.parent.name, capture_date=path.stem.split('_')[1], selection_seed=SEED)
        row.update({key: sc[key] for key in ('mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_qc', 'scale_version', 'geometry_group')})
        row['scale_image_key'] = sc['image_key']
        legacy_key = os.path.normcase(os.path.abspath(legacy_identity_path(path)))
        row['frame_id'] = hashlib.sha256(legacy_key.encode()).hexdigest()[:24]
        samples.append(row)
    if len(samples) != 43198 or Counter((r['camera'] for r in samples)) != Counter(cam0=16034, cam1=19318, cam2=7846):
        raise ValueError("Invalid input or state in inventory (len(samples) != 43198 or Counter((r['camera'] for r in samples)) != Counter(cam0=16034, cam1=19318, cam2=7846))")
    deps = [Path(__file__), CODE_ROOT / 'slot_geometry.py', CODE_ROOT / 'bbox_records.py', CODE_ROOT / 'segmentation_candidates.py', CODE_ROOT / 'anchor_registration.py', CODE_ROOT / 'front_registration.py', CODE_ROOT / 'mask_selection.py', Path(SELECTOR.__file__)]
    deps += [p for p in MAPPER.REFERENCE_ROOT.rglob('*.py')]
    deps += [MAPPER.OCR_MANIFEST, MAPPER.SCALE_MANIFEST, MAPPER.BRIGHT_MANIFEST, MODEL_PATH]
    hashes = {str(p): digest(p) for p in dict.fromkeys(deps)}
    if hashes[str(MODEL_PATH)] != RUNNER.MODEL_SHA256:
        raise ValueError('Invalid input or state in inventory (hashes[str(MODEL_PATH)] != RUNNER.MODEL_SHA256)')
    report = dict(status='FULL_INPUT_LINEAGE_CHECK_PASS', input_images=len(samples), cameras=dict(Counter((r['camera'] for r in samples))), ocr_review_images=sum((r['ocr_action'] == 'REVIEW' for r in samples)), scale_unverified_images=sum((r['scale_verified'] == 'False' for r in samples)), dependency_hashes=hashes, date_ranges={c: [min((r['capture_date'] for r in samples if r['camera'] == c)), max((r['capture_date'] for r in samples if r['camera'] == c))] for c in ('cam0', 'cam1', 'cam2')}, selection_seed=SEED, samples_per_camera=SAMPLES_PER_CAMERA, variant=VERSION)
    fingerprint = hashlib.sha256(encode(report).encode()).hexdigest()
    return (samples, report, fingerprint)

def choose_samples(samples):
    rng = random.Random(SEED)
    chosen = []
    for camera in ('cam0', 'cam1', 'cam2'):
        pool = sorted((r for r in samples if r['camera'] == camera), key=lambda r: (r['capture_date'], r['image_name']))
        dates = sorted({r['capture_date'] for r in pool})
        used = set()
        for bin_index in range(SAMPLES_PER_CAMERA):
            date = dates[min(len(dates) - 1, bin_index * len(dates) // SAMPLES_PER_CAMERA)]
            candidates = [r for r in pool if r['capture_date'] == date and r['frame_id'] not in used]
            diverse = [r for r in candidates if r['image_name'].split('_')[0] not in {x['image_name'].split('_')[0] for x in chosen if x['camera'] == camera}]
            pick = dict(rng.choice(diverse or candidates))
            used.add(pick['frame_id'])
            pick.update(sample_index=bin_index + 1, selection_reason='UNIFORM_AVAILABLE_DATE_STRATA_RANDOM_BED_TIME')
            chosen.append(pick)
    assert len(chosen) == 90 and len({r['frame_id'] for r in chosen}) == 90
    return chosen

class Predictor:

    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(MODEL_PATH))
        self.whole_calls = self.local_calls = 0

    def __call__(self, image, whole=False):
        if whole:
            self.whole_calls += 1
        else:
            self.local_calls += 1
        return self.model.predict(image, imgsz=WHOLE_SIZE if whole else LOCAL_SIZE, conf=0.2, iou=0.7, retina_masks=True, rect=whole, verbose=False, device=DEVICE)[0]

def decode(sample):
    content = Path(sample['input_path']).read_bytes()
    if hashlib.sha256(content).hexdigest() != sample['source_sha256']:
        raise ValueError("Invalid input or state in decode (hashlib.sha256(content).hexdigest() != sample['source_sha256'])")
    source = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None or source.shape[:2] != (int(sample['source_height']), int(sample['source_width'])):
        raise ValueError("Invalid input or state in decode (source is None or source.shape[:2] != (int(sample['source_height']), int(sample['source_width'])))")
    return source

def decisions_for(rows, source, whole, predictor):
    (rows, registration) = REGISTRATION.register_frame(rows, source)
    candidates = RUNNER.candidates(whole, source, [0, 0, source.shape[1], source.shape[0]], 'whole')
    (local, skipped) = ({}, 0)
    for row in rows:
        if not SELECTOR.needs_local(row, source, candidates):
            local[row['record_id']] = []
            skipped += 1
            continue
        box = RUNNER.centered_box(row, source, 1.45)
        if box is None:
            local[row['record_id']] = []
            continue
        result = predictor(source[box[1]:box[3], box[0]:box[2]])
        local[row['record_id']] = RUNNER.candidates(result, source, box, 'local')
    decisions = SELECTOR.select_frame(rows, source, candidates, local)
    return (rows, decisions, registration, skipped)

def mask_blob(mask, bbox):
    (x1, y1, x2, y2) = bbox
    crop = mask[y1:y2, x1:x2]
    return (zlib.compress(np.packbits(crop, bitorder='little').tobytes(), level=3), list(crop.shape), hashlib.sha256(crop.tobytes()).hexdigest())

def read_target_mask(database_path, record_id):
    """Read one saved target record and restore its crop-local binary mask with shape and checksum validation."""
    with sqlite3.connect(Path(database_path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        item = db.execute('SELECT bbox_json,mask FROM slots WHERE record_id=?', (record_id,)).fetchone()
    if item is None:
        raise KeyError(record_id)
    record = json.loads(item[0])
    if item[1] is None:
        return (record, None)
    shape = record['target_mask_shape']
    packed = np.frombuffer(zlib.decompress(item[1]), np.uint8)
    mask = np.unpackbits(packed, bitorder='little', count=int(np.prod(shape))).reshape(shape).astype(bool)
    if hashlib.sha256(mask.tobytes()).hexdigest() != record['target_mask_pixels_sha256']:
        raise ValueError("Invalid input or state in read_target_mask (hashlib.sha256(mask.tobytes()).hexdigest() != record['target_mask_pixels_sha256'])")
    return (record, mask)

def process_frame(sample, source, predictor, refs, sampled):
    whole = predictor(source, whole=True)
    proxy = SimpleNamespace(predict=lambda **_: [whole])
    original_read = refs[0].imread_unicode
    refs[0].imread_unicode = lambda _: source
    try:
        (frame, slots, _) = MAPPER.process_sample(sample, proxy, refs)
    finally:
        refs[0].imread_unicode = original_read
    serialized = [csv_values(r) for r in slots]
    invalid_geometry = any((float(r['cell_width']) <= 0 or float(r['cell_height']) <= 0 for r in slots))
    base = [] if invalid_geometry else [BBOX.build_bbox_record(r, serialized) for r in serialized]
    if invalid_geometry:
        registered = [dict(r, eval_anchor_x=float(r['x_prior']), eval_anchor_y=float(r['y_prior']), anchor_registration_status='INVALID_CELL_GEOMETRY', anchor_registration_review=True) for r in serialized]
        decisions = {r['record_id']: dict(mask=None, status='INVALID_CELL_GEOMETRY', reasons=['FRAME_CELL_GEOMETRY_INVALID'], route='', review_required=True) for r in registered}
        registration = dict(status='INVALID_CELL_GEOMETRY_REVIEW', invalid_slots=[r['physical_slot'] for r in slots if float(r['cell_width']) <= 0 or float(r['cell_height']) <= 0])
        skipped = len(registered)
    elif base:
        (registered, decisions, registration, skipped) = decisions_for(base, source, whole, predictor)
    else:
        (registered, decisions, registration, skipped) = ([], {}, {}, 0)
    if sampled:
        (records, visuals) = RUNNER.export_frame(sample, registered, source, decisions, SAMPLE_ROOT, VERSION, {})
        layout_path = SAMPLE_ROOT / sample['camera'] / 'slot_overlays' / (Path(sample['image_name']).stem + '_slots.jpg')
        (_, meta, segments) = MAPPER.map_image(source, sample['camera'], proxy, refs)
        RUNNER.save_image(layout_path, MAPPER.draw_overlay(source, slots, segments, frame, meta))
        frame.update(visuals, slot_overlay_path=str(layout_path))
    else:
        records = RUNNER.records_without_images(sample, registered, source, decisions, SAMPLE_ROOT, VERSION, {})
        frame.update(overlay_path='', sheet_path='', slot_overlay_path='')
    masks = {}
    registered_by_id = {r['record_id']: r for r in registered}
    for slot in slots:
        reg = registered_by_id.get(slot['record_id'], {})
        slot.update({k: v for (k, v) in reg.items() if k.startswith(('eval_', 'anchor_registration', 'registered_', 'raw_hole'))})
        slot.update(frame_id=sample['frame_id'], experiment_version=VERSION, overlay_path=frame.get('slot_overlay_path', ''), full_batch_inference=True)
    for record in records:
        target = bool(decisions[record['record_id']].get('mask') is not None and decisions[record['record_id']]['mask'].any())
        record.update(frame_id=sample['frame_id'], target_available=target, crop_saved=sampled and target, has_crop=sampled and target, full_batch_inference=True, production_verified=False, target_mask_database=str(STATE_ROOT / 'state.sqlite3'), target_mask_record_id=record['record_id'], target_mask_encoding='ZLIB_PACKBITS_LITTLE_CROP_LOCAL_V1', target_mask_shape='', target_mask_pixels_sha256='')
        if target:
            (blob, shape, sha) = mask_blob(decisions[record['record_id']]['mask'], record['final_bbox'])
            masks[record['record_id']] = blob
            record.update(target_mask_shape=shape, target_mask_pixels_sha256=sha, bbox_handoff_status='TARGET_BODY_READY_REVIEW_PENDING')
            if not sampled:
                record.update(bbox_crop_path='', masked_crop_path='', diagnostic_mask_path='', bbox_crop_sha256='', mask_sha256='', bbox_crop_status='TARGET_BBOX_READY_NOT_CROPPED')
        elif not sampled:
            record.update(bbox_crop_path='', masked_crop_path='', diagnostic_mask_path='')
        record['overlay_path'] = frame.get('overlay_path', '')
    frame.update(frame_id=sample['frame_id'], sample_selected=sampled, status='INVALID_CELL_GEOMETRY_REVIEW' if invalid_geometry else 'PROCESSED' if slots else 'NO_SLOT_GEOMETRY_REVIEW', runner_revision=RUNNER_REVISION, target_count=sum((r['target_available'] for r in records)), skipped_local_slots=skipped, registration=registration, slot_count=len(slots), selection_statuses=dict(Counter((r['selection_status'] for r in records))))
    return (frame, slots, records, masks)

def connect_state():
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE_ROOT / 'state.sqlite3', timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS frames (frame_id TEXT PRIMARY KEY,camera TEXT NOT NULL,status TEXT NOT NULL,data TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS slots (record_id TEXT PRIMARY KEY,frame_id TEXT NOT NULL,slot_json TEXT NOT NULL,bbox_json TEXT NOT NULL,mask BLOB)')
    db.execute('CREATE INDEX IF NOT EXISTS slots_frame ON slots(frame_id)')
    db.commit()
    return db

def save_frame(db, frame, slots, records, masks):
    assert len(slots) == len(records) and {r['record_id'] for r in slots} == {r['record_id'] for r in records}
    slots_by_id = {r['record_id']: r for r in slots}
    with db:
        db.execute('INSERT INTO frames VALUES (?,?,?,?)', (frame['frame_id'], frame['camera'], frame['status'], encode(frame)))
        db.executemany('INSERT INTO slots VALUES (?,?,?,?,?)', [(r['record_id'], frame['frame_id'], encode(slots_by_id[r['record_id']]), encode(r), masks.get(r['record_id'])) for r in records])

def export_table(db, column, path, table='slots'):
    fields = []
    seen = set()
    for (value,) in db.execute(f'SELECT {column} FROM {table} ORDER BY rowid'):
        for k in json.loads(value):
            if k not in seen:
                seen.add(k)
                fields.append(k)
    temp = path.with_suffix('.partial.csv')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or ['status'])
        writer.writeheader()
        for (value,) in db.execute(f'SELECT {column} FROM {table} ORDER BY rowid'):
            writer.writerow(csv_values(json.loads(value)))
    temp.replace(path)

def export_all(db):
    export_table(db, 'slot_json', OUTPUT_ROOT / 'slot_manifest.csv')
    export_table(db, 'bbox_json', OUTPUT_ROOT / 'bbox_manifest.csv')
    export_table(db, 'data', OUTPUT_ROOT / 'frame_manifest.csv', 'frames')

def progress(db, total, started, predictor, status='RUNNING'):
    frames = db.execute('SELECT count(*) FROM frames').fetchone()[0]
    cameras = dict(db.execute('SELECT camera,count(*) FROM frames GROUP BY camera').fetchall())
    errors = db.execute("SELECT count(*) FROM frames WHERE status='FRAME_ERROR'").fetchone()[0]
    geometry_review = db.execute("SELECT count(*) FROM frames WHERE status='NO_SLOT_GEOMETRY_REVIEW'").fetchone()[0]
    invalid_geometry = db.execute("SELECT count(*) FROM frames WHERE status='INVALID_CELL_GEOMETRY_REVIEW'").fetchone()[0]
    rows = db.execute('SELECT count(*) FROM slots').fetchone()[0]
    report = dict(status=status, input_images=total, completed_frames=frames, slot_rows=rows, cameras=cameras, frame_errors=errors, no_slot_geometry_images=geometry_review, invalid_cell_geometry_images=invalid_geometry, runner_revision=RUNNER_REVISION, seconds_current_session=round(time.time() - started, 1), whole_inference_calls_session=predictor.whole_calls, local_inference_calls_session=predictor.local_calls, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), variant=VERSION, full_batch_inference=True, production_review='PENDING', sample_images_requested=90)
    RUNNER.write_json(STATE_ROOT / 'status.json', report)
    return report

def desktop_alert(message):
    print(message, file=sys.stderr, flush=True)

def relocated_completed():
    path = OUTPUT_ROOT / 'completion.json'
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('status') != 'COMPLETE_WITH_REVIEW':
        return None
    if report.get('input_images') != 43198 or report.get('slot_rows') != 717121 or report.get('target_rows') != 244456 or (report.get('frame_errors') != 0):
        raise ValueError("Invalid input or state in relocated_completed (report.get('input_images') != 43198 or report.get('slot_rows') != 717121 or report.get('target_rows') != 244456 or (report.get('frame_errors') != 0))")
    for name in ('frame_manifest.csv', 'slot_manifest.csv', 'bbox_manifest.csv'):
        if not (OUTPUT_ROOT / name).is_file():
            raise FileNotFoundError('Invalid input or state in relocated_completed (not (OUTPUT_ROOT / name).is_file())')
    with sqlite3.connect((STATE_ROOT / 'state.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        if db.execute('SELECT count(*) FROM frames').fetchone()[0] != 43198 or db.execute('SELECT count(*) FROM slots').fetchone()[0] != 717121:
            raise ValueError("Invalid input or state in relocated_completed (db.execute('SELECT count(*) FROM frames').fetchone()[0] != 43198 or db.execute('SELECT count(*) FROM slots').fetchone()[0] != 717121)")
    return dict(status='EXISTING_FULL_BBOX_RELOCATED_PASS', root=str(OUTPUT_ROOT), frames=43198, slots=717121, targets=244456, held_frames=134, inference_calls=0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    completed = relocated_completed()
    if completed:
        print(encode(completed), flush=True)
        return
    clone_record_functions()
    (samples, preflight, fingerprint) = inventory()
    selected = choose_samples(samples)
    if args.check:
        print(encode({**preflight, 'dependency_hashes_count': len(preflight['dependency_hashes']), 'dependency_hashes': None, 'sample_dates': {c: dict(Counter((r['capture_date'] for r in selected if r['camera'] == c))) for c in ('cam0', 'cam1', 'cam2')}}))
        return
    db = connect_state()
    existing = db.execute("SELECT value FROM config WHERE key='fingerprint'").fetchone()
    if existing and existing[0] != fingerprint:
        raise ValueError('Invalid input or state in main (existing and existing[0] != fingerprint)')
    if not existing:
        if (OUTPUT_ROOT / 'slot_manifest.csv').exists() or (OUTPUT_ROOT / 'bbox_manifest.csv').exists():
            raise FileExistsError("Invalid input or state in main ((OUTPUT_ROOT / 'slot_manifest.csv').exists() or (OUTPUT_ROOT / 'bbox_manifest.csv').exists())")
        with db:
            db.execute('INSERT INTO config VALUES (?,?)', ('fingerprint', fingerprint))
    if args.export:
        export_all(db)
        return
    RUNNER.write_json(STATE_ROOT / 'preflight.json', {**preflight, 'fingerprint': fingerprint})
    RUNNER.write_csv(SAMPLE_ROOT / 'sample_manifest.csv', selected)
    selected_ids = {r['frame_id'] for r in selected}
    with db:
        db.execute("DELETE FROM frames WHERE status='FRAME_ERROR'")
    done = {r[0] for r in db.execute('SELECT frame_id FROM frames')}
    pending = [r for r in samples if r['frame_id'] not in done]
    random.Random(SEED + 1).shuffle(pending)
    pending.sort(key=lambda r: r['frame_id'] not in selected_ids)
    started = time.time()
    predictor = Predictor()
    refs = MAPPER.load_references()
    export_all(db)
    progress(db, len(samples), started, predictor)
    notified = False
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as workers:
        futures = {}
        for index in range(min(PREFETCH, len(pending))):
            futures[index] = workers.submit(decode, pending[index])
        bar = tqdm(total=len(samples), initial=len(done), desc='S3 ver2.2 whole', unit='frame', mininterval=10)
        for (index, sample) in enumerate(pending):
            try:
                source = futures.pop(index).result()
                (frame, slots, records, masks) = process_frame(sample, source, predictor, refs, sample['frame_id'] in selected_ids)
                save_frame(db, frame, slots, records, masks)
            except Exception as error:
                frame = dict(sample, status='FRAME_ERROR', error=repr(error), traceback=traceback.format_exc())
                with db:
                    db.execute('INSERT INTO frames VALUES (?,?,?,?)', (sample['frame_id'], sample['camera'], 'FRAME_ERROR', encode(frame)))
                print('\nFRAME ERROR ' + sample['image_name'] + ' ' + repr(error), flush=True)
                if not notified:
                    desktop_alert('main')
                    notified = True
                if db.execute("SELECT count(*) FROM frames WHERE status='FRAME_ERROR'").fetchone()[0] >= 5:
                    raise RuntimeError('Invalid input or state in main (db.execute("SELECT count(*) FROM frames WHERE status=\'FRAME_ERROR\'").fetchone()[0] >= 5)') from error
            finally:
                next_index = index + PREFETCH
                if next_index < len(pending):
                    futures[next_index] = workers.submit(decode, pending[next_index])
            bar.update(1)
            if (index + 1) % 25 == 0 or index == len(pending) - 1:
                state = progress(db, len(samples), started, predictor)
                bar.set_postfix(rows=state['slot_rows'], errors=state['frame_errors'], refresh=False)
            if (index + 1) % CSV_CHECKPOINT_FRAMES == 0 or index + 1 == 90:
                export_all(db)
        bar.close()
    export_all(db)
    for (path, sha) in preflight['dependency_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Invalid input or state in main (digest(path) != sha)')
    count = db.execute('SELECT count(*) FROM frames').fetchone()[0]
    if count != len(samples):
        raise ValueError('Invalid input or state in main (count != len(samples))')
    review = db.execute("SELECT count(*) FROM frames WHERE status!='PROCESSED'").fetchone()[0]
    summary = progress(db, len(samples), started, predictor, 'COMPLETE_WITH_REVIEW' if review else 'COMPLETE_REVIEW_PENDING')
    summary.update(fingerprint=fingerprint, slot_manifest=str(OUTPUT_ROOT / 'slot_manifest.csv'), bbox_manifest=str(OUTPUT_ROOT / 'bbox_manifest.csv'), target_mask_database=str(STATE_ROOT / 'state.sqlite3'), source_images_unchanged=True, upstream_hashes_unchanged=True, numbering='CAM0/CAM2 LEFT1; CAM1 RIGHT12', sample_overlay_counts={c: len(list((SAMPLE_ROOT / c / 'overlays').glob('*.jpg'))) for c in ('cam0', 'cam1', 'cam2')}, selection_statuses=dict(Counter((json.loads(r[0])['selection_status'] for r in db.execute('SELECT bbox_json FROM slots')))), target_rows=sum((json.loads(r[0])['target_available'] for r in db.execute('SELECT bbox_json FROM slots'))), crop_stage_ready=True, source_mask_encoding='ZLIB_PACKBITS_LITTLE_CROP_LOCAL_V1')
    if summary['sample_overlay_counts'] != {'cam0': 30, 'cam1': 30, 'cam2': 30}:
        raise ValueError("Invalid input or state in main (summary['sample_overlay_counts'] != {'cam0': 30, 'cam1': 30, 'cam2': 30})")
    RUNNER.write_json(OUTPUT_ROOT / 'completion.json', summary)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    text = 'Whole-cycle bbox processing complete. Read completion.json for counts and review states. Saved masks must be reused by the crop stage.\n'
    (OUTPUT_ROOT / 'execution_summary.txt').write_text(text, encoding='utf-8')
    print(encode(summary), flush=True)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        detail = traceback.format_exc()
        print(detail, file=sys.stderr, flush=True)
        try:
            RUNNER.write_json(STATE_ROOT / 'fatal_error.json', dict(status='FAILED', traceback=detail, time=time.strftime('%Y-%m-%d %H:%M:%S')))
            status_path = STATE_ROOT / 'status.json'
            old_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            RUNNER.write_json(status_path, {**old_status, 'status': 'FAILED', 'error': detail[-1400:], 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})
            desktop_alert('module')
        except Exception:
            pass
        raise
