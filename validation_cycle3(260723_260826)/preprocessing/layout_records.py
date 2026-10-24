"""Layout records calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[2]
REFERENCE_ROOT = CYCLE.THIRD_ROOT / 'preprocessing'
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
INPUT_ROOT = DATA_ROOT / '3. 밝기보정과 Scale'
OUTPUT_ROOT = DATA_ROOT / '4. 자리배치와 bbox'
TEST_ROOT = OUTPUT_ROOT / 'testimages'
SAMPLE_ROOT = OUTPUT_ROOT / '자리배치_ver1' / 'ver1'
WHOLE_ROOT = OUTPUT_ROOT / '261009_전수_자리배치_ver1'
BRIGHT_MANIFEST = INPUT_ROOT / '_실행기록' / '261009_밝기보정' / 'brightness_manifest.csv'
SCALE_MANIFEST = INPUT_ROOT / '_실행기록' / '261009_SCALE' / 'scale_map.csv'
OCR_POINTER = CYCLE.OCR_POINTER
SEED = 20261009
SAMPLES_PER_CAMERA = 10
EXPECTED_COUNTS = {'cam0': 11837, 'cam1': 13024, 'cam2': 14540}
VERSION = 'S4_SLOT_VER1_S3_FINAL_GEOMETRY_RIGHT12'
COPY_WORKERS = 4
CHECKPOINT_FRAMES = 200
sys.dont_write_bytecode = True
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sqlite3
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def encode(value):
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False, separators=(',', ':'))

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def read_csv(path):
    csv.field_size_limit(10000000)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def csv_value(value):
    return encode(value) if isinstance(value, (list, tuple, dict)) else '' if value is None else value

def write_rows(path, rows, fields):
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for (k, v) in row.items()})
    os.replace(temp, path)

def load_module(name, path):
    spec = resolved_spec(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def inventory():
    pointer = json.loads(OCR_POINTER.read_text(encoding='utf-8-sig'))
    ocr_contract = Path(pointer['manifest']).with_name('02_개선전후_기록.csv')
    ocr_rows = read_csv(ocr_contract)
    ocr = {r['current_path'].casefold(): r for r in ocr_rows}
    bright_rows = read_csv(BRIGHT_MANIFEST)
    scales = read_csv(SCALE_MANIFEST)
    scale = {r['source_path'].casefold(): r for r in scales}
    if len(ocr) != len(ocr_rows) or len(scale) != len(scales):
        raise ValueError('Invalid input or execution contract')
    samples = []
    for b in bright_rows:
        p = Path(b['output_path'])
        camera = p.parent.parent.name
        (s, o) = (scale[p.as_posix().replace('/', '\\').casefold()], ocr[b['source_path'].casefold()])
        if b['status'] != 'OK' or b['label_accepted'] != 'True' or o['label_accepted'] != 'True':
            raise ValueError('Invalid input or execution contract')
        if s['source_sha256'] != b['output_sha256'] or o['sha256'] != b['source_sha256']:
            raise ValueError('Invalid input or execution contract')
        if camera != s['camera'] or camera.upper() != b['camera'] or b['camera'] != o['camera']:
            raise ValueError('Invalid input or execution contract')
        (h, w, _) = json.loads(b['shape'])
        if [w, h] != [int(s['image_width']), int(s['image_height'])]:
            raise ValueError('Invalid input or execution contract')
        if not p.is_file():
            raise FileNotFoundError(p)
        row = dict(camera=camera, input_path=str(p), test_path=str(p), image_name=p.name, frame_id=hashlib.sha256(('4작기|' + s['image_key']).encode()).hexdigest()[:24], source_sha256=b['output_sha256'], source_width=w, source_height=h, brightness_status=b['status'], ocr_current_path=b['source_path'], ocr_action=b['qc_status'], ocr_status=o['ocr_status'], old_key=o['old_key'], new_key=o['new_key'], bed_identity_status='CONFIRMED' if b['identity_verified'] == 'True' else 'REVIEW', ocr_source_sha256=b['source_sha256'], source_camera=b['source_camera'], raw_source_path=b['raw_source_path'], raw_crop_bbox=json.loads(b['crop_bbox']), accepted_number=int(b['accepted_number']), label_source=b['label_source'], identity_verified=b['identity_verified'] == 'True', pair_qc=b['pair_qc'], date_folder=p.parent.name, capture_date=b['date'], selection_seed=SEED, native_width=int(s['native_width']), native_height=int(s['native_height']))
        for k in ('mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_qc', 'scale_version', 'geometry_group'):
            row[k] = s[k]
        row['scale_image_key'] = s['image_key']
        samples.append(row)
    counts = dict(Counter((r['camera'] for r in samples)))
    if counts != EXPECTED_COUNTS or len(samples) != len(scale):
        raise ValueError('Invalid input or execution contract')
    if len({r['frame_id'] for r in samples}) != len(samples):
        raise ValueError('Invalid input or execution contract')
    dependencies = [Path(__file__), BRIGHT_MANIFEST, SCALE_MANIFEST, OCR_POINTER, ocr_contract]
    dependencies += sorted(REFERENCE_ROOT.glob('*.py'))
    support = PROJECT_ROOT / '양상추 생육모델 운영 파이프라인' / 'codex ver2_260913' / 'preprocess'
    dependencies += sorted(support.rglob('*.py'))
    report = dict(version=VERSION, images=len(samples), cameras=counts, numbering={'cam0': 'LEFT_T1_B1', 'cam1': 'RIGHT_T12_B12', 'cam2': 'LEFT_T1_B1'}, seed=SEED, sampling='UNIFORM_WITHOUT_DATE_OR_OCR_STRATIFICATION', dependencies={str(p): sha(p) for p in dependencies})
    fingerprint = hashlib.sha256(encode(report).encode()).hexdigest()
    return (samples, report, fingerprint)

def choose_samples(inputs):
    rng = random.Random(SEED)
    chosen = []
    for camera in EXPECTED_COUNTS:
        pool = sorted((r for r in inputs if r['camera'] == camera), key=lambda r: r['scale_image_key'])
        for (i, source) in enumerate(rng.sample(pool, SAMPLES_PER_CAMERA), 1):
            r = dict(source, sample_index=i, selection_reason='UNIFORM_ALL_DATES', test_path=str(TEST_ROOT / camera / source['image_name']))
            chosen.append(r)
    return chosen

def copy_sample(row):
    (source, target) = (Path(row['input_path']), Path(row['test_path']))
    if sha(source) != row['source_sha256']:
        raise ValueError('선정 원본 hash 변경: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != row['source_sha256']:
            raise FileExistsError('다른 표본은 덮어쓰지 않음: ' + str(target))
    else:
        shutil.copy2(source, target)
    if sha(target) != row['source_sha256']:
        raise ValueError('Invalid input or execution contract')

def load_reference():
    whole = load_module('s4_s3_final_reference', REFERENCE_ROOT / '261001_4. 자리배치_bbox_final_whole.py')
    whole.clone_record_functions()
    whole.MAPPER.RESULT_ROOT = SAMPLE_ROOT
    tree = ast.parse((REFERENCE_ROOT/'slot_geometry.py').read_text(encoding='utf-8-sig'))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'draw_overlay'))

    class Header(ast.NodeTransformer):

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace(' S3 |', ' S4 |')
            return node
    fn = Header().visit(fn)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), '<S4_overlay_header_only>', 'exec'), whole.MAPPER.__dict__)
    refs = whole.MAPPER.load_references()
    return (whole, refs)

def process(sample, whole, refs, model, save_visual):
    source = refs[0].imread_unicode(sample['test_path'])
    if source is None:
        raise ValueError('이미지 안 읽어집니다: ' + sample['test_path'])
    if list(source.shape[:2]) != [sample['source_height'], sample['source_width']]:
        raise ValueError('Invalid input or execution contract')
    if sha(sample['test_path']) != sample['source_sha256']:
        raise ValueError('Invalid input or execution contract')
    (frame, slots, segments) = whole.MAPPER.process_sample(sample, model, refs)
    serialized = [whole.csv_values(r) for r in slots]
    (registered, registration) = whole.REGISTRATION.register_frame(serialized, source)
    reg_by_id = {r['record_id']: r for r in registered}
    for r in slots:
        reg = reg_by_id[r['record_id']]
        r.update({k: v for (k, v) in reg.items() if k.startswith(('eval_', 'anchor_registration', 'registered_', 'cam2_registration', 'raw_hole'))})
        r.update(season_id='4작기', frame_id=sample['frame_id'], source_camera=sample['source_camera'], raw_source_path=sample['raw_source_path'], raw_crop_bbox=sample['raw_crop_bbox'], native_width=sample['native_width'], native_height=sample['native_height'], label_source=sample['label_source'], identity_verified=sample['identity_verified'], pair_qc=sample['pair_qc'], geometry_group=sample['geometry_group'], mapping_version=VERSION, bbox_target_selection_executed=False, downstream_selector='S3_FINAL_ver2.2_INDEPENDENT_ANCHOR_BODY_OWNERSHIP', bbox_handoff_status='SLOT_GEOMETRY_READY_TARGET_SELECTION_PENDING' if r['cell_geometry_ready'] else 'GEOMETRY_REVIEW')
        (x0, y0, _, _) = sample['raw_crop_bbox']
        (r['raw_prior_x'], r['raw_prior_y']) = (r['x_prior'] + x0, r['y_prior'] + y0)
        if str(r['anchor_registration_status']).startswith('REGISTERED'):
            (r['anchor_x'], r['anchor_y']) = (float(r['eval_anchor_x']), float(r['eval_anchor_y']))
        else:
            (r['anchor_x'], r['anchor_y']) = (r['x_prior'], r['y_prior'])
    frame.update(frame_id=sample['frame_id'], season_id='4작기', registration=registration, registration_counts=dict(Counter((r['anchor_registration_status'] for r in slots))), bbox_target_selection_executed=False, full_batch_inference=not save_visual)
    layout_issues = validate_slots(slots, sample)
    frame['layout_validation_issues'] = layout_issues
    if layout_issues:
        frame['processing_status'] = 'REVIEW'
        frame['qc_flags'] += '|' + '|'.join(layout_issues)
        for r in slots:
            r['bbox_handoff_status'] = 'GEOMETRY_REVIEW'
            r['qc_flags'] = frame['qc_flags']
    if save_visual:
        raw_segments = [dict(s, polygon=np.asarray(s['polygon'], np.float32), cx=s['center_x'], cy=s['center_y'], conf=s['confidence']) for s in segments]
        meta = dict(frame['geometry_details'])
        canvas = whole.MAPPER.draw_overlay(source, slots, raw_segments, frame, meta)
        overlay = SAMPLE_ROOT / sample['camera'] / (Path(sample['image_name']).stem + '_overlay.jpg')
        overlay.parent.mkdir(parents=True, exist_ok=True)
        if not refs[0].imwrite_unicode(overlay, canvas):
            raise OSError('오버레이 저장 실패')
        frame['overlay_path'] = str(overlay)
        for r in slots:
            r['overlay_path'] = str(overlay)
    else:
        frame['overlay_path'] = ''
        for r in slots:
            r['overlay_path'] = ''
    return (clean(frame), clean(slots), clean(segments))

def validate_slots(slots, sample):
    issues = []
    ids = [r['record_id'] for r in slots]
    if len(ids) != len(set(ids)):
        raise ValueError('Invalid input or execution contract')
    cols = sorted({r['col'] for r in slots})
    if cols:
        expected = list(range(13 - len(cols), 13)) if sample['camera'] == 'cam1' else list(range(1, len(cols) + 1))
        if cols != expected:
            issues.append('CAMERA_NUMBERING_REVIEW')
    for row_name in ('t', 'b'):
        group = sorted((r for r in slots if r['row'] == row_name), key=lambda r: r['x_prior'])
        if [r['col'] for r in group] != cols:
            issues.append('NONMONOTONIC_OR_INCOMPLETE_ROW_' + row_name.upper())
        for r in group:
            if r['physical_slot'] != row_name + str(r['col']):
                raise ValueError('Invalid input or execution contract')
    if not slots:
        issues.append('NO_SLOT_GEOMETRY')
    if any((not r['cell_geometry_ready'] for r in slots)):
        issues.append('INVALID_CELL_GEOMETRY')
    return issues

def export(db, root):
    for (kind, file_name) in [('frame', 'frame_manifest.csv'), ('slot', 'slot_manifest.csv'), ('segment', 'segment_manifest.csv')]:
        fields = []
        for (payload,) in db.execute('SELECT payload FROM frames ORDER BY sequence'):
            value = json.loads(payload)[kind]
            for r in [value] if kind == 'frame' else value:
                for key in r:
                    if key not in fields:
                        fields.append(key)

        def rows():
            for (payload,) in db.execute('SELECT payload FROM frames ORDER BY sequence'):
                value = json.loads(payload)[kind]
                yield from ([value] if kind == 'frame' else value)
        write_rows(root / file_name, rows(), fields or ['status'])

@contextmanager
def run_lock(root):
    import msvcrt
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a+b') as f:
        if f.seek(0, 2) == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
