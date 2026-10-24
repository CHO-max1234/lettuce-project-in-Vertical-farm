"""Segment top-view beds using the fixed camera-routing and recovery policy."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
RAW_ROOT = CYCLE.RAW_TOP_ROOT
STAGE_ROOT = CYCLE.PREPROCESS_OUTPUT / '1. ROI_BOX'
OUTPUT_ROOT = STAGE_ROOT / 'ROI_BOX'
LOG_ROOT = STAGE_ROOT / '_실행기록' / '261005_윗면_CAM01_전체'
REFERENCE_ROOT = STAGE_ROOT / 'Testimages' / '_실행기록' / '261005_윗면_고정대상_회복검증_ver2'
MODEL_PATH = CYCLE.BED_TOP_MODEL
SURVEY_PATH = CYCLE.GROWTH_FILE
GPU_PYTHON = CYCLE.PYTHON
(START_DATE, END_DATE) = ('20260724', '20260820')
EXPECTED_MODEL_SHA256 = '38e45d96edc6727c1de2c9c7c20079c7b7355403bbb8b602a7e43372cf71b66f'
(BATCH_SIZE, CHUNK_SIZE, IO_WORKERS) = (8, 16, 4)
POLICY = [('baseline_rect1024', 'raw'), ('flip_rect1024', 'flip'), ('rotate5', 'rotate5')]
KNOWN_EQUIPMENT = {'bed00_20260731_105659_cam0.jpg', 'bed00_20260731_110026_cam0.jpg', 'bed00_20260731_110036_cam0.jpg'}
import argparse
import csv
import hashlib
import json
import os
import re
import runpy
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
if __name__ == '__main__' and Path(sys.executable).resolve() != GPU_PYTHON.resolve():
    raise SystemExit(subprocess.call([str(GPU_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()), *sys.argv[1:]]))
os.environ['YOLO_AUTOINSTALL'] = 'false'
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
cv2.setNumThreads(1)
diagnostic = run_source(str(CODE_ROOT / 'top_bed_policy.py'))
sha256 = diagnostic['base']['sha256']
NAME_RE = re.compile('^(bed\\d+)_(20\\d{6})_(\\d{6})_(cam[01])\\.jpg$', re.I)

def discover():
    rows = []
    destinations = set()
    for folder in sorted(RAW_ROOT.iterdir()):
        if not folder.is_dir() or not re.fullmatch('\\d{6}', folder.name):
            continue
        date = '20' + folder.name
        if not START_DATE <= date <= END_DATE:
            continue
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix.lower() != '.jpg':
                continue
            m = NAME_RE.fullmatch(p.name)
            if not m or m[2] != date:
                raise ValueError(f'   /: {p}')
            source_camera = m[4].upper()
            camera = ('CAM1' if source_camera == 'CAM0' else 'CAM0') if date == '20260820' else source_camera
            filename = f'{m[1]}_{date}_{m[3]}_{camera.lower()}.jpg'
            dest = OUTPUT_ROOT / camera / folder.name / filename
            if str(dest).lower() in destinations:
                raise ValueError(f' : {dest}')
            destinations.add(str(dest).lower())
            st = p.stat()
            rows.append(dict(source_path=str(p), source_filename=p.name, date=date, source_camera=source_camera, camera=camera, camera_corrected=date == '20260820', filename=filename, target_path=str(dest), source_size=st.st_size, source_mtime_ns=st.st_mtime_ns))
    if not rows:
        raise RuntimeError('Invalid input or execution contract')
    return rows

def contract(rows):
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    model_hash = sha256(MODEL_PATH)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError('Invalid input or execution contract')
    ref = json.loads((REFERENCE_ROOT / 'completion.json').read_text(encoding='utf-8'))
    if ref['model_sha256'] != model_hash:
        raise ValueError('Invalid input or execution contract')
    return dict(version='season4_top_fixed_target_full_v1', source_fingerprint=digest, source_count=len(rows), code_sha256=sha256(__file__), dependencies_sha256={str(CODE_ROOT / n): sha256(CODE_ROOT / n) for n in ['top_bed_policy.py', 'bed_detection.py']}, model_path=str(MODEL_PATH), model_sha256=model_hash, profiles=ref['profiles'], policy=POLICY, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE, conf_predict=0.25, conf_accept=0.45, fixed_position_rule={'cy': [0.4, 0.68], 'max_height': 0.42, 'max_bottom': 0.8, 'center_distance_gap': 0.08}, camera_correction={'20260820': {'CAM0': 'CAM1', 'CAM1': 'CAM0'}}, crop_rule='selected polygon bounding rectangle, zero margin, original pixels, jpeg quality95', one_target_per_image=True, training=False, ocr=False, bed_identity_verified=False, date_min=START_DATE, date_max=END_DATE, production_scope='top CAM0/CAM1 only')

def read_source(row):
    out = dict(row)
    try:
        p = Path(row['source_path'])
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) != (row['source_size'], row['source_mtime_ns']):
            raise ValueError('Invalid input or execution contract')
        data = p.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Invalid input or execution contract')
        out.update(source_sha256=hashlib.sha256(data).hexdigest(), image_width=image.shape[1], image_height=image.shape[0])
        if row['source_filename'] in KNOWN_EQUIPMENT:
            out.update(status='EXCLUDED_EQUIPMENT', reason='7/31 CAM0 설비 구도 시각 확인')
            return (out, None)
        if min(image.shape[:2]) < 100:
            out.update(status='INVALID_LOW_RESOLUTION', reason='베드 분석 불가 저해상도')
            return (out, None)
        small = cv2.resize(image, (64, 48))
        mean = float(small.mean())
        std = float(small.std())
        out.update(image_mean=round(mean, 3), image_std=round(std, 3))
        if mean < 3 or std < 1:
            out.update(status='INVALID_BLACK_OR_CONSTANT', reason='검정/상수 영상')
            return (out, None)
        return (out, image)
    except Exception as e:
        out.update(status='ERROR_READ', reason=f'{type(e).__name__}: {e}')
        return (out, None)

def choose(detections, w, h, profile):
    (ax, ay) = (profile['anchor'][0] * w, profile['anchor'][1] * h)
    candidates = []
    for d in detections:
        if not diagnostic['acceptable'](d, w, h):
            continue
        (x1, y1, x2, y2) = d['crop_bbox']
        cy = (y1 + y2) / 2 / h
        if 0.4 <= cy <= 0.68 and (y2 - y1) / h <= 0.42 and (y2 / h <= 0.8) and (x1 <= ax <= x2) and (y1 <= ay <= y2):
            candidates.append(d)
    if len(candidates) > 1:
        candidates.sort(key=lambda d: abs((d['crop_bbox'][1] + d['crop_bbox'][3]) / 2 / h - profile['anchor'][1]))
        ds = [abs((d['crop_bbox'][1] + d['crop_bbox'][3]) / 2 / h - profile['anchor'][1]) for d in candidates]
        if ds[1] - ds[0] >= 0.08:
            candidates = candidates[:1]
    return (candidates[0] if len(candidates) == 1 else None, len(candidates))

def infer_group(model, items, profiles):
    pending = list(range(len(items)))
    output = [None] * len(items)
    attempts = defaultdict(list)
    for (variant, kind) in POLICY:
        if not pending:
            break
        transformed = [diagnostic['transform'](items[i][1], kind) for i in pending]
        results = model.predict(transformed, imgsz=1024, rect=True, conf=0.25, iou=0.7, max_det=3, device=0, half=True, verbose=False)
        if len(results) != len(pending):
            raise RuntimeError('Invalid input or execution contract')
        remaining = []
        for (k, i) in enumerate(pending):
            (row, image) = items[i]
            r = results[k]
            if r.orig_img.shape != transformed[k].shape or not np.array_equal(r.orig_img, transformed[k]):
                raise RuntimeError('Invalid input or execution contract')
            found = diagnostic['detections'](r, image, kind)
            (d, n) = choose(found, image.shape[1], image.shape[0], profiles[row['camera'].lower()])
            attempts[i].append(dict(variant=variant, detections=len(found), target_candidates=n, best_confidence=max([x['confidence'] for x in found], default=0)))
            if d is not None:
                output[i] = {**row, 'status': 'SAVED_TARGET_CANDIDATE', 'reason': '', 'selected_variant': variant, 'confidence': d['confidence'], 'bbox': d['crop_bbox'], 'polygon': d['polygon']}
            elif n > 1:
                output[i] = {**row, 'status': 'REVIEW_OVERLAPPING_TARGETS', 'reason': '동일 위치 후보 다수'}
            else:
                remaining.append(i)
        pending = remaining
    for i in pending:
        output[i] = {**items[i][0], 'status': 'NO_FIXED_TARGET', 'reason': '고정 위치 대상 검출 기준 미통과'}
    for (i, row) in enumerate(output):
        row['attempts'] = attempts[i]
        row['actual_bed'] = None
        row['human_qc'] = 'NOT_REVIEWED'
    return output

def infer_safe(model, items, profiles):
    try:
        return infer_group(model, items, profiles)
    except Exception as e:
        if len(items) == 1:
            return [{**items[0][0], 'status': 'ERROR_INFERENCE', 'reason': f'{type(e).__name__}: {e}'}]
        import torch
        torch.cuda.empty_cache()
        print(f'batch  {len(items)}: {type(e).__name__}', flush=True)
        return [r for item in items for r in infer_safe(model, [item], profiles)]

def write_crop(row, image):
    if row['status'] != 'SAVED_TARGET_CANDIDATE':
        return row
    try:
        (x1, y1, x2, y2) = row['bbox']
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError('Invalid input or execution contract')
        path = Path(row['target_path'])
        (ok, encoded) = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ValueError('Invalid input or execution contract')
        output_hash = hashlib.sha256(encoded.tobytes()).hexdigest()
        if path.exists():
            if sha256(path) != output_hash:
                raise FileExistsError(f'미등록 출력의 hash 불일치: {path}')
        else:
            temp = path.with_suffix('.jpg.partial')
            encoded.tofile(str(temp))
            temp.replace(path)
        row.update(output_sha256=output_hash, crop_width=x2 - x1, crop_height=y2 - y1)
    except Exception as e:
        row.update(status='ERROR_WRITE', reason=f'{type(e).__name__}: {e}')
    return row

def open_state(cfg):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(LOG_ROOT / 'state.sqlite3'))
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS records (source_path TEXT PRIMARY KEY, date TEXT NOT NULL, camera TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)')
    value = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    old = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
    if old and old[0] != value:
        raise RuntimeError('Invalid input or execution contract')
    if not old:
        if any(OUTPUT_ROOT.rglob('*.jpg')):
            raise RuntimeError('Invalid input or execution contract')
        db.execute("INSERT INTO metadata VALUES ('contract',?)", (value,))
        db.commit()
    (LOG_ROOT / 'execution_contract.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return db

def database_records(db):
    return [json.loads(r[0]) for r in db.execute('SELECT data FROM records ORDER BY date,camera,source_path')]

def export_report(db, rows, started, final=False, verified=False):
    records = database_records(db)
    counts = Counter((r['status'] for r in records))
    dates = sorted({r['date'] for r in rows})
    stat = []
    for date in dates:
        for camera in ('CAM0', 'CAM1'):
            src = [r for r in rows if r['date'] == date and r['camera'] == camera]
            done = [r for r in records if r['date'] == date and r['camera'] == camera]
            c = Counter((r['status'] for r in done))
            saved = c['SAVED_TARGET_CANDIDATE']
            excluded = sum((v for (k, v) in c.items() if k.startswith('INVALID_') or k == 'EXCLUDED_EQUIPMENT'))
            valid = len(done) - excluded
            stat.append(dict(date=date, camera=camera, source=len(src), processed=len(done), saved=saved, excluded=excluded, not_saved=len(done) - saved - excluded, candidate_save_rate=round(saved / valid * 100, 2) if valid else None))
    totals = []
    for camera in ('CAM0', 'CAM1'):
        data = [r for r in stat if r['camera'] == camera]
        n = sum((r['processed'] for r in data))
        saved = sum((r['saved'] for r in data))
        excluded = sum((r['excluded'] for r in data))
        totals.append(dict(camera=camera, source=sum((r['source'] for r in data)), processed=n, saved=saved, excluded=excluded, not_saved=n - saved - excluded, candidate_save_rate=round(saved / (n - excluded) * 100, 2) if n > excluded else None))
    complete = final and len(records) == len(rows)
    result = dict(status=('COMPLETE_WITH_ROW_ERRORS' if any((k.startswith('ERROR_') for k in counts)) else 'COMPLETE') if complete else 'RUNNING', input_count=len(rows), processed_count=len(records), saved_count=counts['SAVED_TARGET_CANDIDATE'], status_counts=dict(counts), by_camera=totals, by_date_camera=stat, camera_corrected_input_count=sum((r['camera_corrected'] for r in records)), camera_corrected_saved_count=sum((r['camera_corrected'] and r['status'] == 'SAVED_TARGET_CANDIDATE' for r in records)), elapsed_seconds=round(time.monotonic() - started, 2), model_sha256=EXPECTED_MODEL_SHA256, integrity_validation='PASS' if verified else 'PENDING', production=True, training=False, ground_truth_detection_rate=None, survey_bed_missing_count=None, survey_matching_status='PENDING_OCR_BED_SLOT_AND_CROP_QC')
    (LOG_ROOT / 'completion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if final:
        fields = ['date', 'source_camera', 'camera', 'camera_corrected', 'source_filename', 'filename', 'source_path', 'target_path', 'status', 'reason', 'source_sha256', 'output_sha256', 'image_width', 'image_height', 'crop_width', 'crop_height', 'selected_variant', 'confidence', 'bbox', 'polygon', 'actual_bed', 'human_qc']
        for (name, predicate) in [('roi_manifest.csv', lambda r: True), ('failure_manifest.csv', lambda r: r['status'] != 'SAVED_TARGET_CANDIDATE'), ('camera_corrections.csv', lambda r: r['camera_corrected'])]:
            with (LOG_ROOT / name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in records:
                    if predicate(r):
                        writer.writerow({k: json.dumps(r[k], ensure_ascii=False) if k in ('bbox', 'polygon') and k in r else r.get(k, '') for k in fields})
    return result

def verify_all(db, rows):
    records = database_records(db)
    if len(records) != len(rows):
        raise ValueError('Invalid input or execution contract')
    expected = set()
    for r in tqdm(records, desc='Processing', unit='item'):
        if r.get('source_sha256') and sha256(r['source_path']) != r['source_sha256']:
            raise ValueError('Invalid input or execution contract')
        if r['status'] == 'SAVED_TARGET_CANDIDATE':
            p = Path(r['target_path'])
            expected.add(str(p).lower())
            if sha256(p) != r['output_sha256']:
                raise ValueError(f' hash : {p}')
            with Image.open(p) as im:
                if im.size != (r['crop_width'], r['crop_height']):
                    raise ValueError('Invalid input or execution contract')
                im.verify()
            m = NAME_RE.fullmatch(p.name)
            if m[4].upper() != r['camera'] or p.parent.parent.name != r['camera'] or p.parent.name != '' + r['date'][2:]:
                raise ValueError('Invalid input or execution contract')
    actual = {str(p).lower() for p in OUTPUT_ROOT.rglob('*.jpg')}
    if expected != actual:
        raise ValueError('Invalid input or execution contract')

def write_review_contacts(db, cfg):
    records = database_records(db)
    root = LOG_ROOT / '시각검토'
    root.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:\\Windows\\Fonts\\malgun.ttf', 16)
    for (date, camera) in sorted({(r['date'], r['camera']) for r in records}):
        group = [r for r in records if r['date'] == date and r['camera'] == camera]
        saved = [r for r in group if r['status'] == 'SAVED_TARGET_CANDIDATE']
        failures = [r for r in group if r['status'] in ('NO_FIXED_TARGET', 'REVIEW_OVERLAPPING_TARGETS')]
        selected = [saved[int((len(saved) - 1) * v)] for v in (0, 0.5, 1)] if saved else []
        selected += failures[:2]
        if not selected:
            selected = group[:2]
        board = Image.new('RGB', (1100, 40 + len(selected) * 405), 'white')
        draw = ImageDraw.Draw(board)
        draw.text((5, 5), f'{date} {camera} / 원본 / 단일 대상 후보 overlay', font=font, fill='black')
        for (i, r) in enumerate(selected):
            try:
                image = diagnostic['base']['read_image'](r['source_path'])
                overlay = image.copy()
                if r['status'] == 'SAVED_TARGET_CANDIDATE':
                    (x1, y1, x2, y2) = r['bbox']
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 5)
                for (j, arr) in enumerate((image, overlay)):
                    im = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                    im.thumbnail((545, 365))
                    board.paste(im, (j * 550, 40 + i * 405))
                draw.text((5, 40 + i * 405 + 369), r['source_filename'], font=font, fill='black')
                draw.text((555, 40 + i * 405 + 369), r['status'], font=font, fill='black')
            except Exception:
                draw.text((5, 40 + i * 405), '이미지 읽기 실패', font=font, fill='red')
        board.save(root / f'{date}_{camera}.jpg', quality=92)

def verify_pilot(model, cfg):
    references = json.loads((REFERENCE_ROOT / 'prediction_records.json').read_text(encoding='utf-8'))
    refs = [r for r in references if r['date'] <= END_DATE]
    items = []
    for r in refs:
        image = diagnostic['base']['read_image'](r['source_path'])
        camera = ('CAM1' if r['camera'] == 'CAM0' else 'CAM0') if r['date'] == '20260820' else r['camera']
        items.append((dict(camera=camera, filename=r['filename']), image))
    mismatches = []
    bbox_changes = []
    for offset in range(0, len(items), BATCH_SIZE):
        batch = items[offset:offset + BATCH_SIZE]
        groups = defaultdict(list)
        for (i, item) in enumerate(batch):
            groups[item[1].shape].append((i, item))
        for group in groups.values():
            results = infer_group(model, [x[1] for x in group], cfg['profiles'])
            for ((j, _), got) in zip(group, results):
                old = refs[offset + j]
                expected = old['status'] == 'TARGET_POSITION_CANDIDATE'
                if (got['status'] == 'SAVED_TARGET_CANDIDATE') != expected:
                    mismatches.append(old['filename'])
                elif expected and max((abs(a - b) for (a, b) in zip(got['bbox'], old['selected_bbox']))) > 4:
                    a = got['bbox']
                    b = old['selected_bbox']
                    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    bbox_changes.append(dict(filename=old['filename'], old=b, new=a, iou=inter / union, old_method=old['selected_variant'], new_method=got.get('selected_variant')))
    material_changes = [r for r in bbox_changes if r['iou'] < 0.99 or r['old_method'] != r['new_method']]
    result = dict(pilot_inputs=len(refs), status_mismatches=mismatches, bbox_changes_over4px=bbox_changes, bbox_comparison_rule='same selection status; changed box IoU>=0.99 and same variant', material_bbox_changes=material_changes)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if mismatches or material_changes:
        raise RuntimeError('Invalid input or execution contract')
    return result

def main(args):
    started = time.monotonic()
    rows = discover()
    cfg = contract(rows)
    if args.check:
        print(json.dumps(dict(source_count=len(rows), by_camera=dict(Counter((r['camera'] for r in rows))), date_count=len({r['date'] for r in rows}), corrected_count=sum((r['camera_corrected'] for r in rows)), contract=cfg), ensure_ascii=False, indent=2))
        return
    model = None if args.report_only else diagnostic['YOLO'](str(MODEL_PATH))
    if args.verify_pilot:
        verify_pilot(model, cfg)
        return
    db = open_state(cfg)
    for row in rows:
        Path(row['target_path']).parent.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        verify_result = verify_pilot(model, cfg)
        (LOG_ROOT / 'batch_pilot_validation.json').write_text(json.dumps(verify_result, ensure_ascii=False, indent=2), encoding='utf-8')
        done = {r['source_path']: r for r in database_records(db)}
        for r in done.values():
            if r['status'] == 'SAVED_TARGET_CANDIDATE' and sha256(r['target_path']) != r['output_sha256']:
                raise RuntimeError('Invalid input or execution contract')
        pending = [r for r in rows if r['source_path'] not in done]
        export_report(db, rows, started)
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool, tqdm(total=len(rows), initial=len(done), desc='Processing', unit='item', mininterval=3) as progress:
            for (date, camera) in sorted({(r['date'], r['camera']) for r in pending}):
                group = [r for r in pending if r['date'] == date and r['camera'] == camera]
                for offset in range(0, len(group), CHUNK_SIZE):
                    loaded = list(pool.map(read_source, group[offset:offset + CHUNK_SIZE]))
                    finished = []
                    shapes = defaultdict(list)
                    for item in loaded:
                        if item[1] is None:
                            finished.append(item[0])
                        else:
                            shapes[item[1].shape].append(item)
                    for homogeneous in shapes.values():
                        for a in range(0, len(homogeneous), BATCH_SIZE):
                            batch = homogeneous[a:a + BATCH_SIZE]
                            predictions = infer_safe(model, batch, cfg['profiles'])
                            finished += list(pool.map(lambda pair: write_crop(pair[0], pair[1][1]), zip(predictions, batch)))
                    db.executemany('INSERT INTO records VALUES (?,?,?,?,?)', [(r['source_path'], r['date'], r['camera'], r['status'], json.dumps(r, ensure_ascii=False)) for r in finished])
                    db.commit()
                    progress.update(len(finished))
                result = export_report(db, rows, started)
                summary = next((r for r in result['by_date_camera'] if r['date'] == date and r['camera'] == camera))
                print(json.dumps(dict(date=date, camera=camera, summary=summary, total_processed=result['processed_count']), ensure_ascii=False), flush=True)
    verify_all(db, rows)
    write_review_contacts(db, cfg)
    result = export_report(db, rows, started, final=True, verified=True)
    db.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify-pilot', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    main(parser.parse_args())
