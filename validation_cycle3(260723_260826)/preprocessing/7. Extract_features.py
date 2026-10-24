"""Calculate frozen image features and select camera observations."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
HERE = Path(__file__).resolve().parent
INPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '5. bbox to crop, feature'
OUTPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '6. 시각화와 QC'
SOURCE_DATABASE = INPUT_ROOT / '_whole_run' / 'state.sqlite3'
EXPECTED_ROWS = 243025
EXPECTED_KEYS = 198045
WORKERS = 8
CHUNK_ROWS = 64
SYSTEM_RAM_LIMIT_GIB = 50
import argparse, base64, csv, hashlib, json, math, os, re, sqlite3, subprocess
import sys, time, traceback, zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import psutil
from tqdm import tqdm
from feature_calculation import FEATURES, FeatureExtractorAndQC
import camera_selection
from workbook_stream import write_workbook, verify_workbook
VERSION = 'S4_FEATURE_RAW_V1_CAM01_FROZEN_4_5'
META = ['lettuce_key', 'record_id', 'season_id', 'camera', 'observation_view', 'source_camera', 'capture_date', 'capture_time', 'bed_id', 'physical_slot', 'slot_namespace_version', 'crop_path', 'crop_sha256', 'mask_sha256', 'source_path', 'source_sha256', 'crop_source_box', 'raw_final_bbox', 'mask_width', 'mask_height', 'mask_pixels', 'components', 'mm_per_px', 'scale_source', 'scale_version', 'scale_status', 'scale_verified', 'scale_qc', 'identity_verified', 'ocr_action', 'bbox_qc_flags', 'target_body_review_required', 'is_crop_clipped', 'source_boundary_contact', 'neighbor_anchors_in_rgb_crop', 'upstream_crop_status', 'feature_status', 'feature_qc', 'feature_flags', 'error', 'mask_policy', 'physical_units_estimated', 'selected_for_top', 'top_selection_reason', 'top_selection_quality', 'top_selection_hard_valid', 'camera_quality_flags']
ALL_COLUMNS = META + FEATURES
VIEW_META = ['record_id', 'camera', 'source_camera', 'crop_path', 'feature_status', 'feature_qc', 'feature_flags', 'mm_per_px', 'scale_source', 'scale_status', 'scale_verified', 'scale_qc', 'identity_verified', 'bbox_qc_flags', 'target_body_review_required', 'is_crop_clipped', 'source_boundary_contact']
MATCH_COLUMNS = ['lettuce_key', 'season_id', 'capture_date', 'capture_time', 'bed_id', 'physical_slot', 'pair_status', 'preferred_view', 'selected_top_camera', 'top_selection_reason', 'top_selection_hard_valid', 'top_selection_quality', 'cam0_record_id', 'cam1_record_id', 'available_views']
MATCH_COLUMNS += [k + '_' + view for view in ('top', 'front') for k in VIEW_META + FEATURES]
MATCH_COLUMNS += ['volume_proxy_top_front', 'raw_values_preserved', 'visualization_imputed']

def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')

def clean(v):
    if isinstance(v, dict):
        return {k: clean(w) for (k, w) in v.items()}
    if isinstance(v, (tuple, list)):
        return [clean(w) for w in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and (not math.isfinite(v)):
        return None
    return v

def js(v):
    return json.dumps(clean(v), ensure_ascii=False, separators=(',', ':'), allow_nan=False)

def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def yes(v):
    return str(v).lower() in ('true', '1', 'yes')

def read_source():
    wal = Path(str(SOURCE_DATABASE) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise RuntimeError('Source database has uncheckpointed WAL entries')
    return sqlite3.connect(SOURCE_DATABASE.as_uri() + '?mode=ro&immutable=1', uri=True)

def lettuce_key(r):
    if not re.fullmatch('\\d{8}', r['capture_date']) or not re.fullmatch('\\d{6}', r['capture_time']):
        raise ValueError('Invalid input or execution contract')
    if not re.fullmatch('bed\\d+', r['bed_id']) or not re.fullmatch('[tb]\\d+', r['physical_slot']):
        raise ValueError('Invalid input or execution contract')
    return f"{r['capture_date'][2:]}_{r['capture_time']}_{r['bed_id']}_{r['physical_slot']}"

def decode_mask(r):
    if r.get('mask_encoding') != 'packbits_zlib_base64_v1':
        raise ValueError('Invalid input or execution contract')
    (h, w) = (int(r['mask_height']), int(r['mask_width']))
    packed = zlib.decompress(base64.b64decode(r['mask_data'], validate=True))
    if len(packed) != (h * w + 7) // 8:
        raise ValueError('Invalid input or execution contract')
    binary = np.unpackbits(np.frombuffer(packed, np.uint8), count=h * w).reshape(h, w)
    if hashlib.sha256(binary.tobytes()).hexdigest() != r['mask_sha256']:
        raise ValueError('Invalid input or execution contract')
    return binary

def calculate(r):
    out = {k: r.get(k) for k in META}
    out.update(lettuce_key=lettuce_key(r), observation_view='front' if r['camera'] == 'cam2' else 'top', upstream_crop_status=r['status'], mask_policy='EXPLICIT_BINARY_MASK', feature_status='ERROR', selected_for_top=False)
    out.update({k: None for k in FEATURES})
    try:
        data = Path(moved_path(r['crop_path'])).read_bytes()
        if hashlib.sha256(data).hexdigest() != r['crop_sha256']:
            raise ValueError('Invalid input or execution contract')
        crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            raise ValueError('Invalid input or execution contract')
        mask = decode_mask(r)
        if mask.shape != crop.shape[:2]:
            raise ValueError('Invalid input or execution contract')
        if np.any(crop[mask == 0]):
            raise ValueError('Invalid input or execution contract')
        if hashlib.sha256(crop.tobytes()).hexdigest() != r['crop_pixels_sha256']:
            raise ValueError('Invalid input or execution contract')
        try:
            scale = float(r.get('mm_per_px'))
        except (TypeError, ValueError):
            scale = None
        source = str(r.get('scale_source') or '').upper()
        ready = scale is not None and math.isfinite(scale) and (scale > 0) and yes(r.get('scale_ready')) and (source not in ('', 'NONE', 'UNRESOLVED', 'ERROR'))
        scale = scale if ready else None
        out.update(mm_per_px=scale, scale_status=('VERIFIED' if yes(r.get('scale_verified')) and r.get('scale_version') else 'ESTIMATED') if ready else 'MISSING_SCALE', physical_units_estimated=ready and (not yes(r.get('scale_verified'))))
        (values, qc) = FeatureExtractorAndQC().extract_30_features(crop, mask * 255, scale or 1.0, r['camera'] == 'cam2')
        n = cv2.connectedComponents(mask, 8)[0] - 1
        out.update(mask_pixels=int(mask.sum()), components=n, feature_qc=qc['qc_status'], feature_flags=qc.get('qc_flags', qc.get('qc_details', '')))
        if qc['qc_status'] in ('QC_PASS', 'QC_WARNING'):
            if r['camera'] == 'cam2':
                values['front_height_cm'] = math.sqrt(values['front_height_cm2'])
            if any((not math.isfinite(float(v)) for v in values.values())):
                raise ValueError('Invalid input or execution contract')
            if scale is None:
                for k in ('area_cm2', 'front_height_cm', 'front_height_cm2', 'volume_proxy'):
                    values[k] = None
            out.update(values, feature_status='OK')
        else:
            out['feature_status'] = 'INVALID_MASK'
        measurement_record = dict(r, input_record_json={'slot_center_x': r['slot_center_x'], 'slot_center_y': r['slot_center_y']})
        metrics = camera_selection.measure(measurement_record, mask, r['crop_source_box'])
        if out['feature_status'] != 'OK':
            metrics['inference_status'] = 'FEATURE_UNAVAILABLE'
        candidate = {k: r.get(k) for k in ('record_id', 'season_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'source_path', 'raw_source_path', 'status', 'old_key', 'new_key')}
        candidate.update(metrics, candidate_available=out['feature_status'] == 'OK')
        return (clean(out), clean(candidate))
    except Exception as exc:
        out.update(feature_status='ERROR', error=repr(exc), feature_flags=traceback.format_exc())
        candidate = {k: r.get(k) for k in ('record_id', 'season_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'source_path', 'raw_source_path', 'status')}
        candidate.update(candidate_available=False, inference_status='FEATURE_ERROR')
        return (clean(out), clean(candidate))

def init_worker():
    cv2.setNumThreads(1)

def contract():
    completion = json.loads((INPUT_ROOT / 'CROP_CURRENT.json').read_text(encoding='utf-8-sig'))
    if completion['verification'] != 'PASS' or completion['masked_crops'] != EXPECTED_ROWS or completion['errors'] or completion['remaining']:
        raise ValueError('Invalid input or execution contract')
    files = {name: digest(HERE / name) for name in ('7. Extract_features.py', 'feature_calculation.py', 'camera_selection.py', 'cam01_selection_policy.py', 'workbook_stream.py', '261011_04_엑셀서식.mjs')}
    value = dict(version=VERSION, source_crop_signature=completion['signature'], source_database=str(SOURCE_DATABASE), source_database_bytes=SOURCE_DATABASE.stat().st_size, source_database_mtime_ns=SOURCE_DATABASE.stat().st_mtime_ns, source_feature_manifest=str(INPUT_ROOT / 'crops/feature_input_manifest.csv'), expected_rows=EXPECTED_ROWS, expected_keys=EXPECTED_KEYS, code_hashes=files, default_camera_boundary='CAM0 t/b1..4; CAM1 t/b5..12', policy='FROZEN_TRACK_QUALITY_AND_OBSERVATION_FALLBACK', raw_values_preserved=True, imputation_executed=False, feature_definitions='2작기 기본30종+정면3종 동결', camera_measurement_adapter='old measure() on current explicit crop-local mask + native bbox; slot_center preserved; absent confidence uses policy default zero')
    value['signature'] = hashlib.sha256(js(value).encode()).hexdigest()
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    bodies = [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('calculate', 'decode_mask', 'lettuce_key')]
    value['calculation_signature'] = hashlib.sha256(js(dict(input_signature=completion['signature'], source_bytes=value['source_database_bytes'], source_mtime=value['source_database_mtime_ns'], feature_class=files['feature_calculation.py'], measurement=files['camera_selection.py'], calculation_bodies=bodies, metadata=META, features=FEATURES)).encode()).hexdigest()
    return value

def source_records(conn):
    sql = 'SELECT s.payload FROM frames f JOIN slots s INDEXED BY slots_frame_order USING(frame_id) WHERE s.target_available=1 ORDER BY f.sequence,s.slot_order'
    cursor = conn.execute(sql)
    while (batch := cursor.fetchmany(CHUNK_ROWS)):
        yield [json.loads(zlib.decompress(row[0])) for row in batch]

def open_state(c):
    root = OUTPUT_ROOT / '_실행기록' / '261011_Feature'
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / 'feature_state.sqlite3', timeout=60)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('CREATE TABLE IF NOT EXISTS config(key TEXT PRIMARY KEY,value TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS features(id TEXT PRIMARY KEY,lettuce_key TEXT,camera TEXT,sequence INTEGER,payload TEXT,candidate TEXT)')
    existing = conn.execute("SELECT value FROM config WHERE key='calculation_signature'").fetchone()
    if existing and existing[0] != c['calculation_signature']:
        raise ValueError('Invalid input or execution contract')
    if not existing and conn.execute('SELECT count(*) FROM features').fetchone()[0]:
        raise ValueError('Invalid input or execution contract')
    conn.execute("INSERT OR REPLACE INTO config VALUES('calculation_signature',?)", (c['calculation_signature'],))
    conn.execute("INSERT OR REPLACE INTO config VALUES('signature',?)", (c['signature'],))
    conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS features_key_camera ON features(lettuce_key,camera)')
    conn.execute('CREATE TABLE IF NOT EXISTS matching(lettuce_key TEXT PRIMARY KEY,payload TEXT)')
    conn.commit()
    save_json(root / 'execution_contract.json', c)
    return (root, conn)

def extract(conn, source, root, workers, max_new=None):
    done = {r[0] for r in conn.execute("SELECT id FROM features WHERE json_extract(payload,'$.feature_status')!='ERROR'")}
    newly = 0
    sequence = 0
    last_log = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool, tqdm(total=EXPECTED_ROWS, initial=len(done), desc='243025 crop → Feature', unit='crop', mininterval=5) as progress:
        for batch in source_records(source):
            sequence_map = {r['record_id']: sequence + i for (i, r) in enumerate(batch)}
            sequence += len(batch)
            todo = [r for r in batch if r['record_id'] not in done]
            if not todo:
                continue
            while psutil.virtual_memory().used / 2 ** 30 > SYSTEM_RAM_LIMIT_GIB:
                time.sleep(2)
            for (out, candidate) in pool.map(calculate, todo, chunksize=2):
                conn.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?,?,?)', (out['record_id'], out['lettuce_key'], out['camera'], sequence_map[out['record_id']], js(out), js(candidate)))
                newly += 1
                progress.update(1)
            conn.commit()
            if time.monotonic() - last_log > 20:
                report = dict(status='RUNNING', phase='EXTRACT', completed=conn.execute('SELECT count(*) FROM features').fetchone()[0], newly_computed=newly, at=now(), system_ram_used_gib=round(psutil.virtual_memory().used / 2 ** 30, 2))
                save_json(root / 'progress.json', report)
                print(js(report), flush=True)
                last_log = time.monotonic()
            if max_new is not None and newly >= max_new:
                break
    return newly

def feature_rows(conn):
    for (blob,) in conn.execute('SELECT payload FROM features ORDER BY lettuce_key,camera'):
        yield json.loads(blob)

def selection(conn, root):
    candidates = (json.loads(r[0]) for r in conn.execute('SELECT candidate FROM features ORDER BY id'))
    camera_selection.build(conn, candidates)
    for (rid, payload) in tqdm(conn.execute('SELECT id,payload FROM features ORDER BY id'), total=EXPECTED_ROWS, desc='Processing', mininterval=5):
        r = json.loads(payload)
        s = camera_selection.decision(conn, r)
        if r['camera'] in ('cam0', 'cam1'):
            cr = conn.execute('SELECT payload FROM camera_candidates WHERE id=?', (rid,)).fetchone()
            quality = json.loads(cr[0]) if cr else {}
            r.update(selected_for_top=bool(s and s.get('selected_record_id') == rid), top_selection_reason=s.get('selection_reason') if s else 'NOT_SELECTED_ALTERNATE', top_selection_quality=quality.get('quality_score'), top_selection_hard_valid=quality.get('hard_valid'), camera_quality_flags=quality.get('quality_flags'))
        conn.execute('UPDATE features SET payload=? WHERE id=?', (js(r), rid))
    conn.commit()
    counts = Counter((r[0] for r in conn.execute("SELECT json_extract(payload,'$.selection_reason') FROM camera_selections")))
    save_json(root / 'camera_selection_summary.json', dict(reasons=dict(counts), boundary='4/5', policy_sha256=digest(HERE / 'cam01_selection_policy.py'), at=now()))

def make_matching(conn):
    conn.execute('DELETE FROM matching')
    status = Counter()
    batch = []

    def emit(key, records):
        by = {r['camera']: r for r in records}
        seed = records[0]
        top = next((r for r in records if r.get('selected_for_top')), None)
        front = by.get('cam2')
        has_top = any((k in by for k in ('cam0', 'cam1')))
        has_front = front is not None
        pair_status = 'BOTH' if has_top and has_front else 'TOP_ONLY' if has_top else 'FRONT_ONLY'
        pair = f"{seed['season_id']}|{seed['capture_date']}_{seed['capture_time']}_{seed['bed_id']}|{seed['physical_slot']}"
        decision = conn.execute('SELECT payload FROM camera_selections WHERE physical=?', (pair,)).fetchone()
        s = json.loads(decision[0]) if decision else {}
        preferred = 'top' if top and top['feature_status'] == 'OK' else 'front' if front and front['feature_status'] == 'OK' else 'UNAVAILABLE'
        out = {k: seed.get(k) for k in ('lettuce_key', 'season_id', 'capture_date', 'capture_time', 'bed_id', 'physical_slot')}
        out.update(pair_status=pair_status, preferred_view=preferred, selected_top_camera=top['camera'] if top else None, top_selection_reason=s.get('selection_reason'), top_selection_hard_valid=s.get('selected_hard_valid'), top_selection_quality=s.get('selected_quality_score'), cam0_record_id=by.get('cam0', {}).get('record_id'), cam1_record_id=by.get('cam1', {}).get('record_id'), available_views='top;front' if has_top and has_front else 'top' if has_top else 'front', raw_values_preserved=True, visualization_imputed=False)
        for (view, r) in (('top', top), ('front', front)):
            for col in VIEW_META + FEATURES:
                out[col + '_' + view] = r.get(col) if r else None
        a = out.get('area_cm2_top')
        h = out.get('front_height_cm_front')
        out['volume_proxy_top_front'] = a * h if a is not None and h is not None else None
        status[pair_status] += 1
        return (key, js(out))
    key = None
    records = []
    for r in tqdm(feature_rows(conn), total=EXPECTED_ROWS, desc='Processing', mininterval=5):
        if key is not None and r['lettuce_key'] != key:
            batch.append(emit(key, records))
            records = []
            if len(batch) >= 500:
                conn.executemany('INSERT INTO matching VALUES(?,?)', batch)
                conn.commit()
                batch = []
        key = r['lettuce_key']
        records.append(r)
    if records:
        batch.append(emit(key, records))
    conn.executemany('INSERT INTO matching VALUES(?,?)', batch)
    conn.commit()
    return dict(status)

def matching_rows(conn):
    for (blob,) in conn.execute('SELECT payload FROM matching ORDER BY lettuce_key'):
        yield json.loads(blob)

def small_validation(source, root):
    records = []
    for camera in ('cam0', 'cam1', 'cam2'):
        folders = sorted((p for p in (INPUT_ROOT / 'crops' / camera).iterdir() if p.is_dir()))
        for folder in (folders[0], folders[-1]):
            for (_, path) in zip(range(3), folder.glob('*.png')):
                blob = source.execute('SELECT payload FROM slots WHERE target_available=1 AND crop_path=?', (str(path),)).fetchone()[0]
                records.append(json.loads(zlib.decompress(blob)))
    checks = []
    for r in records:
        (out, candidate) = calculate(r)
        if out['feature_status'] == 'ERROR':
            raise ValueError(out['error'])
        checks.append(dict(record_id=r['record_id'], camera=r['camera'], date=r['capture_date'], feature_status=out['feature_status'], feature_qc=out['feature_qc']))
    import ast
    old = CYCLE.MAIN_ROOT / 'preprocessing/feature_calculation.py'

    def class_ast(path):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        class Normalize(ast.NodeTransformer):
            def visit_Expr(self,n):
                return None if isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) else self.generic_visit(n)
            def visit_Assert(self,n):
                self.generic_visit(n);n.msg=None;return n
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureExtractorAndQC')
        return ast.dump(Normalize().visit(cls),include_attributes=False)
    assert class_ast(old) == class_ast(HERE / 'feature_calculation.py'), "Input contract failed: class_ast(old) == class_ast(HERE / 'feature_calculation.py')"
    save_json(root / 'pilot_validation.json', dict(status='PASS', actual_crops=len(checks), checks=checks, feature_class_ast_parity=True, full_run=False))

def export(conn, root):
    for (name, columns, row_iterator, count) in [('Feature_all', ALL_COLUMNS, feature_rows, EXPECTED_ROWS), ('Feature_matching', MATCH_COLUMNS, matching_rows, EXPECTED_KEYS)]:
        spec = dict(name=name, columns=columns, rows=count, sample=[r for (_, r) in zip(range(12), row_iterator(conn))], source='4작기 완료 masked crop 243,025개; 원값 보존', raw_values_preserved=True)
        save_json(root / (name + '_schema.json'), spec)
    summaries = {}
    for (name, columns, row_iterator, count) in [('Feature_all', ALL_COLUMNS, feature_rows, EXPECTED_ROWS), ('Feature_matching', MATCH_COLUMNS, matching_rows, EXPECTED_KEYS)]:
        save_json(root / 'progress.json', dict(status='RUNNING', phase='EXPORT_' + name, at=now()))
        target = OUTPUT_ROOT / (name + '.xlsx')
        write_workbook(root / (name + '_template.xlsx'), target, columns, row_iterator(conn), count)
        save_json(root / 'progress.json', dict(status='RUNNING', phase='VERIFY_' + name, at=now()))
        summaries[name] = verify_workbook(target, columns, row_iterator(conn), count)
        summaries[name].update(path=str(target), sha256=digest(target), bytes=target.stat().st_size)
        save_json(root / 'workbook_verification.json', summaries)
    return summaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--max-new', type=int)
    parser.add_argument('--skip-export', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='Execution option')
    args = parser.parse_args()
    c = contract()
    (root, conn) = open_state(c)
    import msvcrt
    lock = (root / '.run.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    started = now()
    try:
        source = read_source()
        assert source.execute('SELECT count(*) FROM slots WHERE target_available=1').fetchone()[0] == EXPECTED_ROWS
        small_validation(source, root)
        if args.pilot:
            print(js(dict(status='PILOT_PASS', full_run=False, root=str(root))), flush=True)
            return
        if (OUTPUT_ROOT / 'completion.json').is_file():
            completed = json.loads((OUTPUT_ROOT / 'completion.json').read_text())
            if completed.get('signature') == c['signature'] and all((Path(v['path']).is_file() and digest(v['path']) == v['sha256'] for v in completed.get('workbooks', {}).values())):
                print(js(dict(status='ALREADY_COMPLETE', completion=completed)), flush=True)
                return
        if not args.export_only:
            extract(conn, source, root, args.workers, args.max_new)
        counts = dict(conn.execute("SELECT json_extract(payload,'$.feature_status'),count(*) FROM features GROUP BY json_extract(payload,'$.feature_status')"))
        count = conn.execute('SELECT count(*) FROM features').fetchone()[0]
        save_json(root / 'extraction_summary.json', dict(rows=count, status_counts=counts, at=now()))
        if args.max_new is not None or count != EXPECTED_ROWS:
            print(js(dict(status='PARTIAL', rows=count, counts=counts)), flush=True)
            return
        if counts.get('ERROR', 0):
            raise RuntimeError(f"Feature   {counts['ERROR']}:   ")
        save_json(root / 'progress.json', dict(status='RUNNING', phase='CAMERA_SELECTION', at=now()))
        if args.export_only:
            pair_counts = dict(conn.execute("SELECT json_extract(payload,'$.pair_status'),count(*) FROM matching GROUP BY json_extract(payload,'$.pair_status')"))
        else:
            selection(conn, root)
            pair_counts = make_matching(conn)
        assert conn.execute('SELECT count(*) FROM matching').fetchone()[0] == EXPECTED_KEYS
        assert pair_counts == dict(BOTH=22350, TOP_ONLY=97533, FRONT_ONLY=78162), pair_counts
        source_ids = {r[0] for r in source.execute('SELECT record_id FROM slots WHERE target_available=1')}
        assert source_ids == {r[0] for r in conn.execute('SELECT id FROM features')}
        del source_ids
        assert SOURCE_DATABASE.stat().st_mtime_ns == c['source_database_mtime_ns']
        if args.skip_export:
            print(js(dict(status='CALCULATION_COMPLETE_EXPORT_PENDING', counts=counts, pairs=pair_counts)), flush=True)
            return
        books = export(conn, root)
        report = dict(status='COMPLETE_WITH_REVIEW', verification='PASS', version=VERSION, signature=c['signature'], started_at=started, completed_at=now(), input_crops=EXPECTED_ROWS, feature_rows=count, feature_status_counts=counts, matching_rows=EXPECTED_KEYS, pair_counts=pair_counts, errors=counts.get('ERROR', 0), remaining=0, workbooks=books, raw_values_preserved=True, visualization_imputed=False, source_crop_modified=False, growth_survey_matching_executed=False, model_inference_executed=False, full_visual_qc_executed=False)
        save_json(OUTPUT_ROOT / 'completion.json', report)
        save_json(root / 'progress.json', report)
        print(js(report), flush=True)
    except Exception as exc:
        save_json(root / 'failure.json', dict(at=now(), error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        conn.close()
        lock.close()
if __name__ == '__main__':
    main()
