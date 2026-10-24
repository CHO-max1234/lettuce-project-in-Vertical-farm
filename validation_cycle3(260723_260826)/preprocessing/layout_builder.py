"""Layout builder calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import io
import math
import shutil
from collections import deque,defaultdict
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT / '4. 자리배치와 bbox'
BASE_ROOT = CYCLE.LAYOUT_BASE_DB.parent
BASE_DB = CYCLE.LAYOUT_BASE_DB
OUTPUT_ROOT = DATA_ROOT / '자리배치_cam2_최종'
STATE_ROOT = OUTPUT_ROOT / '_실행캐시'
WORKERS = 10
import argparse, copy, csv, json, sqlite3, zlib, hashlib, time, os, importlib.util
from collections import Counter, deque, defaultdict
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def module(name, path):
    spec = resolved_spec(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
G = module('s4_g3', CODE_ROOT / 'front_geometry.py')
V = module('s4_v3_contract', CODE_ROOT / 'layout_geometry.py')
V.VERSION = G.VERSION

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def unpack(blob):
    return json.loads(zlib.decompress(blob))

def pack(d):
    return zlib.compress(V.A.encode(d).encode('utf-8'), 3)

def connect(p, readonly=True):
    return sqlite3.connect(Path(p).as_uri() + '?mode=ro' + ('&immutable=1' if Path(p) in (BASE_DB, OUTPUT_ROOT / 'state.sqlite3') else ''), uri=True) if readonly else sqlite3.connect(p)

def update_payload(d, image):
    f = d['frame']
    oldslots = d['slot']
    sample = dict(f, input_path=f['source_path'])
    (proposed, meta) = G.propose_geometry(image, 'cam2', sample, oldslots, [])
    if meta.get('preserved_ver2_geometry'):
        new = copy.deepcopy(d)
        new['frame']['geometry_metadata'] = meta
        new['frame']['geometry_version'] = G.VERSION
        return new
    rows = V.preserve_contract(proposed, oldslots, sample)
    checks = V.structural_checks(rows, sample)
    states = Counter()
    for r in rows:
        usable = V.usable(r)
        status = 'REVIEW' if usable else 'HOLD'
        states[status] += 1
        r.update(geometry_qc_status=status, bbox_auto_eligible=False, bbox_candidate_eligible=usable, bbox_review_required=True, bbox_stage='NOT_EXECUTED', bbox_input_record_id=r['record_id'], bbox_input_frame_id=f['frame_id'], bbox_raw_segments_source='state.sqlite3:frames.payload.segment', qc_status=status)
        if usable:
            r['bbox_handoff_status'] = 'REVIEW_GEOMETRY_READY_BODY_SELECTION_PENDING'
    state = 'REVIEW' if states['REVIEW'] else 'HOLD'
    new = copy.deepcopy(d)
    new['slot'] = rows
    new['frame'].update(geometry_version=G.VERSION, geometry_status=state, qc_status=state, geometry_metadata=meta, structural_issues=checks, slot_count=len(rows), geometry_supported_slots=0, geometry_review_slots=states['REVIEW'], geometry_hold_slots=states['HOLD'], **V.metrics(rows), bbox_target_selection_executed=False, geometry_update_only=True, raw_seg_inference_status='EXACT_COMPLETED_VER2_SEG_REUSED')
    return V.A.clean(new)

def worker(task):
    (fid, blob, overlay) = task
    d = unpack(blob)
    f = d['frame']
    content = Path(f['source_path']).read_bytes()
    assert hashlib.sha256(content).hexdigest() == f['source_sha256']
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (f['source_height'], f['source_width'])
    new = update_payload(d, image)
    assert new['segment'] == d['segment']
    assert not new.get('error')
    if overlay:
        global MAPPER
        if 'MAPPER' not in globals():
            MAPPER = V.A.load_reference()[0].MAPPER
        raw = [dict(polygon=np.array(r['polygon'], np.float32)) for r in d['segment']]
        canvas = V.draw_overlay(image, new['slot'], raw, f, new['frame']['geometry_metadata'], MAPPER)
        head = max(68, round(83 * max(0.6, min(1.1, f['source_width'] / 2200))))
        canvas[:head] = 0
        title = f"CAM2 S4 ver3 | {new['frame']['geometry_status']} | slots={len(new['slot'])} | SEG REUSED / BODY NOT SELECTED"
        cv2.putText(canvas, title, (8, round(head * 0.38)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(canvas, f['image_name'] + ' | ' + new['frame']['geometry_metadata'].get('ver3_action', ''), (8, round(head * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        p = Path(overlay)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(p)
        new['frame']['overlay_path'] = str(p)
        for r in new['slot']:
            r['overlay_path'] = str(p)
    return (fid, pack(new), dict(frame_id=fid, image_name=f['image_name'], before=f['geometry_status'], after=new['frame']['geometry_status'], before_usable=sum((V.usable(r) for r in d['slot'])), after_usable=sum((V.usable(r) for r in new['slot'])), slots=len(new['slot']), action=new['frame']['geometry_metadata'].get('ver3_action'), reasons=new['frame']['geometry_metadata'].get('failure_reasons'), overlay=overlay))
from types import SimpleNamespace
R = SimpleNamespace(G=G, V=V, BASE_DB=BASE_DB, STATE_ROOT=STATE_ROOT, OUTPUT_ROOT=OUTPUT_ROOT, BASE_ROOT=BASE_ROOT, DATA_ROOT=DATA_ROOT, sha=sha, unpack=unpack, pack=pack, connect=connect, worker=worker, __file__=__file__)
EXPORT = module('s4_cam2_full_export', CODE_ROOT / 'layout_manifest.py')
import sys
sys.modules[EXPORT.__name__] = EXPORT
BASE_DB = CYCLE.LAYOUT_BASE_DB
STATE_ROOT = R.STATE_ROOT
DATABASE = STATE_ROOT / 'state.sqlite3'
OUTPUT_ROOT = R.OUTPUT_ROOT
WORKERS = 10
COMMIT_FRAMES = 64
VALIDATE_CHUNK = 96
GATE_FIELDS = ['record_id', 'frame_id', 'camera', 'physical_slot', 'frame_geometry_status', 'slot_geometry_qc_status', 'slot_geometry_usable', 'bbox_candidate_eligible', 'bbox_review_required', 'bbox_body_selected', 'exclude_reason']

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def candidate(frame, slot):
    status = slot.get('geometry_qc_status', slot.get('geometry_status', 'HOLD'))
    yes = frame['geometry_status'] in ('PASS', 'REVIEW') and status in ('PASS', 'REVIEW') and (slot.get('slot_geometry_usable') is True)
    return yes

def pilot_gate():
    p = OUTPUT_ROOT / 'sample30'
    records = json.loads((p / 'payloads.json').read_text(encoding='utf-8'))
    results = json.loads((p / 'results.json').read_text(encoding='utf-8'))
    assert len(records) == len(results) == 30
    before = R.connect(BASE_DB)
    preserved = 0
    recovered = 0
    bottom_only = 0
    old_holds = 0
    for d in records:
        f = d['frame']
        old = R.unpack(before.execute('SELECT payload FROM frames WHERE frame_id=?', (f['frame_id'],)).fetchone()[0])
        assert d['segment'] == old['segment']
        if old['frame']['geometry_status'] in ('PASS', 'REVIEW'):

            def geometry(s):
                return [{k: v for (k, v) in r.items() if k != 'overlay_path'} for r in s]
            assert geometry(d['slot']) == geometry(old['slot'])
            preserved += 1
        else:
            old_holds += 1
            if f['geometry_status'] == 'REVIEW':
                recovered += 1
                if sum((candidate(f, s) for s in d['slot'])) == 12:
                    bottom_only += 1
        assert not d['error']
        for s in d['slot']:
            if s['geometry_qc_status'] == 'HOLD':
                assert not candidate(f, s)
    before.close()
    assert preserved == 19 and old_holds == 11 and (recovered >= 7) and (bottom_only == 2)
    contract = dict(status='PASS', sample_frames=30, preserved_pass_review_frames=preserved, old_hold_frames=old_holds, recovered_to_review_frames=recovered, bottom_only_recovery_frames=bottom_only, remaining_hold_frames=old_holds - recovered, geometry_code_sha256=R.sha(R.G.__file__), calibration_sha256=R.sha(R.G.CALIBRATION_PATH), pilot_payload_sha256=R.sha(p / 'payloads.json'), scope='CONTRACT_CHECK_PLUS_AGENT_VISUAL_REVIEW_NOT_NEW_HUMAN_APPROVAL', visual_review='All 30 inspected; recovered 17,20,23,25,30 two rows; 24,26 bottom only. Insufficient 12,27,28,29 stay HOLD.')
    save_json(p / 'pilot_gate.json', contract)
    return contract

def run_worker(task):
    return R.worker(task)

def prepare(pilot):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dependencies = {str(p): R.sha(p) for p in [Path(__file__), Path(R.__file__), Path(R.G.__file__), R.G.CALIBRATION_PATH, Path(R.V.__file__), Path(EXPORT.__file__), CODE_ROOT / 'front_geometry_base.py']}
    fingerprint = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    if not DATABASE.exists():
        print('Copying frozen baseline database by SQLite backup', flush=True)
        src = R.connect(BASE_DB)
        dst = sqlite3.connect(DATABASE)
        src.backup(dst, pages=8192)
        src.close()
        dst.close()
    db = sqlite3.connect(DATABASE)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE IF NOT EXISTS cam2_patch(frame_id TEXT PRIMARY KEY,result TEXT NOT NULL)')
    old = db.execute("SELECT value FROM metadata WHERE key='cam2_v3_fingerprint'").fetchone()
    if old and old[0] != fingerprint:
        raise RuntimeError('CAM2_VER3_RESUME_DEPENDENCY_MISMATCH')
    db.execute("INSERT OR IGNORE INTO metadata VALUES('cam2_v3_fingerprint',?)", (fingerprint,))
    db.commit()
    contract = dict(version=R.G.VERSION, baseline_database=str(BASE_DB), working_state_database=str(DATABASE), final_state_database=str(OUTPUT_ROOT / 'state.sqlite3'), changed_camera='cam2', frozen_cameras=['cam0', 'cam1'], input_frames=39401, cam2_frames=14540, dependencies=dependencies, fingerprint=fingerprint, pilot_gate=pilot, segmentation_policy='EXACT_PREVIOUS_COMPLETED_SEGMENTS_NO_NEW_INFERENCE', bbox_candidate_policy='FRAME_AND_SLOT_PASS_OR_REVIEW_AND_GEOMETRY_USABLE; HOLD_EXCLUDED; BODY_SELECTION_PENDING')
    save_json(OUTPUT_ROOT / 'execution_contract.json', contract)
    return (db, contract)

def process_cam2(db):
    sample = json.loads((R.BASE_ROOT / '261009_whole_samples_검토/_support/review_inputs.json').read_text(encoding='utf-8'))['rows']
    overlays = {s['frame_id']: str(OUTPUT_ROOT / 'sample30/cam2' / s['image_name'].replace('.png', '_overlay.jpg')) for s in sample if s['camera'] == 'cam2'}
    source = R.connect(BASE_DB)
    done = {r[0] for r in db.execute('SELECT frame_id FROM cam2_patch')}
    total = source.execute("SELECT count(*) FROM frames WHERE camera='cam2'").fetchone()[0]
    assert total == 14540
    rows = iter(source.execute("SELECT frame_id,payload FROM frames WHERE camera='cam2' ORDER BY sequence"))
    pending = deque()
    updates = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool, tqdm(total=total, initial=len(done), desc='CAM2 only ver3', unit='frame') as bar:

        def fill():
            while len(pending) < WORKERS * 2:
                try:
                    (fid, blob) = next(rows)
                except StopIteration:
                    break
                if fid in done:
                    continue
                pending.append(pool.submit(run_worker, (fid, blob, overlays.get(fid))))
        fill()
        while pending:
            (fid, blob, result) = pending.popleft().result()
            d = R.unpack(blob)
            f = d['frame']
            db.execute('UPDATE frames SET geometry_status=?,slot_count=?,supported=?,segment_count=?,payload=? WHERE frame_id=?', (f['geometry_status'], len(d['slot']), f['geometry_supported_slots'], len(d['segment']), blob, fid))
            db.execute('DELETE FROM slot_index WHERE frame_id=?', (fid,))
            db.executemany('INSERT INTO slot_index VALUES(?,?,?,?,?)', [(s['record_id'], fid, s['physical_slot'], s['geometry_qc_status'], int(s['slot_geometry_usable'])) for s in d['slot']])
            db.execute('INSERT INTO cam2_patch VALUES(?,?)', (fid, json.dumps(result, ensure_ascii=False)))
            updates += 1
            if updates % COMMIT_FRAMES == 0:
                db.commit()
            bar.update(1)
            fill()
    db.commit()
    source.close()
    assert db.execute('SELECT count(*) FROM cam2_patch').fetchone()[0] == total

def validate_chunk(start, stop):
    a = R.connect(BASE_DB)
    b = R.connect(DATABASE)
    errors = []
    counts = Counter()
    frames = defaultdict(Counter)
    slots = defaultdict(Counter)
    groups = defaultdict(Counter)
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, GATE_FIELDS)
    old = list(a.execute('SELECT frame_id,payload FROM frames WHERE sequence>=? AND sequence<? ORDER BY sequence', (start, stop)))
    new = list(b.execute('SELECT frame_id,payload FROM frames WHERE sequence>=? AND sequence<? ORDER BY sequence', (start, stop)))
    assert len(old) == len(new)
    immutable = ['frame_id', 'camera', 'source_path', 'source_sha256', 'image_name', 'source_width', 'source_height', 'native_width', 'native_height', 'source_camera', 'crop_bbox', 'identity_verified', 'scale_verified', 'ocr_status', 'bed_id']
    for ((fid, oblob), (nfid, nblob)) in zip(old, new):
        try:
            assert fid == nfid
            d = R.unpack(nblob)
            f = d['frame']
            cam = f['camera']
            counts['frames'] += 1
            frames[cam][f['geometry_status']] += 1
            counts['slots'] += len(d['slot'])
            counts['segments'] += len(d['segment'])
            assert not d['error']
            if cam in ('cam0', 'cam1'):
                assert oblob == nblob
                counts['cam01_exact_payloads'] += 1
            else:
                od = R.unpack(oblob)
                of = od['frame']
                assert d['segment'] == od['segment']
                counts['cam2_exact_segments'] += 1
                for key in immutable:
                    assert f.get(key) == of.get(key), (key, f.get(key), of.get(key))
                old_by = {s['physical_slot']: s for s in od['slot']}
                new_by = {s['physical_slot']: s for s in d['slot']}
                assert set(old_by) <= set(new_by)
                for (name, s) in old_by.items():
                    assert new_by[name]['record_id'] == s['record_id']
                if of['geometry_status'] in ('PASS', 'REVIEW'):
                    assert len(od['slot']) == len(d['slot'])
                    for (s, ns) in zip(od['slot'], d['slot']):
                        assert {k: v for (k, v) in s.items() if k != 'overlay_path'} == {k: v for (k, v) in ns.items() if k != 'overlay_path'}
                    counts['cam2_preserved_accepted_geometry'] += 1
                elif f['geometry_status'] == 'REVIEW':
                    counts['cam2_recovered_frames'] += 1
                    if sum((candidate(f, s) for s in d['slot'])) < 24:
                        counts['cam2_partial_row_recovered_frames'] += 1
                group = R.G.group_key(f)
                groups[group]['frames'] += 1
                groups[group][f['geometry_status']] += 1
            assert len({s['physical_slot'] for s in d['slot']}) == len(d['slot'])
            for s in d['slot']:
                status = s['geometry_qc_status']
                slots[cam][status] += 1
                eligible = candidate(f, s)
                if eligible:
                    (x, y) = (s['anchor_x'], s['anchor_y'])
                    assert math.isfinite(x) and math.isfinite(y) and (0 <= x < f['source_width']) and (0 <= y < f['source_height'])
                    box = s['cell_bbox']
                    assert 0 <= box[0] < box[2] <= f['source_width'] and 0 <= box[1] < box[3] <= f['source_height']
                    counts[cam + '_bbox_candidates'] += 1
                if status == 'HOLD' or f['geometry_status'] == 'HOLD':
                    assert not eligible
                reason = '' if eligible else 'FRAME_HOLD' if f['geometry_status'] == 'HOLD' else 'SLOT_HOLD_OR_GEOMETRY_UNUSABLE'
                writer.writerow(dict(record_id=s['record_id'], frame_id=fid, camera=cam, physical_slot=s['physical_slot'], frame_geometry_status=f['geometry_status'], slot_geometry_qc_status=status, slot_geometry_usable=s['slot_geometry_usable'], bbox_candidate_eligible=eligible, bbox_review_required=eligible and (status == 'REVIEW' or f['geometry_status'] == 'REVIEW'), bbox_body_selected=False, exclude_reason=reason))
        except Exception as e:
            errors.append(dict(frame_id=fid, error=repr(e)))
    a.close()
    b.close()
    return dict(counts=dict(counts), frame_geometry={k: dict(v) for (k, v) in frames.items()}, slot_geometry={k: dict(v) for (k, v) in slots.items()}, source_groups={k: dict(v) for (k, v) in groups.items()}, errors=errors, gate_csv=buf.getvalue())

def verify(db):
    assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    total = db.execute('SELECT count(*) FROM frames').fetchone()[0]
    assert total == 39401
    pending = deque()
    iterator = iter(range(0, total, VALIDATE_CHUNK))
    counts = Counter()
    frames = defaultdict(Counter)
    slots = defaultdict(Counter)
    groups = defaultdict(Counter)
    errors = []
    gate_path = OUTPUT_ROOT / 'bbox_candidate_manifest.csv'
    partial = gate_path.with_suffix('.csv.partial')
    with partial.open('w', encoding='utf-8-sig', newline='') as gate, ProcessPoolExecutor(max_workers=WORKERS) as pool, tqdm(total=total, desc='Merged lineage and geometry contract', unit='frame') as bar:
        csv.DictWriter(gate, GATE_FIELDS).writeheader()

        def fill():
            while len(pending) < WORKERS:
                try:
                    start = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(validate_chunk, start, min(total, start + VALIDATE_CHUNK)))
        fill()
        while pending:
            r = pending.popleft().result()
            counts.update(r['counts'])
            errors.extend(r['errors'])
            gate.write(r['gate_csv'])
            bar.update(r['counts']['frames'])
            for (k, v) in r['frame_geometry'].items():
                frames[k].update(v)
            for (k, v) in r['slot_geometry'].items():
                slots[k].update(v)
            for (k, v) in r['source_groups'].items():
                groups[k].update(v)
            fill()
    assert counts['cam01_exact_payloads'] == 24861 and counts['cam2_exact_segments'] == 14540 and (counts['cam2_preserved_accepted_geometry'] == 8236)
    assert counts['slots'] == db.execute('SELECT count(*) FROM slot_index').fetchone()[0]
    assert counts['segments'] == 361684
    report = dict(status='FAIL' if errors else 'PASS', scope='FULL_LINEAGE_CAM01_EXACT_PAYLOAD_CAM2_SEG_IMMUTABILITY_GEOMETRY_GATE_NOT_HUMAN_ACCURACY', counts=dict(counts), frame_geometry={k: dict(v) for (k, v) in frames.items()}, slot_geometry={k: dict(v) for (k, v) in slots.items()}, source_groups={k: dict(v) for (k, v) in groups.items()}, failure_count=len(errors), failures=errors[:100], geometry_accuracy_verified=False, bbox_body_selection_executed=False)
    save_json(OUTPUT_ROOT / 'final_verification.json', report)
    if errors:
        raise RuntimeError('MERGED_CONTRACT_FAILED: ' + repr(errors[:4]))
    os.replace(partial, gate_path)
    return report

def audit_overlays(db):
    buckets = defaultdict(list)
    for (fid, result) in db.execute('SELECT frame_id,result FROM cam2_patch ORDER BY frame_id'):
        r = json.loads(result)
        if r['before'] == 'HOLD' and r['after'] == 'REVIEW':
            d = R.unpack(db.execute('SELECT payload FROM frames WHERE frame_id=?', (fid,)).fetchone()[0])
            f = d['frame']
            key = (R.G.group_key(f), r['after_usable'] < 24)
            buckets[key].append((f['image_name'], fid))
    selected = []
    for (key, values) in sorted(buckets.items()):
        values.sort()
        n = min(6, len(values))
        indices = sorted(set((round(i * (len(values) - 1) / max(1, n - 1)) for i in range(n))))
        selected.extend(((key, values[i][1]) for i in indices))
    tasks = []
    for (key, fid) in selected:
        blob = R.connect(BASE_DB)
        try:
            payload = blob.execute('SELECT payload FROM frames WHERE frame_id=?', (fid,)).fetchone()[0]
        finally:
            blob.close()
        d = R.unpack(payload)
        tasks.append((fid, payload, str(OUTPUT_ROOT / 'full_audit/cam2' / d['frame']['image_name'].replace('.png', '_overlay.jpg'))))
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for (fid, blob, r) in pool.map(run_worker, tasks, chunksize=1):
            stored = R.unpack(db.execute('SELECT payload FROM frames WHERE frame_id=?', (fid,)).fetchone()[0])
            drawn = R.unpack(blob)
            assert [(s['anchor_x'], s['anchor_y'], s['geometry_qc_status']) for s in drawn['slot']] == [(s['anchor_x'], s['anchor_y'], s['geometry_qc_status']) for s in stored['slot']]
            results.append(r)
    save_json(OUTPUT_ROOT / 'full_audit/results.json', results)
    import cv2, numpy as np
    for k in range(0, len(results), 4):
        tiles = []
        for (i, r) in enumerate(results[k:k + 4]):
            im = cv2.imdecode(np.fromfile(r['overlay'], np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (1800, round(im.shape[0] * 1800 / im.shape[1])))
            tile = cv2.copyMakeBorder(im, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(240, 240, 240))
            cv2.putText(tile, f"audit {k + i + 1}: {r['before']}->{r['after']} usable={r['after_usable']}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            tiles.append(tile)
        cv2.imencode('.jpg', np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 94])[1].tofile(OUTPUT_ROOT / f'full_audit/contact_{k // 4 + 1:02d}.jpg')
    return len(results)

def publish(db, contract, verified, exported, audit_count, started):
    changed = [p for (p, h) in contract['dependencies'].items() if R.sha(p) != h]
    assert not changed
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    dest = OUTPUT_ROOT / 'state.sqlite3'
    temp = dest.with_suffix('.sqlite3.partial')
    shutil.copy2(DATABASE, temp)
    assert R.sha(temp) == R.sha(DATABASE)
    os.replace(temp, dest)
    completion = dict(status='COMPLETE_WITH_REVIEW', version=R.G.VERSION, processed=39401, cam2_processed=14540, remaining=0, processing_errors=0, changed_camera='cam2', frozen_cameras=['cam0', 'cam1'], final_verification='PASS', frame_geometry=verified['frame_geometry'], slot_geometry=verified['slot_geometry'], counts=verified['counts'], source_groups=verified['source_groups'], working_state_database=str(DATABASE), final_state_database=str(dest), source_hash_verified_cam2_frames=14540, cam01_source_verification='PRIOR_COMPLETE_VER2_REUSED_WITH_EXACT_PAYLOAD', segment_rows=verified['counts']['segments'], slot_rows=verified['counts']['slots'], manifest_export=exported, sample_overlay_count=30, additional_audit_overlay_count=audit_count, bbox_target_selection_executed=False, geometry_accuracy_verified=False, run_seconds=round(time.monotonic() - started, 1), dependency_changes=changed)
    save_json(OUTPUT_ROOT / 'completion.json', completion)
    pointer = dict(status='COMPLETE_WITH_REVIEW', result_root=str(OUTPUT_ROOT), state_database=str(dest), working_state_database=str(DATABASE), frame_manifest=str(OUTPUT_ROOT / 'frame_manifest.csv'), slot_manifest=str(OUTPUT_ROOT / 'slot_manifest.csv'), bbox_candidate_manifest=str(OUTPUT_ROOT / 'bbox_candidate_manifest.csv'), completion=str(OUTPUT_ROOT / 'completion.json'), version=R.G.VERSION, prior_result_root=str(R.BASE_ROOT), bbox_policy='PASS_REVIEW_USABLE; HOLD_EXCLUDED; BODY_SELECTION_PENDING')
    save_json(R.DATA_ROOT / 'LAYOUT_CURRENT.json', pointer)
    save_json(OUTPUT_ROOT / 'active_state.json', dict(state_database=str(DATABASE)))
    print(json.dumps(completion, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-pilot', action='store_true')
    args = parser.parse_args()
    if not args.check_pilot and (OUTPUT_ROOT / 'completion.json').is_file():
        reader = module('layout_cleanup_completed_reader', CODE_ROOT / 'layout_reader.py')
        ready = reader.check()
        print(json.dumps(dict(ready, status='EXISTING_COMPLETE_LAYOUT_READ_ONLY', inference_calls=0), ensure_ascii=False))
        return
    pilot = pilot_gate()
    if args.check_pilot:
        print(json.dumps(pilot, ensure_ascii=False))
        return
    started = time.monotonic()
    (db, contract) = prepare(pilot)
    process_cam2(db)
    verified = verify(db)
    audit_count = audit_overlays(db)
    exported = EXPORT.export_database(DATABASE, OUTPUT_ROOT, workers=8, chunk=64)
    publish(db, contract, verified, exported, audit_count, started)
if __name__ == '__main__':
    main()
