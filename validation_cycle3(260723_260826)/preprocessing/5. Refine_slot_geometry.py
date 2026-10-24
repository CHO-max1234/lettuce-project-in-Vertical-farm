"""Refine CAM2 slot geometry from the preserved base database."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT / '4. 자리배치와 bbox'
FINAL_ROOT = DATA_ROOT / '자리배치_cam2_최종'
FINAL_DATABASE = FINAL_ROOT / 'state.sqlite3'
LINEAGE_BASE_DB = CYCLE.LAYOUT_BASE_DB
PYTHON = CYCLE.PYTHON
import argparse, subprocess, json, importlib.util
import json, math, os, sqlite3, zlib
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, deque
from tqdm import tqdm
LINEAGE_FIELDS = ('record_id', 'camera', 'physical_slot', 'slot_name', 'source_path', 'source_sha256', 'source_width', 'source_height', 'ocr_current_path', 'ocr_action', 'ocr_status', 'bed_identity_status', 'identity_verified', 'label_source', 'pair_qc', 'mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_image_key', 'raw_source_path', 'raw_crop_bbox', 'source_camera', 'native_width', 'native_height')

def lineage_db(path):
    return sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)

def lineage_chunk(start, stop):
    old = lineage_db(LINEAGE_BASE_DB)
    new = lineage_db(FINAL_DATABASE)
    counts = Counter()
    failures = []
    for (fid, blob) in old.execute("SELECT frame_id,payload FROM frames WHERE camera='cam2' AND sequence>=? AND sequence<? ORDER BY sequence", (start, stop)):
        a = json.loads(zlib.decompress(blob))
        b = json.loads(zlib.decompress(new.execute('SELECT payload FROM frames WHERE frame_id=?', (fid,)).fetchone()[0]))
        before = {s['physical_slot']: s for s in a['slot']}
        f = b['frame']
        counts['frames'] += 1
        try:
            for s in b['slot']:
                previous = before.get(s['physical_slot'])
                if previous:
                    for key in LINEAGE_FIELDS:
                        assert s.get(key) == previous.get(key), (s['physical_slot'], key)
                    counts['existing_slots_immutable'] += 1
                else:
                    counts['new_structural_slots'] += 1
                (dx, dy) = f['raw_crop_bbox'][:2]
                assert math.isclose(s['raw_anchor_x'], s['anchor_x'] + dx, abs_tol=1e-06)
                assert math.isclose(s['raw_anchor_y'], s['anchor_y'] + dy, abs_tol=1e-06)
                counts['raw_anchor_offsets_checked'] += 1
                assert s.get('bbox_target_selection_executed') is False
                assert s.get('bbox') is None and s.get('seg_id') is None
                if s.get('geometry_version', '').startswith('S4_CAM2_VER3'):
                    assert s.get('target_available') is None
                if s.get('geometry_qc_status') == 'HOLD':
                    assert not s.get('slot_geometry_usable')
        except Exception as e:
            failures.append(dict(frame_id=fid, error=repr(e)))
    old.close()
    new.close()
    return dict(counts=dict(counts), failures=failures)

def verify_slot_lineage():
    with lineage_db(LINEAGE_BASE_DB) as c:
        n = c.execute('SELECT max(sequence)+1 FROM frames').fetchone()[0]
    pending = deque()
    iterator = iter(range(0, n, 128))
    counts = Counter()
    failures = []
    with ProcessPoolExecutor(max_workers=8) as pool, tqdm(total=14540, desc='CAM2 immutable OCR/scale and raw offsets', unit='frame') as bar:

        def fill():
            while len(pending) < 8:
                try:
                    start = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(lineage_chunk, start, min(n, start + 128)))
        fill()
        while pending:
            r = pending.popleft().result()
            counts.update(r['counts'])
            failures.extend(r['failures'])
            bar.update(r['counts'].get('frames', 0))
            fill()
    assert counts['frames'] == 14540
    report = dict(status='FAIL' if failures else 'PASS', counts=dict(counts), fields=list(LINEAGE_FIELDS), failure_count=len(failures), failures=failures[:100], scope='EXISTING_CAM2_SLOT_LINEAGE_RAW_COORDINATES_AND_NO_BODY_ASSIGNMENT')
    path = FINAL_ROOT / 'slot_lineage_coordinate_verification.json'
    tmp = path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError('CAM2_SLOT_LINEAGE_OR_COORDINATE_FAIL')
import csv, io, json, os, sqlite3, zlib, importlib.util
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
spec = resolved_spec('current_bbox_review_reader', CODE_ROOT / 'layout_reader.py')
READER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(READER)
POLICY_FIELDS = ['record_id', 'frame_id', 'camera', 'physical_slot', 'frame_geometry_status', 'slot_geometry_qc_status', 'slot_geometry_usable', 'bbox_candidate_eligible', 'bbox_review_required', 'bbox_anchor_confirmation_required', 'bbox_body_selected', 'exclude_reason']

def policy_save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def policy_chunk(start, stop):
    db = sqlite3.connect(FINAL_DATABASE.as_uri() + '?mode=ro&immutable=1', uri=True)
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, POLICY_FIELDS)
    counts = Counter()
    errors = []
    for (blob,) in db.execute('SELECT payload FROM frames WHERE sequence>=? AND sequence<? ORDER BY sequence', (start, stop)):
        d = json.loads(zlib.decompress(blob))
        f = d['frame']
        counts['frames'] += 1
        for s in d['slot']:
            eligible = READER.geometry_candidate(f, s)
            status = s['geometry_qc_status']
            allowed = f['geometry_status'] in ('PASS', 'REVIEW') and status in ('PASS', 'REVIEW')
            if allowed and (not eligible):
                errors.append(dict(frame_id=f['frame_id'], physical_slot=s['physical_slot'], reason='REVIEW_PROPOSAL_OUTSIDE_NATIVE_EXCLUDED'))
                counts[f['camera'] + '_invalid_native_proposal_excluded'] += 1
            assert not ((status == 'HOLD' or f['geometry_status'] == 'HOLD') and eligible)
            confirmation = eligible and (s.get('slot_geometry_usable') is not True or s.get('anchor_registration_status') == 'UNKNOWN_REGISTRATION')
            counts['slots'] += 1
            counts[f['camera'] + '_bbox_candidates'] += int(eligible)
            counts[f['camera'] + '_anchor_confirmation_required'] += int(confirmation)
            writer.writerow(dict(record_id=s['record_id'], frame_id=f['frame_id'], camera=f['camera'], physical_slot=s['physical_slot'], frame_geometry_status=f['geometry_status'], slot_geometry_qc_status=status, slot_geometry_usable=s.get('slot_geometry_usable'), bbox_candidate_eligible=eligible, bbox_review_required=eligible and (status == 'REVIEW' or f['geometry_status'] == 'REVIEW'), bbox_anchor_confirmation_required=confirmation, bbox_body_selected=False, exclude_reason='' if eligible else 'FRAME_HOLD' if f['geometry_status'] == 'HOLD' else 'SLOT_HOLD_OR_INVALID_GEOMETRY'))
    db.close()
    return dict(counts=dict(counts), errors=errors, csv=buf.getvalue())

def refresh_candidate_policy():
    c = json.loads((FINAL_ROOT / 'completion.json').read_text(encoding='utf-8'))
    assert c['processed'] == 39401 and c['final_verification'] == 'PASS'
    pending = deque()
    iterator = iter(range(0, 39401, 96))
    counts = Counter()
    errors = []
    path = FINAL_ROOT / 'bbox_candidate_manifest.csv'
    partial = path.with_suffix('.csv.review_policy.partial')
    with partial.open('w', encoding='utf-8-sig', newline='') as f, ProcessPoolExecutor(max_workers=8) as pool, tqdm(total=39401, desc='User accepted PASS/REVIEW candidate contract', unit='frame') as bar:
        csv.DictWriter(f, POLICY_FIELDS).writeheader()

        def fill():
            while len(pending) < 8:
                try:
                    start = next(iterator)
                except StopIteration:
                    break
                pending.append(pool.submit(policy_chunk, start, min(39401, start + 96)))
        fill()
        while pending:
            r = pending.popleft().result()
            counts.update(r['counts'])
            errors.extend(r['errors'])
            f.write(r['csv'])
            bar.update(r['counts']['frames'])
            fill()
    report = dict(status='PASS', policy='USER_ACCEPTED_PASS_REVIEW_VALID_NATIVE_PROPOSAL; HOLD_EXCLUDED; NO_GEOMETRY_CONFIDENCE_UPGRADE', counts=dict(counts), failure_count=0, failures=[], excluded_invalid_native_proposal_count=len(errors), excluded_invalid_native_proposals=errors[:100], body_selection_executed=False, database_modified=False)
    policy_save(FINAL_ROOT / 'bbox_candidate_policy_verification.json', report)
    strict = FINAL_ROOT / 'bbox_candidate_manifest_strict_original.csv'
    if not strict.exists():
        os.replace(path, strict)
    os.replace(partial, path)
    c.setdefault('strict_original_candidate_counts', {cam: c['counts'][cam + '_bbox_candidates'] for cam in ('cam0', 'cam1', 'cam2')})
    c['counts'].update({k: v for (k, v) in counts.items() if '_bbox_candidates' in k or '_anchor_confirmation_required' in k or '_invalid_native_proposal_excluded' in k})
    c['bbox_candidate_policy'] = report['policy']
    c['bbox_candidate_policy_verification'] = 'PASS'
    policy_save(FINAL_ROOT / 'completion.json', c)
    verification = json.loads((FINAL_ROOT / 'final_verification.json').read_text(encoding='utf-8'))
    verification.setdefault('strict_original_candidate_counts', c['strict_original_candidate_counts'])
    verification['counts'].update({k: v for (k, v) in counts.items() if '_bbox_candidates' in k or '_anchor_confirmation_required' in k or '_invalid_native_proposal_excluded' in k})
    verification['bbox_candidate_policy_verification'] = 'PASS'
    verification['bbox_candidate_policy'] = report['policy']
    policy_save(FINAL_ROOT / 'final_verification.json', verification)
    pointer = json.loads(READER.CURRENT_POINTER.read_text(encoding='utf-8'))
    pointer['bbox_policy'] = report['policy']
    policy_save(READER.CURRENT_POINTER, pointer)
    policy_save(FINAL_ROOT / 'bbox_reader_check.json', READER.check())
    print(json.dumps(report, ensure_ascii=False), flush=True)
import csv, hashlib, importlib.util, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor

def report_load(name, path):
    spec = resolved_spec(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
READER = report_load('cam2_latest_reader', CODE_ROOT / 'layout_reader.py')

def report_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def report_save(path, data):
    p = Path(path)
    t = p.with_suffix(p.suffix + '.partial')
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(t, p)

def write_handoff():
    """Write the current verified geometry contract for the crop stage."""
    check = READER.check()
    report_save(FINAL_ROOT / 'bbox_handoff_contract.json', dict(status='BBOX_HANDOFF_PASS', reader=str(Path(READER.__file__)), current_pointer=str(READER.CURRENT_POINTER), checks=check, review_policy='PASS/REVIEW candidates; HOLD excluded; identity and scale review retained', body_selection_executed=False))
    print(json.dumps(check, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true', help='Execution option')
    args = parser.parse_args()
    if not args.finalize and READER.CURRENT_POINTER.is_file():
        ready = READER.check()
        print(json.dumps(dict(ready, status='EXISTING_COMPLETE_LAYOUT_READ_ONLY', inference_calls=0), ensure_ascii=False))
        return
    if not args.finalize:
        subprocess.run([str(PYTHON), '-B', '-u', '-X', 'utf8', str(CODE_ROOT / 'layout_builder.py')], cwd=CODE_ROOT, check=True)
    verify_slot_lineage()
    refresh_candidate_policy()
    write_handoff()
if __name__ == '__main__':
    main()
