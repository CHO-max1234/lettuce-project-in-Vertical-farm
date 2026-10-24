"""Layout reader calculations used by the final cycle pipeline."""
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
CURRENT_POINTER = DATA_ROOT / 'LAYOUT_CURRENT.json'
import argparse, json, math, sqlite3, zlib

def resolve_layout():
    if not CURRENT_POINTER.exists():
        raise FileNotFoundError('최신 자리배치 완료 포인터가 없습니다: ' + str(CURRENT_POINTER))
    p = json.loads(CURRENT_POINTER.read_text(encoding='utf-8'))
    root = Path(p['result_root'])
    completion = json.loads((root / 'completion.json').read_text(encoding='utf-8'))
    if completion.get('status') != 'COMPLETE_WITH_REVIEW' or completion.get('final_verification') != 'PASS' or completion.get('processed') != 39401 or (completion.get('remaining') != 0):
        raise RuntimeError('Invalid input or execution contract')
    for name in ('frame_manifest', 'slot_manifest', 'bbox_candidate_manifest'):
        if not Path(p[name]).is_file():
            raise FileNotFoundError(p[name])
    database = Path(p['state_database'])
    if not database.is_file():
        raise FileNotFoundError(database)
    return dict(p, database=str(database), completion_data=completion)

def connect_readonly(database=None):
    path = Path(database or resolve_layout()['database'])
    return sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)

def read_layout_frame(frame_id, database=None):
    with connect_readonly(database) as db:
        row = db.execute('SELECT payload FROM frames WHERE frame_id=?', (frame_id,)).fetchone()
    if row is None:
        raise KeyError(frame_id)
    return json.loads(zlib.decompress(row[0]))

def iter_layout_frames(database=None, camera=None):
    db = connect_readonly(database)
    try:
        sql = 'SELECT payload FROM frames' + (' WHERE camera=?' if camera else '') + ' ORDER BY sequence'
        for (blob,) in db.execute(sql, (camera,) if camera else ()):
            yield json.loads(zlib.decompress(blob))
    finally:
        db.close()

def geometry_candidate(frame, slot):
    if frame.get('geometry_status') not in ('PASS', 'REVIEW') or slot.get('geometry_qc_status', slot.get('geometry_status')) not in ('PASS', 'REVIEW'):
        return False
    try:
        (x, y) = (float(slot['anchor_x']), float(slot['anchor_y']))
        box = slot['cell_bbox']
        (w, h) = (frame['source_width'], frame['source_height'])
        return math.isfinite(x) and math.isfinite(y) and (0 <= x < w) and (0 <= y < h) and (0 <= box[0] < box[2] <= w) and (0 <= box[1] < box[3] <= h)
    except (KeyError, TypeError, ValueError, IndexError):
        return False

def iter_bbox_inputs(database=None, camera=None):
    for d in iter_layout_frames(database, camera):
        f = d['frame']
        for s in d['slot']:
            if not geometry_candidate(f, s):
                continue
            yield dict(frame=f, slot=s, raw_segments=d['segment'], bbox_candidate_eligible=True, bbox_review_required=f['geometry_status'] == 'REVIEW' or s['geometry_qc_status'] == 'REVIEW', bbox_anchor_confirmation_required=s.get('slot_geometry_usable') is not True or s.get('anchor_registration_status') == 'UNKNOWN_REGISTRATION', body_selection_executed=False, target_confirmation='BODY_SELECTION_PENDING')

def check():
    p = resolve_layout()
    c = p['completion_data']
    db = connect_readonly(p['database'])
    try:
        assert db.execute('SELECT count(*) FROM frames').fetchone()[0] == 39401
        candidates = {cam: n for (cam, n) in db.execute("SELECT f.camera,count(*) FROM slot_index s JOIN frames f USING(frame_id) WHERE f.geometry_status IN ('PASS','REVIEW') AND s.geometry_qc_status IN ('PASS','REVIEW') GROUP BY f.camera")}
        policy = json.loads((Path(p['result_root']) / 'bbox_candidate_policy_verification.json').read_text(encoding='utf-8'))
        assert policy['status'] == 'PASS' and policy['counts']['frames'] == 39401
        candidates = {cam: n - policy['counts'].get(cam + '_invalid_native_proposal_excluded', 0) for (cam, n) in candidates.items()}
        for (cam, n) in candidates.items():
            assert n == c['counts'][cam + '_bbox_candidates']
        assert db.execute("SELECT count(*) FROM frames WHERE camera='cam2'").fetchone()[0] == 14540
    finally:
        db.close()
    return dict(status='PASS', result_root=p['result_root'], database=p['database'], frames=39401, candidate_slots=candidates, policy='PASS_REVIEW_VALID_NATIVE_PROPOSAL; FRAME_AND_SLOT_HOLD_EXCLUDED; ANCHOR_CONFIRMATION_PENDING_WHERE_REQUIRED', bbox_body_selection_executed=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.parse_args()
    print(json.dumps(check(), ensure_ascii=False, indent=2))
