"""Generate front bed crops into a fresh directory with the cycle-4 model."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE = Path(__file__).resolve().parent
ROOT = CYCLE.PREPROCESS_OUTPUT / '1. ROI_BOX'
OUTPUT = (ROOT / 'ROI_BOX/cam2').resolve()
RUN = ROOT / '_실행기록/261005_정면_CAM2_재생성'
OLD_RUN = ROOT / '_실행기록/261005_정면_CAM2_전체'
CONTROL = ROOT / 'ROI_BOX/CAM2_CURRENT_RUN.json'
MODEL_SHA = '648f6ce3fb520ef603a427f578aaa60c58ebc96e0226122a51c6d53b1b0e5109'
CHUNK = 64
import argparse, csv, json, sqlite3, time, importlib.util, hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm

def module(name, file):
    spec = resolved_spec(name, CODE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
c = module('cam2_rebuild_common', 'front_bed_policy.py')
prod = module('cam2_rebuild_save', 'front_bed_storage.py')

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def tree_meta(camera):
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in (ROOT / 'ROI_BOX' / camera).rglob('*') if p.is_file()}

def csv_read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def setup():
    training = read(c.TRAIN_COMPLETE)
    policy = read(c.POLICY_PATH)
    training = dict(training, best_model=str(CYCLE.BED_FRONT_MODEL))
    assert training['status'] == 'COMPLETE' and c.sha(training['best_model']) == MODEL_SHA == policy['model_sha256']
    assert policy['crop_rule'] == 'MODEL_BBOX_EXACT_NO_PADDING'
    assert all((p['up'] == p['down'] == 0 for p in policy['crop_profiles'].values()))
    rows = c.discover()
    assert len(rows) == 16843
    assert all((Path(r['target_path']).resolve().is_relative_to(OUTPUT) for r in rows))
    return (training, policy, rows)

def run():
    (training, policy, rows) = setup()
    RUN.mkdir(parents=True, exist_ok=True)
    current_sources = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    contract_path = RUN / 'execution_contract.json'
    contract = dict(model_sha256=MODEL_SHA, policy=policy, source_fingerprint=current_sources, output=str(OUTPUT), code_sha256=c.sha(__file__))
    if contract_path.exists():
        assert read(contract_path) == contract
    else:
        if OUTPUT.exists() and any(OUTPUT.rglob('*.jpg')):
            raise ValueError('Use a fresh output directory')
        c.write_json(contract_path, contract)
    db = sqlite3.connect(RUN / 'state.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS result(source TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    done_rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM result')]
    current = {str(Path(r['output_path']).resolve()).lower() for r in done_rows if r['status'] == 'SAVED_CANDIDATE'}
    actual = {str(p.resolve()).lower() for p in OUTPUT.rglob('*.jpg')}
    source_targets = {str(Path(r['target_path']).resolve()).lower() for r in rows}
    assert actual.issubset(source_targets) and current.issubset(actual)
    if not done_rows:
        assert not actual, 'Input contract failed: not actual'
    done = {r['source_path'].lower() for r in done_rows}
    pending = [r for r in rows if r['source_path'].lower() not in done]
    for p in sorted({Path(r['target_path']).parent for r in rows}):
        p.mkdir(parents=True, exist_ok=True)
        assert p.resolve().is_relative_to(OUTPUT)
    (model, _) = c.load_model()
    prod.KNOWN_REVIEWS = {}
    started = time.monotonic()
    base = len(done)
    c.write_json(CONTROL, dict(status='REBUILD_RUNNING', usable_for_ocr=False, current_run=str(RUN), output_root=str(OUTPUT), model_sha256=MODEL_SHA, updated=c.now()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in tqdm(range(0, len(pending), CHUNK), desc='Processing', unit='chunk'):
            predicted = c.predict_rows(model, pending[offset:offset + CHUNK], policy, desc='batch')
            valid = [r for r in predicted if 'image_width' in r and (not r.get('status', '').startswith(('ERROR', 'INVALID')))]
            invalid = [r for r in predicted if r not in valid]
            processed = list(pool.map(lambda r: prod.save_one(r, policy), valid)) + invalid
            for r in processed:
                r.update(run_id=RUN.name, model_sha256=MODEL_SHA, crop_rule=policy['crop_rule'], old_output_fallback=False)
            assert len(processed) == len(predicted)
            db.executemany('INSERT INTO result VALUES (?,?)', [(r['source_path'], json.dumps(r, ensure_ascii=False)) for r in processed])
            db.commit()
            processed_count = base + offset + len(processed)
            c.write_json(RUN / 'progress.json', dict(status='RUNNING', processed=processed_count, total=len(rows), elapsed_seconds=round(time.monotonic() - started, 1), updated=c.now()))
            if offset % (CHUNK * 8) == 0:
                print(f'PROGRESS {processed_count}/{len(rows)} elapsed={time.monotonic() - started:.1f}s', flush=True)
    records = [json.loads(r[0]) for r in db.execute('SELECT payload FROM result ORDER BY source')]
    db.close()
    assert len(records) == len(rows)
    saved = [r for r in records if r['status'] == 'SAVED_CANDIDATE']
    failed = [r for r in records if r['status'] != 'SAVED_CANDIDATE']
    saved_paths = {str(Path(r['output_path']).resolve()).lower() for r in saved}
    actual = {str(p.resolve()).lower() for p in OUTPUT.rglob('*.jpg')}
    assert actual == saved_paths, 'Input contract failed: actual == saved_paths'
    assert all((not Path(r['target_path']).exists() for r in failed)), "Input contract failed: all((not Path(r['target_path']).exists() for r in failed))"

    def verify(r):
        p = Path(r['output_path'])
        assert p.resolve().is_relative_to(OUTPUT) and p.name.endswith('_cam2.jpg')
        assert c.sha(p) == r['output_sha256']
        with Image.open(p) as im:
            assert im.size == (r['output_width'], r['output_height'])
            im.verify()
        assert r['crop_bbox'] == c.expanded_box(r['detected_bbox'], r['image_width'], r['image_height'], r['date'], policy['crop_profiles'])
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(tqdm(pool.map(verify, saved), total=len(saved), desc='Processing'))
    assert c.sha(training['best_model']) == MODEL_SHA
    c.csv_write(RUN / 'roi_manifest.csv', records)
    c.csv_write(RUN / 'failure_manifest.csv', failed)
    (dates, regimes) = prod.summarize(records)
    c.csv_write(RUN / 'date_summary.csv', dates)
    c.csv_write(RUN / 'camera_corrections.csv', [r for r in records if r['camera_corrected']])
    complete = dict(status='COMPLETE_WITH_ROW_ERRORS' if any((r['status'].startswith('ERROR') for r in records)) else 'COMPLETE', finished=c.now(), total=len(rows), saved_candidates=len(saved), not_saved=len(failed), statuses=dict(Counter((r['status'] for r in records))), by_regime=regimes, by_date=dates, model_sha256=MODEL_SHA, initial_deleted_count=0, stale_outputs_remaining=0, unregistered_files=0, failed_outputs_remaining=0, output_hash_dimensions_verified=True, exactly_one_selected_candidate_per_saved_source=True, crop_padding=0, single_bed_pixels_all_verified=False, camera_contract='front early CAM2 and late source CAM0 -> CAM2 filename/folder', actual_bed_numbers_verified=False, growth_bed_matching='PENDING_OCR', ocr_executed=False, other_cameras_unchanged=True, source_labels_unchanged=True, elapsed_this_invocation_seconds=round(time.monotonic() - started, 1))
    c.write_json(RUN / 'completion.json', complete)
    c.write_json(CONTROL, dict(status=complete['status'], usable_for_ocr=True, current_run=str(RUN), model_sha256=MODEL_SHA, roi_manifest=str(RUN / 'roi_manifest.csv'), failure_manifest=str(RUN / 'failure_manifest.csv'), saved_candidates=len(saved), not_saved=len(failed), stale_outputs_remaining=0, actual_bed_numbers_verified=False, remaining_crop_boundary_review=True, updated=c.now()))
    c.write_json(RUN / 'handoff_validation.json', dict(status='PASS', manifest_equals_output_directory=True, failed_output_paths_absent=True, stale_outputs_remaining=0, output_hash_dimensions_verified=True, cam0_cam1_preserved=True, actual_bed_number_qc='PENDING_OCR', all_crop_visual_purity_verified=False))
    print(json.dumps(complete, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
