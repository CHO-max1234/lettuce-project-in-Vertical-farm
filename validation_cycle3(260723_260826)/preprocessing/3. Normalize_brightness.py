"""Normalize reviewed bed crops using the shared reference calculations."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CODE = CYCLE.THIRD_ROOT / 'preprocessing/3. Normalize_brightness.py'
INPUT_ROOT = CYCLE.SOURCE_ROOT / '1. ROI_BOX/ROI_BOX'
OCR_POINTER = CYCLE.OCR_POINTER
OUTPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '3. 밝기보정과 Scale'
RECORD_ROOT = OUTPUT_ROOT / '_실행기록' / '261009_밝기보정'
EXPECTED_COUNTS = {'CAM0': 11837, 'CAM1': 13024, 'CAM2': 14540}
WORKERS = 4
CHUNK_SIZE = 8
import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
sys.dont_write_bytecode = True
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda : f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def save_json(path, obj):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

@contextmanager
def run_lock():
    import msvcrt
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    with (RECORD_ROOT / '.run.lock').open('a+b') as f:
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

def load_inputs():
    pointer = json.loads(OCR_POINTER.read_text(encoding='utf-8-sig'))
    contract = Path(pointer['manifest']).with_name('02_개선전후_기록.csv')
    csv.field_size_limit(10000000)
    all_rows = []
    with contract.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            all_rows.append(r)
    selected = [r for r in all_rows if r['label_accepted'] == 'True']
    counts = dict(Counter((r['camera'] for r in selected)))
    if len(all_rows) != 41322 or counts != EXPECTED_COUNTS:
        raise ValueError(f'OCR   : {len(all_rows)}, {counts}')
    seen = set()
    for r in selected:
        source = Path(r['current_path']).resolve()
        rel = source.relative_to(INPUT_ROOT.resolve())
        if len(rel.parts) != 3 or rel.parts[0].lower() != r['camera'].lower():
            raise ValueError(f'CAM/  : {source}')
        if rel.parts[1] != r['date'][2:] or source.stem.startswith('bed00_'):
            raise ValueError(f'/  : {source}')
        target = OUTPUT_ROOT / rel.parts[0].lower() / rel.parts[1] / (source.stem + '.png')
        key = str(target).casefold()
        if key in seen:
            raise ValueError(f'  : {target}')
        if not source.is_file():
            raise FileNotFoundError(source)
        seen.add(key)
        r['_target'] = str(target)
    return (contract, selected)

def load_journal(path):
    rows = {}
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[r['source_path'].casefold()] = r
    return rows

def work(r, previous, reference):
    (source, target) = (Path(r['current_path']), Path(r['_target']))
    row = {k: r.get(k, '') for k in ['capture_key', 'date', 'camera', 'accepted_number', 'label_accepted', 'label_source', 'label_confidence', 'qc_status', 'review_required', 'review_reasons', 'label_review_reasons', 'pair_qc', 'identity_verified', 'human_label', 'old_key', 'old_filename', 'source_camera', 'crop_bbox']}
    row.update(source_path=str(source), raw_source_path=r['source_path'], source_sha256=r['sha256'], output_path=str(target), status='ERROR')
    try:
        if sha(source) != r['sha256']:
            raise ValueError('Invalid input or execution contract')
        if previous and previous.get('status') == 'OK' and target.exists():
            if previous['source_sha256'] != r['sha256'] or sha(target) != previous['output_sha256']:
                raise ValueError('Invalid input or execution contract')
            return previous
        before = reference['read_image'](source)
        after = reference['correct_brightness'](before)
        if before.shape != after.shape:
            raise ValueError('Invalid input or execution contract')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not np.array_equal(reference['read_image'](target), after):
                raise FileExistsError(f'다른 보정 결과 존재: {target}')
        else:
            (ok, encoded) = cv2.imencode('.png', after)
            if not ok:
                raise ValueError('Invalid input or execution contract')
            temp = target.with_suffix('.png.partial')
            encoded.tofile(str(temp))
            os.rename(temp, target)
        if not np.array_equal(reference['read_image'](target), after):
            raise ValueError('Invalid input or execution contract')
        if sha(source) != r['sha256']:
            raise ValueError('Invalid input or execution contract')
        (bmax, amax) = (before.max(axis=2), after.max(axis=2))
        (bg, ag) = (cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY))
        row.update(status='OK', output_sha256=sha(target), shape=list(before.shape), input_unchanged=True, coordinates_unchanged=True, before_mean_gray=float(bg.mean()), after_mean_gray=float(ag.mean()), before_dark50_pct=float((bg < 50).mean() * 100), after_dark50_pct=float((ag < 50).mean() * 100), new_any250_pixels=int(((bmax < 250) & (amax >= 250)).sum()), new_any255_pixels=int(((bmax < 255) & (amax == 255)).sum()), new_black_pixels=int(((bmax > 0) & (amax == 0)).sum()))
    except Exception as exc:
        row.update(error=f'{type(exc).__name__}: {exc}')
    return row

def verify_outputs(inputs, results):
    expected = {str(Path(r['_target']).resolve()).casefold() for r in inputs}
    actual = {str(p.resolve()).casefold() for cam in EXPECTED_COUNTS for p in (OUTPUT_ROOT / cam.lower()).rglob('*.png')}
    errors = []
    for r in tqdm(results, desc='Independent SHA verification', unit='image', mininterval=3):
        if r['status'] != 'OK':
            errors.append(r['source_path'])
            continue
        if sha(r['source_path']) != r['source_sha256'] or sha(r['output_path']) != r['output_sha256']:
            errors.append(r['source_path'])
    return dict(expected=len(expected), actual=len(actual), missing=len(expected - actual), extra=len(actual - expected), hash_errors=errors)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    reference = run_source(str(REFERENCE_CODE), run_name='brightness_reference')
    if reference['SHADOW_STRENGTH'] != 0.65 or reference['HIGHLIGHT_REDUCTION'] != 0.02:
        raise ValueError('Invalid input or execution contract')
    (contract, inputs) = load_inputs()
    fingerprint = dict(code_sha256=sha(__file__), reference_code=str(REFERENCE_CODE), reference_sha256=sha(REFERENCE_CODE), pointer_sha256=sha(OCR_POINTER), input_contract=str(contract), input_contract_sha256=sha(contract), input_root=str(INPUT_ROOT), output_root=str(OUTPUT_ROOT), accepted_count=len(inputs), camera_counts=EXPECTED_COUNTS, parameters={'shadow_strength': 0.65, 'highlight_reduction': 0.02}, workers=WORKERS, chunk_size=CHUNK_SIZE, method='Exact season-3 correct_brightness; common RGB multiplier; once only')
    with run_lock():
        contract_file = RECORD_ROOT / 'execution_contract.json'
        if contract_file.exists():
            if json.loads(contract_file.read_text(encoding='utf-8')) != fingerprint:
                raise ValueError('Invalid input or execution contract')
        else:
            save_json(contract_file, fingerprint)
        journal_file = RECORD_ROOT / 'brightness_records.jsonl'
        previous = load_journal(journal_file)
        todo = inputs
        if args.pilot:
            todo = []
            for cam in EXPECTED_COUNTS:
                group = [r for r in inputs if r['camera'] == cam]
                todo.extend([group[0], group[len(group) // 2]])
        results = []
        ok_count = error_count = 0
        save_json(RECORD_ROOT / 'completion.json', dict(status='RUNNING', expected=len(inputs), pilot=args.pilot, updated=datetime.now().astimezone().isoformat()))
        print(json.dumps(dict(selected=len(inputs), processing=len(todo), cameras=EXPECTED_COUNTS), ensure_ascii=False), flush=True)
        with journal_file.open('a', encoding='utf-8') as journal, ThreadPoolExecutor(max_workers=WORKERS) as pool:
            with tqdm(total=len(todo), desc='Season-4 brightness', unit='image', mininterval=3) as bar:
                for start in range(0, len(todo), CHUNK_SIZE):
                    chunk = todo[start:start + CHUNK_SIZE]
                    for result in pool.map(lambda r: work(r, previous.get(r['current_path'].casefold()), reference), chunk):
                        results.append(result)
                        ok_count += result['status'] == 'OK'
                        error_count += result['status'] != 'OK'
                        journal.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                        bar.update(1)
                    journal.flush()
                    save_json(RECORD_ROOT / 'progress.json', dict(processed=len(results), total=len(todo), ok=ok_count, errors=error_count, elapsed_seconds=round(time.perf_counter() - started, 1), updated=datetime.now().astimezone().isoformat()))
        good = [r for r in results if r['status'] == 'OK']
        failed = [r for r in results if r['status'] != 'OK']
        if args.pilot:
            verification = dict(scope='6 camera samples; decoded PNG equals exact reference function')
            status = 'PILOT_COMPLETE' if not failed else 'PILOT_FAILED'
        else:
            verification = verify_outputs(inputs, results)
            unchanged = all((sha(path) == fingerprint[key] for (path, key) in [(contract, 'input_contract_sha256'), (OCR_POINTER, 'pointer_sha256'), (REFERENCE_CODE, 'reference_sha256'), (Path(__file__), 'code_sha256')]))
            verification['contracts_unchanged'] = unchanged
            status = 'COMPLETE' if not failed and (not verification['hash_errors']) and (not verification['missing']) and (not verification['extra']) and unchanged else 'COMPLETE_WITH_ERRORS'
            fields = list(dict.fromkeys((k for r in results for k in r)))
            with (RECORD_ROOT / 'brightness_manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for (k, v) in r.items()} for r in results))
        summary = dict(status=status, expected=len(inputs), processed=len(results), ok=len(good), errors=len(failed), pilot=args.pilot, camera_counts=dict(Counter((r['camera'] for r in good))), output_root=str(OUTPUT_ROOT), parameters=fingerprint['parameters'], elapsed_seconds=round(time.perf_counter() - started, 1), verification=verification, source_sha_verified_during_processing=len(good), decoded_png_reference_equal=len(good), diagnostics={k: sum((r[k] for r in good)) for k in ['new_any250_pixels', 'new_any255_pixels', 'new_black_pixels']}, errors_detail=failed, completed_at=datetime.now().astimezone().isoformat(), scale_executed=False, ocr_states_preserved=True)
        save_json(RECORD_ROOT / ('pilot_summary.json' if args.pilot else 'summary.json'), summary)
        save_json(RECORD_ROOT / 'completion.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if failed or status == 'COMPLETE_WITH_ERRORS':
            raise SystemExit(1)
if __name__ == '__main__':
    main()
