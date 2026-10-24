"""Extract saved target masks and plant crops using bounded worker processes."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
RUNNER_PATH = Path(__file__).with_name('crop_runtime.py')
PROCESS_WORKERS = 8
import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import traceback
import numpy as np
WORKER = None
SAVES = None
SAMPLE_IDS = set()

def load_runner():
    spec = resolved_spec('s4_process_runner', RUNNER_PATH)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    spec.loader.exec_module(obj)
    return obj

def initialize(sample_ids=()):
    global WORKER, SAVES, SAMPLE_IDS
    WORKER = load_runner()
    import torch
    torch.set_num_threads(2)
    SAVES = ThreadPoolExecutor(max_workers=4)
    SAMPLE_IDS = set(sample_ids)

def pipeline(item):
    while WORKER.memory_used_gib() > WORKER.SYSTEM_RAM_LIMIT_GIB - 2:
        time.sleep(0.25)
    decoded = WORKER.decode(item)
    inferred = WORKER.infer_inputs(decoded, 1)
    return WORKER.postprocess(inferred, SAMPLE_IDS, SAVES)

def verify_case(item, benchmark=False):
    r = WORKER
    selected = r.select(r.infer_inputs(r.decode(item), 1))
    (_, payload, source, rows, decisions, calls, local_images) = selected
    if benchmark:
        db = sqlite3.connect(r.STATE_DATABASE.as_uri() + '?mode=ro', uri=True)
        expected = {r.unpack(blob)['record_id']: r.unpack(blob) for (blob,) in db.execute('SELECT payload FROM slots WHERE frame_id=?', (item['frame_id'],))}
        db.close()
    else:
        p = r.APPROVED_BBOX_ROOT / item['camera'] / 'records' / (Path(payload['frame']['image_name']).stem + '.json')
        expected = {x['record_id']: x for x in json.loads(p.read_text(encoding='utf-8'))['targets']}
    mismatches = []
    targets = 0
    for s in payload['slot']:
        (d, e) = (decisions[s['record_id']], expected[s['record_id']])
        available = d.get('mask') is not None and d['mask'].any()
        if d['status'] != e['selection_status'] or bool(available) != e['target_available']:
            mismatches.append({'record_id': s['record_id'], 'reason': 'STATUS_OR_AVAILABILITY'})
            continue
        if available:
            targets += 1
            (_, box) = r.target_box(d['mask'], s, source.shape[:2])
            (x1, y1, x2, y2) = box
            digest = hashlib.sha256(d['mask'][y1:y2, x1:x2].tobytes()).hexdigest()
            expected_hash = e.get('target_mask_pixels_sha256') if benchmark else e['mask_pixels_sha256']
            if box != e['final_bbox'] or digest != expected_hash:
                mismatches.append({'record_id': s['record_id'], 'reason': 'MASK_PIXELS_OR_BBOX'})
    if benchmark and (not mismatches):
        (records, _, _) = r.build_records(payload, source, rows, decisions, False, SAVES)
        for rec in records:
            if rec['target_available']:
                assert rec['crop_sha256'] == expected[rec['record_id']]['crop_sha256']
    return dict(frames=1, slots=len(expected), targets=targets, mismatches=mismatches)

def verify_or_benchmark(workers, benchmark=False, benchmark_frames=64):
    r = load_runner()
    (items, _, _) = r.inventory()
    if benchmark:
        with sqlite3.connect(r.STATE_DATABASE.as_uri() + '?mode=ro', uri=True) as db:
            chosen = {x[0] for x in db.execute('SELECT frame_id FROM frames ORDER BY sequence LIMIT ?', (benchmark_frames,))}
    else:
        chosen = {s['frame_id'] for s in json.loads(r.CORE.SAMPLE_MANIFEST.read_text(encoding='utf-8'))}
    tasks = [i for i in items if i['frame_id'] in chosen]
    started = time.monotonic()
    counts = Counter()
    mismatches = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(verify_case, i, benchmark) for i in tasks]
        for f in futures:
            result = f.result()
            counts.update({k: result[k] for k in ('frames', 'slots', 'targets')})
            mismatches.extend(result['mismatches'])
    elapsed = time.monotonic() - started
    report = dict(status='PASS' if not mismatches else 'FAIL', scope=f'{len(tasks)}_FRAME_MATERIALIZATION_BENCHMARK' if benchmark else 'VER2_30_FRAME_PARITY', counts=dict(counts), process_workers=workers, local_batch=1, elapsed_seconds=round(elapsed, 2), fps=round(len(tasks) / elapsed, 3), mismatches=mismatches, runner_sha256=r.sha(RUNNER_PATH), engine_sha256=r.sha(__file__), checked_at=r.now())
    r.save_json(r.STATE_ROOT / (f'process_benchmark_{workers}_{len(tasks)}.json' if benchmark else f'process_ver2_parity_{workers}.json'), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(2)

def configure_export(db, runner):
    used = runner.memory_used_gib()
    memory_sort = runner.SYSTEM_RAM_LIMIT_GIB - used >= 16
    db.execute('PRAGMA temp_store=' + ('MEMORY' if memory_sort else 'FILE'))
    db.execute('PRAGMA cache_size=' + ('-524288' if memory_sort else '-131072'))
    db.execute('PRAGMA mmap_size=' + (str(8 * 2 ** 30) if memory_sort else '0'))
    db.execute('CREATE INDEX IF NOT EXISTS slots_frame_order ON slots(frame_id,slot_order)')
    db.execute('CREATE INDEX IF NOT EXISTS slots_camera_target_status ON slots(camera,target_available,selection_status)')
    db.execute('CREATE INDEX IF NOT EXISTS frames_camera ON frames(camera)')
    db.execute('ANALYZE')
    query = 'SELECT s.payload FROM slots s JOIN frames f USING(frame_id) ORDER BY f.sequence,s.slot_order'
    report = dict(at=runner.now(), system_ram_used_gib_before=used, system_ram_limit_gib=runner.SYSTEM_RAM_LIMIT_GIB, temp_store='MEMORY' if memory_sort else 'FILE', cache_mib=512 if memory_sort else 128, mmap_limit_gib=8 if memory_sort else 0, query_plan=db.execute('EXPLAIN QUERY PLAN ' + query).fetchall(), calculations_changed=False, output_order='frame.sequence,slot.slot_order')
    runner.save_json(runner.STATE_ROOT / 'export_execution_settings.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)

def run_full_processes(workers):
    import msvcrt
    r = load_runner()
    (items, sampled, _) = r.inventory()
    proof = r.STATE_ROOT / f'process_ver2_parity_{workers}.json'
    benchmark = r.STATE_ROOT / f'process_benchmark_{workers}_256.json'
    assert json.loads(proof.read_text(encoding='utf-8'))['status'] == 'PASS'
    assert json.loads(benchmark.read_text(encoding='utf-8'))['status'] == 'PASS'
    contract = r.runtime_contract(1, 4, sampled)
    signature = hashlib.sha256(r.encoded(contract).encode()).hexdigest()
    r.STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with (r.STATE_ROOT / '.run.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        db = r.database()
        existing = db.execute("SELECT value FROM config WHERE key='signature'").fetchone()
        if existing and existing[0] != signature:
            raise RuntimeError('Invalid input or execution contract')
        with db:
            db.execute('INSERT OR IGNORE INTO config VALUES(?,?)', ('signature', signature))
        completed_path = r.OUTPUT_ROOT / 'completion.json'
        if completed_path.exists():
            previous = json.loads(completed_path.read_text(encoding='utf-8'))
            if previous.get('signature') == signature and previous.get('verification') == 'PASS' and (previous.get('frames') == r.EXPECTED_FRAMES) and (previous.get('slot_rows') == r.EXPECTED_SLOTS) and (db.execute('SELECT count(*) FROM frames').fetchone()[0] == r.EXPECTED_FRAMES) and (db.execute('SELECT count(*) FROM slots').fetchone()[0] == r.EXPECTED_SLOTS) and all((Path(previous[k]).is_file() for k in ('crop_manifest', 'feature_input_manifest', 'frame_manifest', 'sample_gallery'))):
                db.close()
                print(json.dumps(dict(status='ALREADY_COMPLETE', completion=str(completed_path), frames=previous['frames'], masked_crops=previous['masked_crops']), ensure_ascii=False))
                return
        done = {x[0] for x in db.execute('SELECT frame_id FROM frames')}
        pending = [i for i in items if i['frame_id'] not in done]
        sample_ids = [s['frame_id'] for s in sampled]
        execution = dict(engine='ISOLATED_PROCESS_PIPELINE', engine_path=str(Path(__file__).resolve()), engine_sha256=r.sha(__file__), runner_path=str(RUNNER_PATH), runner_sha256=r.sha(RUNNER_PATH), original_contract_signature=signature, process_workers=workers, local_batch=1, png_save_threads_per_process=4, system_ram_limit_gib=r.SYSTEM_RAM_LIMIT_GIB, resumed_frames=len(done), started_at=r.now(), parity_proof=str(proof), benchmark_proof=str(benchmark))
        r.save_json(r.STATE_ROOT / 'process_execution_contract.json', execution)
        if not (r.STATE_ROOT / 'execution_contract.json').exists():
            r.save_json(r.STATE_ROOT / 'execution_contract.json', dict(contract, signature=signature))
        r.save_json(r.SAMPLE_ROOT / 'sample_manifest.json', sampled)
        counters = Counter(completed=0, inference_calls=0, local_images=0, gpu_workers=workers, local_batch=1)
        started = time.monotonic()
        r.status(db, started, counters)
        bar = r.tqdm(total=len(pending), desc='S4 FULL frozen ver2 / isolated processes', unit='frame', mininterval=5)
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(sample_ids,), mp_context=multiprocessing.get_context('spawn')) as pool:
                iterator = iter(pending)
                futures = set()
                exhausted = False
                last_report = time.monotonic()
                while futures or not exhausted:
                    while not exhausted and len(futures) < workers * 3 and (r.memory_used_gib() < r.SYSTEM_RAM_LIMIT_GIB - 4):
                        try:
                            item = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        futures.add(pool.submit(pipeline, item))
                    (finished, _) = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED) if futures else (set(), set())
                    for future in finished:
                        futures.remove(future)
                        result = future.result()
                        r.save_result(db, result)
                        frame = result[1]
                        counters['completed'] += 1
                        counters['inference_calls'] += frame['inference_calls']
                        counters['local_images'] += frame['local_input_images']
                        bar.update(1)
                    if finished and counters['completed'] % r.CHECKPOINT_FRAMES == 0 or time.monotonic() - last_report > 20:
                        report = r.status(db, started, counters)
                        report['execution_engine'] = 'ISOLATED_PROCESS_PIPELINE'
                        r.save_json(r.OUTPUT_ROOT / 'progress.json', report)
                        db.execute('PRAGMA wal_checkpoint(PASSIVE)')
                        last_report = time.monotonic()
                    if not futures and (not exhausted):
                        time.sleep(0.25)
            bar.close()
            r.status(db, started, counters, 'EXPORTING')
            configure_export(db, r)
            r.finalize(db, items, sampled, signature)
            p = r.OUTPUT_ROOT / 'completion.json'
            completed = json.loads(p.read_text(encoding='utf-8'))
            completed['execution_engine'] = execution
            for target in (p, r.OUTPUT_ROOT / 'CROP_CURRENT.json', r.OUTPUT_ROOT / 'progress.json'):
                r.save_json(target, completed)
        except BaseException as error:
            r.save_json(r.STATE_ROOT / 'failure.json', dict(status='FAILED_RESTARTABLE', at=r.now(), error=str(error), traceback=traceback.format_exc()))
            r.status(db, started, counters, 'FAILED_RESTARTABLE')
            raise
        finally:
            db.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=PROCESS_WORKERS)
    parser.add_argument('--verify-ver2', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-frames', type=int, default=64)
    args = parser.parse_args()
    if args.verify_ver2 or args.benchmark:
        verify_or_benchmark(args.workers, args.benchmark, args.benchmark_frames)
    else:
        run_full_processes(args.workers)
