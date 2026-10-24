"""Processing utilities for pipeline runner."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import time
import traceback
from dataclasses import asdict
import pipeline_paths as P
from pipeline_contracts import connect_bbox
import pipeline_events as events
from bbox_artifacts import write as write_bbox_artifacts
from bbox_artifacts import choose_overlays

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def encoded(value):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    with temp.open('w', encoding='utf-8') as f:
        f.write(encoded(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        rows = list(rows)
        fields = list(dict.fromkeys((k for r in rows for k in r)))
    if not fields:
        fields = ['record_id', 'status']
    tmp = path.with_suffix('.writing')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: encoded(v) if isinstance(v, (dict, list, tuple)) else clean(v) for (k, v) in row.items()})
    os.replace(tmp, path)

def keypath(path):
    return str(Path(path).resolve()).casefold()

def cache(root, key, signature, run, output_resolver=None):
    ident = hashlib.sha256(str(key).encode()).hexdigest()[:24]
    path = Path(root) / '_state' / f'{ident}.json'
    if path.is_file():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['signature'] == signature and (not state.get('retry', False)):
            valid = True
            for (name, sha) in state['outputs'].items():
                output = Path(output_resolver(name, sha) if output_resolver else name)
                if not output.is_file() or digest(output) != sha:
                    valid = False
                    break
            if valid:
                return state['payload']
    (payload, outputs) = run()
    retry = payload.get('status') == 'ERROR'
    if retry:
        details = payload.get('info') or payload.get('row') or payload
        events.report_error(key, details.get('error') or details.get('reason') or details.get('qc') or details, payload.get('traceback', ''))
    save_json(path, dict(signature=signature, payload=payload, retry=retry, outputs={str(p): digest(p) for p in outputs}))
    return payload

def cached_payload(root, key):
    ident = hashlib.sha256(str(key).encode()).hexdigest()[:24]
    return json.loads((Path(root) / '_state' / f'{ident}.json').read_text(encoding='utf-8'))['payload']

def fingerprint(*values):
    return hashlib.sha256(encoded(values).encode()).hexdigest()

def stage_signatures(codes, model_hashes):
    import inspect
    names = {'scale': [P.STAGE_FILES['scale']], 'bright': [P.STAGE_FILES['bright']], 'bbox': [P.STAGE_FILES[k] for k in ('name0', 'name1', 'name2', 'bbox01', 'bbox2')] + ['pipeline_contracts.py', 'bbox_artifacts.py']}
    names['bbox'] += [name for name in codes if name.startswith('_support' + os.sep) and 'cam01_selection_policy' not in name]
    functions = {'scale': scale_stage, 'bright': bright_stage, 'bbox': bbox_stage}
    identity = (inspect.getsource(P.identity), P.NAME_RE.pattern, P.NAME_RE.flags)
    helpers = [inspect.getsource(fn) for fn in (clean, encoded, fingerprint, keypath, digest, cache)]
    settings = {'scale': dict(output=str(P.SCALE_ROOT), input=str(P.OCR_IMAGES)), 'bright': dict(output=str(P.BRIGHT_ROOT), input=str(P.OCR_IMAGES)), 'bbox': dict(device=P.DEVICE, season=P.SEASON_ID, model=model_hashes['lettuce'], identity=identity, source=str(P.BRIGHT_IMAGES), output=str(P.BBOX_ROOT), overlay_limit=P.BBOX_OVERLAY_LIMIT, overlay_seed=P.BBOX_OVERLAY_SEED, save_bbox_crops=P.BBOX_SAVE_CROPS)}
    if P.REUSE_BED:
        for value in settings.values():
            value['profile'] = P.PROFILE
            value['existing_bed_code'] = codes['pipeline_existing_bed.py']
    return {key: fingerprint('stage_cache_v2', {n: codes[n] for n in names[key]}, inspect.getsource(functions[key]), helpers, settings[key]) for key in names}

def inspect_sources():
    if P.REUSE_BED:
        from pipeline_existing_bed import inspect_sources as inspect_existing
        return inspect_existing()
    if not P.INPUT_ROOT.is_dir():
        raise FileNotFoundError(P.INPUT_ROOT)
    if P.INPUT_ROOT == P.OUTPUT_ROOT or P.INPUT_ROOT in P.OUTPUT_ROOT.parents:
        raise ValueError('Output must be outside the source tree')
    for stage in [P.BED_ROOT, P.OCR_ROOT, P.SCALE_ROOT, P.BRIGHT_ROOT, P.NAME_ROOT, P.BBOX_ROOT, P.CROP_ROOT, P.FEATURE_ROOT, P.MANAGEMENT]:
        if stage == P.INPUT_ROOT or stage in P.INPUT_ROOT.parents or P.INPUT_ROOT in stage.parents:
            raise ValueError('Source and stage output paths overlap: ' + str(stage))
    images = sorted((p for p in P.INPUT_ROOT.rglob('*') if p.is_file() and p.suffix.lower() in P.IMAGE_EXTS))
    if not images:
        raise ValueError('No source images')
    paths = {str(p.resolve()) for p in images}
    copy_path = P.OUTPUT_ROOT / '0. 원본 선발' / 'copy_manifest.csv'
    copies = {}
    if copy_path.is_file():
        copies = {str(Path(r['test_path']).resolve()): r for r in read_csv(copy_path)}
        if set(copies) != paths:
            raise ValueError('Invalid input or state in inspect_sources (rejected condition: set(copies) != paths)')
    records = []
    for image in images:
        info = P.identity(image)
        from datetime import datetime
        datetime.strptime(info['capture_date'] + info['capture_time'], '%Y%m%d%H%M%S')
        sha = digest(image)
        old = copies.get(str(image.resolve()), {})
        if old and sha != old['test_sha256']:
            raise ValueError("Invalid input or state in inspect_sources (rejected condition: old and sha != old['test_sha256'])")
        records.append(dict(**info, raw_source_path=str(image.resolve()), raw_sha256=sha, original_key=image.stem, original_source_path=old.get('source_path', str(image.resolve())), relative_path=image.relative_to(P.INPUT_ROOT).as_posix()))
    triplets = defaultdict(list)
    for row in records:
        triplets[row['key']].append(row['camera'])
    duplicates = [k for (k, cameras) in triplets.items() if len(cameras) != len(set(cameras))]
    if duplicates:
        raise ValueError('Duplicate source KEY/camera: ' + str(duplicates[:10]))
    return records

def preflight():
    records = inspect_sources()
    modules = {k: P.load_stage(k) for k in P.STAGE_FILES}
    model_hashes = {}
    for (key, path) in {**({} if P.REUSE_BED else P.BED_MODELS), 'lettuce': P.LETTUCE_MODEL}.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        model_hashes[key] = digest(path)
    if model_hashes['lettuce'] != P.LETTUCE_SHA256:
        raise ValueError("Invalid input or state in preflight (rejected condition: model_hashes['lettuce'] != P.LETTUCE_SHA256)")
    import torch, easyocr, openpyxl, pandas
    import importlib.util
    if P.DEVICE != 'cpu' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable; set LETTUCE_DEVICE=cpu or check the environment')
    ocr_models = CYCLE.OCR_MODEL_DIR
    if not all(((ocr_models / name).is_file() for name in ('craft_mlt_25k.pth', 'english_g2.pth'))):
        raise FileNotFoundError('Local EasyOCR model files required: ' + str(ocr_models))
    codes = {str(f.relative_to(P.CODE_ROOT)): digest(f) for f in P.CODE_ROOT.rglob('*.py') if 'tests' not in f.relative_to(P.CODE_ROOT).parts}
    provenance = json.loads((P.SUPPORT_ROOT / 'sources.json').read_text(encoding='utf-8'))
    for item in provenance:
        if digest(P.CODE_ROOT / item['local']) != item['sha256']:
            raise ValueError("Invalid input or state in preflight (rejected condition: digest(P.CODE_ROOT / item['local']) != item['sha256'])")
    signature = fingerprint(codes, model_hashes, P.SEASON_ID, P.DEVICE, P.PROFILE)
    summary = dict(images=len(records), cameras=dict(Counter((r['camera'] for r in records))), beds=dict(Counter((r['bed_id'] for r in records))), input_root=str(P.INPUT_ROOT), output_root=str(P.OUTPUT_ROOT), modules_imported=len(modules), support_modules=len(provenance), model_hashes=model_hashes, device=P.DEVICE, season_id=P.SEASON_ID, inference=False, generated_images=0, signature=signature, stage_signatures=stage_signatures(codes, model_hashes))
    if P.REUSE_BED:
        summary.update(profile=P.PROFILE, existing_bed_images=sum((bool(r['existing_bed_path']) for r in records)), missing_bed_images=sum((not r['existing_bed_path'] for r in records)), raw_offset_status='UNAVAILABLE_EXISTING_BED', bbox_save_crops=P.BBOX_SAVE_CROPS)
    print(encoded(summary))
    return (records, modules, signature, summary)

def resolve_bed_output(path, sha, journal):
    row = journal.get(keypath(path))
    if row is None:
        return Path(path)
    if row['sha256'] != sha:
        raise ValueError("Invalid input or state in resolve_bed_output (rejected condition: row['sha256'] != sha)")
    candidates = {Path(row[k]) for k in ('old_path', 'current_path', 'new_path') if row.get(k)}
    matches = [p for p in candidates if p.is_file() and digest(p) == sha]
    if len(matches) != 1:
        raise ValueError('Bed image missing, duplicated or changed after OCR rename: ' + str(path))
    return matches[0]

def bed_stage(raw, modules, signature):
    if P.REUSE_BED:
        from pipeline_existing_bed import bed_stage as reuse_beds
        return reuse_beds(raw)
    from tqdm import tqdm
    from ultralytics import YOLO
    import torch
    journal_path = P.OCR_LOG / 'manifest.csv'
    journal = {keypath(r['old_path']): r for r in modules['ocr'].load_manifest(journal_path, [P.OCR_IMAGES.resolve()])} if journal_path.is_file() else {}
    all_rows = []
    for (cameras, key) in [(('cam0', 'cam1'), 'bed01'), (('cam2',), 'bed2')]:
        mod = modules[key]
        model = None
        bed_signature = fingerprint('bed_inplace_ocr_v1', digest(Path(mod.__file__)), digest(mod.MODEL_PATH), P.DEVICE)
        for source in tqdm([r for r in raw if r['camera'] in cameras], desc=key, unit='image'):
            path = Path(source['raw_source_path'])
            target = P.BED_IMAGES / source['camera'] / source['relative_path']

            def run():
                nonlocal model
                if keypath(target) in journal:
                    raise RuntimeError('Bed regeneration conflicts with OCR rename; inspect the bed inputs, settings and journal: ' + str(target))
                try:
                    image = mod.read_image(path)
                    if model is None:
                        model = YOLO(str(mod.MODEL_PATH))
                    result = model.predict(image, imgsz=1024, conf=0.25, iou=0.7, max_det=3 if key == 'bed01' else 5, device=P.DEVICE, half=P.DEVICE != 'cpu', verbose=False)[0]
                    (row, crop, _) = mod.evaluate_prediction(result, image, path, target)
                    row.update(source, bed_status=row['status'], status='OK' if crop is not None else 'REVIEW')
                    if crop is not None:
                        mod.write_image(target, crop)
                    row['bed_sha256'] = digest(target) if crop is not None else ''
                    return (row, [target] if crop is not None else [])
                except Exception as exc:
                    return (dict(**source, target_path=str(target), status='ERROR', bed_status='error', reason=repr(exc), traceback=traceback.format_exc()), [])
            result = cache(P.BED_ROOT, path, fingerprint(bed_signature, source), run, lambda p, h: resolve_bed_output(p, h, journal))
            if result['status'] == 'OK':
                result['current_path'] = str(resolve_bed_output(result['target_path'], result['bed_sha256'], journal))
            all_rows.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_csv(P.BED_ROOT / 'bed_manifest.csv', all_rows)
    return all_rows

def reconcile_ocr_contract(previous, current, mod):
    if previous == clean(current):
        return current
    ignored = {'ocr_code', 'ocr_code_history'}
    before = {k: v for (k, v) in previous.items() if k not in ignored}
    after = {k: v for (k, v) in clean(current).items() if k not in ignored}
    if before != after:
        raise RuntimeError('OCR input changed: preserve prior results and use a new OUTPUT_ROOT')
    old_code = previous.get('ocr_code')
    if old_code != current['ocr_code'] and old_code not in getattr(mod, 'COMPATIBLE_CODE_HASHES', set()):
        raise RuntimeError('Unrecognized OCR code change: explicit compatibility check required')
    history = list(previous.get('ocr_code_history', []))
    if old_code != current['ocr_code'] and old_code not in history:
        history.append(old_code)
    if history:
        current['ocr_code_history'] = history
    return current

def ocr_stage(beds, mod, signature):
    if P.REUSE_BED:
        from pipeline_existing_bed import ocr_stage as reference_ocr
        return reference_ocr(beds, mod, signature)
    successful = [r for r in beds if r['status'] == 'OK']
    if not successful:
        raise RuntimeError('No successful bed segmentation')
    mapping = {}
    for row in successful:
        target = P.OCR_IMAGES / row['camera'] / row['relative_path']
        mapping[keypath(target)] = row
    contract = dict(mode='inplace_bed_rename', source_images={str(r['target_path']): r['bed_sha256'] for r in successful}, ocr_code=digest(Path(mod.__file__)))
    contract_path = P.MANAGEMENT / 'ocr_input_contract.json'
    if contract_path.is_file():
        contract = reconcile_ocr_contract(json.loads(contract_path.read_text(encoding='utf-8')), contract, mod)
    if not (P.OCR_LOG / 'manifest.csv').exists():
        for (oldkey, row) in mapping.items():
            target = P.OCR_IMAGES / row['camera'] / row['relative_path']
            if not target.is_file() or digest(target) != row['bed_sha256']:
                raise ValueError('OCR bed image missing or changed: ' + str(target))
    save_json(contract_path, contract)
    mod.GPU = P.DEVICE
    result = mod.run(str(P.OCR_IMAGES), str(P.OCR_LOG), previews=False)
    frames = []
    for row in read_csv(P.OCR_LOG / 'manifest.csv'):
        bed = mapping[keypath(row['old_path'])]
        current = Path(row['current_path'])
        if not current.is_file() or digest(current) != row['sha256']:
            raise ValueError('OCR current path/checksum mismatch: ' + str(current))
        row.update({k: bed[k] for k in ('raw_source_path', 'raw_sha256', 'original_source_path', 'original_key')})
        row['bed_box'] = [bed[k] for k in ('x1', 'y1', 'x2', 'y2')]
        frames.append(row)
    if len(frames) != len(successful):
        raise ValueError('OCR input/output count mismatch')
    save_csv(P.OCR_ROOT / 'frame_manifest.csv', frames)
    current = {keypath(r['old_path']): r['current_path'] for r in frames}
    for bed in beds:
        if bed['status'] == 'OK':
            bed['current_path'] = current[keypath(bed['target_path'])]
    save_csv(P.BED_ROOT / 'bed_manifest.csv', beds)
    if any((r['status'] in ('READY', 'DISCOVERED', 'RENAME_ERROR', 'COLLISION') for r in frames)):
        raise RuntimeError('OCR rename incomplete or conflicting; inspect frame_manifest.csv')
    return frames

def scale_stage(frames, mod, signature):
    from tqdm import tqdm
    P.SCALE_ROOT.mkdir(parents=True, exist_ok=True)
    P.MANAGEMENT.mkdir(parents=True, exist_ok=True)
    if mod.SAVE_OVERLAYS:
        raise ValueError('Invalid input or state in scale_stage (rejected condition: mod.SAVE_OVERLAYS)')
    db = sqlite3.connect(P.MANAGEMENT / 'scale_state.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS frames(path TEXT PRIMARY KEY, signature TEXT, payload TEXT)')

    def calculate(frame):
        path = Path(frame['current_path'])
        if P.REUSE_BED:
            from pipeline_existing_bed import logical_relative
            relative = logical_relative(frame)
            (row, candidates, audit) = mod.process_final(path, Path(frame['external_bed_root']), P.SCALE_ROOT, image_key=relative.as_posix(), image_name=relative.name)
        else:
            (row, candidates, audit) = mod.process_final(path, P.OCR_IMAGES, P.SCALE_ROOT)
        return dict(row=row, candidates=candidates, audit=audit)
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=mod.WORKERS) as pool, tqdm(total=len(frames), desc='scale', unit='image') as bar:
            for start in range(0, len(frames), mod.CHUNK_SIZE):
                chunk = frames[start:start + mod.CHUNK_SIZE]
                done = {}
                todo = []
                for frame in chunk:
                    key = keypath(frame['current_path'])
                    sig = fingerprint(signature, frame)
                    old = db.execute('SELECT signature,payload FROM frames WHERE path=?', (key,)).fetchone()
                    if old and old[0] == sig and (json.loads(old[1])['row']['scale_source'] != 'ERROR'):
                        done[key] = json.loads(old[1])
                    else:
                        todo.append(frame)
                for (frame, result) in zip(todo, pool.map(calculate, todo)):
                    key = keypath(frame['current_path'])
                    done[key] = result
                    db.execute('INSERT OR REPLACE INTO frames VALUES(?,?,?)', (key, fingerprint(signature, frame), encoded(result)))
                    if result['row']['scale_source'] == 'ERROR':
                        events.report_error(frame['current_path'], result['row'].get('qc', 'scale ERROR'), result['audit'].get('traceback'))
                db.commit()
                for frame in chunk:
                    rows.append(done[keypath(frame['current_path'])]['row'])
                    bar.update()

        def candidates():
            for frame in frames:
                payload = db.execute('SELECT payload FROM frames WHERE path=?', (keypath(frame['current_path']),)).fetchone()[0]
                yield from json.loads(payload)['candidates']
        mod.complete_scale_rows(rows)
        mod.save_scale_csv(rows, P.SCALE_ROOT)
        mod.export_workbook(rows, P.SCALE_ROOT, candidate_data=candidates())
        return {keypath(r['source_path']): r for r in rows}
    finally:
        db.close()

def bright_stage(frames, mod, signature):
    from tqdm import tqdm

    def one(frame):
        path = Path(frame['current_path'])

        def run():
            if P.REUSE_BED:
                from pipeline_existing_bed import logical_relative
                result = mod.work(path, P.BRIGHT_ROOT, mod.SHADOW_STRENGTH, mod.HIGHLIGHT_REDUCTION, output_relative=logical_relative(frame))
            else:
                result = mod.work(path, P.BRIGHT_ROOT, mod.SHADOW_STRENGTH, mod.HIGHLIGHT_REDUCTION, P.OCR_IMAGES)
            return (result, [result['output_path']] if result['status'] == 'OK' else [])
        return cache(P.BRIGHT_ROOT, path, fingerprint(signature, frame), run)
    results = []
    chunk_size = max(1, mod.WORKERS * 2)
    with ThreadPoolExecutor(max_workers=mod.WORKERS) as pool, tqdm(total=len(frames), desc='brightness', unit='image') as bar:
        for start in range(0, len(frames), chunk_size):
            for result in pool.map(one, frames[start:start + chunk_size]):
                results.append(result)
                bar.update()
    save_json(P.BRIGHT_ROOT / 'manifest.json', results)
    save_json(P.BRIGHT_ROOT / 'summary.json', dict(images=len(results), statuses=dict(Counter((r['status'] for r in results)))))
    errors = [r for r in results if r['status'] != 'OK']
    if errors:
        raise RuntimeError(f'Invalid input or state in bright_stage (rejected condition: errors): {len(errors)} ')
    return {keypath(r['source_path']): r for r in results}

def bbox_stage(frames, scales, brights, modules, signature):
    from tqdm import tqdm
    from ultralytics import YOLO
    model = None
    top = modules['bbox01']
    front = modules['bbox2']
    (core, lettuce, cam0, cam1) = top._load_frozen_modules()
    source_keys = []
    summaries = []
    fieldnames = {}
    ids = set()
    output_count = 0
    overlay_paths = choose_overlays([brights[keypath(f['current_path'])]['output_path'] for f in frames], P.BBOX_OVERLAY_LIMIT, P.BBOX_OVERLAY_SEED)
    for frame in tqdm(frames, desc='name / bbox', unit='image'):
        skey = keypath(frame['current_path'])
        bright = brights[skey]
        scale = scales[skey]
        path = Path(bright['output_path'])
        camera = frame['camera']
        save_overlay = str(path) in overlay_paths

        def run():
            nonlocal model
            if model is None:
                model = YOLO(str(P.LETTUCE_MODEL))
                model.overrides['device'] = P.DEVICE
            try:
                image = core.imread_unicode(path)
                if image is None:
                    raise ValueError('Invalid input or state in run (rejected condition: image is None)')
                if camera == 'cam2':
                    (summary, rows) = front.process_image(model, path, P.BRIGHT_IMAGES / 'cam2', image=image)
                    info = asdict(summary)
                    dropped = []
                    if info['error']:
                        events.event('REVIEW', f"run: | Input:{path} | {info['error']}", input_file=str(path))
                        return (dict(status='REVIEW', info=info, rows=[], unexported_slots=[]), write_bbox_artifacts(path, [], P.BBOX_ROOT, save_overlay, image=image, save_crops=P.BBOX_SAVE_CROPS))
                else:
                    (slots, meta) = top._map_one(image, camera, model, lettuce, cam0, cam1)
                    rows = []
                    dropped = []
                    for slot in slots:
                        row = top._slot_to_row(slot, meta, path, camera, image.shape[1], image.shape[0])
                        if row is None:
                            dropped.append(slot.to_dict())
                        else:
                            rows.append(row)
                    info = dict(map_status=meta.get('status'), qc_flags=meta.get('qc_flags'), slot_count=len(slots), unexported_slot_count=len(dropped))
                connected = [connect_bbox(row, frame, scale, bright) for row in rows]
                outputs = write_bbox_artifacts(path, connected, P.BBOX_ROOT, save_overlay, image=image, save_crops=P.BBOX_SAVE_CROPS)
                return (dict(status='OK' if connected else 'REVIEW', info=info, rows=connected, unexported_slots=dropped), outputs)
            except Exception as exc:
                return (dict(status='ERROR', info={'error': repr(exc)}, traceback=traceback.format_exc(), rows=[], unexported_slots=[]), [])
        result = cache(P.BBOX_ROOT, path, fingerprint(signature, frame, bright, scale, save_overlay), run)
        source_keys.append(path)
        for row in result['rows']:
            if row['record_id'] in ids:
                raise ValueError("Invalid input or state in bbox_stage (rejected condition: row['record_id'] in ids)")
            ids.add(row['record_id'])
            fieldnames.update(dict.fromkeys(row))
            output_count += 1
        summaries.append(dict(raw_source_path=frame['raw_source_path'], source_path=str(path), camera=camera, old_key=frame['old_key'], new_key=frame.get('new_key') or frame['old_key'], status=result['status'], exported_slots=len(result['rows']), details=result['info'], unexported_slots=result['unexported_slots']))

    def output_rows():
        for key in source_keys:
            yield from cached_payload(P.BBOX_ROOT, key)['rows']
    save_csv(P.BBOX_MANIFEST, output_rows(), list(fieldnames))
    save_csv(P.NAME_ROOT / 'frame_summary.csv', summaries)
    fields = ['record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'legacy_slot_id', 'slot_namespace_version', 'anchor_x', 'anchor_y', 'mapping_version', 'source_path', 'raw_source_path', 'old_key', 'new_key', 'ocr_action']
    save_csv(P.NAME_ROOT / 'slot_manifest.csv', ({k: r.get(k, '') for k in fields} for r in output_rows()), fields)
    errors = [r for r in summaries if r['status'] == 'ERROR']
    if errors:
        raise RuntimeError(f'Invalid input or state in bbox_stage (rejected condition: errors): {len(errors)} ')
    if not output_count:
        raise RuntimeError('Invalid input or state in bbox_stage (rejected condition: not output_count)')
    return output_count

def reconcile(raw, beds, frames):
    bedmap = {keypath(r['raw_source_path']): r for r in beds}
    frame_map = {keypath(r['raw_source_path']): r for r in frames}
    byraw = defaultdict(Counter)
    feature_path = P.FEATURE_ROOT / 'features_all.csv'
    if feature_path.exists():
        with feature_path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if not row.get('raw_source_path'):
                    raise ValueError("Invalid input or state in reconcile (rejected condition: not row.get('raw_source_path'))")
                byraw[keypath(row['raw_source_path'])][row['feature_status']] += 1
    if set(byraw) - set(bedmap):
        raise ValueError('Invalid input or state in reconcile (rejected condition: set(byraw) - set(bedmap))')
    ledger = []
    for source in raw:
        k = keypath(source['raw_source_path'])
        bed = bedmap[k]
        ocr = frame_map.get(k, {})
        fs = byraw[k]
        ledger.append(dict(**source, bed_status=bed['bed_status'], bed_reason=bed.get('reason', ''), ocr_status=ocr.get('ocr_status', 'NOT_RUN'), ocr_action=ocr.get('status', 'NOT_RUN'), new_key=ocr.get('new_key', ''), feature_rows=sum(fs.values()), features_ok=fs['OK'], feature_status_counts=dict(fs)))
    save_csv(P.MANAGEMENT / 'input_outcomes.csv', ledger)
    return ledger

def output_report():
    from output_inventory import collect, summarize
    target = P.MANAGEMENT / 'output_inventory.csv'
    temporary = target.with_suffix('.writing')
    fields = ['relative_path', 'stage', 'kind', 'bytes', 'data_rows', 'sheet_rows']
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()

        def records():
            for row in collect(P.OUTPUT_ROOT, excluded_roots=[P.INPUT_ROOT, temporary]):
                writer.writerow({**row, 'sheet_rows': encoded(row['sheet_rows'])})
                yield row
        counts = summarize(records())
    os.replace(temporary, target)
    counts['inventory_scope'] = 'Excludes input/selection files and two inventory files; includes execution records'
    save_json(P.MANAGEMENT / 'output_counts.json', counts)
    print(f"output report: Output inventory complete |{counts['total_images']:,}output report: | CSV{counts['counts_by_kind'].get('csv', 0)}output report: | Excel{counts['counts_by_kind'].get('xlsx', 0)}output report: ")
    print('Output inventory: ' + str(target))
    return counts

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='main: /feature .')
    parser.add_argument('--until', choices=['bed', 'ocr', 'scale', 'bright', 'bbox', 'crop', 'feature', 'growth_qc'], default='growth_qc')
    parser.add_argument('--from', dest='start_from', choices=['crop', 'feature', 'growth_qc'], help='main: 6/7/8 . Source')
    args = parser.parse_args(argv)
    if args.start_from:
        return resume_from_stage(args)
    (raw, modules, signature, summary) = preflight()
    if not args.run:
        return 0
    stage_keys = summary.get('stage_signatures', {})
    P.MANAGEMENT.mkdir(parents=True, exist_ok=True)
    events.configure(P.MANAGEMENT)
    import msvcrt
    with (P.MANAGEMENT / '.run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        source_contract = P.MANAGEMENT / 'source_contract.json'
        contract = dict(input_root=str(P.INPUT_ROOT), season_id=P.SEASON_ID, images={r['raw_source_path']: r['raw_sha256'] for r in raw})
        if P.REUSE_BED:
            contract.update(profile=P.PROFILE, bed_images={r['existing_bed_path']: r['existing_bed_sha256'] for r in raw if r['existing_bed_path']})
        if source_contract.exists() and json.loads(source_contract.read_text(encoding='utf-8')) != clean(contract):
            raise ValueError("Invalid input or state in main (rejected condition: source_contract.exists() and json.loads(source_contract.read_text(encoding='utf-8')) != clean(contract))")
        save_json(source_contract, contract)
        save_json(P.MANAGEMENT / 'preflight.json', summary)

        def checkpoint(stage):
            stop = args.until == stage
            save_json(P.MANAGEMENT / 'run_summary.json', dict(state='stopped_at_requested_stage' if stop else 'running', completed_through=stage, requested_until=args.until, signature=signature))
            return stop

        def execute(label, function, *values):
            events.start(label)
            started = time.monotonic()
            result = function(*values)
            events.event('COMPLETE', f'Stage complete ({time.monotonic() - started:.1f} s)')
            return result

        def checked_bed():
            result = bed_stage(raw, modules, signature)
            if any((r['status'] == 'ERROR' for r in result)):
                raise RuntimeError('Bed segmentation failed; inspect bed_manifest.csv before resuming')
            return result

        def checked_main(module, arguments):
            if module.main(arguments):
                raise RuntimeError(f'Invalid input or state in checked_main (rejected condition: module.main(arguments)): {Path(module.__file__).name} ')
        try:
            save_json(P.MANAGEMENT / 'run_summary.json', dict(state='running', requested_until=args.until, signature=signature))
            beds = execute('1 | Existing bed inputs' if P.REUSE_BED else '1 | Bed segmentation', checked_bed)
            if checkpoint('bed'):
                return 0
            frames = execute('2 | Bed-number OCR', ocr_stage, beds, modules['ocr'], signature)
            if checkpoint('ocr'):
                return 0
            scales = execute('3 | Scale estimation', scale_stage, frames, modules['scale'], stage_keys.get('scale', signature))
            if checkpoint('scale'):
                return 0
            bright = execute('4 | Brightness normalization', bright_stage, frames, modules['bright'], stage_keys.get('bright', signature))
            if checkpoint('bright'):
                return 0
            execute('5-6 | Slot assignment and bounding boxes', bbox_stage, frames, scales, bright, modules, stage_keys.get('bbox', signature))
            if checkpoint('bbox'):
                return 0
            execute('7 | Plant crop extraction', checked_main, modules['crop'], ['--input', str(P.BBOX_MANIFEST), '--output', str(P.CROP_ROOT), '--device', P.DEVICE])
            if checkpoint('crop'):
                return 0
            execute('8 | Feature extraction', checked_main, modules['feature'], ['--input', str(P.CROP_ROOT / 'crop_manifest.csv'), '--output', str(P.FEATURE_ROOT)])
            ledger = reconcile(raw, beds, frames)
            if args.until != 'feature':
                execute('9 | Growth QC', checked_main, modules['growth_qc'], ['--input', str(P.FEATURE_ROOT / 'crop_features.xlsx'), '--output', str(P.OUTPUT_ROOT / '9. 생육QC')])
            save_json(P.MANAGEMENT / 'run_summary.json', dict(state='complete', input_images=len(raw), feature_rows=sum((r['feature_rows'] for r in ledger)), completed_through=args.until, signature=signature))
            execute('Output inventory', output_report)
        except BaseException as exc:
            events.report_error(P.INPUT_ROOT, exc)
            save_json(P.MANAGEMENT / 'run_summary.json', dict(state='interrupted' if isinstance(exc, KeyboardInterrupt) else 'needs_review', error=repr(exc), signature=signature))
            raise
    return 0
ORDER = ('bed', 'ocr', 'scale', 'bright', 'bbox', 'crop', 'feature', 'growth_qc')
LABELS = {'bbox': '5-6 | Slot assignment and bounding boxes', 'crop': '7 | Plant crop extraction', 'feature': '8 | Feature extraction', 'growth_qc': '9 | Growth QC'}

def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required resume file is missing: ' + str(path))
    return path

def read_table(path):
    path = require_file(path)
    if path.suffix == '.xlsx':
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            header = next(rows)
            return [dict(zip(header, ('' if v is None else v for v in row))) for row in rows]
        finally:
            workbook.close()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def inspect_resume(start, until):
    if ORDER.index(start) > ORDER.index(until):
        raise ValueError('--until must be at or after --from.')
    contract = json.loads(require_file(P.MANAGEMENT / 'source_contract.json').read_text(encoding='utf-8'))
    if contract.get('profile', {}) != P.PROFILE or contract.get('season_id') != P.SEASON_ID or keypath(contract['input_root']) != keypath(P.INPUT_ROOT):
        raise ValueError("Invalid input or state in inspect_resume (rejected condition: contract.get('profile', {}) != P.PROFILE or contract.get('season_id') != P.SEASON_ID or keypath(contract['input_root']) != keypath(P.INPUT_ROOT))")
    prerequisite = ORDER[ORDER.index(start) - 1]
    last_state = None
    with require_file(P.MANAGEMENT / 'events.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('stage') == LABELS[prerequisite] and event.get('kind') in ('START', 'COMPLETE', 'ERROR'):
                last_state = event['kind']
    if last_state != 'COMPLETE':
        raise ValueError(f"Invalid input or state in inspect_resume (rejected condition: last_state != 'COMPLETE'): {LABELS[prerequisite]} ")
    inputs = {'crop': P.BBOX_MANIFEST, 'feature': P.CROP_ROOT / 'crop_manifest.csv', 'growth_qc': P.FEATURE_ROOT / 'crop_features.xlsx'}
    source = require_file(inputs[start])
    count = 0
    checked_sources = set()
    if source.suffix == '.csv':
        with source.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'record_id', 'raw_source_path', 'source_path'} if start == 'crop' else {'record_id', 'crop_path'}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f'{source}: Missing required columns {sorted(missing)}')
            for row in reader:
                if not row.get('record_id'):
                    raise ValueError(f'{source}: Empty record_id at row {reader.line_num}')
                if start == 'crop':
                    image = row.get('source_path', '')
                    if not image:
                        raise ValueError(f'{source}: Empty source_path at row {reader.line_num}')
                    if image not in checked_sources:
                        require_file(image)
                        checked_sources.add(image)
                count += 1
        if not count:
            raise ValueError('Invalid input or state in inspect_resume (rejected condition: not count)')
    beds = read_table(P.BED_ROOT / ('bed_reference.xlsx' if P.REUSE_BED else 'bed_manifest.csv'))
    frames = read_table(P.OCR_ROOT / 'frame_manifest.csv')
    byraw = {keypath(r['raw_source_path']): r for r in beds}
    expected = {keypath(k): v for (k, v) in contract['images'].items()}
    if len(byraw) != len(beds) or set(byraw) != set(expected):
        raise ValueError('Invalid input or state in inspect_resume (rejected condition: len(byraw) != len(beds) or set(byraw) != set(expected))')
    if any((r.get('raw_sha256') != expected[k] for (k, r) in byraw.items())):
        raise ValueError("Invalid input or state in inspect_resume (rejected condition: any((r.get('raw_sha256') != expected[k] for (k, r) in byraw.items())))")
    summary = dict(start_from=start, requested_until=until, input_manifest=str(source), input_rows=count if source.suffix == '.csv' else None, original_images=len(beds), skipped_stages=list(ORDER[:ORDER.index(start)]), source_images_rechecked=False, inference=False)
    print(encoded(summary), flush=True)
    return (beds, frames, summary)

def resume_from_stage(args):
    import pipeline_events as events
    (beds, frames, summary) = inspect_resume(args.start_from, args.until)
    if not args.run:
        return 0
    import msvcrt
    with (P.MANAGEMENT / '.run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        events.configure(P.MANAGEMENT)
        completed = ORDER[ORDER.index(args.start_from) - 1]
        signature = fingerprint(summary, {k: digest(P.CODE_ROOT / P.STAGE_FILES[k]) for k in ORDER[ORDER.index(args.start_from):ORDER.index(args.until) + 1]})

        def checkpoint(state, **extra):
            save_json(P.MANAGEMENT / 'run_summary.json', dict(state=state, completed_through=completed, requested_until=args.until, start_from=args.start_from, signature=signature, **extra))
        try:
            checkpoint('running')
            arguments = {'crop': ['--input', str(P.BBOX_MANIFEST), '--output', str(P.CROP_ROOT), '--device', P.DEVICE], 'feature': ['--input', str(P.CROP_ROOT / 'crop_manifest.csv'), '--output', str(P.FEATURE_ROOT)], 'growth_qc': ['--input', str(P.FEATURE_ROOT / 'crop_features.xlsx'), '--output', str(P.OUTPUT_ROOT / '9. 생육QC')]}
            for stage in ORDER[ORDER.index(args.start_from):ORDER.index(args.until) + 1]:
                events.start(LABELS[stage])
                started = time.monotonic()
                module = P.load_stage(stage)
                if module.main(arguments[stage]):
                    raise RuntimeError(f'Invalid input or state in resume_from_stage (rejected condition: module.main(arguments[stage])): {P.STAGE_FILES[stage]} ')
                events.event('COMPLETE', f'Stage complete ({time.monotonic() - started:.1f} s)')
                completed = stage
                checkpoint('running')
            if ORDER.index(args.until) >= ORDER.index('feature'):
                contract = json.loads((P.MANAGEMENT / 'source_contract.json').read_text(encoding='utf-8'))
                raw = [{'raw_source_path': path, 'raw_sha256': digest} for (path, digest) in contract['images'].items()]
                reconcile(raw, beds, frames)
            output_report()
            checkpoint('complete' if args.until == 'growth_qc' else 'stopped_at_requested_stage')
        except BaseException as exc:
            events.report_error(summary['input_manifest'], exc)
            checkpoint('interrupted' if isinstance(exc, KeyboardInterrupt) else 'needs_review', error=repr(exc))
            raise
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
