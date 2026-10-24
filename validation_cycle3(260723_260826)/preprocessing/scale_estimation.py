"""Scale estimation calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CODE = CYCLE.THIRD_ROOT / 'preprocessing/4. Estimate_scale.py'
ROOT = CYCLE.PREPROCESS_OUTPUT
BRIGHT_ROOT = ROOT / '3. 밝기보정과 Scale'
MANIFEST = BRIGHT_ROOT / '_실행기록' / '261009_밝기보정' / 'brightness_manifest.csv'
ROI_MANIFESTS = [ROOT / '1. ROI_BOX' / '_실행기록' / '261005_윗면_CAM01_전체' / 'roi_manifest.csv', ROOT / '1. ROI_BOX' / '_실행기록' / '261005_정면_CAM2_재생성' / 'roi_manifest.csv']
PILOT_SOURCE = CYCLE.SOURCE_ROOT / '3. 밝기보정과 Scale/_실행기록/261009_SCALE_표본점검/261009_SCALE_표본점검.json'
RUN_DIR = BRIGHT_ROOT / '_실행기록' / '261009_SCALE'
(WORKERS, CHUNK_SIZE, EXPECTED) = (8, 48, 39401)
VERSION = 'season4_261009_x_window_fallback_native_groups'
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
sys.dont_write_bytecode = True
import cv2
import numpy as np
from tqdm import tqdm

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_reference():
    spec = resolved_spec('scale_frozen_season3', REFERENCE_CODE)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cv2.setNumThreads(1)
    return ref

def load_inputs():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED or any((r['label_accepted'] != 'True' or r['status'] != 'OK' for r in rows)):
        raise ValueError('brightness accepted/status/count contract mismatch')
    metadata = {}
    for path in ROI_MANIFESTS:
        with path.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if not r['status'].startswith('SAVED'):
                    continue
                metadata[r['source_path'].casefold()] = dict(native_width=int(r['image_width']), native_height=int(r['image_height']), bbox=json.loads(r.get('crop_bbox') or r.get('bbox')), polygon=json.loads(r.get('detected_polygon') or r.get('polygon')))
    seen = set()
    for r in rows:
        p = Path(r['output_path'])
        key = p.relative_to(BRIGHT_ROOT).as_posix()
        if key in seen or p.parent.parent.name.upper() != r['camera']:
            raise ValueError('duplicate key or camera path mismatch: ' + key)
        seen.add(key)
        m = metadata[r['raw_source_path'].casefold()]
        if list(map(int, json.loads(r['crop_bbox']))) != list(map(int, m['bbox'])):
            raise ValueError('crop bbox mismatch: ' + key)
        r.update(metadata=m, image_key=key)
    return rows

def broaden(poly, shape, target):
    (h, w) = shape
    q = poly.astype(float).copy()
    (left, right) = (20, w - 21)
    for (i, j, x) in [(0, 1, left), (3, 2, left), (1, 0, right), (2, 3, right)]:
        slope = (poly[j, 1] - poly[i, 1]) / (poly[j, 0] - poly[i, 0])
        q[i, 1] = poly[i, 1] + slope * (x - poly[i, 0])
        q[i, 0] = x
    q[:, 1] = np.clip(q[:, 1], 0, h - 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [q.astype(np.int32)], 255)
    target_mask = np.zeros_like(mask)
    cv2.fillPoly(target_mask, [target.astype(np.int32)], 255)
    return (cv2.bitwise_and(mask, target_mask), q.astype(np.int32))

def process(r, ref, with_image=False):
    start = time.perf_counter()
    (key, path) = (r['image_key'], Path(r['output_path']))
    row = dict.fromkeys(ref.SCALE_FIELDS)
    m = r['metadata']
    row.update(image_key=key, image_name=path.name, camera=r['camera'].lower(), source_path=str(path), detected=False, n_candidates=0, scale_ready=False, scale_verified=False, scale_version=VERSION, reference_count=0, scale_source='UNRESOLVED', capture_key=r['capture_key'], date=r['date'], accepted_number=r['accepted_number'], label_accepted=True, label_source=r['label_source'], ocr_qc_status=r['qc_status'], identity_verified=r['identity_verified'], label_review_reasons=r['label_review_reasons'], source_camera=r['source_camera'], native_width=m['native_width'], native_height=m['native_height'], crop_bbox=r['crop_bbox'], roi_source_path=r['source_path'], raw_source_path=r['raw_source_path'], roi_source_sha256=r['source_sha256'], input_hash_verified=False, window_mode='BASELINE', baseline_n_candidates=0, accepted_diameter_range_ratio=None)
    audit = dict(image_key=key)
    out = []
    im = None
    try:
        stat_before = path.stat()
        data = path.read_bytes()
        row['source_sha256'] = hashlib.sha256(data).hexdigest()
        if row['source_sha256'] != r['output_sha256']:
            raise ValueError('brightness output SHA mismatch')
        row['input_hash_verified'] = True
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        del data
        if im is None:
            raise ValueError('cannot decode image')
        (h, w) = im.shape[:2]
        row.update(image_width=w, image_height=h)
        bbox = m['bbox']
        if (w, h) != (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])):
            raise ValueError('native crop dimensions do not match decoded image')
        target = np.asarray(m['polygon'], dtype=np.float32).reshape(-1, 2) - np.array(bbox[:2], np.float32)
        (roi, poly, geometry, green) = ref.search_region(im, row['camera'])
        candidates = ref.detect_holes(im, roi, green)
        baseline = sum((c['accepted'] for c in candidates))
        row['baseline_n_candidates'] = baseline
        final_poly = poly
        if not baseline and row['camera'] in ('cam0', 'cam1'):
            (expanded_mask, expanded_poly) = broaden(poly, (h, w), target)
            extra = ref.detect_holes(im, expanded_mask, green)
            if any((c['accepted'] for c in extra)):
                (candidates, final_poly) = (extra, expanded_poly)
                row['window_mode'] = 'X_EXPANDED_ON_MISS_TARGET_MASK'
            else:
                row['window_mode'] = 'X_EXPANDED_ON_MISS_NO_RECOVERY'
        accepted = sorted((c for c in candidates if c['accepted']), key=lambda c: c['id'])
        selected = max(accepted, key=lambda c: c['score']) if accepted else None
        (xl, yl) = map(int, poly[3])
        (xr, yr) = map(int, poly[2])
        if xr <= xl:
            raise ValueError('invalid baseline lower edge')
        angle = math.degrees(math.atan2(yr - yl, xr - xl))
        legacy_group = ref.geometry_group(row['camera'], angle, w)
        group = f"{legacy_group}__native{m['native_width']}x{m['native_height']}__{r['source_camera'].lower()}"
        row.update(bottom_left_x=xl, bottom_left_y=yl, bottom_right_x=xr, bottom_right_y=yr, bottom_angle_deg=angle, legacy_geometry_group=legacy_group, geometry_group=group, detected=bool(accepted), n_candidates=len(accepted))
        for (no, c) in enumerate(accepted, 1):
            cand = dict(zip(ref.CANDIDATE_FIELDS, [key, path.name, row['camera'], f'candidate_{no}', f"H{c['id']}", c is selected, c['cx'], c['cy'], c['major_px'], c['minor_px'], 50.0 / c['major_px'], c['angle_deg'], c['score']]))
            out.append(cand)
            if c is selected:
                row.update(selected=cand['candidate'], selected_overlay_id=cand['overlay_id'], selected_x_px=c['cx'], selected_y_px=c['cy'], selected_diameter_px=c['major_px'], selected_minor_px=c['minor_px'], selected_mm_per_px=50.0 / c['major_px'])
        flags = []
        if any(('REVIEW' in v.get('method', '') for v in geometry.values())):
            flags.append('STRUCTURE_WINDOW_REVIEW')
        if geometry['top'].get('framing_review'):
            flags.append('OTHER_FRAMING')
        if row['window_mode'] == 'X_EXPANDED_ON_MISS_TARGET_MASK':
            flags.append('SEARCH_X_EXPANDED')
        if selected and cv2.pointPolygonTest(target, (selected['cx'], selected['cy']), False) < 0:
            flags.append('SELECTED_OUTSIDE_TARGET_POLYGON_REVIEW')
        diameters = [c['major_px'] for c in accepted]
        if len(diameters) > 1:
            row['accepted_diameter_range_ratio'] = (max(diameters) - min(diameters)) / statistics.median(diameters)
            if row['accepted_diameter_range_ratio'] > 0.25:
                flags.append('WITHIN_FRAME_DIAMETER_SPREAD_GT25PCT')
        row['qc'] = ';'.join(flags)
        after = path.stat()
        if (stat_before.st_size, stat_before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('input file changed during processing')
        audit.update(baseline_polygon=poly.tolist(), roi_polygon=final_poly.tolist(), target_polygon=target.tolist(), structure=geometry, accepted_candidates=accepted, rejected_flag_counts=dict(Counter((f for c in candidates if not c['accepted'] for f in c.get('flags', [])))), selected_id=selected['id'] if selected else None)
    except Exception as e:
        row.update(scale_source='ERROR', detected=False, n_candidates=0, qc=f'{type(e).__name__}: {e}')
        out = []
        audit['error'] = row['qc']
    row['seconds'] = time.perf_counter() - start
    result = dict(row=row, candidates=out, audit=audit)
    if with_image:
        result['image'] = im
    return result

def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def init_process_worker():
    global PROCESS_REFERENCE
    PROCESS_REFERENCE = load_reference()

def process_worker(row):
    return process(row, PROCESS_REFERENCE)

def pilot(inputs, ref):
    report = json.loads(PILOT_SOURCE.read_text(encoding='utf-8'))
    previous = {s['bright_path']: s for g in report['groups'] for s in g['samples']}
    names = {'bed76_20260818_020051_cam2_rf.png', 'bed33_20260818_025922_cam2.png', 'bed75_20260818_034543_cam2.png'}
    picks = [r for r in inputs if r['output_path'] in previous or Path(r['output_path']).name in names]
    if len(picks) != 30:
        raise ValueError('pilot input selection mismatch')
    directory = RUN_DIR / 'pilot'
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 25)
    for r in picks:
        result = process(r, ref, with_image=True)
        row = result['row']
        prior = previous.get(r['output_path'])
        if prior and prior['bright_result']['detected']:
            if row['selected_diameter_px'] != prior['bright_result']['selected_diameter_px']:
                raise ValueError('previous successful selection changed: ' + row['image_key'])
        im = result.pop('image')
        if im is not None and 'roi_polygon' in result['audit']:
            annotated = ref.overlay(im, np.asarray(result['audit']['roi_polygon'], np.int32), result['audit']['accepted_candidates'], row['window_mode'])
            view = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            view.thumbnail((1900, 1100))
            canvas = Image.new('RGB', (1900, view.height + 100), '#f4f5f6')
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 8), Path(row['source_path']).name, font=font, fill='#17222a')
            draw.text((15, 44), f"{row['n_candidates']} holes | {row['selected_diameter_px']} px | {row['qc']}", font=font, fill='#17222a')
            canvas.paste(view, (0, 100))
            canvas.save(directory / (Path(row['image_name']).stem + '_scale.png'))
        results.append(result)
    summary = dict(total=len(results), detected=sum((x['row']['detected'] for x in results)), unchanged_previous_successes=26, source_hashes_verified=all((x['row']['input_hash_verified'] for x in results)), errors=[x['row']['qc'] for x in results if x['row']['scale_source'] == 'ERROR'], results=results, production=False)
    write_json(directory / 'pilot_summary.json', summary)
    print(json.dumps({k: v for (k, v) in summary.items() if k != 'results'}, ensure_ascii=False))
    if summary['errors'] or summary['detected'] != 30:
        raise ValueError('pilot failed')

def finish(results, ref):
    rows = [x['row'] for x in results]
    for row in rows:
        if row['detected']:
            row.update(scale_source='DIRECT', scale_ready=True, diameter_px=row['selected_diameter_px'], mm_per_px=50.0 / row['selected_diameter_px'])
    outliers = ref.flag_direct_group_outliers(rows)
    excluded = set()
    for row in rows:
        if row['detected'] and any((flag in row['qc'] for flag in ('DIRECT_DIAMETER_OUTSIDE_GROUP_30PCT', 'SELECTED_OUTSIDE_TARGET_POLYGON_REVIEW'))):
            excluded.add(row['image_key'])
    seeds = [row for row in rows if row['detected'] and row['image_key'] not in excluded]
    missing = [row for row in rows if not row['detected']]
    ref.complete_scale_rows(seeds + missing)
    by_key = {r['image_key']: r for r in rows}
    for row in rows:
        row['reference_seed_eligible'] = row['detected'] and row['image_key'] not in excluded
    for row in rows:
        for i in range(1, row['reference_count'] + 1):
            rr = by_key[row[f'reference_{i}']]
            if rr['geometry_group'] != row['geometry_group'] or rr['scale_source'] != 'DIRECT' or (not rr['reference_seed_eligible']):
                raise ValueError('invalid reference linkage')
        if row['reference_count'] and any((by_key[row[f'reference_{i}']]['qc'] for i in range(1, row['reference_count'] + 1))):
            row['qc'] += (';' if row['qc'] else '') + 'REFERENCE_HAS_QC_REVIEW'
        row['scale_qc'] = row.get('qc') or ''
        if row['scale_ready'] and (not row['diameter_px'] or not math.isclose(row['mm_per_px'] * row['diameter_px'], 50, abs_tol=1e-10)):
            raise ValueError('50mm identity failed')
    if len(rows) != EXPECTED or len(by_key) != EXPECTED:
        raise ValueError('final row count/uniqueness failed')
    fields = ref.SCALE_FIELDS + list(dict.fromkeys((k for r in rows for k in r if k not in ref.SCALE_FIELDS)))
    for (name, headers, records) in [('scale_map.csv', fields, rows), ('candidates.csv', ref.CANDIDATE_FIELDS, (c for x in results for c in x['candidates']))]:
        path = RUN_DIR / name
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temp, path)
    groups = defaultdict(Counter)
    for r in rows:
        groups[r['geometry_group']][r['scale_source']] += 1
    summary = dict(version=VERSION, created_at=datetime.now().isoformat(timespec='seconds'), input_images=len(rows), camera_counts=dict(Counter((r['camera'] for r in rows))), status_counts=dict(Counter((r['scale_source'] for r in rows))), window_counts=dict(Counter((r['window_mode'] for r in rows))), qc_counts=dict(Counter((flag for r in rows for flag in (r['qc'] or '').split(';') if flag))), geometry_groups={k: dict(v) for (k, v) in groups.items()}, direct_group_outliers=outliers, reference_seed_excluded=len(excluded), candidates=sum((r['n_candidates'] for r in rows)), hashes_verified=sum((r['input_hash_verified'] for r in rows)), physical_scale_verified=False, real_diameter_mm=50, diameter_definition='ellipse_major_axis', images_resized=False, reference_rows_max=3, input_rows_preserved=True, errors=[dict(image_key=r['image_key'], error=r['qc']) for r in rows if r['scale_source'] == 'ERROR'])
    write_json(RUN_DIR / 'summary.json', summary)
    write_json(RUN_DIR / 'workbook_data.json', dict(scale_fields=ref.SCALE_FIELDS, candidate_fields=ref.CANDIDATE_FIELDS, rows=[{k: r[k] for k in ref.SCALE_FIELDS} for r in rows], candidates=[c for x in results for c in x['candidates']], summary=summary))
    write_json(RUN_DIR / 'completion.json', dict(status='SCALE_CALCULATED_XLSX_PENDING', **summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args([] if 'ipykernel' in sys.modules else None)
    ref = load_reference()
    inputs = load_inputs()
    if args.pilot:
        return pilot(inputs, ref)
    pilot_path = RUN_DIR / 'pilot' / 'pilot_summary.json'
    if not pilot_path.exists() or json.loads(pilot_path.read_text(encoding='utf-8'))['detected'] != 30:
        raise ValueError('run --pilot first')
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    contract = dict(script_sha256=digest(__file__), reference_sha256=digest(REFERENCE_CODE), brightness_manifest_sha256=digest(MANIFEST), roi_manifest_sha256=[digest(p) for p in ROI_MANIFESTS], real_diameter_mm=50, total=EXPECTED, workers=WORKERS, chunk_size=CHUNK_SIZE, input=str(BRIGHT_ROOT), output=str(RUN_DIR), policy='Frozen hole conditions; X expansion on CAM0/1 misses only, clipped by target polygon; native resolution/source-camera/legacy geometry reference groups; direct values preserved; outlier/outside-target seeds excluded; no height correction')
    cp = RUN_DIR / 'execution_contract.json'
    if cp.exists() and json.loads(cp.read_text(encoding='utf-8')) != contract:
        raise ValueError('resume contract changed')
    lock = RUN_DIR / 'run.lock'
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        write_json(cp, contract)
        completed = {}
        journal_paths = sorted(RUN_DIR.glob('detection_records*.jsonl'))
        for journal in journal_paths:
            valid_end = 0
            with journal.open('rb') as f:
                while (line := f.readline()):
                    try:
                        record = json.loads(line.decode('utf-8'))
                        key = record['row']['image_key']
                        if key in completed:
                            raise ValueError('duplicate resume record')
                        completed[key] = record
                        valid_end = f.tell()
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        break
            with journal.open('r+b') as f:
                f.truncate(valid_end)
        input_keys = {r['image_key'] for r in inputs}
        if any((k not in input_keys for k in completed)):
            raise ValueError('unknown resume key')
        pending = [r for r in inputs if r['image_key'] not in completed]
        chunk_number = len(list(RUN_DIR.glob('detection_records_chunk*.jsonl')))
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_process_worker) as pool, tqdm(total=EXPECTED, initial=len(completed), desc='SCALE', unit='image', mininterval=10) as bar:
            for start in range(0, len(pending), CHUNK_SIZE):
                batch = list(pool.map(process_worker, pending[start:start + CHUNK_SIZE]))
                chunk_number += 1
                destination = RUN_DIR / f'detection_records_chunk{chunk_number:05d}.jsonl'
                temp = destination.with_suffix('.jsonl.tmp')
                payload = ''.join((json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n' for result in batch))
                for attempt in range(3):
                    try:
                        temp.write_text(payload, encoding='utf-8')
                        os.replace(temp, destination)
                        break
                    except PermissionError:
                        if attempt == 2:
                            raise
                        time.sleep(2)
                for result in batch:
                    completed[result['row']['image_key']] = result
                    bar.update(1)
        for (path, expected) in [(REFERENCE_CODE, contract['reference_sha256']), (MANIFEST, contract['brightness_manifest_sha256'])]:
            if digest(path) != expected:
                raise ValueError('upstream code/manifest changed')
        finish([completed[r['image_key']] for r in inputs], ref)
    finally:
        lock.unlink(missing_ok=True)
if __name__ == '__main__':
    main()
