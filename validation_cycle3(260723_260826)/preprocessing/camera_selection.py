"""Camera selection calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import cv2
import numpy as np
import cam01_selection_policy as policy
VERSION = 'cam01_quality_selection_260905_adapted_v1'
METRICS = ['inference_status', 'target_conf', 'center_distance_norm', 'edge_margin_norm', 'mask_edge_touch', 'mask_aspect_ratio', 'mask_area_ratio_input', 'reference_level', 'reference_n', 'aspect_robust_z', 'size_robust_z', 'hard_valid', 'quality_score', 'quality_flags']
SELECTION_FIELDS = ['observation_view', 'selection_status', 'selection_reason', 'selected_camera', 'selected_record_id', 'default_camera', 'track_primary_camera', 'track_primary_reason', 'track_dual_n', 'track_cam0_hard_valid_rate', 'track_cam1_hard_valid_rate', 'track_cam0_median_quality', 'track_cam1_median_quality', 'selection_version', 'selected_quality_score', 'selected_hard_valid']

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def dumps(value):
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False)

def measure(record, mask, box):
    if mask is None or not mask.any():
        return dict(inference_status='NO_DETECTION')
    (ys, xs) = np.where(mask)
    (height, width) = mask.shape
    short = max(1, min(width, height))
    (contours, _) = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moment = cv2.moments(max(contours, key=cv2.contourArea))
    if moment['m00'] <= 0:
        return dict(inference_status='EMPTY_TARGET_MASK')
    cx = moment['m10'] / moment['m00'] + box[0]
    cy = moment['m01'] / moment['m00'] + box[1]
    original = record.get('input_record_json', {})
    if isinstance(original, str):
        original = json.loads(original)

    def point(name, fallback):
        try:
            return float(original.get(name))
        except (TypeError, ValueError):
            return fallback
    px = point('slot_center_x', record.get('anchor_x', cx))
    py = point('slot_center_y', record.get('anchor_y', cy))
    if record.get('source_coordinate_space') == 'raw_camera_pixels':
        px += float(record['bed_x1_raw'])
        py += float(record['bed_y1_raw'])
    margin = min(int(xs.min()), int(ys.min()), width - int(xs.max()) - 1, height - int(ys.max()) - 1)
    blocked = any((s in str(record.get('status', '')) for s in ('NEIGHBOR_CLAIM', 'NEIGHBOR_OWNERSHIP', 'IDENTITY_CONFLICT', 'AMBIGUOUS')))
    return dict(inference_status='IDENTITY_OR_NEIGHBOR_REVIEW' if blocked else 'OK', target_conf=float(record.get('target_conf', 0)), center_distance_norm=math.hypot(cx - px, cy - py) / short, edge_margin_norm=margin / short, mask_edge_touch=int(margin <= 1), mask_aspect_ratio=(int(xs.max()) - int(xs.min()) + 1) / (int(ys.max()) - int(ys.min()) + 1), mask_area_ratio_input=int(np.count_nonzero(mask)) / max(1, width * height))

def keys(record):
    season = str(record.get('season_id') or 'unspecified')
    (bed, date, time, slot) = (str(record[k]) for k in ('bed_id', 'capture_date', 'capture_time', 'physical_slot'))
    pair = f'{season}|{date}_{time}_{bed}|{slot}'
    if record.get('pair_id') and record['pair_id'] != pair:
        raise ValueError('카메라 선택 pair_id 불일치: ' + record['record_id'])
    return (pair, f'{season}|{bed}_{slot}')

def references(db):
    result = {}
    levels = [('date', ['date', 'camera', 'row_name']), ('camera_row', ['camera', 'row_name']), ('camera', ['camera']), ('all', [])]
    for (label, cols) in levels:
        groups = list(db.execute('SELECT DISTINCT ' + ','.join(cols) + ' FROM camera_candidates WHERE clean=1')) if cols else [()]
        for group in groups:
            condition = 'clean=1' + ''.join((f' AND {col}=?' for col in cols))
            cursor = db.execute('SELECT log_aspect,log_size FROM camera_candidates WHERE ' + condition, group)
            values = np.fromiter((v for pair in cursor for v in pair), dtype=np.float64).reshape(-1, 2)
            count = len(values)
            if not count:
                continue
            stats = []
            for (column, floor) in [(0, 0.05), (1, 0.1)]:
                vector = values[:, column]
                center = float(np.median(vector))
                spread = max(floor, 1.4826 * float(np.median(np.abs(vector - center))))
                stats.extend([center, spread])
            result[(label, *group)] = (*stats, count)
    return result

def build(db, records):
    db.execute('CREATE TABLE IF NOT EXISTS camera_candidates(id TEXT PRIMARY KEY,physical TEXT,track TEXT,camera TEXT,date TEXT,row_name TEXT,clean INTEGER,log_aspect REAL,log_size REAL,payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS camera_selections(physical TEXT PRIMARY KEY,id TEXT,payload TEXT)')
    db.execute('DELETE FROM camera_candidates')
    db.execute('DELETE FROM camera_selections')
    for (number, record) in enumerate(records, 1):
        (pair, track) = keys(record)
        fields = ['record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'source_path', 'raw_source_path', 'status', 'old_key', 'new_key'] + METRICS
        item = {k: record.get(k) for k in fields}
        item.update(physical_key=pair, track_id=track, row=record['physical_slot'][0], original_status=record['status'])
        usable = record.get('candidate_available', False) and item.get('inference_status') == 'OK'
        if not usable:
            item['inference_status'] = item.get('inference_status') or 'NO_DETECTION'
        top = item['camera'] in ('cam0', 'cam1')
        valid = int(top and usable and (item['mask_edge_touch'] == 0) and (item['center_distance_norm'] <= policy.HARD_CENTER_LIMIT))
        aspect = math.log(max(1e-08, item['mask_aspect_ratio'])) if usable else None
        size = math.log(max(1e-08, item['mask_area_ratio_input'])) if usable else None
        db.execute('INSERT INTO camera_candidates VALUES(?,?,?,?,?,?,?,?,?,?)', (item['record_id'], pair, track, item['camera'], item['capture_date'], item['row'], valid, aspect, size, dumps(item)))
        if number % 500 == 0:
            db.commit()
    db.execute('CREATE INDEX IF NOT EXISTS selection_reference ON camera_candidates(clean,date,camera,row_name)')
    db.execute('CREATE INDEX IF NOT EXISTS selection_camera_reference ON camera_candidates(clean,camera,row_name)')
    db.execute('CREATE INDEX IF NOT EXISTS selection_track ON camera_candidates(track,camera)')
    db.execute('CREATE INDEX IF NOT EXISTS selection_output ON camera_candidates(physical,camera,id)')
    refs = references(db)
    cursor = db.execute("SELECT id,payload FROM camera_candidates WHERE camera IN ('cam0','cam1') ORDER BY id")
    while (batch := cursor.fetchmany(512)):
        rows = [json.loads(p) for (_, p) in batch]
        if not refs:
            for row in rows:
                if row['inference_status'] == 'OK':
                    row['inference_status'] = 'NO_ROBUST_REFERENCE'
        policy.add_quality_metrics(rows, reference=refs)
        db.executemany('UPDATE camera_candidates SET payload=? WHERE id=?', [(dumps(r), r['record_id']) for r in rows])
    db.commit()
    tracks = db.execute("SELECT DISTINCT track FROM camera_candidates WHERE camera IN ('cam0','cam1') ORDER BY track").fetchall()
    for (track,) in tracks:
        rows = [json.loads(p) for (p,) in db.execute("SELECT payload FROM camera_candidates INDEXED BY selection_track WHERE track=? AND camera IN ('cam0','cam1') ORDER BY physical,camera,id", (track,))]
        (primary, evidence) = policy.track_primary_cameras(rows)
        groups = defaultdict(list)
        for row in rows:
            groups[row['physical_key']].append(row)
        for (pair, items) in groups.items():
            (selected, reason) = policy.choose_candidate(items, primary[track])
            if selected is None and any((r['inference_status'] == 'NO_ROBUST_REFERENCE' for r in items)):
                reason = 'NO_ROBUST_REFERENCE'
            representative = selected or next((r for r in items if r['camera'] == primary[track]), items[0])
            status = 'SELECTED' if selected and selected['hard_valid'] and (not selected['original_status'].startswith('REVIEW')) else 'SELECTED_REVIEW' if selected else 'NO_USABLE_DETECTION'
            info = dict(observation_view='top', selection_status=status, selection_reason=reason, selected_camera=selected['camera'] if selected else '', selected_record_id=selected['record_id'] if selected else '', default_camera='cam0' if int(representative['physical_slot'][1:]) <= 4 else 'cam1', selected_quality_score=selected['quality_score'] if selected else None, selected_hard_valid=selected['hard_valid'] if selected else 0, selection_version=VERSION, **evidence[track])
            db.execute('INSERT INTO camera_selections VALUES(?,?,?)', (pair, representative['record_id'], dumps(info)))
    db.commit()

def decision(db, record):
    if record['camera'] == 'cam2':
        return dict(observation_view='front', selection_status='FRONT_PRESERVED', selection_reason='CAM2_SEPARATE_VIEW', selected_camera='cam2', selected_record_id=record['record_id'], selection_version=VERSION)
    (pair, _) = keys(record)
    row = db.execute('SELECT id,payload FROM camera_selections WHERE physical=?', (pair,)).fetchone()
    if row is None:
        raise ValueError('카메라 선택 기록 누락: ' + pair)
    return json.loads(row[1]) if row[0] == record['record_id'] else None

def export(db, out):
    out = Path(out)

    def write(path, rows):
        rows = iter(rows)
        first = next(rows, None)
        temp = path.with_suffix('.writing')
        with temp.open('w', encoding='utf-8-sig', newline='') as stream:
            fields = list(first) if first else ['record_id']
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            if first:
                writer.writerow(first)
            writer.writerows(rows)
        os.replace(temp, path)

    def candidates():
        for (payload, selection) in db.execute('SELECT c.payload,s.payload FROM camera_candidates c LEFT JOIN camera_selections s ON c.physical=s.physical ORDER BY c.physical,c.camera,c.id'):
            r = json.loads(payload)
            s = json.loads(selection) if selection else {}
            yield {**r, 'selected_for_top': r['camera'] in ('cam0', 'cam1') and s.get('selected_record_id') == r['record_id'], 'selection_reason': s.get('selection_reason') if r['camera'] != 'cam2' else 'CAM2_SEPARATE_VIEW'}
    write(out / 'camera_candidates.csv', candidates())
    write(out / 'camera_selection.csv', ({'pair_id': pair, 'representative_record_id': rid, **json.loads(payload)} for (pair, rid, payload) in db.execute('SELECT physical,id,payload FROM camera_selections ORDER BY physical')))
