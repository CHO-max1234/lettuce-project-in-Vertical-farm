"""Refine only failed or unresolved slot geometry using independent raw-image evidence. Retain unresolved states when evidence is insufficient."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
WHOLE_FILE = CYCLE.PREPROCESS_CODE / '5. Build_plant_bboxes.py'
OUTPUT_ROOT = CYCLE.BBOX_OUTPUT
STATE_ROOT = OUTPUT_ROOT / '_whole_run'
REPAIR_VERSION = '1.3_RAW_HOLE_ROWS_AND_CAM2_RAIL_RECOVERY'
import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
spec = importlib.util.spec_from_file_location('s3_coordinate_repair_whole', WHOLE_FILE)
W = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = W
spec.loader.exec_module(W)
import cv2
import numpy as np

class RepairUnavailable(ValueError):
    pass

def wide_holes(source, adapter):
    original = (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND)
    try:
        (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND) = ((0.0, 0.5), (0.5, 0.85))
        raw = adapter.detect_hole_evidence(source)
    finally:
        (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND) = original
    return [dict(id=i, x=h['x'], y=h['y'], score=h['score'], detector_row=row) for (i, (row, h)) in enumerate(((row, h) for (row, holes) in raw.items() for h in holes))]

def cam2_geometry_holes(source, adapter):
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    (yy, xx) = np.ogrid[:source.shape[0], :source.shape[1]]
    radius = max(9.0, 0.008 * source.shape[1])
    found = []
    for gamma in (0.35, 0.5):
        lut = np.round(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
        view = cv2.LUT(source, lut)
        raw = adapter.detect_hole_evidence(view)
        for group in raw.values():
            for h in group:
                if h['score'] < 60:
                    continue
                d2 = (xx - h['x']) ** 2 + (yy - h['y']) ** 2
                inner = d2 <= (0.65 * radius) ** 2
                ring = (d2 > (1.25 * radius) ** 2) & (d2 <= (1.9 * radius) ** 2)
                if not inner.any() or not ring.any():
                    continue
                contrast = float(gray[ring].mean() - gray[inner].mean())
                neutral = float((hsv[:, :, 1][ring] < 90).mean())
                if contrast < 10 or neutral < 0.4:
                    continue
                if not 0.04 * source.shape[0] <= h['y'] <= 0.85 * source.shape[0]:
                    continue
                candidate = dict(h, geometry_gamma=gamma, original_gray_contrast=contrast, original_neutral_ring=neutral)
                close = next((i for (i, v) in enumerate(found) if math.hypot(v['x'] - h['x'], v['y'] - h['y']) < 0.75 * radius), None)
                if close is None:
                    found.append(candidate)
                elif h['score'] > found[close]['score']:
                    found[close] = candidate
    return [dict(h, id=i, detector_row='t' if h['y'] < 0.5 * source.shape[0] else 'b') for (i, h) in enumerate(found)]

def cam2_raw_board(source, cam2, adapter):
    raw = cam2_geometry_holes(source, adapter)
    (front, _, _) = cam2.detect_front_rail_y(source)
    a = float(cam2.PREVIEW.REF_RAIL_A)
    proposal = W.SimpleNamespace(a=a, b=-a * front, x_at=lambda y: a * (y - front))
    (original_top, original_bottom) = cam2.transfer_board(proposal, front, source.shape[1] / cam2.PREVIEW.REF_WIDTH)
    (points, details) = ({}, {})
    for (name, prior) in (('t', original_top), ('b', original_bottom)):
        pitch = float(np.median(np.diff(prior[:, 0])))
        line = W.REGISTRATION._line([tuple(p) for p in prior])
        possible = [h for h in raw if abs(h['y'] - (line[0] * h['x'] + line[1])) < 0.28 * pitch]
        first = [h for h in possible if math.hypot(h['x'] - prior[0, 0], h['y'] - prior[0, 1]) < 0.3 * pitch]
        if not first:
            raise RepairUnavailable('CAM2_FIRST_PHYSICAL_HOLE_UNAVAILABLE_' + name)
        start = min(first, key=lambda h: abs(h['x'] - prior[0, 0]))
        seeds = [dict(start, col=1, slot_name=name + '1')]
        previous = start
        col = 1
        local_pitch = pitch
        for hole in sorted((h for h in possible if h['x'] > start['x'] + 0.5 * pitch), key=lambda h: h['x']):
            gap = hole['x'] - previous['x']
            steps = max(1, round(gap / local_pitch))
            new_pitch = gap / steps
            if steps > 4 or not 0.75 * local_pitch <= new_pitch <= 1.25 * local_pitch:
                continue
            if col + steps > 12:
                continue
            col += steps
            seeds.append(dict(hole, col=col, slot_name=name + str(col)))
            previous = hole
            local_pitch = new_pitch
        (kept, fit) = W.REGISTRATION._robust_row(seeds, max(5.0, 0.05 * pitch))
        if len(kept) < 4 or not any((h['col'] == 1 for h in kept)) or kept[-1]['col'] - kept[0]['col'] < 6:
            raise RepairUnavailable('CAM2_CALIBRATED_RAW_HOLE_ROW_UNSUPPORTED_' + name)
        points[name] = np.asarray([W.REGISTRATION._interpolate(kept, c) for c in range(1, 13)], np.float64)
        details[name] = dict(seeds=kept, line=list(fit), numbering_from='LEFT_RAW_PHYSICAL_HOLE_1')
    if np.any(points['b'][:, 1] - points['t'][:, 1] < 0.3 * np.median(np.diff(points['t'][:, 0]))):
        raise RepairUnavailable('CAM2_RAW_BOARD_ROW_ORDER_CONFLICT')
    return (points['t'], points['b'], raw, details)

def raw_two_rows(slots, source, adapter):
    pitch = float(np.median([float(r['cell_width']) for r in slots if float(r['cell_width']) > 0]))
    priors = [float(r['x_prior']) for r in slots]
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    neutral = (hsv[:, :, 1] <= 82) & (hsv[:, :, 2] >= 38) & (hsv[:, :, 2] <= 232)
    (yy, xx) = np.ogrid[:source.shape[0], :source.shape[1]]
    raw = wide_holes(source, adapter)
    holes = []
    for h in raw:
        if h['score'] < 80 or not min(priors) - 0.35 * pitch <= h['x'] <= max(priors) + 0.52 * pitch:
            continue
        radius = max(6.0, 0.1 * pitch)
        d2 = (xx - h['x']) ** 2 + (yy - h['y']) ** 2
        ring = (d2 > radius ** 2) & (d2 <= (2 * radius) ** 2)
        share = float(neutral[ring].mean()) if ring.any() else 0.0
        if share >= 0.35:
            holes.append(dict(h, neutral_ring_share=share))
    lines = []
    for (i, a) in enumerate(holes):
        for b in holes[i + 1:]:
            if abs(b['x'] - a['x']) < 2 * pitch:
                continue
            slope = (b['y'] - a['y']) / (b['x'] - a['x'])
            if abs(slope) > 0.18:
                continue
            intercept = a['y'] - slope * a['x']
            kept = [h for h in holes if abs(h['y'] - slope * h['x'] - intercept) <= max(5.0, 0.08 * pitch)]
            if len(kept) < 3 or max((h['x'] for h in kept)) - min((h['x'] for h in kept)) < 2.5 * pitch:
                continue
            fit = W.REGISTRATION._line([(h['x'], h['y']) for h in kept])
            lines.append(dict(line=fit, holes=kept, ids={h['id'] for h in kept}))
    pairs = []
    mid = source.shape[1] / 2
    for (i, a) in enumerate(lines):
        for b in lines[i + 1:]:
            if a['ids'] & b['ids']:
                continue
            ay = a['line'][0] * mid + a['line'][1]
            by = b['line'][0] * mid + b['line'][1]
            if not 0.45 * pitch <= abs(by - ay) <= 2.2 * pitch:
                continue
            (upper, lower) = (a, b) if ay < by else (b, a)
            gaps = [(lower['line'][0] - upper['line'][0]) * x + lower['line'][1] - upper['line'][1] for x in (min(priors), max(priors))]
            if min(gaps) < 0.3 * pitch:
                continue
            score = (len(a['holes']) + len(b['holes']), sum((h['score'] for h in a['holes'] + b['holes'])))
            pairs.append((score, upper, lower))
    if not pairs:
        raise RepairUnavailable('TWO_INDEPENDENT_RAW_HOLE_ROWS_UNAVAILABLE')
    (_, upper, lower) = max(pairs, key=lambda x: x[0])
    return (dict(t=upper, b=lower), raw, pitch)

def repair_slots(slots, source, refs):
    if not slots:
        raise RepairUnavailable('NO_SLOT_GEOMETRY_AFTER_WIDE_HOLE_DETECTION')
    output = [dict(r) for r in slots]
    camera = output[0]['camera']
    detail = dict(version=REPAIR_VERSION, camera=camera, uses_seg_centroids=False)
    if camera in ('cam0', 'cam1'):
        (rows, raw, pitch) = raw_two_rows(output, source, refs[2])
        for name in ('t', 'b'):
            group = [r for r in output if r['row'] == name]
            if camera == 'cam1':
                (slope, intercept) = rows[name]['line']
                for r in group:
                    x = float(r['x_prior'])
                    y = slope * x + intercept
                    r.update(y_prior=y, slot_center_x=x, slot_center_y=y, evidence_kind='template')
                detail[name] = dict(raw_holes=rows[name]['holes'], line=list(rows[name]['line']))
                continue
            seeds = []
            used = set()
            for h in sorted(rows[name]['holes'], key=lambda h: -h['score']):
                r = min(group, key=lambda r: abs(float(r['x_prior']) - h['x']))
                if r['col'] in used or abs(float(r['x_prior']) - h['x']) > 0.52 * pitch:
                    continue
                used.add(r['col'])
                seeds.append(dict(h, col=int(r['col']), slot_name=r['slot_name']))
            seeds.sort(key=lambda h: h['col'])
            if len(seeds) < 3 or seeds[-1]['col'] - seeds[0]['col'] < 3:
                raise RepairUnavailable('INSUFFICIENT_CANONICAL_RAW_HOLE_MATCHES_' + name)
            if any((b['x'] <= a['x'] for (a, b) in zip(seeds, seeds[1:]))):
                raise RepairUnavailable('RAW_HOLE_PHASE_CONFLICT_' + name)
            for r in group:
                col = int(r['col'])
                if max(seeds[0]['col'] - col, col - seeds[-1]['col'], 0) > 1:
                    raise RepairUnavailable('OUTSIDE_RAW_HOLE_SPAN_' + r['physical_slot'])
                (x, y) = W.REGISTRATION._interpolate(seeds, col)
                r.update(x_prior=x, y_prior=y, slot_center_x=x, slot_center_y=y, evidence_kind='template')
            detail[name] = dict(raw_hole_seeds=seeds, line=list(rows[name]['line']))
    elif camera == 'cam2':
        (registered, meta) = W.REGISTRATION.register_frame([W.csv_values(r) for r in output], source)
        for name in ('t', 'b'):
            row_meta = meta.get('rows', {}).get(name, {})
            seeds = row_meta.get('registered_holes', [])
            pitch = float(row_meta.get('pitch_px', 0))
            if len(seeds) < 3 or max((s['col'] for s in seeds)) - min((s['col'] for s in seeds)) < 4 or pitch <= 0:
                raise RepairUnavailable('CAM2_INDEPENDENT_RAW_ROW_SUPPORT_INSUFFICIENT_' + name)
            if not any((s['col'] <= 2 and s['prior_match_distance_px'] < 0.25 * pitch for s in seeds)):
                raise RepairUnavailable('CAM2_LEFT_PHYSICAL_PHASE_UNSUPPORTED_' + name)
            if float(np.median([s['prior_match_distance_px'] for s in seeds])) >= 0.25 * pitch:
                raise RepairUnavailable('CAM2_COARSE_PHASE_RESIDUAL_TOO_LARGE_' + name)
        by_name = {r['physical_slot']: r for r in registered}
        for r in output:
            reg = by_name[r['physical_slot']]
            r['before_coordinate_repair'] = {k: r.get(k) for k in ('x_prior', 'y_prior', 'slot_center_x', 'slot_center_y', 'cell_bbox', 'cell_polygon', 'cell_width', 'cell_height', 'evidence_kind')}
            r.update(coordinate_repair_version=REPAIR_VERSION, coordinate_repair_source='CAM2_INDEPENDENT_RAW_HOLES_VALIDATE_COARSE_FRONT_RAIL_BOOTSTRAP', rail_bootstrap_review=True)
            if str(reg.get('anchor_registration_status', '')).startswith('REGISTERED'):
                (ax, ay) = (float(reg['eval_anchor_x']), float(reg['eval_anchor_y']))
                r.update(x_prior=ax, y_prior=ay, slot_center_x=ax, slot_center_y=ay, anchor_x=ax, anchor_y=ay, evidence_kind='template')
            if float(r['cell_width']) <= 0 or float(r['cell_height']) <= 0:
                raise RepairUnavailable('CAM2_CELL_GEOMETRY_INVALID')
        detail['registration'] = meta
        detail['rail_bootstrap_review'] = True
        return (output, detail)
    else:
        raise RepairUnavailable('UNSUPPORTED_CAMERA')
    (registered, meta) = W.REGISTRATION.register_frame([W.csv_values(r) for r in output], source)
    if any((not str(r.get('anchor_registration_status', '')).startswith('REGISTERED') for r in registered)):
        raise RepairUnavailable('RAW_HOLE_REGISTRATION_INCOMPLETE')
    by_name = {r['physical_slot']: r for r in registered}
    original_by_name = {r['physical_slot']: r for r in slots}
    cols = sorted({int(r['col']) for r in output})
    pairs = {c: (by_name['t' + str(c)], by_name['b' + str(c)]) for c in cols}
    centers = [0.5 * (float(t['eval_anchor_x']) + float(b['eval_anchor_x'])) for (t, b) in pairs.values()]
    if len(centers) < 2 or any((b <= a for (a, b) in zip(centers, centers[1:]))):
        raise RepairUnavailable('NONMONOTONIC_CELL_COLUMNS')
    mids = [0.5 * (a + b) for (a, b) in zip(centers, centers[1:])]
    left = [centers[0] - 0.55 * (centers[1] - centers[0])] + mids
    right = mids + [centers[-1] + 0.55 * (centers[-1] - centers[-2])]
    (h, w) = source.shape[:2]
    for (i, col) in enumerate(cols):
        (top, bottom) = pairs[col]
        (ty, by) = (float(top['eval_anchor_y']), float(bottom['eval_anchor_y']))
        gap = by - ty
        if gap < max(8.0, 0.2 * (right[i] - left[i])):
            raise RepairUnavailable('INVALID_REGISTERED_ROW_GAP')
        (x1, x2) = (max(0, round(left[i])), min(w, round(right[i])))
        (y1, mid, y2) = (max(0, round(ty - 0.55 * gap)), round(0.5 * (ty + by)), min(h, round(by + 0.55 * gap)))
        for (name, a, b) in (('t', y1, mid), ('b', mid, y2)):
            r = next((r for r in output if r['physical_slot'] == name + str(col)))
            reg = by_name[r['physical_slot']]
            if not (x2 > x1 and 0 <= a < b <= h):
                raise RepairUnavailable('REBUILT_CELL_OUT_OF_SOURCE')
            orig = original_by_name[r['physical_slot']]
            r['before_coordinate_repair'] = {k: orig.get(k) for k in ('x_prior', 'y_prior', 'slot_center_x', 'slot_center_y', 'cell_bbox', 'cell_polygon', 'cell_width', 'cell_height', 'evidence_kind')}
            (ax, ay) = (float(reg['eval_anchor_x']), float(reg['eval_anchor_y']))
            r.update(x_prior=ax, y_prior=ay, slot_center_x=ax, slot_center_y=ay, anchor_x=ax, anchor_y=ay, evidence_kind='template', cell_bbox=[x1, a, x2, b], cell_polygon=[[x1, a], [x2, a], [x2, b], [x1, b]], cell_x1=x1, cell_y1=a, cell_x2=x2, cell_y2=b, cell_width=x2 - x1, cell_height=b - a, cell_geometry_ready=True, coordinate_repair_version=REPAIR_VERSION, coordinate_repair_source='INDEPENDENT_RAW_HOLE_ROWS', bbox_handoff_status='SLOT_CSV_READY')
    detail['registration'] = meta
    return (output, detail)

def process_repaired(sample, predictor, refs, save_visuals=False):
    original = W.MAPPER.process_sample
    original_raw = W.REGISTRATION._raw_holes
    adapter = refs[2]
    source = W.decode(sample)
    info = {}
    cam2 = refs[4]
    original_choose = cam2.PREVIEW.choose_left_rail
    original_transfer = cam2.transfer_board
    cam2_helper_name = '_cam2_physical_anchor_registration_ver12'
    cam2_helper = None
    original_cam2_detector = None
    recovered_board = None
    recovered_raw = None

    def choose_recoverable(image, front_y):
        try:
            return original_choose(image, front_y)
        except RuntimeError as error:
            if str(error) != 'LEFT_CONTINUOUS_RAIL_NOT_FOUND':
                raise
            a = float(cam2.PREVIEW.REF_RAIL_A)
            rail = W.SimpleNamespace(a=a, b=-a * front_y, score=0.0, x_at=lambda y: a * (y - front_y))
            info['rail_bootstrap'] = dict(status='RAW_HOLE_VALIDATION_REQUIRED', front_y=front_y, a=a, b=-a * front_y, detected=False)
            return (rail, 0)

    def transfer_recoverable(rail, front_y, scale):
        nonlocal recovered_board, recovered_raw
        if 'rail_bootstrap' not in info:
            return original_transfer(rail, front_y, scale)
        if recovered_board is None:
            cam2.transfer_board = original_transfer
            try:
                (top, bottom, raw, detail) = cam2_raw_board(source, cam2, adapter)
            finally:
                cam2.transfer_board = transfer_recoverable
            recovered_board = (top, bottom)
            recovered_raw = raw
            info['cam2_raw_board'] = detail
        return recovered_board

    def wrapped(*args):
        (frame, slots, segs) = original(*args)
        if not slots and sample['camera'] == 'cam0':
            bands = (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND)
            try:
                (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND) = ((0.0, 0.5), (0.5, 0.85))
                (frame, slots, segs) = original(*args)
            finally:
                (adapter.TOP_ROW_BAND, adapter.BOTTOM_ROW_BAND) = bands
        (repaired, detail) = repair_slots(slots, source, refs)
        info.update(detail)
        frame['coordinate_repair'] = detail
        return (frame, repaired, segs)
    W.MAPPER.process_sample = wrapped
    if sample['camera'] == 'cam0':
        W.REGISTRATION._raw_holes = lambda src: wide_holes(src, adapter)
    if sample['camera'] == 'cam2':
        cam2.PREVIEW.choose_left_rail = choose_recoverable
        cam2.transfer_board = transfer_recoverable
        cam2_helper = sys.modules.get(cam2_helper_name)
        if cam2_helper is None:
            cam2_helper = W.load_module(cam2_helper_name, CODE_ROOT / 'front_registration.py')
        original_cam2_detector = cam2_helper.detect_raw_holes
        cam2_helper.detect_raw_holes = lambda src: [dict(h) for h in recovered_raw] if recovered_raw is not None else original_cam2_detector(src)
    original_sample_root = W.SAMPLE_ROOT
    original_state_root = W.STATE_ROOT
    if save_visuals:
        W.SAMPLE_ROOT = CODE_ROOT / '_coordinate_repair_ver1.3_tests'
        W.STATE_ROOT = STATE_ROOT
    try:
        (frame, slots, records, masks) = W.process_frame(sample, source, predictor, refs, save_visuals)
    finally:
        W.MAPPER.process_sample = original
        W.REGISTRATION._raw_holes = original_raw
        cam2.PREVIEW.choose_left_rail = original_choose
        cam2.transfer_board = original_transfer
        if cam2_helper is not None:
            cam2_helper.detect_raw_holes = original_cam2_detector
        W.SAMPLE_ROOT = original_sample_root
        W.STATE_ROOT = original_state_root
    frame.update(status='COORDINATES_REPAIRED', coordinate_repair=info, coordinate_repair_version=REPAIR_VERSION)
    return (frame, slots, records, masks)

def run_refinement():
    W.clone_record_functions()
    W.RUNNER.write_json(STATE_ROOT / 'coordinate_repair_status.json', dict(status='WAITING_MAIN_RUN', pid=__import__('os').getpid(), version=REPAIR_VERSION))
    if not (OUTPUT_ROOT / 'completion.json').is_file():
        raise FileNotFoundError('Complete bbox assignment before coordinate refinement')
    completion = json.loads((OUTPUT_ROOT / 'completion.json').read_text(encoding='utf-8'))
    if completion.get('completed_frames') != 43198:
        raise ValueError("Invalid input or state in run_refinement (completion.get('completed_frames') != 43198)")
    predictor = W.Predictor()
    refs = W.MAPPER.load_references()
    db = W.connect_state()
    todo = [json.loads(r[0]) for r in db.execute("SELECT data FROM frames WHERE status IN ('INVALID_CELL_GEOMETRY_REVIEW','NO_SLOT_GEOMETRY_REVIEW','FRAME_ERROR') ORDER BY rowid")]
    audit = []
    for old_frame in W.tqdm(todo, desc='Coordinate repair', unit='frame'):
        frame_id = old_frame['frame_id']
        old_slots = db.execute('SELECT slot_json,bbox_json FROM slots WHERE frame_id=?', (frame_id,)).fetchall()
        try:
            (frame, slots, records, masks) = process_repaired(old_frame, predictor, refs, False)
            if old_slots:
                before_ids = {json.loads(r[0])['record_id'] for r in old_slots}
                before_names = {json.loads(r[0])['physical_slot'] for r in old_slots}
                if before_ids != {r['record_id'] for r in slots} or before_names != {r['physical_slot'] for r in slots}:
                    raise RepairUnavailable('CANONICAL_SLOT_ID_CHANGE_REJECTED')
            entry = dict(frame_id=frame_id, image_name=old_frame['image_name'], before_status=old_frame['status'], after_status=frame['status'], before_frame=old_frame, before_slots=[dict(slot=json.loads(r[0]), bbox=json.loads(r[1])) for r in old_slots], slot_count=len(slots), target_count=frame['target_count'], registration=frame['coordinate_repair'])
            W.RUNNER.write_json(STATE_ROOT / 'coordinate_repair_history' / (frame_id + '.json'), entry)
            with db:
                db.execute('DELETE FROM slots WHERE frame_id=?', (frame_id,))
                db.execute('DELETE FROM frames WHERE frame_id=?', (frame_id,))
                W.save_frame(db, frame, slots, records, masks)
            audit.append({k: v for (k, v) in entry.items() if k not in ('before_frame', 'before_slots', 'registration')})
        except RepairUnavailable as error:
            audit.append(dict(frame_id=frame_id, image_name=old_frame['image_name'], before_status=old_frame['status'], after_status='REPAIR_UNAVAILABLE', reason=str(error)))
        W.RUNNER.write_json(STATE_ROOT / 'coordinate_repair_status.json', dict(status='REPAIRING', total=len(todo), finished=len(audit), results=audit))
    W.export_all(db)
    counts = dict(db.execute('SELECT status,count(*) FROM frames GROUP BY status').fetchall())
    if sum(counts.values()) != 43198:
        raise ValueError('Invalid input or state in run_refinement (sum(counts.values()) != 43198)')
    report = dict(status='COORDINATE_REPAIR_COMPLETE_REVIEW_PENDING', version=REPAIR_VERSION, input_images=43198, candidates=len(todo), repaired=sum((r['after_status'] == 'COORDINATES_REPAIRED' for r in audit)), unavailable=sum((r['after_status'] == 'REPAIR_UNAVAILABLE' for r in audit)), frame_statuses=counts, results=audit, script_sha256=W.digest(__file__), frozen_whole_runner_sha256=W.digest(WHOLE_FILE), slot_rows=db.execute('SELECT count(*) FROM slots').fetchone()[0], target_rows=sum((json.loads(r[0])['target_available'] for r in db.execute('SELECT bbox_json FROM slots'))), target_selection_method_unchanged='ver2.2', finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    completion.update(coordinate_repair=report, slot_rows=report['slot_rows'], target_rows=report['target_rows'], no_slot_geometry_images=counts.get('NO_SLOT_GEOMETRY_REVIEW', 0), invalid_cell_geometry_images=counts.get('INVALID_CELL_GEOMETRY_REVIEW', 0), selection_statuses=dict(Counter((json.loads(r[0])['selection_status'] for r in db.execute('SELECT bbox_json FROM slots')))), frame_errors=counts.get('FRAME_ERROR', 0))
    W.RUNNER.write_json(OUTPUT_ROOT / 'completion.json', completion)
    live_status = json.loads((STATE_ROOT / 'status.json').read_text(encoding='utf-8'))
    live_status.update(slot_rows=report['slot_rows'], coordinate_repair_complete=True, no_slot_geometry_images=counts.get('NO_SLOT_GEOMETRY_REVIEW', 0), invalid_cell_geometry_images=counts.get('INVALID_CELL_GEOMETRY_REVIEW', 0), coordinate_repaired_images=counts.get('COORDINATES_REPAIRED', 0))
    W.RUNNER.write_json(STATE_ROOT / 'status.json', live_status)
    W.RUNNER.write_json(STATE_ROOT / 'coordinate_repair_completion.json', report)
    W.RUNNER.write_json(STATE_ROOT / 'coordinate_repair_status.json', report)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    print(W.encode(report), flush=True)

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    required = [OUTPUT_ROOT / 'completion.json', STATE_ROOT / 'state.sqlite3']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.check:
        print(json.dumps({'status': 'READY', 'writes': 0, 'database': str(required[1])}))
        return 0
    run_refinement()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
