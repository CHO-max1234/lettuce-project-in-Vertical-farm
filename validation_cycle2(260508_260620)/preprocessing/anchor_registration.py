"""Register physical anchors from independent raw-image holes and retain unsupported geometry as review."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
SUPPORT_ROOT = CYCLE.MAIN_PREPROCESS / '_support' / 'cam01'
MIN_HOLE_SCORE = 80.0
INITIAL_X_GATE_PITCH = 0.52
INITIAL_Y_GATE_ROW_GAP = 0.42
ROW_RESIDUAL_GATE_ROW_GAP = 0.15
MIN_REGISTERED_HOLES = 3
MAX_EXTRAPOLATION_COLS = 1.0
VERSION = 'raw_hole_registered_physical_anchor_ver1.2'
CAM2_HELPER_FILE = Path(__file__).with_name('front_registration.py')
import importlib
import importlib.util
import math
import re
import sys
import cv2
import numpy as np

def _number(row, key, default=None):
    try:
        value = float(row.get(key, ''))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default

def _col(row):
    match = re.fullmatch('[tb](\\d+)', str(row.get('slot_name', row.get('physical_slot', ''))))
    return int(match[1]) if match else int(_number(row, 'col', 0))

def _point(row):
    if not row.get('seg_id') and 'hole' in str(row.get('evidence_kind', '')):
        (x, y) = (_number(row, 'slot_center_x'), _number(row, 'slot_center_y'))
        if x is not None and y is not None:
            return (x, y, 'SEG_FREE_HOLE_CORRECTION')
    return (_number(row, 'x_prior', _number(row, 'slot_center_x', 0.0)), _number(row, 'y_prior', _number(row, 'slot_center_y', 0.0)), 'PHYSICAL_PRIOR')

def _pitch(rows):
    points = sorted(((_col(r), _number(r, 'x_prior')) for r in rows))
    values = [(x2 - x1) / (c2 - c1) for ((c1, x1), (c2, x2)) in zip(points, points[1:]) if x1 is not None and x2 is not None and (c2 > c1) and (x2 > x1)]
    widths = [_number(r, 'cell_width') for r in rows]
    values = values or [v for v in widths if v is not None and v > 0]
    return max(1.0, float(np.median(values))) if values else 1.0

def _line(points):
    if len(points) < 2 or np.ptp([p[0] for p in points]) < 1:
        return (0.0, float(np.median([p[1] for p in points])))
    (slope, intercept) = np.polyfit(np.asarray(points, float)[:, 0], np.asarray(points, float)[:, 1], 1)
    return (float(slope), float(intercept))

def _robust_row(seeds, tolerance):
    """Find a straight physical row; leaf-shadow and adjacent-bed holes are outliers."""
    if len(seeds) < 3:
        return ([], None)
    candidates = []
    for (i, a) in enumerate(seeds):
        for b in seeds[i + 1:]:
            dx = b['x'] - a['x']
            if abs(dx) < 1:
                continue
            slope = (b['y'] - a['y']) / dx
            if abs(slope) > 0.3:
                continue
            intercept = a['y'] - slope * a['x']
            kept = [s for s in seeds if abs(s['y'] - slope * s['x'] - intercept) <= tolerance]
            span = max((s['col'] for s in kept), default=0) - min((s['col'] for s in kept), default=0)
            candidates.append((len(kept), span, sum((s['score'] for s in kept)), kept))
    if not candidates:
        return ([], None)
    kept = max(candidates, key=lambda item: item[:3])[3]
    for _ in range(2):
        (slope, intercept) = _line([(s['x'], s['y']) for s in kept])
        cleaned = [s for s in seeds if abs(s['y'] - slope * s['x'] - intercept) <= tolerance]
        if len(cleaned) < 3:
            break
        kept = cleaned
    kept.sort(key=lambda s: s['col'])
    if any((b['x'] <= a['x'] for (a, b) in zip(kept, kept[1:]))):
        return ([], None)
    if len(kept) < MIN_REGISTERED_HOLES or kept[-1]['col'] - kept[0]['col'] < 2:
        return ([], None)
    return (kept, _line([(s['x'], s['y']) for s in kept]))

def _interpolate(seeds, col):
    """Piecewise physical-hole interpolation retains perspective and row offsets."""
    if col <= seeds[0]['col']:
        (a, b) = seeds[:2]
    elif col >= seeds[-1]['col']:
        (a, b) = seeds[-2:]
    else:
        (a, b) = next(((a, b) for (a, b) in zip(seeds, seeds[1:]) if a['col'] <= col <= b['col']))
    fraction = (col - a['col']) / (b['col'] - a['col'])
    return (float(a['x'] + fraction * (b['x'] - a['x'])), float(a['y'] + fraction * (b['y'] - a['y'])))

def _raw_holes(source):
    path = str(SUPPORT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    adapter = importlib.import_module('cam0_geometry')
    raw = adapter.detect_hole_evidence(source)
    holes = []
    for (provisional_row, entries) in raw.items():
        for entry in entries:
            holes.append({'id': len(holes), 'x': float(entry['x']), 'y': float(entry['y']), 'score': float(entry['score']), 'detector_row': provisional_row})
    return holes

def _cam1_right_phase(group, holes, pitch, row_gap, name):
    """Register CAM1 holes from approved right 12, before any slot-nearest match.

    Its lens compresses the left columns. A gap representing two compressed
    columns must not be forced into one column merely by a displaced prior.
    Raw row consensus excludes holes on the bed below before counting gaps.
    """
    candidates = [h for h in holes if h['score'] >= MIN_HOLE_SCORE and h['registered_row_candidate'] == name and (h['prior_row_residual_px'] <= INITIAL_Y_GATE_ROW_GAP * row_gap)]
    tolerance = max(5.0, ROW_RESIDUAL_GATE_ROW_GAP * row_gap)
    line_options = []
    for (i, a) in enumerate(candidates):
        for b in candidates[i + 1:]:
            if abs(b['x'] - a['x']) < pitch:
                continue
            slope = (b['y'] - a['y']) / (b['x'] - a['x'])
            if abs(slope) > 0.3:
                continue
            intercept = a['y'] - slope * a['x']
            kept = [h for h in candidates if abs(h['y'] - slope * h['x'] - intercept) <= tolerance]
            line_options.append((len(kept), sum((h['score'] for h in kept)), max((h['x'] for h in kept), default=0) - min((h['x'] for h in kept), default=0), kept))
    if not line_options:
        return ([], None, {'status': 'UNKNOWN_RIGHT_PHASE'})
    consistent = max(line_options, key=lambda v: v[:3])[3]
    if len(consistent) < 3:
        return ([], None, {'status': 'INSUFFICIENT_RAW_ROW_HOLES'})
    line = _line([(h['x'], h['y']) for h in consistent])
    consistent = [h for h in consistent if abs(h['y'] - line[0] * h['x'] - line[1]) <= tolerance]
    right = max(group, key=_col)
    if _col(right) != 12:
        return ([], None, {'status': 'APPROVED_RIGHT_12_UNAVAILABLE'})
    prior_x = _number(right, 'x_prior', _point(right)[0])
    endpoint_candidates = [h for h in consistent if abs(h['x'] - prior_x) <= 0.7 * pitch]
    if not endpoint_candidates:
        return ([], None, {'status': 'RIGHT_12_RAW_HOLE_UNAVAILABLE'})
    endpoint = min(endpoint_candidates, key=lambda h: abs(h['x'] - prior_x) / pitch + 0.35 * max(0, 160 - h['score']) / 160)
    slot_by_col = {_col(r): r for r in group}
    seeds = [{**endpoint, 'col': 12, 'slot_name': slot_by_col[12]['slot_name'], 'association_cost': 0.0}]
    xs = sorted((h['x'] for h in consistent if h['x'] <= endpoint['x']))
    estimates = []
    for (x1, x2) in zip(xs, xs[1:]):
        gap = x2 - x1
        steps = max(1, int(round(gap / pitch)))
        local = gap / steps
        if 0.7 * pitch <= local <= 1.25 * pitch and x2 >= endpoint['x'] - 4 * pitch:
            estimates.append(local)
    reference = float(np.median(estimates)) if estimates else pitch
    (previous, previous_col, previous_pitch) = (endpoint, 12, reference)
    discarded = []
    for hole in sorted((h for h in consistent if h['x'] < endpoint['x']), key=lambda h: h['x'], reverse=True):
        gap = previous['x'] - hole['x']
        possibilities = []
        for steps in range(1, min(6, previous_col)):
            local_pitch = gap / steps
            if not 0.4 * reference <= local_pitch <= 1.25 * previous_pitch:
                continue
            cost = abs(math.log(local_pitch / previous_pitch))
            possibilities.append((cost, steps, local_pitch))
        if not possibilities:
            discarded.append({**hole, 'reason': 'INCONSISTENT_RIGHT_ANCHORED_PITCH'})
            continue
        (_, steps, local_pitch) = min(possibilities)
        col = previous_col - steps
        if col in slot_by_col:
            seeds.append({**hole, 'col': col, 'slot_name': slot_by_col[col]['slot_name'], 'association_cost': float(abs(local_pitch - reference) / reference)})
        else:
            discarded.append({**hole, 'inferred_col': col, 'reason': 'OUTSIDE_APPROVED_VISIBLE_COLUMNS'})
        (previous, previous_col, previous_pitch) = (hole, col, local_pitch)
    seeds.sort(key=lambda h: h['col'])
    if len(seeds) < 3 or seeds[-1]['col'] - seeds[0]['col'] < 2:
        return ([], None, {'status': 'INSUFFICIENT_RIGHT_REGISTERED_HOLES', 'discarded_holes': discarded})
    return (seeds, line, {'status': 'RIGHT_12_PHASE_REGISTERED', 'reference_pitch_px': reference, 'right_endpoint_hole': endpoint, 'discarded_holes': discarded})

def _cam0_raw_reselect(group, holes, initial_seeds, pitch, row_gap, name, source):
    """Two independent raw-hole passes prevent a leaf shadow tilting the row.

    A clean gray-board subset establishes physical row/column positions. The
    second pass can recover a displaced true hole that was farther from the old
    prior than a leaf shadow; neither pass examines a SEG centroid or polygon.
    """
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    neutral = (hsv[:, :, 1] <= 82) & (hsv[:, :, 2] >= 38) & (hsv[:, :, 2] <= 232)
    (yy, xx) = np.ogrid[:source.shape[0], :source.shape[1]]
    radius = max(6.0, 0.1 * pitch)
    gray_holes = {}
    for hole in holes:
        if hole['score'] < MIN_HOLE_SCORE or hole['registered_row_candidate'] != name:
            continue
        d2 = (xx - hole['x']) ** 2 + (yy - hole['y']) ** 2
        annulus = (d2 > radius ** 2) & (d2 <= (2 * radius) ** 2)
        support = float(neutral[annulus].mean()) if annulus.any() else 0.0
        if support >= 0.35:
            gray_holes[hole['id']] = {**hole, 'neutral_ring_share': support}
    gray_seeds = [{**seed, 'neutral_ring_share': gray_holes[seed['id']]['neutral_ring_share']} for seed in initial_seeds if seed['id'] in gray_holes]
    (clean, first_line) = _robust_row(gray_seeds, max(5.0, 0.08 * row_gap))
    detail = {'status': 'UNKNOWN_GRAY_BOARD_CONSENSUS', 'gray_raw_holes': list(gray_holes.values()), 'first_pass_holes': clean}
    if not clean:
        return ([], None, detail)
    y_from_col = _line([(h['col'], h['y']) for h in clean])
    possible = []
    for row in group:
        col = _col(row)
        (expected_x, _) = _interpolate(clean, col)
        expected_y = y_from_col[0] * col + y_from_col[1]
        for hole in gray_holes.values():
            (dx, dy) = (abs(hole['x'] - expected_x), abs(hole['y'] - expected_y))
            if dx > 0.4 * pitch or dy > max(6.0, 0.1 * row_gap):
                continue
            cost = dx / pitch + dy / row_gap + 0.35 * max(0, 160 - hole['score']) / 160
            possible.append((cost, row['slot_name'], col, hole['id']))
    (seeds, slots_used, holes_used) = ([], set(), set())
    for (cost, slot, col, hole_id) in sorted(possible):
        if slot in slots_used or hole_id in holes_used:
            continue
        slots_used.add(slot)
        holes_used.add(hole_id)
        seeds.append({**gray_holes[hole_id], 'slot_name': slot, 'col': col, 'association_cost': float(cost)})
    (kept, line) = _robust_row(seeds, max(5.0, 0.08 * row_gap))
    detail.update(status='GRAY_BOARD_ROW_AND_COLUMN_RESELECTED' if kept else 'UNKNOWN_SECOND_PASS', second_pass_holes=kept, first_pass_line_y_from_x=list(first_line), first_pass_line_y_from_col=list(y_from_col))
    return (kept, line, detail)

def _opposite_row_review(updated, holes, metadata, source):
    """Borrow only physical phase from a good row, retaining a review boundary.

    Three target-row raw holes establish its own position and slope. The opposite
    row never supplies y coordinates. Hole annuli must look like the gray board,
    and at least one inter-row path must show board pixels. These checks support
    a review proposal; they do not prove that two visible boards share identity.
    """
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    neutral = (hsv[:, :, 1] <= 82) & (hsv[:, :, 2] >= 38) & (hsv[:, :, 2] <= 232)
    (yy, xx) = np.ogrid[:source.shape[0], :source.shape[1]]
    count = 0
    for target_name in ('t', 'b'):
        other_name = 'b' if target_name == 't' else 't'
        target_meta = metadata['rows'].get(target_name, {})
        donor_meta = metadata['rows'].get(other_name, {})
        if target_meta.get('status') != 'UNKNOWN' or donor_meta.get('status') != 'REGISTERED':
            continue
        if donor_meta.get('phase_registration', {}).get('status') != 'RIGHT_12_PHASE_REGISTERED':
            continue
        targets = [r for r in updated if r.get('row', r['slot_name'][:1]) == target_name]
        donors = {_col(r): r for r in updated if r.get('row', r['slot_name'][:1]) == other_name and str(r.get('anchor_registration_status', '')).startswith('REGISTERED')}
        (pitch, row_gap) = (target_meta['pitch_px'], target_meta['row_gap_px'])
        candidate_pairs = []
        for hole in holes:
            if hole['score'] < MIN_HOLE_SCORE or hole['registered_row_candidate'] != target_name or hole['prior_row_residual_px'] > 0.45 * row_gap:
                continue
            radius = max(6.0, 0.1 * pitch)
            radius_squared = (xx - hole['x']) ** 2 + (yy - hole['y']) ** 2
            annulus = (radius_squared > radius ** 2) & (radius_squared <= (2 * radius) ** 2)
            neutral_ring = float(neutral[annulus].mean()) if annulus.any() else 0.0
            if neutral_ring < 0.4:
                continue
            for target in targets:
                col = _col(target)
                if col not in donors:
                    continue
                donor = donors[col]
                dx = abs(hole['x'] - donor['eval_anchor_x'])
                if dx > 0.55 * pitch:
                    continue
                (prior_x, prior_y, prior_source) = _point(target)
                prior_residual = abs(hole['x'] - prior_x) / pitch
                cost = dx / pitch
                if prior_source == 'SEG_FREE_HOLE_CORRECTION':
                    cost = 0.6 * cost + 0.4 * prior_residual
                candidate_pairs.append((cost, target['slot_name'], hole['id'], col, neutral_ring))
        (selected, slots_used, holes_used) = ([], set(), set())
        for (cost, slot, hole_id, col, neutral_ring) in sorted(candidate_pairs):
            if slot in slots_used or hole_id in holes_used:
                continue
            slots_used.add(slot)
            holes_used.add(hole_id)
            selected.append({**holes[hole_id], 'slot_name': slot, 'col': col, 'association_cost': float(cost), 'neutral_ring_share': neutral_ring})
        if len(selected) < 3 or max((h['col'] for h in selected)) - min((h['col'] for h in selected)) < 3:
            target_meta['opposite_row_review'] = {'status': 'INSUFFICIENT_TARGET_ROW_RAW_HOLES'}
            continue
        selected.sort(key=lambda h: h['col'])
        if any((b['x'] <= a['x'] for (a, b) in zip(selected, selected[1:]))):
            target_meta['opposite_row_review'] = {'status': 'NONMONOTONIC_TARGET_ROW'}
            continue
        y_line = _line([(h['col'], h['y']) for h in selected])
        tolerance = max(5.0, 0.12 * row_gap)
        selected = [h for h in selected if abs(h['y'] - y_line[0] * h['col'] - y_line[1]) <= tolerance]
        if len(selected) < 3:
            target_meta['opposite_row_review'] = {'status': 'TARGET_ROW_POSE_CONFLICT'}
            continue
        bridges = []
        for hole in selected:
            donor = donors[hole['col']]
            points = np.linspace([hole['x'], hole['y']], [donor['eval_anchor_x'], donor['eval_anchor_y']], 30)[5:25]
            xs = np.clip(np.rint(points[:, 0]).astype(int), 0, source.shape[1] - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(int), 0, source.shape[0] - 1)
            bridges.append(float(neutral[ys, xs].mean()))
        if max(bridges, default=0.0) < 0.2:
            target_meta['opposite_row_review'] = {'status': 'NO_BOARD_SUPPORT_BETWEEN_ROWS', 'bridge_neutral_shares': bridges}
            continue
        y_line = _line([(h['col'], h['y']) for h in selected])
        cols = [h['col'] for h in selected]
        shifts = [h['x'] - donors[h['col']]['eval_anchor_x'] for h in selected]
        if np.ptp(shifts) > 0.75 * pitch or any((abs(shift) > 0.55 * pitch for shift in shifts)):
            target_meta['opposite_row_review'] = {'status': 'OPPOSITE_ROW_PHASE_WARP_CONFLICT'}
            continue
        direct = {h['slot_name']: h for h in selected}
        proposal_points = []
        for target in targets:
            col = _col(target)
            if col not in donors:
                continue
            donor = donors[col]
            x = float(donor['eval_anchor_x'] + np.interp(col, cols, shifts))
            y = float(y_line[0] * col + y_line[1])
            separation = donor['eval_anchor_y'] - y if target_name == 't' else y - donor['eval_anchor_y']
            if separation < 0.5 * row_gap:
                continue
            target.update(eval_anchor_x=x, eval_anchor_y=y, anchor_registration_status='REGISTERED_OPPOSITE_ROW_REVIEW', anchor_registration_source='OPPOSITE_ROW_PHASE_AND_TARGET_RAW_HOLE_POSE', anchor_registration_reasons='PHASE_BORROWED_SAME_BOARD_IDENTITY_REVIEW', anchor_registration_review=True, eval_hole_visible=False, eval_hole_evidence='POSE_PROPOSAL_REVIEW', eval_hole_strength='REVIEW')
            if target['slot_name'] in direct:
                hole = direct[target['slot_name']]
                target.update(eval_anchor_x=hole['x'], eval_anchor_y=hole['y'], eval_hole_x=hole['x'], eval_hole_y=hole['y'], eval_hole_score=hole['score'], eval_hole_evidence='RAW_HOLE_LOCATION_REVIEW')
            count += 1
            proposal_points.append({'slot_name': target['slot_name'], 'x': target['eval_anchor_x'], 'y': target['eval_anchor_y']})
        target_meta.update(status='REGISTERED_OPPOSITE_ROW_REVIEW', registered_holes=selected, opposite_row_review={'status': 'REVIEW_PHASE_AND_TARGET_POSE', 'donor_row': other_name, 'line_y_from_col': list(y_line), 'raw_hole_col_span': [min(cols), max(cols)], 'bridge_neutral_shares': bridges, 'proposed_anchors': proposal_points, 'same_board_identity_confirmed': False})
    return count

def register_frame(rows, source):
    """Return copied rows and JSON-safe metadata; never alter upstream evidence.

    Reliable raw-hole matches become VISIBLE_HOLE evidence. No detected hole does
    not mean occupied, and an unreliable row remains UNKNOWN. Occupancy decisions
    must still distinguish leaves crossing the independent anchor from a body.
    """
    updated = [dict(r) for r in rows]
    camera = str(updated[0].get('camera', '')) if updated else ''
    metadata = {'version': VERSION, 'camera': camera, 'uses_seg_centroids': False, 'raw_holes': [], 'rows': {}, 'status': 'UNKNOWN'}
    for row in updated:
        (x, y, point_source) = _point(row)
        row.update(eval_anchor_x=x, eval_anchor_y=y, anchor_registration_status='UNKNOWN_REGISTRATION', anchor_registration_source=point_source, anchor_registration_version=VERSION, anchor_registration_review=False, eval_hole_visible=False, eval_hole_x='', eval_hole_y='', eval_hole_score='', eval_hole_evidence='UNKNOWN', eval_hole_strength='UNKNOWN', anchor_registration_reasons='')
    if camera == 'cam2':
        if not CAM2_HELPER_FILE.is_file():
            metadata.update(status='CAM2_REGISTRATION_HELPER_UNAVAILABLE', error=str(CAM2_HELPER_FILE))
            return (updated, metadata)
        module_name = '_cam2_physical_anchor_registration_ver12'
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, CAM2_HELPER_FILE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        (output, detail) = module.register_frame(rows, source)
        detail.update(main_anchor_registration_version=VERSION, delegate_helper_path=str(CAM2_HELPER_FILE), uses_seg_centroids=False)
        for row in output:
            row['anchor_registration_review'] = bool(row.get('cam2_registration_review', False))
        return (output, detail)
    if not updated or camera not in ('cam0', 'cam1'):
        metadata['status'] = 'UNSUPPORTED_CAMERA' if updated else 'NO_ROWS'
        return (updated, metadata)
    try:
        holes = _raw_holes(source)
    except Exception as exc:
        metadata.update(status='HOLE_DETECTOR_UNAVAILABLE', error=str(exc))
        return (updated, metadata)
    metadata['raw_holes'] = holes
    grouped = {name: [r for r in updated if str(r.get('row', r.get('slot_name', '')[:1])) == name] for name in ('t', 'b')}
    row_models = {}
    for (name, group) in grouped.items():
        points = [(_number(r, 'x_prior'), _number(r, 'y_prior')) for r in group]
        points = [p for p in points if None not in p]
        if points:
            row_models[name] = _line(points)
    if len(row_models) < 2:
        metadata['status'] = 'ROW_GEOMETRY_UNAVAILABLE'
        return (updated, metadata)
    middle_x = source.shape[1] / 2.0
    row_gap = max(16.0, abs((row_models['b'][0] - row_models['t'][0]) * middle_x + row_models['b'][1] - row_models['t'][1]))
    for hole in holes:
        residuals = {name: abs(hole['y'] - line[0] * hole['x'] - line[1]) for (name, line) in row_models.items()}
        best = min(residuals, key=residuals.get)
        hole['registered_row_candidate'] = best
        hole['prior_row_residual_px'] = float(residuals[best])
    registered_count = 0
    for (name, group) in grouped.items():
        if not group:
            continue
        pitch = _pitch(group)
        possible = []
        for (row_index, row) in enumerate(group):
            (x, y, _) = _point(row)
            evidence = str(row.get('evidence_kind', ''))
            if 'hole' not in evidence and evidence != 'template':
                continue
            if evidence == 'template' and (not (0 <= x < source.shape[1] and 0 <= y < source.shape[0])):
                continue
            for hole in holes:
                if hole['score'] < MIN_HOLE_SCORE or hole['registered_row_candidate'] != name or hole['prior_row_residual_px'] > INITIAL_Y_GATE_ROW_GAP * row_gap:
                    continue
                (dx, dy) = (abs(hole['x'] - x), abs(hole['y'] - y))
                if dx > INITIAL_X_GATE_PITCH * pitch or dy > INITIAL_Y_GATE_ROW_GAP * row_gap:
                    continue
                cost = dx / pitch + dy / row_gap + 0.35 * max(0, 160.0 - hole['score']) / 160.0
                possible.append((cost, row_index, hole['id']))
        (taken_rows, taken_holes, seeds) = (set(), set(), [])
        for (cost, row_index, hole_id) in sorted(possible):
            if row_index in taken_rows or hole_id in taken_holes:
                continue
            (row, hole) = (group[row_index], holes[hole_id])
            taken_rows.add(row_index)
            taken_holes.add(hole_id)
            seeds.append({**hole, 'col': _col(row), 'slot_name': row['slot_name'], 'association_cost': float(cost)})
        tolerance = max(5.0, ROW_RESIDUAL_GATE_ROW_GAP * row_gap)
        phase_details = {}
        if camera == 'cam1':
            (kept, line, phase_details) = _cam1_right_phase(group, holes, pitch, row_gap, name)
        else:
            (kept, line, phase_details) = _cam0_raw_reselect(group, holes, seeds, pitch, row_gap, name, source)
        metadata['rows'][name] = {'pitch_px': float(pitch), 'row_gap_px': float(row_gap), 'initial_seeds': seeds, 'registered_holes': kept, 'residual_tolerance_px': float(tolerance), 'status': 'REGISTERED' if kept else 'UNKNOWN', 'line_y_from_x': list(line) if line is not None else None, 'phase_registration': phase_details}
        if not kept:
            for row in group:
                row['anchor_registration_reasons'] = 'INSUFFICIENT_CONSISTENT_RAW_HOLES'
            continue
        direct = {seed['slot_name']: seed for seed in kept}
        for row in group:
            col = _col(row)
            extra = max(kept[0]['col'] - col, col - kept[-1]['col'], 0)
            if extra > MAX_EXTRAPOLATION_COLS:
                row['anchor_registration_reasons'] = 'OUTSIDE_REGISTERED_HOLE_SPAN'
                continue
            (x, y) = _interpolate(kept, col)
            row.update(eval_anchor_x=x, eval_anchor_y=y, anchor_registration_source='RAW_HOLES_WITH_APPROVED_CANONICAL_NUMBERS', anchor_registration_status='REGISTERED_EXTRAPOLATED' if extra else 'REGISTERED_INTERPOLATED')
            if row['slot_name'] in direct:
                hole = direct[row['slot_name']]
                row.update(eval_anchor_x=hole['x'], eval_anchor_y=hole['y'], anchor_registration_status='REGISTERED_RAW_HOLE', eval_hole_visible=True, eval_hole_x=hole['x'], eval_hole_y=hole['y'], eval_hole_score=hole['score'], eval_hole_evidence='VISIBLE_HOLE', eval_hole_strength='STRONG' if hole['score'] >= 110.0 else 'MODERATE')
            else:
                weak = [h for h in holes if 50.0 <= h['score'] < MIN_HOLE_SCORE and h['registered_row_candidate'] == name and (abs(h['x'] - x) <= 0.12 * pitch) and (abs(h['y'] - y) <= 0.1 * row_gap) and (abs(h['y'] - line[0] * h['x'] - line[1]) <= tolerance)]
                if weak:
                    hole = min(weak, key=lambda h: math.hypot((h['x'] - x) / pitch, (h['y'] - y) / row_gap))
                    row.update(eval_anchor_x=hole['x'], eval_anchor_y=hole['y'], anchor_registration_source='BOUNDED_WEAK_RAW_HOLE_LOCATION_ONLY', eval_hole_x=hole['x'], eval_hole_y=hole['y'], eval_hole_score=hole['score'], eval_hole_evidence='WEAK_HOLE_LOCATION_ONLY', eval_hole_strength='WEAK')
            registered_count += 1
    if camera == 'cam1':
        registered_count += _opposite_row_review(updated, holes, metadata, source)
    has_review = any((r.get('anchor_registration_review') for r in updated))
    metadata['status'] = ('REGISTERED_WITH_REVIEW' if has_review else 'REGISTERED') if registered_count == len(updated) else 'PARTIALLY_REGISTERED' if registered_count else 'UNKNOWN'
    metadata['registered_slot_count'] = registered_count
    metadata['slot_count'] = len(updated)
    return (updated, metadata)
