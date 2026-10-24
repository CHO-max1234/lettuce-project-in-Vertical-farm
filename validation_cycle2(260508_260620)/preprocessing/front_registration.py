"""Register front-view holes with the cycle-specific rail and row constraints."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
SUPPORT_ROOT = CYCLE.MAIN_PREPROCESS / '_support' / 'cam01'
VERSION = 'cam2_raw_frontal_hole_registration_ver1.2'
MIN_HOLE_SCORE = 60.0
DISTANCE_GATE_PITCH = 0.4
ROW_GATE_GAP = 0.4
ROW_RESIDUAL_GAP = 0.1
MIN_ROW_HOLES = 3
MAX_EXTRAPOLATION_COLS = 1
import importlib
import math
import re
import sys
from typing import Any, Mapping, Sequence
import numpy as np

def _number(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ''))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def _col(row: Mapping[str, Any]) -> int | None:
    match = re.fullmatch('[tb](\\d+)', str(row.get('physical_slot') or row.get('slot_name') or ''))
    if match:
        return int(match[1])
    value = _number(row, 'col')
    return int(value) if value is not None and value > 0 else None

def _line(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    xy = np.asarray(points, dtype=float)
    if len(xy) < 2 or float(np.ptp(xy[:, 0])) < 1.0:
        return (0.0, float(np.median(xy[:, 1])))
    (slope, intercept) = np.polyfit(xy[:, 0], xy[:, 1], 1)
    return (float(slope), float(intercept))

def _pitch(rows: Sequence[Mapping[str, Any]]) -> float | None:
    points = sorted(((_col(row), _number(row, 'x_prior')) for row in rows if _col(row) is not None and _number(row, 'x_prior') is not None))
    gaps = [(x2 - x1) / (c2 - c1) for ((c1, x1), (c2, x2)) in zip(points, points[1:]) if c2 > c1 and x2 > x1]
    return float(np.median(gaps)) if gaps else None

def detect_raw_holes(source: np.ndarray) -> list[dict[str, Any]]:
    """Reuse the frozen CPU detector; provisional image-band row is diagnostic."""
    support = str(SUPPORT_ROOT)
    if support not in sys.path:
        sys.path.insert(0, support)
    adapter = importlib.import_module('cam0_geometry')
    detected = adapter.detect_hole_evidence(source)
    result = []
    for (provisional_row, holes) in detected.items():
        for hole in holes:
            result.append({'id': len(result), 'x': float(hole['x']), 'y': float(hole['y']), 'score': float(hole['score']), 'detector_row': provisional_row})
    return result

def _robust_row(seeds: Sequence[Mapping[str, Any]], tolerance: float):
    """Select a physical row using raw-hole agreement, without plant evidence."""
    if len(seeds) < MIN_ROW_HOLES:
        return ([], None, None)
    options = []
    for (i, first) in enumerate(seeds):
        for second in seeds[i + 1:]:
            dx = second['x'] - first['x']
            if abs(dx) < 1.0:
                continue
            slope = (second['y'] - first['y']) / dx
            if abs(slope) > 0.2:
                continue
            intercept = first['y'] - slope * first['x']
            kept = [dict(seed) for seed in seeds if abs(seed['y'] - slope * seed['x'] - intercept) <= tolerance]
            if len(kept) < MIN_ROW_HOLES:
                continue
            span = max((seed['col'] for seed in kept)) - min((seed['col'] for seed in kept))
            if span < 2:
                continue
            residual = sum((abs(seed['y'] - slope * seed['x'] - intercept) for seed in kept))
            options.append((len(kept), span, -residual, sum((seed['score'] for seed in kept)), kept))
    if not options:
        return ([], None, None)
    kept = max(options, key=lambda option: option[:4])[4]
    for _ in range(2):
        line = _line([(seed['x'], seed['y']) for seed in kept])
        proposed = [dict(seed) for seed in seeds if abs(seed['y'] - line[0] * seed['x'] - line[1]) <= tolerance]
        if len(proposed) >= MIN_ROW_HOLES:
            kept = proposed
    kept.sort(key=lambda seed: seed['col'])
    if len(kept) < MIN_ROW_HOLES or kept[-1]['col'] - kept[0]['col'] < 2 or any((second['x'] <= first['x'] for (first, second) in zip(kept, kept[1:]))):
        return ([], None, None)
    line = _line([(seed['x'], seed['y']) for seed in kept])
    rms = math.sqrt(float(np.mean([(seed['y'] - line[0] * seed['x'] - line[1]) ** 2 for seed in kept])))
    return (kept, line, rms)

def _interpolate(seeds, col):
    if col <= seeds[0]['col']:
        (first, second) = seeds[:2]
    elif col >= seeds[-1]['col']:
        (first, second) = seeds[-2:]
    else:
        (first, second) = next(((a, b) for (a, b) in zip(seeds, seeds[1:]) if a['col'] <= col <= b['col']))
    fraction = (col - first['col']) / (second['col'] - first['col'])
    return (float(first['x'] + fraction * (second['x'] - first['x'])), float(first['y'] + fraction * (second['y'] - first['y'])))

def register_frame(rows: Sequence[Mapping[str, Any]], source: np.ndarray):
    """Register copied CAM2 rows; metadata contains raw evidence and rejections."""
    if source.ndim < 2:
        raise ValueError('CAM2 source requires an image')
    (height, width) = source.shape[:2]
    updated = [dict(row) for row in rows]
    metadata = {'version': VERSION, 'camera': 'cam2', 'source_width': int(width), 'source_height': int(height), 'slot_count': len(updated), 'registered_slot_count': 0, 'raw_holes': [], 'rows': {}, 'rejected_holes': [], 'status': 'UNKNOWN', 'uses_seg_centroids': False, 'uses_evaluation_occupancy': False, 'association_policy': 'COARSE_PHYSICAL_PRIOR_EUCLIDEAN_DISTANCE_LT_0_40_PITCH', 'row_policy': 'TWO_PHYSICAL_PRIOR_LINES_THEN_RAW_HOLE_CONSENSUS', 'original_fields_preserved': True}
    for row in updated:
        (x, y) = (_number(row, 'x_prior'), _number(row, 'y_prior'))
        valid = x is not None and y is not None and (_col(row) is not None)
        row.update(eval_anchor_x=x if x is not None else -1.0, eval_anchor_y=y if y is not None else -1.0, anchor_registration_status='UNKNOWN_REGISTRATION', anchor_registration_source='APPROVED_CAM2_COARSE_PHYSICAL_PRIOR', anchor_registration_reasons='NO_CONSISTENT_RAW_FRONTAL_HOLE_ROW' if valid else 'INVALID_APPROVED_PHYSICAL_PRIOR', eval_hole_visible=False, eval_hole_x='', eval_hole_y='', eval_hole_score='', eval_hole_evidence='UNKNOWN', eval_hole_strength='UNKNOWN', cam2_registration_prior_in_fov=bool(valid and 0 <= x < width and (0 <= y < height)), cam2_registration_review=True)
    if not updated:
        metadata['status'] = 'NO_ROWS'
        return (updated, metadata)
    if any((str(row.get('camera', 'cam2')) != 'cam2' for row in updated)):
        metadata['status'] = 'UNSUPPORTED_CAMERA'
        return (updated, metadata)
    grouped = {name: [row for row in updated if str(row.get('row') or str(row.get('slot_name', ''))[:1]) == name] for name in ('t', 'b')}
    prior_lines = {}
    pitches = {}
    for (name, group) in grouped.items():
        points = [(_number(row, 'x_prior'), _number(row, 'y_prior')) for row in group]
        points = [point for point in points if None not in point]
        if len(points) >= 2:
            prior_lines[name] = _line(points)
        pitches[name] = _pitch(group)
    if len(prior_lines) != 2 or any((value is None for value in pitches.values())):
        metadata['status'] = 'ROW_GEOMETRY_UNAVAILABLE'
        return (updated, metadata)
    middle_x = width / 2.0
    row_gap = abs((prior_lines['b'][0] - prior_lines['t'][0]) * middle_x + prior_lines['b'][1] - prior_lines['t'][1])
    if row_gap < 16.0:
        metadata['status'] = 'PHYSICAL_ROW_GAP_UNAVAILABLE'
        return (updated, metadata)
    try:
        holes = detect_raw_holes(source)
    except Exception as error:
        metadata.update(status='HOLE_DETECTOR_UNAVAILABLE', error=repr(error))
        return (updated, metadata)
    metadata['raw_holes'] = holes
    for hole in holes:
        residuals = {name: abs(hole['y'] - line[0] * hole['x'] - line[1]) for (name, line) in prior_lines.items()}
        name = min(residuals, key=residuals.get)
        hole.update(physical_row_candidate=name, prior_row_residual_px=float(residuals[name]))
        if hole['score'] < MIN_HOLE_SCORE:
            metadata['rejected_holes'].append({'hole_id': hole['id'], 'reason': 'WEAK_DARK_ELLIPSE_CONTRAST'})
        elif residuals[name] >= ROW_GATE_GAP * row_gap:
            metadata['rejected_holes'].append({'hole_id': hole['id'], 'reason': 'OUTSIDE_TARGET_PHYSICAL_ROW_WINDOW'})
    registered_count = 0
    for (name, group) in grouped.items():
        pitch = pitches[name]
        possible = []
        for (row_index, row) in enumerate(group):
            (x, y, col) = (_number(row, 'x_prior'), _number(row, 'y_prior'), _col(row))
            if None in (x, y, col):
                continue
            for hole in holes:
                if hole['score'] < MIN_HOLE_SCORE or hole['physical_row_candidate'] != name or hole['prior_row_residual_px'] >= ROW_GATE_GAP * row_gap:
                    continue
                distance = math.hypot(hole['x'] - x, hole['y'] - y)
                if distance >= DISTANCE_GATE_PITCH * pitch:
                    continue
                possible.append((distance, -hole['score'], row_index, hole['id']))
        (used_rows, used_holes, seeds) = (set(), set(), [])
        for (distance, _, row_index, hole_id) in sorted(possible):
            if row_index in used_rows or hole_id in used_holes:
                continue
            used_rows.add(row_index)
            used_holes.add(hole_id)
            (row, hole) = (group[row_index], holes[hole_id])
            seeds.append({**hole, 'col': _col(row), 'slot_name': row['slot_name'], 'prior_match_distance_px': float(distance)})
        tolerance = max(5.0, ROW_RESIDUAL_GAP * row_gap)
        (kept, line, rms) = _robust_row(seeds, tolerance)
        metadata['rows'][name] = {'status': 'REGISTERED' if kept else 'UNKNOWN', 'pitch_px': float(pitch), 'row_gap_px': float(row_gap), 'prior_line_y_from_x': list(prior_lines[name]), 'line_y_from_x': list(line) if line is not None else None, 'line_rms_px': rms, 'residual_tolerance_px': float(tolerance), 'initial_seeds': seeds, 'registered_holes': kept, 'registered_col_span': [kept[0]['col'], kept[-1]['col']] if kept else []}
        kept_ids = {seed['id'] for seed in kept}
        for seed in seeds:
            if seed['id'] not in kept_ids:
                metadata['rejected_holes'].append({'hole_id': seed['id'], 'reason': 'RAW_HOLE_ROW_CONSENSUS_OUTLIER'})
        if not kept:
            continue
        direct = {seed['slot_name']: seed for seed in kept}
        for row in group:
            col = _col(row)
            if col is None:
                continue
            (x, y) = _interpolate(kept, col)
            extra = max(kept[0]['col'] - col, col - kept[-1]['col'], 0)
            row.update(cam2_registration_row_rms_px=float(rms), cam2_registration_raw_hole_count=len(kept), cam2_registration_col_span=[kept[0]['col'], kept[-1]['col']], cam2_registration_extrapolated_cols=int(extra), cam2_registration_proposed_x=x, cam2_registration_proposed_y=y, cam2_registration_proposed_in_fov=bool(0 <= x < width and 0 <= y < height))
            if extra > MAX_EXTRAPOLATION_COLS:
                row['anchor_registration_reasons'] = 'OUTSIDE_SUPPORTED_RAW_HOLE_SPAN'
                continue
            if extra and (not row['cam2_registration_prior_in_fov']):
                row['anchor_registration_reasons'] = 'OUT_OF_FOV_PRIOR_UNSUPPORTED_EXTRAPOLATION'
                continue
            row.update(eval_anchor_x=x, eval_anchor_y=y, anchor_registration_status='REGISTERED_EXTRAPOLATED' if extra else 'REGISTERED_INTERPOLATED', anchor_registration_source='CAM2_RAW_FRONTAL_HOLES_APPROVED_PHYSICAL_COLUMNS', anchor_registration_reasons='ONE_COLUMN_EXTRAPOLATION_REVIEW' if extra else '', cam2_registration_review=bool(extra))
            if row['slot_name'] in direct:
                hole = direct[row['slot_name']]
                row.update(eval_anchor_x=hole['x'], eval_anchor_y=hole['y'], anchor_registration_status='REGISTERED_RAW_HOLE', anchor_registration_reasons='', cam2_registration_review=False, eval_hole_visible=True, eval_hole_x=hole['x'], eval_hole_y=hole['y'], eval_hole_score=hole['score'], eval_hole_evidence='VISIBLE_HOLE', eval_hole_strength='STRONG' if hole['score'] >= 100.0 else 'MODERATE')
            registered_count += 1
    metadata['registered_slot_count'] = registered_count
    metadata['status'] = 'REGISTERED' if registered_count == len(updated) else 'PARTIALLY_REGISTERED' if registered_count else 'UNKNOWN'
    metadata['unknown_slot_count'] = len(updated) - registered_count
    return (updated, metadata)
