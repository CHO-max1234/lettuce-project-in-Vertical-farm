"""Top geometry calculations used by the final cycle pipeline."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from copy import deepcopy
import math
import cv2
import numpy as np
VERSION = 'S4_CAM01_VER2_HOLE_PAIR_SHARED_CURVATURE'

def _odd(value):
    return max(3, int(round(value)) // 2 * 2 + 1)

def detect_holes(image):
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 65)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    close = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(w * 0.06), _odd(w * 0.05))))
    contrast = close.astype(np.float32) - blurred
    masks = [(contrast > threshold) & (gray < 205) & ~green for threshold in (24, 38, 52)]
    holes = []
    for (mask_index, mask) in enumerate(masks):
        binary = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 12:
                continue
            (x, y, cw, ch) = cv2.boundingRect(contour)
            if not 0.01 * w <= max(cw, ch) <= 0.045 * w or not 0.007 * w <= min(cw, ch):
                continue
            area = cv2.contourArea(contour)
            if area <= 25:
                continue
            ((cx, cy), (aa, bb), angle) = cv2.fitEllipse(contour)
            (major, minor) = (max(aa, bb), min(aa, bb))
            if not 0.012 * w <= major <= 0.045 * w or minor / max(major, 1) < 0.48:
                continue
            fill = area / max(math.pi * aa * bb / 4, 1)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / max(hull_area, 1)
            if not 0.65 <= fill <= 1.2 or solidity < 0.75:
                continue
            center_mask = np.zeros((ch, cw), np.uint8)
            local = contour - np.array([[[x, y]]])
            cv2.drawContours(center_mask, [local], -1, 1, -1)
            means = contrast[y:y + ch, x:x + cw][center_mask.astype(bool)]
            support = float(np.mean(means)) if means.size else 0.0
            native_values = gray[y:y + ch, x:x + cw][center_mask.astype(bool)]
            dark_fraction = float(np.mean(native_values < 120)) if native_values.size else 0.0
            if dark_fraction < 0.24 or float(np.percentile(native_values, 15)) > 105:
                continue
            candidate = dict(x=float(cx), y=float(cy), diameter=float(major), minor=float(minor), score=float(fill * solidity + support / 120), contour=contour[:, 0, :].tolist(), dark_fraction=dark_fraction, method='LOCAL_BLACKHAT_ELLIPSE', mask_index=mask_index)
            if candidate['score'] < 1.05:
                continue
            matching = [i for (i, p) in enumerate(holes) if math.hypot(p['x'] - cx, p['y'] - cy) < 0.01 * w]
            if matching:
                old = matching[0]
                if candidate['score'] > holes[old]['score']:
                    holes[old] = candidate
            else:
                holes.append(candidate)
    return sorted(holes, key=lambda p: (p['y'], p['x']))

def detect_plate_boundaries(image):
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    nonleaf = (hsv[:, :, 1] < 95).astype(np.uint8) * 255
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 100)
    edges &= nonleaf
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(40, round(0.04 * w)), minLineLength=round(0.18 * w), maxLineGap=round(0.018 * w))
    result = []
    for line in [] if lines is None else lines.reshape(-1, 4):
        (x1, y1, x2, y2) = map(float, line)
        if abs(x2 - x1) < 0.18 * w or abs((y2 - y1) / (x2 - x1)) > 0.11:
            continue
        if x2 < x1:
            (x1, y1, x2, y2) = (x2, y2, x1, y1)
        result.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, slope=(y2 - y1) / (x2 - x1), span=x2 - x1))
    return result

def _row_candidates(holes, w, h, x_min, x_max):
    selected = [p for p in holes if x_min < p['x'] < x_max]
    if len(selected) < 2:
        return []
    x = np.array([p['x'] for p in selected])
    y = np.array([p['y'] for p in selected])
    tolerance = max(8.0, 0.008 * w)
    fits = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if abs(x[j] - x[i]) < 0.18 * w:
                continue
            slope = (y[j] - y[i]) / (x[j] - x[i])
            if abs(slope) > 0.105:
                continue
            intercept = y[i] - slope * x[i]
            support = np.flatnonzero(np.abs(y - (slope * x + intercept)) <= tolerance)
            if len(support) < 2 or np.ptp(x[support]) < 0.2 * w:
                continue
            (xs, ys) = (x[support], y[support])
            model = np.polyfit(xs, ys, 1)
            residual = np.abs(ys - np.polyval(model, xs))
            score = len(support) * 4 + np.ptp(xs) / w * 3 - float(np.median(residual)) / tolerance
            key = tuple(sorted(support.tolist()))
            if any((v['support_key'] == key for v in fits)):
                continue
            fits.append(dict(coefficients=model.tolist(), support_key=key, holes=[selected[k] for k in support], score=float(score), center_y=float(np.polyval(model, 0.55 * w)), residual_median=float(np.median(residual)), span=float(np.ptp(xs))))
    return sorted(fits, key=lambda p: -p['score'])

def _as_float(record, key):
    return float(record[key])

def _register_x_lattice(points, baseline_x, camera, width):
    baseline_x = np.asarray(baseline_x, float)
    old_pitch = float(np.median(np.diff(baseline_x)))
    count = len(baseline_x)
    result = dict(status='REVIEW', method='UPSTREAM_X_PRESERVED', matched_holes=0)
    if len(points) < 3:
        return (baseline_x.copy(), result)
    values = np.array(sorted((p['x'] for p in points)))
    (lower, upper) = (0.7 * old_pitch, 1.35 * old_pitch)
    pitch_candidates = [old_pitch]
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            for steps in range(1, count):
                candidate = (values[b] - values[a]) / steps
                if lower <= candidate <= upper:
                    pitch_candidates.append(candidate)
    best = None
    for pitch in pitch_candidates:
        for value in values:
            for index in range(count):
                origin = value - index * pitch
                coordinates = origin + np.arange(count) * pitch
                anchor_index = 0 if camera == 'cam0' else count - 1
                if abs(coordinates[anchor_index] - baseline_x[anchor_index]) > 0.55 * old_pitch:
                    continue
                if camera == 'cam0' and (coordinates[0] < 0.125 * width or coordinates[-1] > 0.995 * width):
                    continue
                labels = np.rint((values - origin) / pitch).astype(int)
                residual = np.abs(values - (origin + labels * pitch))
                keep = (labels >= 0) & (labels < count) & (residual < 0.12 * pitch)
                ids = labels[keep]
                if len(ids) < 3 or len(set(ids)) != len(ids) or np.ptp(ids) < 2:
                    continue
                score = float(len(ids) * 5 + np.ptp(ids) * 0.3 - np.median(residual[keep]) / pitch - abs(pitch - old_pitch) / old_pitch * 0.3)
                if best is None or score > best[0]:
                    best = (score, ids, values[keep], pitch, origin)
    if best is None:
        return (baseline_x.copy(), result)
    (score, ids, observed, pitch, origin) = best
    labels = np.rint((values - origin) / pitch).astype(int)
    residual = np.abs(values - (origin + labels * pitch))
    expanded = (labels >= 0) & (labels < count) & (residual < 0.24 * pitch)
    if len(set(labels[expanded])) == sum(expanded):
        (ids, observed) = (labels[expanded], values[expanded])
    degree = 2 if len(ids) >= 4 and np.ptp(ids) >= 4 else 1
    coefficient = np.polyfit(ids, observed, degree)
    coordinates = np.polyval(coefficient, np.arange(count))
    diffs = np.diff(coordinates)
    fit_residual = np.abs(np.polyval(coefficient, ids) - observed)
    if np.min(diffs) < 0.65 * pitch or np.max(diffs) > 1.4 * pitch or np.max(fit_residual) > 0.11 * pitch or (abs(coordinates[0 if camera == 'cam0' else -1] - baseline_x[0 if camera == 'cam0' else -1]) > 0.55 * old_pitch):
        return (baseline_x.copy(), dict(result, reason='BOUNDED_LATTICE_FIT_FAILED'))
    unmatched = [float(v) for v in values if min(abs(v - coordinates)) > 0.15 * pitch]
    result = dict(status='SUPPORTED', method='PHYSICAL_HOLE_LATTICE_FIXED_IDS', matched_holes=len(ids), fixed_local_indices=ids.tolist(), observed_x=observed.tolist(), coefficients=coefficient.tolist(), model_degree=degree, pitch=float(np.median(diffs)), fit_residual_max=float(np.max(fit_residual)), unmatched_hole_x=unmatched)
    return (coordinates, result)

def _interpolate_knots(indices, observed, requested):
    order = np.argsort(indices)
    indices = np.asarray(indices, float)[order]
    observed = np.asarray(observed, float)[order]
    requested = np.asarray(requested, float)
    values = np.interp(requested, indices, observed)
    left = (observed[1] - observed[0]) / (indices[1] - indices[0])
    right = (observed[-1] - observed[-2]) / (indices[-1] - indices[-2])
    values = np.where(requested < indices[0], observed[0] + (requested - indices[0]) * left, values)
    values = np.where(requested > indices[-1], observed[-1] + (requested - indices[-1]) * right, values)
    return values

def _register_cam1_perspective(points, records, width, bottom_reference=None, camera='cam1'):
    baseline = np.array([float(r['x_prior']) for r in records])
    pitch = float(np.median(np.diff(baseline)))
    out = dict(status='REVIEW', method='UPSTREAM_X_PRESERVED', matched_holes=0)
    points = sorted(points, key=lambda p: p['x'])
    if len(points) < 2:
        return (baseline.copy(), out)
    requested = np.array([int(str(r['physical_slot'])[1:]) for r in records])
    endpoint = points[-1]
    if camera == 'cam0':
        endpoint = points[0]
        index = int(np.argmin(abs(baseline - endpoint['x'])))
        anchor_error = float(abs(endpoint['x'] - baseline[index]))
        seed = int(requested[index])
        if anchor_error > 0.45 * pitch:
            return (baseline.copy(), dict(out, reason='LEFT_FIXED_ANCHOR_PHASE_UNSUPPORTED'))
    elif bottom_reference is None:
        if endpoint['x'] < 0.84 * width or abs(endpoint['x'] - baseline[-1]) > 0.9 * pitch:
            return (baseline.copy(), dict(out, reason='RIGHT12_PHYSICAL_ENDPOINT_NOT_SUPPORTED'))
        seed = 12
        anchor_error = float(abs(endpoint['x'] - baseline[-1]))
    else:
        candidates = np.arange(1, 13)
        bx = _interpolate_knots(bottom_reference['physical_columns'], bottom_reference['observed_x'], candidates)
        index = int(np.argmin(abs(bx - endpoint['x'])))
        local_pitch = float(np.median(np.diff(bx[max(0, index - 1):min(len(bx), index + 2)])))
        anchor_error = float(abs(bx[index] - endpoint['x']))
        if anchor_error > 0.34 * local_pitch:
            return (baseline.copy(), dict(out, reason='TOP_COLUMN_PHASE_UNSUPPORTED_BY_BOTTOM'))
        seed = int(candidates[index])
    physical = [seed]
    errors = []
    steps = []
    expected = []
    ordered_pairs = list(zip(points[:-1], points[1:]))
    if camera == 'cam1':
        ordered_pairs.reverse()
    for (left, right) in ordered_pairs:
        local = 4.2 * 0.5 * (left['diameter'] + right['diameter'])
        ratio = (right['x'] - left['x']) / local
        step = int(round(ratio))
        if not 1 <= step <= 6:
            return (baseline.copy(), dict(out, reason='LOCAL_PITCH_STEP_OUT_OF_RANGE'))
        physical.append(physical[-1] + (step if camera == 'cam0' else -step))
        errors.append(abs(ratio - step))
        steps.append(step)
        expected.append(local)
    if camera == 'cam1':
        physical = list(reversed(physical))
    observed = [p['x'] for p in points]
    coordinates = _interpolate_knots(physical, observed, requested)
    increments = np.diff(coordinates)
    if np.min(increments) < 0.045 * width or np.max(increments) > 0.165 * width:
        return (baseline.copy(), dict(out, reason='PERSPECTIVE_LATTICE_SPACING_OUT_OF_RANGE'))
    flags = []
    if max(errors, default=0) > 0.34:
        flags.append('INTEGER_HOLE_STEP_AMBIGUOUS')
    if camera == 'cam1' and bottom_reference is None and (anchor_error > 0.4 * pitch):
        flags.append('RIGHT12_ENDPOINT_DISAGREES_WITH_UPSTREAM')
    if min(physical) < min(requested) or max(physical) > max(requested):
        flags.append('OBSERVED_EXTRA_COLUMNS_OUTSIDE_BASELINE')
    out = dict(status='SUPPORTED_PROPOSAL', method='LEFT_FIXED_OBSERVED_KNOT_PERSPECTIVE' if camera == 'cam0' else 'RIGHT12_OBSERVED_KNOT_PERSPECTIVE', matched_holes=len(points), physical_columns=physical, observed_x=observed, anchor_column=seed, anchor_x=float(endpoint['x']), anchor_error_px=anchor_error, local_pitch_diameter_ratio=4.2, scale_ratio_verified=False, integer_steps_right_to_left=steps, integer_step_residuals=errors, expected_local_pitch_px=expected, phase_flags=flags, numbering_verified=False, unmatched_hole_x=[])
    return (coordinates, out)

def propose_geometry(image, camera, sample, baseline_slots, raw_segments):
    slots = deepcopy(baseline_slots)
    (h, w) = image.shape[:2]
    camera = str(camera).lower()
    meta = dict(version=VERSION, camera=camera, method='HOLE_PAIR_SHARED_CURVATURE', native_width=w, native_height=h, production=False, qc_flags=[])
    for r in slots:
        for key in ('x_prior', 'y_prior', 'anchor_x', 'anchor_y', 'slot_center_x', 'slot_center_y', 'cell_bbox', 'cell_polygon', 'bbox', 'cell_geometry_ready'):
            if key in r and 'upstream_' + key not in r:
                r['upstream_' + key] = deepcopy(r[key])
    if camera not in ('cam0', 'cam1') or not slots:
        meta.update(status='HOLD', geometry_status='HOLD', slot_geometry_usable=False, qc_flags=['NO_CAM01_BASELINE_SLOTS'])
        return (slots, meta)
    holes = detect_holes(image)
    holes = [p for p in holes if p['diameter'] >= (0.022 if camera == 'cam0' else 0.015) * w]
    if camera == 'cam1':
        holes = [p for p in holes if p['score'] >= 1.2]
    meta['hole_candidates'] = holes
    boundaries = detect_plate_boundaries(image)
    meta['plate_boundary_candidates'] = boundaries
    by_row = {row: sorted([s for s in slots if s['row'] == row], key=lambda s: float(s['x_prior'])) for row in ('t', 'b')}
    if any((len(rr) < 2 for rr in by_row.values())):
        meta.update(status='HOLD', geometry_status='HOLD', slot_geometry_usable=False, qc_flags=['INCOMPLETE_BASELINE_ROWS'])
        return (slots, meta)
    baseline_bottom = float(np.median([float(s['y_prior']) for s in by_row['b']]))
    baseline_gap = float(np.median([float(b['y_prior']) - float(t['y_prior']) for (t, b) in zip(by_row['t'], by_row['b'])]))
    x_min = 0.135 * w if camera == 'cam0' else 0.015 * w
    x_max = 0.965 * w if camera == 'cam0' else 0.93 * w
    fits = _row_candidates(holes, w, h, x_min, x_max)
    meta['row_candidate_models'] = [{k: v for (k, v) in f.items() if k != 'support_key'} for f in fits[:20]]
    target_gap_range = (0.083 * w, 0.145 * w) if camera == 'cam0' else (0.08 * w, 0.133 * w)
    pairs = []
    for bottom in fits:
        by = bottom['center_y']
        if abs(by - baseline_bottom) > 0.07 * w or not 0.3 * h < by < 0.95 * h:
            continue
        for top in fits:
            gap = by - top['center_y']
            if camera == 'cam0' and top['center_y'] > 0.51 * h:
                continue
            if not target_gap_range[0] <= gap <= target_gap_range[1]:
                continue
            slope_delta = abs(bottom['coefficients'][0] - top['coefficients'][0])
            if camera == 'cam1' and (bottom['coefficients'][0] > 0.02 or top['coefficients'][0] > 0.025):
                continue
            if slope_delta > 0.06:
                continue
            crossing = []
            for edge in boundaries:
                ex = 0.5 * (edge['x1'] + edge['x2'])
                ey = 0.5 * (edge['y1'] + edge['y2'])
                ty = float(np.polyval(top['coefficients'], ex))
                byx = float(np.polyval(bottom['coefficients'], ex))
                if ty + 0.17 * gap < ey < byx - 0.17 * gap:
                    crossing.append(edge)
            crossing_span = max([e['span'] / w for e in crossing], default=0.0)
            score = bottom['score'] + top['score'] - abs(by - baseline_bottom) / w * 30 - slope_delta * 80 - 30 * crossing_span
            pairs.append((score, top, bottom, gap, crossing_span))
    partial_top = False
    if not pairs and camera == 'cam1':
        lower = [f for f in fits if len(f['holes']) >= 3 and f['span'] > 0.55 * w and (abs(f['center_y'] - baseline_bottom) < 0.07 * w) and (f['coefficients'][0] < 0.025)]
        for bfit in lower:
            lone = [p for p in holes if x_min < p['x'] < x_max and target_gap_range[0] < np.polyval(bfit['coefficients'], p['x']) - p['y'] < target_gap_range[1]]
            if not lone:
                continue
            point = max(lone, key=lambda p: p['score'])
            local_gap = float(np.polyval(bfit['coefficients'], point['x']) - point['y'])
            tfit = dict(coefficients=[bfit['coefficients'][0], bfit['coefficients'][1] - local_gap], holes=[point], score=0.0, center_y=bfit['center_y'] - local_gap, residual_median=0.0, span=0.0)
            pairs.append((bfit['score'], tfit, bfit, local_gap, 0.0))
            partial_top = True
            meta['qc_flags'].append('TOP_CLIPPED_SINGLE_HOLE_PROPOSAL_ONLY')
            break
    if not pairs:
        meta.update(status='HOLD', geometry_status='HOLD', slot_geometry_usable=False, qc_flags=['TARGET_PLATE_TWO_ROW_HOLE_EVIDENCE_INSUFFICIENT'], baseline_bottom_y=baseline_bottom, baseline_row_gap=baseline_gap)
        for r in slots:
            r['geometry_proposal_status'] = 'HOLD'
            r['geometry_proposal_version'] = VERSION
            r['geometry_handoff_status'] = 'GEOMETRY_HOLD'
            r['cell_geometry_ready'] = False
            r['geometry_status'] = 'HOLD'
            r['slot_geometry_usable'] = False
            r['geometry_reason'] = 'TARGET_PLATE_TWO_ROW_HOLE_EVIDENCE_INSUFFICIENT'
        return (slots, meta)
    pairs.sort(key=lambda p: -p[0])
    (score, top, bottom, gap, crossing_span) = pairs[0]
    meta['interior_boundary_crossing_fraction'] = crossing_span
    if crossing_span > 0.2:
        meta['qc_flags'].append('TARGET_PAIR_STRADDLES_PLATE_BOUNDARY')
    for fit in (top, bottom):
        nearby = [p for p in holes if x_min < p['x'] < x_max and abs(p['y'] - np.polyval(fit['coefficients'], p['x'])) < 0.023 * w]
        if len(nearby) >= 3 and np.ptp([p['x'] for p in nearby]) > 0.55 * w:
            ux = np.array([(p['x'] - 0.5 * w) / w for p in nearby])
            vy = np.array([p['y'] for p in nearby])
            q = np.polyfit(ux, vy, 2)
            residual = np.abs(np.polyval(q, ux) - vy)
            if abs(q[0]) < 0.22 * w and np.max(residual) < 0.011 * w:
                fit['holes'] = nearby
    xs = np.array([p['x'] for p in bottom['holes']])
    ys = np.array([p['y'] for p in bottom['holes']])
    xc = 0.5 * w
    u = (xs - xc) / w
    degree = 2 if len(xs) >= 4 and np.ptp(xs) > 0.58 * w else 1
    coef = np.polyfit(u, ys, degree)
    if degree == 2 and abs(coef[0]) > 0.055 * w:
        coef = np.polyfit(u, ys, 1)
        degree = 1
    bottom_fn = lambda xx: float(np.polyval(coef, (float(xx) - xc) / w))
    gaps = np.array([bottom_fn(p['x']) - p['y'] for p in top['holes']])
    gap = float(np.median(gaps))
    gap_mad = float(np.median(np.abs(gaps - gap)))
    if gap_mad > 0.014 * w:
        meta['qc_flags'].append('TOP_BOTTOM_NONPARALLEL_HOLE_EVIDENCE')
    all_points = [(p['x'], p['y'], 0.0) for p in top['holes']] + [(p['x'], p['y'], 1.0) for p in bottom['holes']]
    xp = np.array([v[0] for v in all_points])
    yp = np.array([v[1] for v in all_points])
    rp = np.array([v[2] for v in all_points])
    up = (xp - xc) / w
    joint_degree = 2 if len(bottom['holes']) >= 3 and len(top['holes']) >= 3 else 1
    matrix = np.column_stack([np.ones(len(up)), up] + ([up * up] if joint_degree == 2 else []) + [rp, rp * up])
    validation_matrix = matrix
    joint = np.linalg.lstsq(matrix, yp, rcond=None)[0]
    curvature = joint[2] if joint_degree == 2 else 0.0
    gap_index = 3 if joint_degree == 2 else 2
    gap_slope = float(joint[gap_index + 1])
    if abs(curvature) > 0.1 * w or abs(gap_slope) > 0.03 * w:
        adjusted_y = yp - rp * gap
        base = np.polyfit(up, adjusted_y, 1)
        joint = np.array([base[1], base[0], gap, 0.0])
        joint_degree = 1
        gap_index = 2
        validation_matrix = np.column_stack([np.ones(len(up)), up, rp])
        meta['qc_flags'].append('GAP_TREND_CLAMPED_TO_PARALLEL')
    top_fn = lambda xx: float(joint[0] + joint[1] * ((float(xx) - xc) / w) + (joint[2] * ((float(xx) - xc) / w) ** 2 if joint_degree == 2 else 0.0))
    bottom_fn = lambda xx: float(top_fn(xx) + joint[gap_index] + joint[gap_index + 1] * ((float(xx) - xc) / w))
    gap = float(joint[gap_index])
    loo = []
    if len(top['holes']) >= 3 and len(bottom['holes']) >= 3:
        for held in range(len(yp)):
            keep = np.arange(len(yp)) != held
            restricted = validation_matrix[keep]
            if np.linalg.matrix_rank(restricted) < restricted.shape[1]:
                continue
            fitted = np.linalg.lstsq(restricted, yp[keep], rcond=None)[0]
            loo.append(float(abs(validation_matrix[held] @ fitted - yp[held])))
    meta['leave_one_hole_out_errors_px'] = loo
    meta['leave_one_hole_out_max_px'] = max(loo) if loo else None
    if loo and max(loo) > 0.02 * w:
        meta['qc_flags'].append('HELD_OUT_HOLE_PREDICTION_REVIEW')
    residuals = []
    for (row, fit) in [('t', top), ('b', bottom)]:
        fn = top_fn if row == 't' else bottom_fn
        residuals.extend((abs(p['y'] - fn(p['x'])) for p in fit['holes']))
    independent = len(top['holes']) >= 3 and len(bottom['holes']) >= 3
    meta.update(row_gap=gap, row_gap_mad=gap_mad, bottom_model_degree=degree, bottom_coefficients_native_normalized_x=coef.tolist(), top_hole_support=len(top['holes']), bottom_hole_support=len(bottom['holes']), independent_validation=independent, residual_median=float(np.median(residuals)), residual_max=float(max(residuals)), chosen_top_holes=top['holes'], chosen_bottom_holes=bottom['holes'], target_plate_top=[top_fn(0) - 0.25 * gap, top_fn(w) - 0.25 * gap], target_plate_bottom=[bottom_fn(0) + 0.25 * gap, bottom_fn(w) + 0.25 * gap], baseline_row_gap=baseline_gap, baseline_bottom_y=baseline_bottom, candidate_pair_score=score)
    meta.update(shared_curve_degree=joint_degree, shared_curve_coefficients=joint.tolist(), gap_slope_native=float(joint[gap_index + 1]))
    if not independent:
        meta['qc_flags'].append('SPARSE_HOLE_ROW_VALIDATION_REVIEW')
    status = 'PROPOSED_REVIEW' if meta['qc_flags'] else 'GEOMETRY_SUPPORTED'
    if not independent and gap_mad > 0.014 * w:
        status = 'HOLD'
    cam1_bottom_registration = None
    extra_physical_column_range = None
    if camera == 'cam0':
        registrations = {row: _register_cam1_perspective(fit['holes'], by_row[row], w, camera=camera)[1] for (row, fit) in [('t', top), ('b', bottom)]}
        observed_columns = [value for registration in registrations.values() for value in registration.get('physical_columns', [])]
        if observed_columns:
            extra_physical_column_range = max(observed_columns)
    for (row, fit) in [('b', bottom), ('t', top)]:
        rr = by_row[row]
        baseline_x = np.array([float(r['x_prior']) for r in rr])
        pitch = float(np.median(np.diff(baseline_x)))
        row_holes = fit['holes']
        (adjusted_x, lattice) = _register_cam1_perspective(row_holes, rr, w, cam1_bottom_registration if row == 't' and camera == 'cam1' else None, camera=camera)
        if camera == 'cam1' and row == 'b' and (lattice['status'] == 'SUPPORTED_PROPOSAL'):
            cam1_bottom_registration = lattice
        if lattice['status'] == 'SUPPORTED_PROPOSAL':
            meta['qc_flags'].append('PHYSICAL_COLUMN_NUMBERING_PROPOSAL_REVIEW_' + row.upper())
            for flag in lattice.get('phase_flags', []):
                meta['qc_flags'].append(flag + '_' + row.upper())
        meta.setdefault('horizontal_registration', {})[row] = lattice
        offsets = []
        matched = []
        for p in row_holes:
            index = int(np.argmin(np.abs(baseline_x - p['x'])))
            offset = float(p['x'] - baseline_x[index])
            if abs(offset) <= 0.35 * pitch:
                offsets.append(offset)
                matched.append(index)
        offset = float(np.median(offsets)) if len(set(matched)) >= 2 and lattice['status'] != 'SUPPORTED_PROPOSAL' else 0.0
        if offsets and lattice['status'] != 'SUPPORTED_PROPOSAL' and (float(np.median(np.abs(np.array(offsets) - offset))) > 0.12 * pitch):
            offset = 0.0
            meta['qc_flags'].append('HORIZONTAL_PHASE_INCONSISTENT_' + row.upper())
        if lattice['status'] == 'SUPPORTED_PROPOSAL':
            baseline_x = adjusted_x
            pitch = float(np.median(np.diff(baseline_x)))
            if lattice.get('unmatched_hole_x'):
                meta['qc_flags'].append('UNMATCHED_HOLE_X_' + row.upper())
            if camera == 'cam0' and extra_physical_column_range is not None:
                final_existing = max((int(str(r['physical_slot'])[1:]) for r in rr))
                for column in range(final_existing + 1, min(12, extra_physical_column_range) + 1):
                    xx = float(_interpolate_knots(lattice['physical_columns'], lattice['observed_x'], [column])[0])
                    if not 0 <= xx < w:
                        continue
                    proposal = deepcopy(rr[-1])
                    proposal['physical_slot'] = row + str(column)
                    for key in ('slot', 'slot_name', 'slot_id'):
                        if key in proposal:
                            proposal[key] = row + str(column)
                    if 'col' in proposal:
                        proposal['col'] = column
                    if 'column' in proposal:
                        proposal['column'] = column
                    proposal['record_id'] = str(proposal.get('record_id', '')) + '::ver2_extra_' + row + str(column)
                    proposal['geometry_appended_slot'] = True
                    proposal['slot_origin'] = 'NEW_CAM01_EXTRA_COLUMN_GEOMETRY'
                    proposal['geometry_status'] = 'HOLD'
                    proposal['slot_geometry_usable'] = False
                    proposal['geometry_proposal_status'] = 'HOLD_EXTRA_COLUMN_NUMBERING_REVIEW'
                    proposal['geometry_handoff_status'] = 'GEOMETRY_HOLD'
                    proposal['geometry_numbering_verified'] = False
                    proposal['geometry_reason'] = 'EXTRA_VISIBLE_COLUMN_CANDIDATE_PRESERVED_WITHOUT_COMPRESSING_EXISTING_COLUMNS'
                    proposal['geometry_position_evidence'] = 'OBSERVED_HOLE_KNOT' if column in lattice['physical_columns'] else 'CONSTRAINED_ROW_MODEL'
                    proposal['geometry_proposal_version'] = VERSION
                    proposal['x_prior'] = proposal['anchor_x'] = proposal['slot_center_x'] = xx
                    yy = top_fn(xx) if row == 't' else bottom_fn(xx)
                    proposal['y_prior'] = proposal['anchor_y'] = proposal['slot_center_y'] = yy
                    for key in ('seg_id', 'bbox', 'bbox_source', 'seg_bbox', 'bbox_seg', 'bbox_pixel', 'bbox_xyxy', 'segment_id', 'mask', 'crop_path', 'plant_crop_path'):
                        if key in proposal:
                            proposal[key] = None
                    for key in ('target_available', 'seg_assigned', 'bbox_available', 'cell_geometry_ready'):
                        proposal[key] = False
                    for key in list(proposal):
                        if key.startswith('upstream_'):
                            proposal.pop(key)
                    local_pitch = float(_interpolate_knots(lattice['physical_columns'], lattice['observed_x'], [column])[0] - _interpolate_knots(lattice['physical_columns'], lattice['observed_x'], [column - 1])[0])
                    left = xx - 0.5 * local_pitch
                    right = xx + 0.5 * local_pitch
                    fn = top_fn if row == 't' else bottom_fn
                    proposal['cell_polygon'] = [[left, fn(left) - 0.5 * gap], [right, fn(right) - 0.5 * gap], [right, fn(right) + 0.5 * gap], [left, fn(left) + 0.5 * gap]]
                    proposal['cell_bbox'] = [max(0, int(math.floor(left))), max(0, int(math.floor(min((v[1] for v in proposal['cell_polygon']))))), min(w, int(math.ceil(right))), min(h, int(math.ceil(max((v[1] for v in proposal['cell_polygon'])))))]
                    proposal['direct_hole_support'] = column in lattice['physical_columns']
                    proposal['assignment_status'] = 'UNASSIGNED_EXTRA_COLUMN_REVIEW'
                    proposal['geometry_row_hole_support'] = len(row_holes)
                    proposal['geometry_row_gap'] = gap
                    slots.append(proposal)
                    meta.setdefault('extra_column_proposals', []).append(dict(physical_slot=proposal['physical_slot'], x=xx, y=yy, evidence=proposal['geometry_position_evidence'], geometry_status='HOLD'))
        horizontal_error = [float(np.min(np.abs(baseline_x + offset - p['x']))) for p in row_holes]
        meta.setdefault('hole_x_grid_error', {})[row] = horizontal_error
        if sum((v > 0.2 * pitch for v in horizontal_error)) > len(horizontal_error) * 0.25:
            meta['qc_flags'].append('HOLE_X_GRID_MISMATCH_' + row.upper())
        for (index, r) in enumerate(rr):
            for key in ('x_prior', 'y_prior', 'anchor_x', 'anchor_y', 'slot_center_x', 'slot_center_y', 'cell_bbox', 'cell_polygon', 'bbox', 'cell_geometry_ready'):
                if key in r and 'upstream_' + key not in r:
                    r['upstream_' + key] = deepcopy(r[key])
            xx = float(baseline_x[index] + offset)
            yy = top_fn(xx) if row == 't' else bottom_fn(xx)
            r['x_prior'] = r['anchor_x'] = r['slot_center_x'] = xx
            r['y_prior'] = r['anchor_y'] = r['slot_center_y'] = yy
            r.update(geometry_proposal_status=status, geometry_proposal_version=VERSION, geometry_handoff_status='GEOMETRY_HOLD' if status == 'HOLD' else 'SLOT_GEOMETRY_REVIEW', geometry_row_hole_support=len(row_holes), geometry_row_gap=gap)
            r['geometry_numbering_verified'] = False
            r['geometry_row_usable'] = len(row_holes) >= 3 and (row == 'b' or not partial_top)
            if partial_top and row == 't':
                r['geometry_handoff_status'] = 'GEOMETRY_HOLD'
                r['geometry_proposal_status'] = 'HOLD_TOP_ROW_SINGLE_HOLE'
                r['geometry_status'] = 'HOLD'
                r['slot_geometry_usable'] = False
                r['geometry_reason'] = 'TOP_ROW_PHASE_AND_GAP_NOT_INDEPENDENTLY_VERIFIED'
            left = (baseline_x[index - 1] + baseline_x[index]) / 2 + offset if index else xx - 0.5 * pitch
            right = (baseline_x[index + 1] + baseline_x[index]) / 2 + offset if index + 1 < len(rr) else xx + 0.5 * pitch
            fn = top_fn if row == 't' else bottom_fn
            half_gap = 0.5 * gap
            poly = [[left, fn(left) - half_gap], [right, fn(right) - half_gap], [right, fn(right) + half_gap], [left, fn(left) + half_gap]]
            r['cell_polygon'] = poly
            r['cell_bbox'] = [max(0, int(math.floor(left))), max(0, int(math.floor(min((v[1] for v in poly))))), min(w, int(math.ceil(right))), min(h, int(math.ceil(max((v[1] for v in poly)))))]
            r['cell_geometry_ready'] = status != 'HOLD' and (not (partial_top and row == 't')) and (r['cell_bbox'][2] > r['cell_bbox'][0]) and (r['cell_bbox'][3] > r['cell_bbox'][1])
            if not (0 <= xx < w and 0 <= yy < h):
                r['geometry_handoff_status'] = 'GEOMETRY_HOLD'
                r['cell_geometry_ready'] = False
                r['geometry_proposal_status'] = 'HOLD_OUTSIDE_ROI'
                r['slot_geometry_usable'] = False
                r['geometry_status'] = 'HOLD'
                r['geometry_reason'] = 'FINAL_ANCHOR_OUTSIDE_NATIVE_ROI'
                meta['qc_flags'].append('PARTIAL_SLOT_OUTSIDE_NATIVE_ROI')
    if meta['qc_flags'] and status != 'HOLD':
        status = 'PROPOSED_REVIEW'
    if crossing_span > 0.2 or max(residuals) > 0.024 * w:
        status = 'HOLD'
    meta['status'] = status
    meta['geometry_status'] = 'HOLD' if status == 'HOLD' else 'ACCEPTED' if status == 'GEOMETRY_SUPPORTED' else 'REVIEW'
    meta['slot_geometry_usable'] = meta['geometry_status'] == 'ACCEPTED'
    meta['row_geometry_status'] = {row: 'HOLD' if row == 't' and partial_top else 'REVIEW' if meta['qc_flags'] else 'ACCEPTED' for row in ('t', 'b')}
    for r in slots:
        if str(r.get('geometry_proposal_status', '')).startswith('HOLD'):
            continue
        r['geometry_proposal_status'] = status
        r['geometry_status'] = meta['geometry_status']
        r['slot_geometry_usable'] = meta['slot_geometry_usable'] and r['cell_geometry_ready']
        r['geometry_handoff_status'] = 'GEOMETRY_HOLD' if status == 'HOLD' else 'SLOT_GEOMETRY_REVIEW'
        if status == 'HOLD':
            r['cell_geometry_ready'] = False
        nearest = min((math.hypot(p['x'] - r['x_prior'], p['y'] - r['y_prior']) for p in (top if r['row'] == 't' else bottom)['holes']), default=float('inf'))
        r['geometry_position_evidence'] = 'OBSERVED_HOLE_NEARBY' if nearest < 0.025 * w else 'CONSTRAINED_ROW_MODEL'
        r['geometry_reason'] = '|'.join(sorted(set(meta['qc_flags']))) or 'TWO_ROW_PHYSICAL_HOLE_SUPPORT'
    meta['qc_flags'] = sorted(set(meta['qc_flags']))
    return (slots, meta)
