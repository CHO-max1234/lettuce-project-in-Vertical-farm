"""Front geometry base calculations used by the final cycle pipeline."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
import copy
import itertools
import json
import math
import hashlib
import cv2
import numpy as np
REFERENCE_WIDTH = 2439.0
VERSION = 'S4_CAM2_STRUCTURAL_CURVED_BOARD_VER2'
MIN_ROW_HOLES = 3

def _value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value

def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def detect_native_holes(image):
    (h, w) = image.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = []
    thresholds = (35, 55, 75, 95, 115)
    local = cv2.GaussianBlur(gray, (0, 0), max(5.0, w * 0.007))
    masks = [np.uint8(gray < t) * 255 for t in thresholds]
    masks.append(np.uint8((local.astype(float) - gray.astype(float) > 24) & (gray < 125)) * 255)
    for (level, mask) in enumerate(masks):
        (contours, _) = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 12:
                continue
            area = cv2.contourArea(contour)
            if not w * w * 1.8e-05 < area < w * w * 0.0007:
                continue
            try:
                ((x, y), (a, b), angle) = cv2.fitEllipse(contour)
            except cv2.error:
                continue
            (major, minor) = (max(a, b), min(a, b))
            if not (0.007 * w < major < 0.043 * w and 0.0015 * w < minor < 0.026 * w and (0.17 < minor / major < 0.92)):
                continue
            ellipse_area = math.pi * a * b / 4
            fill = area / max(ellipse_area, 1.0)
            if not 0.64 < fill < 1.23:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            box = cv2.boundingRect(contour)
            if box[2] > major * 1.4 or box[3] > major * 1.4:
                continue
            rad = round(major * 0.82 + 3)
            (x0, x1) = (max(0, round(x) - rad), min(w, round(x) + rad + 1))
            (y0, y1) = (max(0, round(y) - rad), min(h, round(y) + rad + 1))
            inner = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.ellipse(inner, ((x - x0, y - y0), (a * 0.74, b * 0.74), angle), 1, -1)
            ring = np.zeros_like(inner)
            cv2.ellipse(ring, ((x - x0, y - y0), (a * 1.52, b * 1.52), angle), 1, -1)
            inner_full = np.zeros_like(inner)
            cv2.ellipse(inner_full, ((x - x0, y - y0), (a * 1.12, b * 1.12), angle), 1, -1)
            ring[inner_full > 0] = 0
            if inner.sum() < 8 or ring.sum() < 8:
                continue
            ip = gray[y0:y1, x0:x1][inner > 0].astype(float)
            rp = gray[y0:y1, x0:x1][ring > 0].astype(float)
            contrast = float(np.median(rp) - np.median(ip))
            rg = hsv[y0:y1, x0:x1][ring > 0]
            green = (rg[:, 0] > 26) & (rg[:, 0] < 96) & (rg[:, 1] > 65) & (rg[:, 2] > 45)
            neutral = float(1 - np.mean(green))
            if contrast < 22 or neutral < 0.6:
                continue
            perimeter = max(cv2.arcLength(contour, True), 1.0)
            c = dict(x=float(x), y=float(y), major=float(major), minor=float(minor), angle=float(angle), contrast=contrast, neutral_ring=neutral, fill_ratio=float(fill), threshold=level, score=contrast * neutral * min(fill, 1.0))
            near = next((j for (j, p) in enumerate(candidates) if abs(p['x'] - x) < 0.22 * max(p['major'], major) and abs(p['y'] - y) < 0.28 * max(p['minor'], minor)), None)
            if near is None:
                candidates.append(c)
            elif c['score'] > candidates[near]['score']:
                candidates[near] = c
    return sorted(candidates, key=lambda c: (c['x'], c['y']))

def _robust_row_models(holes, w, h, expected_y):
    pool = [p for p in holes if 0.08 * w < p['x'] < 0.955 * w and 0.01 * w < p['major'] < 0.029 * w and (p['minor'] / p['major'] > 0.28) and (p['score'] > 42) and (abs(p['y'] - expected_y) < 0.055 * w)]
    if len(pool) < MIN_ROW_HOLES:
        return []
    xy = np.array([[p['x'] / w, p['y']] for p in pool])
    u = xy[:, 0] - 0.5
    design = np.column_stack([u * u, u, np.ones(len(u))])
    options = []
    choices = list(itertools.combinations(range(len(pool)), 3))
    if len(choices) > 4200:
        rng = np.random.default_rng(20261009)
        choices = [choices[i] for i in sorted(rng.choice(len(choices), 4200, replace=False))]
    for indices in choices:
        sample_xy = xy[list(indices)]
        if np.ptp(sample_xy[:, 0]) < 0.4:
            continue
        if min((abs(a - b) for (a, b) in itertools.combinations(sample_xy[:, 0], 2))) < 0.08:
            continue
        try:
            coef = np.linalg.solve(design[list(indices)], sample_xy[:, 1])
        except np.linalg.LinAlgError:
            continue
        if not -0.2 * w < coef[0] < 0.018 * w or abs(coef[1]) > 0.05 * w:
            continue
        mid = coef[2]
        if not 0.025 * h < mid < 0.94 * h or abs(mid - expected_y) > 0.045 * w:
            continue
        residual = np.abs(design @ coef - xy[:, 1])
        kept = residual < max(8.0, 0.004 * w)
        if kept.sum() < MIN_ROW_HOLES or np.ptp(xy[kept, 0]) < 0.45:
            continue
        for _ in range(2):
            coef = np.linalg.lstsq(design[kept], xy[kept, 1], rcond=None)[0]
            residual = np.abs(design @ coef - xy[:, 1])
            kept = residual < max(8.0, 0.004 * w)
        if kept.sum() < MIN_ROW_HOLES:
            continue
        active = []
        for i in sorted(np.flatnonzero(kept), key=lambda i: residual[i]):
            if all((abs(xy[i, 0] - xy[j, 0]) > 0.018 for j in active)):
                active.append(int(i))
        if len(active) < MIN_ROW_HOLES:
            continue
        coef = np.linalg.lstsq(design[active], xy[active, 1], rcond=None)[0]
        residual = np.abs(design @ coef - xy[:, 1])
        score = len(active) + 2.0 * np.ptp(xy[active, 0]) - float(np.mean(residual[active])) / 18 - 0.25 * abs(coef[2] - expected_y) / max(w * 0.03, 1)
        at = coef[0] * np.array([-0.35, 0, 0.35]) ** 2 + coef[1] * np.array([-0.35, 0, 0.35]) + coef[2]
        if any((np.linalg.norm(at - p['shape']) < 12 for p in options)):
            continue
        options.append(dict(coef=coef, shape=at, score=float(score), holes=[pool[i] for i in active], rms=float(np.sqrt(np.mean(residual[active] ** 2)))))
    return sorted(options, key=lambda p: p['score'], reverse=True)[:18]

def _row_y(coef, x, w):
    u = np.asarray(x, float) / w - 0.5
    return coef[0] * u * u + coef[1] * u + coef[2]

def _initial_x(w, row):
    (left, right, k) = (0.101, 0.92, 0.062) if row == 't' else (0.073, 0.946, 0.073)
    z = (np.arange(1, 13) - 6.5) / 5.5
    return w * ((left + right) / 2 + (right - left) / 2 * z + k * z * (1 - z * z))

def _plate_edges(image, models, side='left'):
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edge = cv2.Canny(gray, 25, 75)
    if side == 'left':
        edge[:, round(0.205 * w):] = 0
    else:
        edge[:, :round(0.8 * w)] = 0
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 32, minLineLength=max(50, round(0.11 * h)), maxLineGap=15)
    if lines is None:
        return []
    (t, b) = (models['t'], models['b'])
    top = float(_row_y(t['coef'], 0.12 * w, w))
    bottom = float(_row_y(b['coef'], 0.08 * w, w))
    gap = bottom - top
    answer = []
    for (x1, y1, x2, y2) in lines.reshape(-1, 4):
        if abs(y2 - y1) < max(40, 0.55 * gap):
            continue
        a = (x2 - x1) / (y2 - y1)
        bb = x1 - a * y1
        if side == 'left' and (not -1.45 < a < -0.3):
            continue
        if side == 'right' and (not 0.25 < a < 1.45):
            continue
        (lo, hi) = sorted([y1, y2])
        if min(hi, bottom) - max(lo, top) < 0.5 * gap:
            continue
        yy = np.linspace(max(lo, top), min(hi, bottom), 40)
        xx = a * yy + bb
        if side == 'left' and (not 0.018 * w < np.median(xx) < 0.16 * w):
            continue
        if side == 'right' and (not 0.83 * w < np.median(xx) < 0.983 * w):
            continue
        iy = np.clip(np.rint(yy).astype(int), 0, h - 1)
        l = np.clip(np.rint(xx - 10).astype(int), 0, w - 1)
        r = np.clip(np.rint(xx + 10).astype(int), 0, w - 1)
        face = hsv[iy, r if side == 'left' else l]
        neutral = ~((face[:, 0] > 26) & (face[:, 0] < 96) & (face[:, 1] > 65))
        brightness = float(np.median(gray[iy, r].astype(float) - gray[iy, l].astype(float)))
        if np.mean(neutral) < 0.82 or (brightness < 9 if side == 'left' else abs(brightness) < 9):
            continue
        answer.append(dict(a=float(a), b=float(bb), top_y=top, bottom_y=bottom, top_x=float(a * top + bb), bottom_x=float(a * bottom + bb), contrast=brightness, neutral=float(np.mean(neutral)), score=float((1 if side == 'left' else -1) * np.median(xx) + 0.7 * abs(brightness))))
    return sorted(answer, key=lambda e: e['score'], reverse=True)

def _register_x(model, row, w, left_edge=None, right_edge=None):
    init = _initial_x(w, row)
    phase_from_boundary = False
    left_boundary_used = False
    right_boundary_used = False
    (first, last) = (float(init[0]), float(init[-1]))
    if left_edge is not None:
        y = float(_row_y(model['coef'], init[0], w))
        edge_x = left_edge['a'] * y + left_edge['b']
        proposed_first = edge_x + 0.5 * (init[1] - init[0])
        shift = proposed_first - init[0]
        if abs(shift) < 0.075 * w:
            first = proposed_first
            phase_from_boundary = True
            left_boundary_used = True
    if right_edge is not None:
        y = float(_row_y(model['coef'], init[-1], w))
        edge_x = right_edge['a'] * y + right_edge['b']
        proposed_last = edge_x - 0.5 * (init[-1] - init[-2])
        if abs(proposed_last - init[-1]) < 0.075 * w:
            last = proposed_last
            phase_from_boundary = True
            right_boundary_used = True
    factor = (last - first) / (init[-1] - init[0])
    init = first + (init - init[0]) * factor
    pitch = np.median(np.diff(init))
    associations = []
    for p in model['holes']:
        col = int(np.argmin(abs(init - p['x']))) + 1
        if abs(init[col - 1] - p['x']) < 0.35 * pitch:
            associations.append(dict(p, col=col, initial_dx=float(p['x'] - init[col - 1])))
    associations = [p for p in associations if p['x'] < (0.945 if row == 't' else 0.971) * w]
    by = {}
    for p in associations:
        if p['col'] not in by or p['score'] > by[p['col']]['score']:
            by[p['col']] = p
    points = sorted(by.values(), key=lambda p: p['col'])
    if len(points) < 3:
        return (init, points, dict(status='INSUFFICIENT_PHASE_HOLES', phase_confirmed=False))
    cols = np.array([p['col'] for p in points], float)
    xx = np.array([p['x'] for p in points], float)
    z = (cols - 6.5) / 5.5
    degree = min(3, len(points) - 1)
    mat = np.column_stack([z ** i for i in range(degree, -1, -1)])
    gridz = (np.arange(1, 13) - 6.5) / 5.5
    gmat = np.column_stack([gridz ** i for i in range(degree, -1, -1)])
    coef = np.linalg.lstsq(np.vstack([mat, 0.13 * gmat]), np.r_[xx, 0.13 * init], rcond=None)[0]
    pred = gmat @ coef
    for _ in range(2):
        residual = abs(mat @ coef - xx)
        keep = residual < 0.11 * pitch
        if keep.sum() < 3:
            break
        coef = np.linalg.lstsq(np.vstack([mat[keep], 0.13 * gmat]), np.r_[xx[keep], 0.13 * init], rcond=None)[0]
        pred = gmat @ coef
    residual = abs(mat @ coef - xx)
    points = [p for (p, e) in zip(points, residual) if e < 0.11 * pitch]
    strong_span = left_boundary_used and left_edge['contrast'] >= 20 and (left_edge['neutral'] >= 0.9) and (len(points) >= 6) and (min((p['col'] for p in points)) <= 2) and (max((p['col'] for p in points)) >= 10)
    shift_limit = 0.5 if strong_span else 0.3
    max_shift_pitch = float(max(abs(pred - init)) / pitch)
    phase = len(points) >= 3 and any((p['col'] <= 2 for p in points)) and (max((p['col'] for p in points)) - min((p['col'] for p in points)) >= 5) and np.all(np.diff(pred) > 0.1 * w / 12) and (max_shift_pitch < shift_limit)
    return (pred, points, dict(status='RAW_HOLE_LATTICE_REGISTERED' if phase else 'PHASE_OR_SPAN_REVIEW', phase_confirmed=bool(phase), rms_px=float(np.sqrt(np.mean(residual ** 2))), left_plate_boundary_used=left_boundary_used, right_plate_boundary_used=right_boundary_used, max_seed_shift_pitch=max_shift_pitch, seed_shift_limit_pitch=shift_limit, strong_raw_hole_span=bool(strong_span), first_supported_col=min((p['col'] for p in points)) if points else None, last_supported_col=max((p['col'] for p in points)) if points else None, registered_columns=[p['col'] for p in points], x_curve_coefficients=coef.tolist()))

def _cell(top, bottom, index, row, w, h):

    def boundaries(points):
        return np.vstack([points[0] - (points[1] - points[0]) * 0.5, (points[:-1] + points[1:]) * 0.5, points[-1] + (points[-1] - points[-2]) * 0.5])
    (a, b) = (boundaries(top), boundaries(bottom))
    (v0, v1) = (-0.5, 0.5) if row == 't' else (0.5, 1.5)
    poly = np.array([a[index] + v0 * (b[index] - a[index]), a[index + 1] + v0 * (b[index + 1] - a[index + 1]), a[index + 1] + v1 * (b[index + 1] - a[index + 1]), a[index] + v1 * (b[index] - a[index])])
    box = [max(0, math.floor(poly[:, 0].min())), max(0, math.floor(poly[:, 1].min())), min(w, math.ceil(poly[:, 0].max())), min(h, math.ceil(poly[:, 1].max()))]
    return (poly.tolist(), box)

def propose_geometry(image, camera, sample, baseline_slots, raw_segments):
    slots = copy.deepcopy(baseline_slots)
    meta = dict(version=VERSION, camera=camera, frame_status='HOLD', failure_reasons=[], evidence={}, row_support={}, geometry_only=True, bbox_target_selection_executed=False, full_batch_validated=False)
    if camera != 'cam2':
        meta['failure_reasons'] = ['CAMERA_NOT_SUPPORTED']
        return (slots, meta)
    (h, w) = image.shape[:2]
    holes = detect_native_holes(image)
    meta['evidence']['raw_holes'] = holes
    for r in slots:
        r['slot_geometry_usable'] = False
        r['geometry_review_reasons'] = ['CAM2_GEOMETRY_NOT_REGISTERED']
        r.update(row_geometry_usable=False, cell_crop_usable=False, cell_geometry_ready=False, row_geometry_review_reasons=['CAM2_GEOMETRY_NOT_REGISTERED'], geometry_version=VERSION)
    geometry = _value(sample.get('geometry_details', {}))
    front = _float(geometry.get('front_rail_y'), 0.86 * h)
    scale = w / REFERENCE_WIDTH
    current_prior = {row: np.array([[_float(r['x_prior']), _float(r['y_prior'])] for r in slots if r['row'] == row]) for row in ('t', 'b')}
    if any((len(current_prior[row]) < 2 for row in ('t', 'b'))):
        meta['failure_reasons'] = ['BASELINE_PHYSICAL_NAMESPACE_UNAVAILABLE']
        return (slots, meta)
    expected = {row: front + scale * (np.median(current_prior[row][:, 1]) - front) for row in ('t', 'b')}
    models = {row: _robust_row_models(holes, w, h, expected[row]) for row in ('t', 'b')}
    pairs = []
    for t in models['t']:
        for b in models['b']:
            xs = np.linspace(0.1 * w, 0.94 * w, 20)
            gap = _row_y(b['coef'], xs, w) - _row_y(t['coef'], xs, w)
            if min(gap) < 0.028 * w or max(gap) > 0.074 * w or max(gap) / min(gap) > 1.65:
                continue
            slopes = (2 * (b['coef'][0] - t['coef'][0]) * (xs / w - 0.5) + b['coef'][1] - t['coef'][1]) / w
            if max(abs(slopes)) > 0.09:
                continue
            score = t['score'] + b['score'] - 0.2 * abs(np.median(gap) - 0.05 * w) / (w * 0.02)
            pairs.append((score, t, b, gap))
    meta['evidence']['model_counts'] = {r: len(m) for (r, m) in models.items()}
    meta['evidence']['isotropic_seed_row_y'] = expected
    if not pairs:
        meta['failure_reasons'] = ['NO_TWO_ROW_PLATE_CONSENSUS']
        return (slots, meta)
    proposals = []
    for (pair_score, t, b, gap) in sorted(pairs, key=lambda p: p[0], reverse=True)[:22]:
        fitted = {'t': t, 'b': b}
        edges = _plate_edges(image, fitted, 'left')
        right_edges = _plate_edges(image, fitted, 'right')
        variants = [('NORMALIZED_GROUP_SEED', None, None)]
        if edges:
            variants.append(('LEFT_FACE_EDGE', edges[0], None))
        if right_edges:
            variants.append(('RIGHT_FACE_EDGE', None, right_edges[0]))
        if edges and right_edges:
            variants.append(('BOTH_FACE_EDGES', edges[0], right_edges[0]))
        registered = {}
        for (row, model) in fitted.items():
            choices = []
            for (name, l, r) in variants:
                (x, points, info) = _register_x(model, row, w, l, r)
                support_score = len(points) * 2 + (5 if info['phase_confirmed'] else 0) - info.get('rms_px', 100) / 30
                choices.append((support_score, x, points, dict(info, initialization_variant=name)))
            registered[row] = max(choices, key=lambda p: p[0])
        (xt, xb) = (registered['t'][1], registered['b'][1])
        if abs(xt[0] - xb[0]) > 0.075 * w or max(abs(xt - xb)) > 0.1 * w:
            continue
        score = registered['t'][0] + registered['b'][0] + pair_score * 0.16
        proposals.append((score, t, b, gap, registered, edges, right_edges))
    if not proposals:
        meta['failure_reasons'] = ['NO_TWO_ROW_LATTICE_PHASE_CONSENSUS']
        return (slots, meta)
    (_, t, b, gap, registered, edges, right_edges) = max(proposals, key=lambda p: p[0])
    fitted = {'t': t, 'b': b}
    meta['evidence']['left_plate_border_candidates'] = edges
    meta['evidence']['right_plate_border_candidates'] = right_edges
    curves = {}
    supports = {}
    phase = {}
    for (row, model) in fitted.items():
        (_, x, points, info) = registered[row]
        curves[row] = np.column_stack([x, _row_y(model['coef'], x, w)])
        supports[row] = points
        phase[row] = info
        meta['row_support'][row] = dict(info, raw_holes=len(points), y_rms_px=model['rms'], y_curve_coefficients=model['coef'].tolist(), holes=points)
    meta['evidence'].update(board_top=curves['t'].tolist(), board_bottom=curves['b'].tolist(), row_gap_range_px=[float(min(gap)), float(max(gap))], uses_seg_centroids=False, physical_columns=12, phase_policy='LEFT_PLATE_LATTICE_PLUS_RAW_HOLE_NEAREST_COL_CONSENSUS')
    present = {(str(r['row']), int(r['col'])) for r in slots}
    for row in ('t', 'b'):
        group = [r for r in slots if r['row'] == row]
        if not group:
            continue
        missing = [col for col in range(1, 13) if (row, col) not in present]
        for col in missing if phase[row]['phase_confirmed'] else []:
            (first, last) = (phase[row].get('first_supported_col'), phase[row].get('last_supported_col'))
            if first is None or col < first - 1 or col > last + 1:
                continue
            r = copy.deepcopy(max(group, key=lambda r: int(r['col'])))
            name = row + str(col)
            namespace = r.get('slot_namespace_version', 'image_left_to_right_v1')
            path = str(sample.get('input_path') or r.get('source_path') or '')
            r.update(col=col, local_col=col, reference_local_col=col, slot_name=name, canonical_id=name, physical_slot=name, reference_slot_name=name, slot_origin='NEW_PLATE_LATTICE_GEOMETRY', record_id=hashlib.sha256((path + '|' + namespace + '|' + name).encode()).hexdigest()[:24], bbox=None, seg_id=None, seg_conf=0.0, seg_center_x=None, seg_center_y=None, seg_bbox=None, seg_area=0.0, seg_status='NO_BODY_SELECTION_EXECUTED', target_available=False)
            slots.append(r)
    for r in slots:
        row = str(r['row'])
        col = int(r['col'])
        i = col - 1
        if row not in curves or not 0 <= i < 12:
            continue
        (x, y) = map(float, curves[row][i])
        (poly, box) = _cell(curves['t'], curves['b'], i, row, w, h)
        info = phase[row]
        reasons = []
        if not info['phase_confirmed']:
            reasons.append('LEFT_PHASE_OR_ROW_SPAN_UNCONFIRMED')
        (first, last) = (info.get('first_supported_col'), info.get('last_supported_col'))
        if first is None or col < first - 1 or col > last + 1:
            reasons.append('EXTRAPOLATION_OVER_ONE_COLUMN')
        if not (0 <= x < w and 0 <= y < h):
            reasons.append('ANCHOR_OUTSIDE_NATIVE_ROI')
        if box[2] <= box[0] or box[3] <= box[1]:
            reasons.append('CELL_OUTSIDE_NATIVE_ROI')
        row_reasons = list(reasons)
        if not all((phase[k]['phase_confirmed'] for k in ('t', 'b'))):
            reasons.append('SHARED_CELL_OTHER_ROW_PHASE_UNCONFIRMED')
        near = next((p for p in supports[row] if p['col'] == col), None)
        r.update(x_prior=x, y_prior=y, anchor_x=x, anchor_y=y, eval_anchor_x=x, eval_anchor_y=y, slot_center_x=x, slot_center_y=y, cell_polygon=poly, cell_bbox=box, cell_geometry_ready=not reasons, slot_geometry_usable=not reasons, row_geometry_usable=not row_reasons, cell_crop_usable=not reasons, row_geometry_review_reasons=row_reasons, cell_width=box[2] - box[0], cell_height=box[3] - box[1], geometry_review_reasons=reasons, geometry_version=VERSION, anchor_registration_status='REGISTERED_RAW_HOLE' if near else 'REGISTERED_INTERPOLATED' if not reasons else 'UNKNOWN_REGISTRATION', anchor_registration_source='S4_NATIVE_CURVED_BOARD_RAW_HOLES', geometry_hole_x=near['x'] if near else None, geometry_hole_y=near['y'] if near else None, geometry_hole_visible=near is not None, geometry_hole_residual_px=float(math.hypot(near['x'] - x, near['y'] - y)) if near else None, geometry_evidence='RAW_HOLE_LATTICE' if near else 'REGISTERED_TWO_ROW_STRUCTURAL_LATTICE', slot_origin=r.get('slot_origin', 'BASELINE_ID_PRESERVED_GEOMETRY_REGISTERED'))
    usable = sum((bool(r.get('slot_geometry_usable')) for r in slots))
    meta['frame_status'] = 'ACCEPTED' if usable == len(slots) else 'REVIEW' if usable else 'HOLD'
    meta['failure_reasons'] = sorted({v for r in slots for v in r.get('geometry_review_reasons', [])})
    meta['slot_geometry_usable_count'] = usable
    meta['slot_count'] = len(slots)
    return (slots, meta)
