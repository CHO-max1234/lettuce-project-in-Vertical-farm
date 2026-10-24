"""Front geometry calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = CYCLE.PROJECT_ROOT / 'preprocessing/front_geometry_calibration.json'
VERSION = 'S4_CAM2_VER3_GROUP_LOWER_ROW_JOINT_TOP'
import copy, json, hashlib, itertools
import importlib.util
import numpy as np
spec = resolved_spec('s4_cam2_v3_v2', CODE_ROOT / 'front_geometry_base.py')
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
CAL = json.loads(CALIBRATION_PATH.read_text(encoding='utf-8'))['groups']

def group_key(sample):
    return f"{sample['source_camera']}_{sample['native_width']}x{sample['native_height']}"

def row_y(coef, x, w):
    return OLD._row_y(coef, x, w)

def unique_points(points):
    out = []
    for p in sorted(points, key=lambda p: p['score'], reverse=True):
        if all((abs(p['x'] - q['x']) > max(p['major'], q['major']) * 0.42 or abs(p['y'] - q['y']) > max(p['minor'], q['minor']) * 0.65 for q in out)):
            out.append(p)
    return out

def fit_x(points, seed, w):
    pitch = np.median(np.diff(seed))
    out = []
    for delta in (-0.3, -0.15, 0, 0.15, 0.3):
        grid = seed + delta * pitch
        for _ in range(3):
            by = {}
            for p in points:
                col = int(np.argmin(abs(grid - p['x'])))
                local = (grid[min(col + 1, 11)] - grid[max(col - 1, 0)]) / (1 if col in (0, 11) else 2)
                if abs(grid[col] - p['x']) > max(8, 0.3 * local):
                    continue
                if col not in by or p['score'] > by[col]['score']:
                    by[col] = dict(p, col=col + 1)
            pp = list(by.values())
            if len(pp) < 2:
                break
            indices = np.array([p['col'] - 1 for p in pp])
            xx = np.array([p['x'] for p in pp])
            mat = np.column_stack([seed[indices] / w, np.ones(len(pp))])
            reg = np.column_stack([seed / w, np.ones(12)])
            coef = np.linalg.lstsq(np.vstack([mat, 0.35 * reg]), np.r_[xx / w, 0.35 * seed / w], rcond=None)[0]
            if not 0.93 < coef[0] < 1.07:
                break
            grid = reg @ coef * w
        if len(pp) < 2 or np.max(abs(grid - seed)) > 0.47 * pitch:
            continue
        pp = [p for p in pp if abs(grid[p['col'] - 1] - p['x']) < max(8, 0.18 * pitch)]
        if len(pp) < 2:
            continue
        span = (max((p['col'] for p in pp)) - min((p['col'] for p in pp))) / 11
        err = np.mean([abs(grid[p['col'] - 1] - p['x']) for p in pp]) / pitch
        score = len(pp) + 2 * span - 3 * err - 0.3 * np.max(abs(grid - seed)) / pitch
        out.append((score, grid, pp))
    return max(out, key=lambda p: p[0]) if out else None

def bottom_models(holes, w, h, cal, expected):
    pool = unique_points([p for p in holes if 0.065 * w < p['x'] < 0.965 * w and 0.008 * w < p['major'] < 0.031 * w and (p['minor'] / p['major'] > 0.25) and (p['score'] > 35) and (p['neutral_ring'] > 0.7)])
    seed = np.array(cal['x']['b']) * w
    valid = []
    for p in pool:
        col = int(np.argmin(abs(seed - p['x'])))
        if abs(seed[col] - p['x']) < 0.4 * np.median(np.diff(seed)):
            valid.append(p)
    if len(valid) < 4:
        return []
    xy = np.array([[p['x'] / w - 0.5, p['y']] for p in valid])
    u = xy[:, 0]
    design = np.column_stack([u * u, u, np.ones(len(u))])
    out = []
    comb = list(itertools.combinations(range(len(valid)), 3))
    if len(comb) > 1500:
        rng = np.random.default_rng(20261010)
        comb = [comb[i] for i in sorted(rng.choice(len(comb), 1500, replace=False))]
    for inds in comb:
        if np.ptp(u[list(inds)]) < 0.38 or min((abs(a - b) for (a, b) in itertools.combinations(u[list(inds)], 2))) < 0.07:
            continue
        try:
            coef = np.linalg.solve(design[list(inds)], xy[list(inds), 1])
        except np.linalg.LinAlgError:
            continue
        if not -0.25 * w < coef[0] < 0.025 * w or abs(coef[1]) > 0.05 * w:
            continue
        if not 0.32 * h < coef[2] < 0.93 * h or abs(coef[2] - expected) > 0.065 * w:
            continue
        mask = abs(design @ coef - xy[:, 1]) < max(7, 0.0032 * w)
        if mask.sum() < 4 or np.ptp(u[mask]) < 0.46:
            continue
        coef = np.linalg.lstsq(design[mask], xy[mask, 1], rcond=None)[0]
        if not -0.25 * w < coef[0] < 0.025 * w or abs(coef[1]) > 0.05 * w:
            continue
        mask = abs(design @ coef - xy[:, 1]) < max(7, 0.0032 * w)
        pp = [valid[i] for i in np.flatnonzero(mask)]
        reg = fit_x(pp, seed, w)
        if reg is None:
            continue
        (score, grid, pp) = reg
        if len(pp) < 4 or max((p['col'] for p in pp)) - min((p['col'] for p in pp)) < 6:
            continue
        if min((p['col'] for p in pp)) > 3:
            continue
        shape = row_y(coef, np.array([0.15, 0.5, 0.85]) * w, w)
        if any((np.linalg.norm(shape - m['shape']) < 10 for m in out)):
            continue
        rms = float(np.sqrt(np.mean([(p['y'] - row_y(coef, p['x'], w)) ** 2 for p in pp])))
        out.append(dict(coef=coef, grid=grid, holes=pp, shape=shape, rms=rms, score=score - 0.7 * abs(coef[2] - expected) / (0.05 * w)))
    return sorted(out, key=lambda m: m['score'], reverse=True)[:10]

def top_joint(holes, bottom, cal, w, h):
    seed = np.array(cal['x']['t']) * w
    gap = np.array(cal['gap']) * w
    prior = bottom['coef'] - gap
    options = []
    for dy in (-0.014 * w, 0, 0.014 * w):
        coef = prior.copy()
        coef[2] += dy
        pp = unique_points([p for p in holes if 0.07 * w < p['x'] < 0.96 * w and 0.008 * w < p['major'] < 0.032 * w and (p['score'] > 35) and (p['neutral_ring'] > 0.7) and (abs(p['y'] - row_y(coef, p['x'], w)) < max(10, 0.01 * w))])
        reg = fit_x(pp, seed, w)
        if reg is None:
            continue
        (score, grid, pp) = reg
        if len(pp) < 2:
            continue
        u = np.array([p['x'] / w - 0.5 for p in pp])
        yy = np.array([p['y'] for p in pp])
        design = np.column_stack([u * u, u, np.ones(len(u))])
        prior_mat = np.diag([0.27, 0.34, 0.025])
        coef = np.linalg.lstsq(np.vstack([design, prior_mat]), np.r_[yy, prior_mat @ prior], rcond=None)[0]
        pp = [p for p in pp if abs(p['y'] - row_y(coef, p['x'], w)) < max(9, 0.0045 * w)]
        if len(pp) < 2:
            continue
        if max((p['col'] for p in pp)) - min((p['col'] for p in pp)) < 2:
            continue
        x = np.linspace(0.1 * w, 0.94 * w, 24)
        diff = row_y(bottom['coef'], x, w) - row_y(coef, x, w)
        if min(diff) < 0.03 * w or max(diff) > 0.085 * w or max(diff) / min(diff) > 1.85:
            continue
        if max(abs((2 * (bottom['coef'][0] - coef[0]) * (x / w - 0.5) + bottom['coef'][1] - coef[1]) / w)) > 0.12:
            continue
        if abs(coef[0] - prior[0]) > 0.045 * w or abs(coef[1] - prior[1]) > 0.026 * w:
            continue
        if not -0.23 * w < coef[0] < 0.025 * w or abs(coef[1]) > 0.055 * w:
            continue
        rms = float(np.sqrt(np.mean([(p['y'] - row_y(coef, p['x'], w)) ** 2 for p in pp])))
        score -= rms / 10 + abs(coef[2] - prior[2]) / (0.02 * w)
        options.append(dict(coef=coef, grid=grid, holes=pp, rms=rms, score=score, prior=prior))
    return max(options, key=lambda m: m['score']) if options else None

def propose_geometry(image, camera, sample, baseline_slots, raw_segments):
    assert camera == 'cam2'
    slots = copy.deepcopy(baseline_slots)
    old = sample.get('geometry_metadata', {})
    if sample.get('geometry_status') in ('PASS', 'REVIEW'):
        meta = copy.deepcopy(old)
        meta.update(version=VERSION, ver3_action='PRESERVED_VER2_' + sample['geometry_status'], preserved_ver2_geometry=True)
        return (slots, meta)
    (w, h) = (sample['source_width'], sample['source_height'])
    key = group_key(sample)
    meta = dict(version=VERSION, camera='cam2', frame_status='HOLD', failure_reasons=[], row_support={}, evidence={}, ver3_action='JOINT_REPAIR', geometry_only=True, bbox_target_selection_executed=False)
    if key not in CAL:
        meta['failure_reasons'] = ['UNKNOWN_SOURCE_GEOMETRY_GROUP']
        return (slots, meta)
    cal = CAL[key]
    holes = old.get('evidence', {}).get('raw_holes')
    if holes is None:
        holes = OLD.detect_native_holes(image)
    meta['evidence'] = dict(raw_holes=holes, group_key=key, prior_scope='STRUCTURAL_INITIALIZATION_ONLY', uses_seg_centroids=False)
    expected = old.get('evidence', {}).get('isotropic_seed_row_y', {}).get('b', 0.7 * h)
    models = bottom_models(holes, w, h, cal, expected)
    proposals = []
    for b in models:
        t = top_joint(holes, b, cal, w, h)
        if t is None:
            continue
        fitted = {'t': t, 'b': b}
        left = OLD._plate_edges(image, fitted, 'left')
        right = OLD._plate_edges(image, fitted, 'right')

        def phase(m):
            return min((p['col'] for p in m['holes'])) <= 2 and max((p['col'] for p in m['holes'])) >= 8
        sparse = len(t['holes']) < 4 or max((p['col'] for p in t['holes'])) - min((p['col'] for p in t['holes'])) < 5
        boundary = bool(left) and any((abs(e['top_x'] - (t['grid'][0] - 0.5 * (t['grid'][1] - t['grid'][0]))) < 0.8 * (t['grid'][1] - t['grid'][0]) for e in left[:5]))
        top_span = max((p['col'] for p in t['holes'])) - min((p['col'] for p in t['holes']))
        tphase = phase(t) or (len(t['holes']) >= 3 and min((p['col'] for p in t['holes'])) <= 2 and (top_span >= 5)) or (sparse and min((p['col'] for p in t['holes'])) <= 3 and (len(b['holes']) >= 5) and boundary)
        if not phase(b) or not tphase:
            continue
        proposals.append((b['score'] + t['score'] + (1 if boundary else 0), t, b, left, right, sparse, boundary))
    meta['evidence']['bottom_candidate_models'] = len(models)
    if not proposals:
        meta['failure_reasons'] = ['VER3_INSUFFICIENT_CURRENT_FRAME_TWO_ROW_PHASE']
        meta['previous_failure_reasons'] = old.get('failure_reasons', [])
        for r in slots:
            r.update(slot_geometry_usable=False, cell_geometry_ready=False, cell_crop_usable=False, geometry_status='HOLD', geometry_review_reasons=meta['failure_reasons'], geometry_version=VERSION)
        return (slots, meta)
    (_, t, b, left, right, sparse, boundary) = max(proposals, key=lambda p: p[0])
    curves = {r: np.column_stack([m['grid'], row_y(m['coef'], m['grid'], w)]) for (r, m) in [('t', t), ('b', b)]}
    horizontal_difference = t['grid'] / w - np.array(cal['x']['t']) - (b['grid'] / w - np.array(cal['x']['b']))
    top_phase_review = bool(np.max(abs(horizontal_difference[:3])) > 0.023)
    cell_curves = copy.deepcopy(curves)
    if top_phase_review:
        cell_curves['t'][:, 0] = b['grid'] + w * (np.array(cal['x']['t']) - np.array(cal['x']['b']))
        cell_curves['t'][:, 1] = row_y(t['coef'], cell_curves['t'][:, 0], w)
    meta['evidence'].update(board_top=curves['t'].tolist(), board_bottom=curves['b'].tolist(), left_plate_border_candidates=left, right_plate_border_candidates=right, sparse_top_joint_fit=sparse, left_boundary_confirmed=boundary, physical_columns=12, top_horizontal_phase_review=top_phase_review, cross_row_horizontal_difference=horizontal_difference.tolist())
    for (row, m) in [('t', t), ('b', b)]:
        meta['row_support'][row] = dict(phase_confirmed=not (row == 't' and top_phase_review), raw_holes=len(m['holes']), registered_columns=[p['col'] for p in m['holes']], y_curve_coefficients=m['coef'].tolist(), y_rms_px=m['rms'], holes=m['holes'], source='CURRENT_IMAGE_RAW_HOLES', sparse_joint_fit=sparse and row == 't')
    present = {(r['row'], int(r['col'])) for r in slots}
    for row in ('t', 'b'):
        src = max([r for r in slots if r['row'] == row], key=lambda r: int(r['col']))
        for col in range(1, 13):
            if (row, col) in present:
                continue
            r = copy.deepcopy(src)
            name = row + str(col)
            r.update(col=col, local_col=col, reference_local_col=col, slot_name=name, canonical_id=name, physical_slot=name, reference_slot_name=name, slot_origin='NEW_VER3_STRUCTURAL_GEOMETRY', record_id=hashlib.sha256((sample['source_path'] + '|' + r.get('slot_namespace_version', 'image_left_to_right_v1') + '|' + name).encode()).hexdigest()[:24])
            slots.append(r)
    for r in slots:
        row = r['row']
        i = int(r['col']) - 1
        m = t if row == 't' else b
        (x, y) = map(float, curves[row][i])
        (poly, box) = OLD._cell(cell_curves['t'], cell_curves['b'], i, row, w, h)
        near = next((p for p in m['holes'] if p['col'] == i + 1), None)
        reasons = []
        if not (0 <= x < w and 0 <= y < h):
            reasons.append('ANCHOR_OUTSIDE_NATIVE_ROI')
        if box[2] <= box[0] or box[3] <= box[1]:
            reasons.append('CELL_OUTSIDE_NATIVE_ROI')
        if row == 't' and top_phase_review:
            reasons.append('TOP_START_HORIZONTAL_PHASE_UNCONFIRMED')
        review = sparse or near is None
        r.update(x_prior=x, y_prior=y, anchor_x=x, anchor_y=y, eval_anchor_x=x, eval_anchor_y=y, slot_center_x=x, slot_center_y=y, cell_polygon=poly, cell_bbox=box, cell_width=box[2] - box[0], cell_height=box[3] - box[1], slot_geometry_usable=not reasons, cell_crop_usable=not reasons, cell_geometry_ready=not reasons, row_geometry_usable=not reasons, geometry_review_reasons=reasons, geometry_review_required=review, geometry_status='HOLD' if reasons else 'REVIEW', geometry_version=VERSION, geometry_source='VER3_CURRENT_IMAGE_JOINT_TWO_ROW', geometry_hole_x=near['x'] if near else None, geometry_hole_y=near['y'] if near else None, geometry_hole_visible=near is not None, direct_hole_support=near is not None, geometry_hole_residual_px=float(np.hypot(near['x'] - x, near['y'] - y)) if near else None, geometry_evidence='RAW_HOLE' if near else 'STRUCTURAL_JOINT_PREDICTION', ver3_geometry_candidate=True)
    meta.update(frame_status='REVIEW', failure_reasons=['VER3_JOINT_GEOMETRY_REQUIRES_BODY_OWNERSHIP_CHECK'], slot_geometry_usable_count=sum((r['slot_geometry_usable'] for r in slots)), slot_count=len(slots))
    return (slots, meta)

def build_group_calibration(base_database=None, output_path=None):
    import sqlite3, zlib
    from collections import defaultdict
    BASE_DB = Path(base_database) if base_database else CYCLE.PREPROCESS_OUTPUT / '4. 자리배치와 bbox/자리배치_ver2/state.sqlite3'
    CALIBRATION = Path(output_path) if output_path else CALIBRATION_PATH
    db = sqlite3.connect(BASE_DB.as_uri() + '?mode=ro&immutable=1', uri=True)
    groups = defaultdict(list)
    for (fid, blob) in db.execute("SELECT frame_id,payload FROM frames WHERE camera='cam2' AND geometry_status='PASS' ORDER BY sequence"):
        d = json.loads(zlib.decompress(blob))
        f = d['frame']
        m = f['geometry_metadata']
        w = f['source_width']
        g = f"{f['source_camera']}_{f['native_width']}x{f['native_height']}"
        s = m.get('row_support', {})
        if not all((r in s and s[r].get('raw_holes', 0) >= 5 and (s[r].get('first_supported_col', 9) <= 2) and (s[r].get('last_supported_col', 0) >= 10) for r in ('t', 'b'))):
            continue
        xy = {r: np.array(m['evidence']['board_top' if r == 't' else 'board_bottom'], float) / w for r in ('t', 'b')}
        if any((len(xy[r]) != 12 for r in xy)):
            continue
        yc = {r: np.asarray(s[r]['y_curve_coefficients'], float) / w for r in ('t', 'b')}
        groups[g].append(dict(frame_id=fid, x={r: xy[r][:, 0].tolist() for r in xy}, y={r: yc[r].tolist() for r in yc}))
    out = {'version': 'S4_CAM2_VER3_GROUP_INITIALIZATION', 'scope': 'STRUCTURAL_PRIOR_ONLY_NOT_OBSERVED_HOLES', 'groups': {}}
    for (g, rows) in groups.items():
        out['groups'][g] = dict(frames=len(rows), frame_ids=[r['frame_id'] for r in rows], x={r: np.median([q['x'][r] for q in rows], axis=0).tolist() for r in ('t', 'b')}, y={r: np.median([q['y'][r] for q in rows], axis=0).tolist() for r in ('t', 'b')}, gap=np.median([np.array(q['y']['b']) - np.array(q['y']['t']) for q in rows], axis=0).tolist())
        print(g, json.dumps({k: v for (k, v) in out['groups'][g].items() if k != 'frame_ids'}), flush=True)
    assert len(out['groups']) == 3
    CALIBRATION.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SHA256', hashlib.sha256(CALIBRATION.read_bytes()).hexdigest())
