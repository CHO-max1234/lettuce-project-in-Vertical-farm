"""Final anchored target-mask selection with detached-only shared-pixel ownership. The numerical base and ownership refinement are kept together."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
MAX_CENTER_DISTANCE = 0.55
MIN_CORE_COVERAGE = 0.2
LOCAL_ONLY_MAX_DISTANCE = 0.3
LOCAL_ONLY_MIN_CORE = 0.45
AGREEMENT_IOU = 0.65
UNION_IOU = 0.8

def _base_anchor(row):
    return (float(row['eval_anchor_x']), float(row['eval_anchor_y']))

def _base_core_coverage(mask, point, radius):
    (x, y) = point
    (h, w) = mask.shape
    (x1, x2) = (max(0, int(x - radius)), min(w, int(x + radius) + 1))
    (y1, y2) = (max(0, int(y - radius)), min(h, int(y + radius) + 1))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    (yy, xx) = np.ogrid[y1:y2, x1:x2]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return float((mask[y1:y2, x1:x2] & disk).sum() / max(1, disk.sum()))

def _base_near_independent_anchor(mask, point, radius):
    (x, y) = point
    (h, w) = mask.shape
    (x1, x2) = (max(0, int(x - radius) - 1), min(w, int(x + radius) + 2))
    (y1, y2) = (max(0, int(y - radius) - 1), min(h, int(y + radius) + 2))
    if x2 <= x1 or y2 <= y1:
        return False
    (yy, xx) = np.where(mask[y1:y2, x1:x2])
    return bool(len(xx) and np.min((xx + x1 - x) ** 2 + (yy + y1 - y) ** 2) <= radius ** 2)

def _base_features(row, candidate):
    (ax, ay) = _base_anchor(row)
    (cw, ch) = (float(row['cell_width']), float(row['cell_height']))
    radius = max(5.0, 0.1 * min(cw, ch))
    return dict(distance=math.hypot((candidate['cx'] - ax) / cw, (candidate['cy'] - ay) / ch), core=_base_core_coverage(candidate['mask'], (ax, ay), radius), near=_base_near_independent_anchor(candidate['mask'], (ax, ay), max(3.0, 0.055 * min(cw, ch))))

def _base_supported(row, candidate):
    f = _base_features(row, candidate)
    return f['distance'] <= MAX_CENTER_DISTANCE and (f['core'] >= MIN_CORE_COVERAGE or (f['distance'] <= 0.35 and f['near']))

def _base_visible_empty_core(row, source):
    if not row.get('eval_hole_visible'):
        return False
    (ax, ay) = _base_anchor(row)
    radius = max(3.0, 0.045 * min(float(row['cell_width']), float(row['cell_height'])))
    (h, w) = source.shape[:2]
    (x1, x2) = (max(0, int(ax - radius)), min(w, int(ax + radius) + 1))
    (y1, y2) = (max(0, int(ay - radius)), min(h, int(ay + radius) + 1))
    if x2 <= x1 or y2 <= y1:
        return False
    hsv = cv2.cvtColor(source[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    (yy, xx) = np.ogrid[y1:y2, x1:x2]
    disk = (xx - ax) ** 2 + (yy - ay) ** 2 <= radius ** 2
    green = (hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 35)
    dark = hsv[:, :, 2] < 100
    n = max(1, disk.sum())
    return float((dark & disk).sum() / n) >= 0.5 and float((green & disk).sum() / n) < 0.15

def _base_clean_anchor_components(mask, row):
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 2:
        return (mask, 0)
    point = _base_anchor(row)
    radius = max(5.0, 0.1 * min(float(row['cell_width']), float(row['cell_height'])))
    supports = [(_base_core_coverage(labels == i, point, radius), int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)]
    (support, area, label) = max(supports)
    if support < MIN_CORE_COVERAGE:
        return (None, 0)
    selected = labels == label
    removed = int(mask.sum() - selected.sum())
    if removed / max(1, int(mask.sum())) > 0.05:
        return (mask, 0)
    return (selected, removed)

def _base_overlap(a, b):
    inter = int((a & b).sum())
    return inter / max(1, int((a | b).sum()))

def _base_internal_edge(candidate, shape):
    (x1, y1, x2, y2) = candidate['prediction_box']
    (bx1, by1, bx2, by2) = candidate['box']
    (h, w) = shape
    return x1 > 0 and bx1 <= x1 + 1 or (y1 > 0 and by1 <= y1 + 1) or (x2 < w and bx2 >= x2 - 1) or (y2 < h and by2 >= y2 - 1)

def _base_select_frame(rows, source, whole_candidates, local_candidates_by_record):
    results = {}
    eligible = []
    for row in rows:
        key = str(row['record_id'])
        (ax, ay) = _base_anchor(row)
        if not (0 <= ax < source.shape[1] and 0 <= ay < source.shape[0]):
            results[key] = dict(mask=None, status='OUT_OF_FOV', reasons=['INDEPENDENT_ANCHOR_OUTSIDE_SOURCE'], route='')
        elif row.get('anchor_registration_status') == 'UNKNOWN_REGISTRATION':
            results[key] = dict(mask=None, status='REVIEW_ANCHOR_REGISTRATION', reasons=['NO_RELIABLE_INDEPENDENT_ANCHOR'], route='')
        elif _base_visible_empty_core(row, source) and (not any((_base_supported(row, c) for c in whole_candidates))):
            results[key] = dict(mask=None, status='VISIBLE_EMPTY_CORE', reasons=['REGISTERED_RAW_HOLE_AND_DARK_NON_GREEN_CORE'], route='')
        else:
            eligible.append(row)
    (n, m) = (len(eligible), len(whole_candidates))
    cost = np.full((n, m + n), 1000000.0, dtype=float)
    for (i, row) in enumerate(eligible):
        cost[i, m + i] = 2.0
        for (j, candidate) in enumerate(whole_candidates):
            if _base_supported(row, candidate):
                f = _base_features(row, candidate)
                cost[i, j] = f['distance'] - 0.15 * f['core'] - 0.05 * candidate['conf']
    assigned = {}
    if n:
        (ii, jj) = linear_sum_assignment(cost)
        assigned = {int(i): whole_candidates[int(j)] for (i, j) in zip(ii, jj) if j < m and cost[i, j] < 2.0}
    for (i, row) in enumerate(eligible):
        key = str(row['record_id'])
        whole = assigned.get(i)
        local = [c for c in local_candidates_by_record.get(key, []) if _base_supported(row, c)]
        local.sort(key=lambda c: (_base_features(row, c)['distance'], -_base_features(row, c)['core'], -c['conf']))
        selected = None
        reasons = []
        route = ''
        agreement = None
        family_added = 0
        if whole is not None:
            body = whole['mask'].copy()
            for candidate in whole_candidates:
                if candidate['index'] == whole['index'] or not _base_supported(row, candidate):
                    continue
                if _base_overlap(whole['mask'], candidate['mask']) < AGREEMENT_IOU:
                    continue
                other_core = False
                for (j, other) in enumerate(eligible):
                    if j == i or j not in assigned:
                        continue
                    radius = max(5.0, 0.1 * min(float(other['cell_width']), float(other['cell_height'])))
                    if _base_core_coverage(candidate['mask'], _base_anchor(other), radius) >= 0.6:
                        other_core = True
                        break
                if not other_core:
                    body |= candidate['mask']
            family_added = int(body.sum() - whole['mask'].sum())
            whole = {**whole, 'mask': body}
            compatible = [c for c in local if _base_overlap(c['mask'], whole['mask']) >= AGREEMENT_IOU]
            best = max(compatible, key=lambda c: _base_overlap(c['mask'], whole['mask'])) if compatible else None
            if best is not None:
                agreement = _base_overlap(best['mask'], whole['mask'])
                if family_added > 0:
                    selected = whole['mask'].copy()
                    route = 'whole_family_preserve_body'
                    reasons.append('COMPATIBLE_SAME_CORE_WHOLE_MASKS_UNION')
                elif _base_internal_edge(best, source.shape[:2]):
                    selected = whole['mask'].copy()
                    route = 'whole_preserve_body'
                    reasons.append('LOCAL_INTERNAL_EDGE_CLIPS_BODY')
                elif agreement >= UNION_IOU:
                    selected = whole['mask'] | best['mask']
                    route = 'local_whole_agreement_union'
                else:
                    selected = whole['mask'] | best['mask']
                    route = 'local_whole_body_preserving_union'
                    reasons.append('NEVER_REPLACE_SUPPORTED_WHOLE_WITH_SMALLER_LOCAL')
            else:
                selected = whole['mask'].copy()
                route = 'whole_supported_recovery'
                reasons.append('LOCAL_UNSUPPORTED_NO_ARBITRARY_SHAPE_FALLBACK')
        elif local:
            best = local[0]
            f = _base_features(row, best)
            if f['distance'] <= LOCAL_ONLY_MAX_DISTANCE and f['core'] >= LOCAL_ONLY_MIN_CORE and (not _base_internal_edge(best, source.shape[:2])):
                selected = best['mask'].copy()
                route = 'local_independent_core'
                reasons.append('FULL_BED_NO_SUPPORTED_TARGET_LOCAL_CLOSED_BODY')
            else:
                reasons.append('LOCAL_ONLY_EDGE_OR_WEAK_CORE_MAY_BE_NEIGHBOR')
        if selected is None:
            results[key] = dict(mask=None, status='NO_SUPPORTED_TARGET', reasons=reasons or ['NO_CANDIDATE_SUPPORTS_INDEPENDENT_ANCHOR'], route='')
            continue
        (selected, removed) = _base_clean_anchor_components(selected, row)
        if selected is None:
            results[key] = dict(mask=None, status='REVIEW_COMPONENT_SUPPORT', reasons=['NO_CONNECTED_COMPONENT_SUPPORTS_INDEPENDENT_CORE'], route='')
            continue
        claims = []
        for (j, other) in enumerate(eligible):
            if j == i or j not in assigned:
                continue
            point = _base_anchor(other)
            radius = max(5.0, 0.1 * min(float(other['cell_width']), float(other['cell_height'])))
            if _base_core_coverage(selected, point, radius) >= 0.6:
                claims.append(other['physical_slot'])
        if claims:
            results[key] = dict(mask=None, status='REVIEW_OTHER_SUPPORTED_BODY', reasons=['OTHER_SUPPORTED_CORE:' + ','.join(claims)], route=route)
        else:
            results[key] = dict(mask=selected, status='SELECTED', reasons=reasons, route=route, candidate_id='whole:' + str(whole['index']) if whole is not None else 'local:' + key, context_iou=agreement, removed_detached_pixels=removed, whole_family_added_pixels=family_added, center_definition='REGISTERED_HOLE_GEOMETRY_INDEPENDENT_OF_SEG')
    for a in range(len(rows)):
        ka = str(rows[a]['record_id'])
        if results[ka]['mask'] is None:
            continue
        for b in range(a + 1, len(rows)):
            kb = str(rows[b]['record_id'])
            (ma, mb) = (results[ka]['mask'], results[kb]['mask'])
            if ma is None or mb is None:
                continue
            shared = int((ma & mb).sum()) / max(1, min(int(ma.sum()), int(mb.sum())))
            if shared >= 0.75:
                for key in (ka, kb):
                    results[key] = dict(mask=None, status='REVIEW_DUPLICATE_BODY', reasons=['LARGE_SHARED_TARGET_MASK'], route='')
    return results
OTHER_DISTANCE_RATIO = 0.65
MIN_DISTANCE_ADVANTAGE_PX = 8.0
DISTANCE_ADVANTAGE_ANCHOR_FRACTION = 0.08
PROTECTED_CORE_CELL_FRACTION = 0.15
MIN_PROTECTED_CORE_RADIUS_PX = 8.0
COMPONENT_CONNECTIVITY = 8
MAX_REMOVED_FRACTION = 0.2
AMBIGUOUS_SHARED_FRACTION = 0.02
MIN_AMBIGUOUS_SHARED_PIXELS = 128
# Original frozen ver1.3 digest retained after merging its calculations below.
BASE_SELECTOR_SHA256 = '8fb735ad96f223511bdb785edf9c862593907eb9dc2d464ca7b299df014b6df3'
POLICY = 'DETACHED_TARGET_SHARED_PIXELS_CLAIMED_BY_OTHER_ANCHORED_COMPONENT'
import math
import cv2
import numpy as np

def needs_local(row, source, whole_candidates):
    """Exact initial ver1.3 eligibility; registry fields retain caller types.

    Automatic visibility EMPTY / NO_SEG never provides a skip condition.
    The freshly registered eval_hole_visible boolean and raw source are used
    by frozen ver1.3's own visible_empty_core predicate without reinterpretation.
    """
    (ax, ay) = _base_anchor(row)
    if not (0 <= ax < source.shape[1] and 0 <= ay < source.shape[0]):
        return False
    elif row.get('anchor_registration_status') == 'UNKNOWN_REGISTRATION':
        return False
    elif _base_visible_empty_core(row, source) and (not any((_base_supported(row, c) for c in whole_candidates))):
        return False
    else:
        return True

def _truth(value):
    return str(value).strip().lower() in ('true', '1', 'yes')

def _geometry(row, shape):
    if not str(row.get('anchor_registration_status', '')).startswith('REGISTERED') or _truth(row.get('anchor_registration_review', False)):
        return None
    try:
        (ax, ay) = (float(row['eval_anchor_x']), float(row['eval_anchor_y']))
        (width, height) = (float(row['cell_width']), float(row['cell_height']))
    except (KeyError, TypeError, ValueError):
        return None
    if not all((math.isfinite(v) for v in (ax, ay, width, height))):
        return None
    if width <= 0 or height <= 0 or (not (0 <= ax < shape[1] and 0 <= ay < shape[0])):
        return None
    return (ax, ay, max(MIN_PROTECTED_CORE_RADIUS_PX, PROTECTED_CORE_CELL_FRACTION * min(width, height)))

def _anchored_components(mask, geometry):
    """Every complete 8-connected component touching the protected core."""
    if geometry is None:
        return (np.zeros_like(mask), 0)
    (ax, ay, radius) = geometry
    (count, labels) = cv2.connectedComponents(mask.astype(np.uint8), connectivity=COMPONENT_CONNECTIVITY)
    (h, w) = mask.shape
    (x1, x2) = (max(0, int(ax - radius)), min(w, int(ax + radius) + 1))
    (y1, y2) = (max(0, int(ay - radius)), min(h, int(ay + radius) + 1))
    (yy, xx) = np.ogrid[y1:y2, x1:x2]
    disk = (xx - ax) ** 2 + (yy - ay) ** 2 <= radius ** 2
    protected = np.unique(labels[y1:y2, x1:x2][disk & mask[y1:y2, x1:x2]])
    protected = protected[protected != 0]
    return (np.isin(labels, protected), int(len(protected)))

def apply_shared_ownership(rows, decisions):
    """CPU replay API. Inputs remain unchanged; all claims use original masks."""
    row_by_key = {str(row['record_id']): row for row in rows}
    if len(row_by_key) != len(rows):
        raise ValueError('Duplicate record_id in one frame')
    output = {str(key): {**value, 'reasons': list(value.get('reasons', []))} for (key, value) in decisions.items()}
    masks = {}
    shape = None
    for (key, decision) in output.items():
        mask = decision.get('mask')
        if mask is None or decision.get('status') != 'SELECTED':
            continue
        if key not in row_by_key:
            raise ValueError('Selected decision has no input row: ' + key)
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError('Ownership requires full-source 2D bool masks')
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise ValueError('Selected mask source shapes differ')
        masks[key] = mask.copy()
    if not masks:
        return output
    geometries = {key: _geometry(row_by_key[key], shape) for key in masks}
    anchored_info = {key: _anchored_components(mask, geometries[key]) for (key, mask) in masks.items()}
    anchored = {key: info[0] for (key, info) in anchored_info.items()}
    missing_anchored = {key: anchored_info[key][1] == 0 for key in masks}
    proposals = {key: np.zeros(shape, dtype=bool) for key in masks}
    shared_sets = {key: np.zeros(shape, dtype=bool) for key in masks}
    claims = {key: [] for key in masks}
    weak_shared = {key: False for key in masks}
    keys = list(masks)
    for (i, ka) in enumerate(keys):
        for kb in keys[i + 1:]:
            shared = masks[ka] & masks[kb]
            shared_count = int(shared.sum())
            if not shared_count:
                continue
            shared_sets[ka] |= shared
            shared_sets[kb] |= shared
            if missing_anchored[ka] or missing_anchored[kb]:
                weak_shared[ka] = weak_shared[kb] = True
                for (target, other) in ((ka, kb), (kb, ka)):
                    claims[target].append(dict(record_id=other, physical_slot=row_by_key[other].get('physical_slot', ''), shared_pixels=shared_count, proposed_removed_pixels=0, evidence='NO_RELIABLE_ANCHORED_COMPONENT_RETAINED'))
                continue
            (ax, ay, ar) = geometries[ka]
            (bx, by, br) = geometries[kb]
            advantage = max(MIN_DISTANCE_ADVANTAGE_PX, DISTANCE_ADVANTAGE_ANCHOR_FRACTION * math.hypot(bx - ax, by - ay))
            (yy, xx) = np.where(shared)
            (da, db) = (np.hypot(xx - ax, yy - ay), np.hypot(xx - bx, yy - by))
            for (target, other, own_distance, other_distance) in ((ka, kb, da, db), (kb, ka, db, da)):
                eligible = ~anchored[target][yy, xx] & anchored[other][yy, xx]
                take = eligible & (other_distance <= OTHER_DISTANCE_RATIO * own_distance) & (own_distance - other_distance >= advantage)
                proposals[target][yy[take], xx[take]] = True
                claims[target].append(dict(record_id=other, physical_slot=row_by_key[other].get('physical_slot', ''), shared_pixels=shared_count, eligible_detached_shared_pixels=int(eligible.sum()), proposed_removed_pixels=int(take.sum()), min_distance_advantage_px=round(advantage, 4), evidence='DETACHED_SHARED_OTHER_ANCHORED_CLAIM' if take.any() else 'OWN_COMPONENT_OR_AMBIGUOUS_RETAINED'))
    excessive = {key: int(proposals[key].sum()) / max(1, int(mask.sum())) > MAX_REMOVED_FRACTION for (key, mask) in masks.items()}
    cleaned = {key: mask.copy() if excessive[key] else mask & ~proposals[key] for (key, mask) in masks.items()}
    remaining_shared = {key: np.zeros(shape, dtype=bool) for key in masks}
    for (i, ka) in enumerate(keys):
        for kb in keys[i + 1:]:
            shared = cleaned[ka] & cleaned[kb]
            remaining_shared[ka] |= shared
            remaining_shared[kb] |= shared
    for (key, original) in masks.items():
        decision = output[key]
        area = int(original.sum())
        proposed_count = int(proposals[key].sum())
        removed_count = 0 if excessive[key] else proposed_count
        retained_shared = int(remaining_shared[key].sum())
        ambiguous = retained_shared >= max(MIN_AMBIGUOUS_SHARED_PIXELS, int(math.ceil(AMBIGUOUS_SHARED_FRACTION * area)))
        review = bool(excessive[key] or ambiguous or weak_shared[key] or missing_anchored[key] or _truth(decision.get('review_required', False)))
        if excessive[key]:
            decision['reasons'].append('OWNERSHIP_REMOVAL_OVER_20_PERCENT_KEEP_ORIGINAL_REVIEW')
        elif removed_count:
            decision['reasons'].append('ONLY_DETACHED_SHARED_PIXELS_CLAIMED_BY_OTHER_ANCHORED_COMPONENT_REMOVED')
        if ambiguous:
            decision['reasons'].append('AMBIGUOUS_SHARED_PIXELS_RETAINED_REVIEW')
        if missing_anchored[key] or weak_shared[key]:
            decision['reasons'].append('NO_RELIABLE_ANCHORED_COMPONENT_RETAINED_REVIEW')
        decision.update(mask=cleaned[key], ownership_policy=POLICY, ownership_base_selector_sha256=BASE_SELECTOR_SHA256, ownership_shared_pixels=int(shared_sets[key].sum()), ownership_proposed_removed_pixels=proposed_count, ownership_removed_pixels=removed_count, ownership_retained_shared_pixels=retained_shared, ownership_removed_fraction=removed_count / max(1, area), ownership_anchored_component_count=anchored_info[key][1], ownership_anchored_component_pixels=int(anchored[key].sum()), ownership_missing_anchored_component=missing_anchored[key], ownership_claims=claims[key], review_required=review)
    return output

def select_frame(rows, source, whole_candidates, local_candidates_by_record):
    """Same candidate API as frozen ver1.3; no model loading or inference."""
    return apply_shared_ownership(rows, _base_select_frame(rows, source, whole_candidates, local_candidates_by_record))
DEPENDENCY_PATHS = ()
