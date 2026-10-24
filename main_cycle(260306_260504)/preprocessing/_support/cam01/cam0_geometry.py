"""Top-view geometry used by the final preprocessing adapters."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[3]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import importlib
_slot_core = importlib.import_module('slot_geometry')
_lettuce_seg = importlib.import_module('plant_segmentation')
Visibility = _slot_core.Visibility
RegStatus = _slot_core.RegStatus
build_slots_from_columns = _slot_core.build_slots_from_columns
apply_bounded_residual = _slot_core.apply_bounded_residual
build_cell_bboxes = _slot_core.build_cell_bboxes
N_PHYSICAL_COLS = _slot_core.N_PHYSICAL_COLS
CAM0_MIN_VISIBLE_COLS = 7
CAM0_MAX_VISIBLE_COLS = 7
PITCH_PRIOR_RATIO = 0.125
PITCH_MIN_RATIO = 0.085
PITCH_MAX_RATIO = 0.165
COLUMN_MERGE_PITCH_RATIO = 0.45
ROW_DEDUP_PITCH_RATIO = 0.6
GAP_REVIEW_PITCH_RATIO = 1.65
CAM0_EVIDENCE_X_MIN_RATIO = 0.07
TOP_ROW_BAND = (0.08, 0.5)
BOTTOM_ROW_BAND = (0.46, 0.94)
ROW_SPLIT_RATIO = 0.5
GREEN_HSV_LOW = (22, 35, 35)
GREEN_HSV_HIGH = (95, 255, 255)
GREEN_PROFILE_THRESHOLD_RATIO = 0.18
GREEN_MIN_PIXELS = 80
BED_NEUTRAL_S_MAX = 82
BED_NEUTRAL_V_MIN = 38
BED_NEUTRAL_V_MAX = 232
BED_PROFILE_MIN_SUPPORT = 0.14
BED_X_CLOSE_RATIO = 0.075
BED_Y_CLOSE_RATIO = 0.075
BED_MIN_WIDTH_RATIO = 0.42
BED_CELL_PITCH_MIN_RATIO = 0.65
BED_CELL_PITCH_MAX_RATIO = 1.4
HOLE_MIN_RADIUS_RATIO = 0.008
HOLE_MAX_RADIUS_RATIO = 0.05
HOLE_MIN_CIRCULARITY = 0.42
HOLE_MIN_DARK_CONTRAST = 7.0
MAX_SHIFT_RATIO = 0.22
EVIDENCE_REJECT_RATIO = 0.55

def _row_name(y: float, image_h: int) -> Optional[str]:
    ratio = y / max(image_h, 1)
    if TOP_ROW_BAND[0] <= ratio <= TOP_ROW_BAND[1]:
        return 't'
    if BOTTOM_ROW_BAND[0] <= ratio <= BOTTOM_ROW_BAND[1]:
        return 'b'
    return None

def _odd(value: int, minimum: int=3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1

def _weighted_merge(items: List[dict], kind: str) -> dict:
    weights = np.asarray([max(float(item.get('score', 0.0)), 1e-06) for item in items])
    xs = np.asarray([float(item['x']) for item in items])
    ys = np.asarray([float(item['y']) for item in items])
    return {'x': float(np.average(xs, weights=weights)), 'y': float(np.average(ys, weights=weights)), 'score': float(np.max(weights)), 'kind': kind}

def detect_hole_evidence(image: np.ndarray) -> Dict[str, List[dict]]:
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    proc = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    proc = cv2.GaussianBlur(proc, (5, 5), 0)
    (_, otsu) = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    (_, fixed) = cv2.threshold(proc, 92, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.bitwise_and(otsu, fixed)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    (contours, _) = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_radius = HOLE_MIN_RADIUS_RATIO * w
    max_radius = HOLE_MAX_RADIUS_RATIO * w
    found: Dict[str, List[dict]] = {'t': [], 'b': []}
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < HOLE_MIN_CIRCULARITY:
            continue
        ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
        if not min_radius <= radius <= max_radius:
            continue
        row = _row_name(cy, h)
        if row is None:
            continue
        (x, y, bw, bh) = cv2.boundingRect(contour)
        if max(bw, bh) / max(min(bw, bh), 1) > 2.1:
            continue
        (yy, xx) = np.ogrid[:h, :w]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inner = r2 <= (0.72 * radius) ** 2
        ring = (r2 > (0.9 * radius) ** 2) & (r2 <= (1.65 * radius) ** 2)
        if inner.sum() < 20 or ring.sum() < 20:
            continue
        dark_contrast = float(proc[ring].mean() - proc[inner].mean())
        if dark_contrast < HOLE_MIN_DARK_CONTRAST:
            continue
        found[row].append({'x': float(cx), 'y': float(cy), 'score': float(dark_contrast + 20.0 * circularity), 'kind': 'hole'})
    return found

def _green_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(GREEN_HSV_LOW), np.asarray(GREEN_HSV_HIGH))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def _profile_peaks(profile: np.ndarray, min_distance: int, threshold: float) -> List[int]:
    if profile.size < 3 or float(profile.max()) <= 0:
        return []
    candidates = [idx for idx in range(1, len(profile) - 1) if profile[idx] >= profile[idx - 1] and profile[idx] >= profile[idx + 1] and (profile[idx] >= threshold)]
    selected: List[int] = []
    for idx in sorted(candidates, key=lambda value: profile[value], reverse=True):
        if all((abs(idx - kept) >= min_distance for kept in selected)):
            selected.append(idx)
    return sorted(selected)

def detect_lettuce_evidence(image: np.ndarray) -> Tuple[Dict[str, List[dict]], np.ndarray]:
    (h, w) = image.shape[:2]
    mask = _green_mask(image)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    pitch_prior = PITCH_PRIOR_RATIO * w
    min_distance = max(20, int(round(0.55 * pitch_prior)))
    smooth_kernel = _odd(int(round(0.035 * w)))
    found: Dict[str, List[dict]] = {'t': [], 'b': []}
    for (row, (lo, hi)) in (('t', TOP_ROW_BAND), ('b', BOTTOM_ROW_BAND)):
        y1 = int(round(lo * h))
        y2 = int(round(hi * h))
        row_mask = mask[y1:y2]
        profile = row_mask.astype(np.float32).sum(axis=0) / 255.0
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_kernel, 1), 0).ravel()
        positive = profile[profile > 0]
        if positive.size == 0:
            continue
        threshold = max(GREEN_PROFILE_THRESHOLD_RATIO * float(profile.max()), float(np.percentile(positive, 55)))
        for peak_x in _profile_peaks(profile, min_distance, threshold):
            half_window = max(20, int(round(0.38 * pitch_prior)))
            x1 = max(0, peak_x - half_window)
            x2 = min(w, peak_x + half_window + 1)
            roi_distance = distance[y1:y2, x1:x2]
            roi_mask = row_mask[:, x1:x2]
            if int(np.count_nonzero(roi_mask)) < GREEN_MIN_PIXELS:
                continue
            (_, max_value, _, max_loc) = cv2.minMaxLoc(roi_distance)
            cx = float(x1 + max_loc[0])
            cy = float(y1 + max_loc[1])
            found[row].append({'x': cx, 'y': cy, 'score': float(max(max_value, 1.0)), 'kind': 'lettuce'})
    return (found, mask)

def _estimate_pitch(rows: Dict[str, List[dict]], image_w: int) -> float:
    lower = PITCH_MIN_RATIO * image_w
    upper = PITCH_MAX_RATIO * image_w
    diffs: List[float] = []
    for evidence in rows.values():
        xs = sorted((float(item['x']) for item in evidence))
        diffs.extend((b - a for (a, b) in zip(xs[:-1], xs[1:]) if lower <= b - a <= upper))
    return float(np.median(diffs)) if diffs else PITCH_PRIOR_RATIO * image_w

def _merge_row_evidence(items: List[dict], pitch: float) -> List[dict]:
    if not items:
        return []
    groups: List[List[dict]] = []
    tolerance = ROW_DEDUP_PITCH_RATIO * pitch
    for item in sorted(items, key=lambda value: value['x']):
        if not groups or abs(item['x'] - np.mean([v['x'] for v in groups[-1]])) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    merged = []
    for group in groups:
        kinds = {item['kind'] for item in group}
        kind = 'hole+lettuce' if len(kinds) > 1 else next(iter(kinds))
        merged.append(_weighted_merge(group, kind))
    return merged

def _merge_columns(rows: Dict[str, List[dict]], pitch: float) -> List[dict]:
    tagged = []
    for row in ('t', 'b'):
        tagged.extend(({**item, 'row': row} for item in rows[row]))
    if not tagged:
        return []
    tolerance = COLUMN_MERGE_PITCH_RATIO * pitch
    groups: List[List[dict]] = []
    for item in sorted(tagged, key=lambda value: value['x']):
        if not groups or abs(item['x'] - np.mean([v['x'] for v in groups[-1]])) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    columns = []
    for (local_col, group) in enumerate(groups, start=1):
        by_row: Dict[str, dict] = {}
        for row in ('t', 'b'):
            row_items = [item for item in group if item['row'] == row]
            if row_items:
                kinds = {item['kind'] for item in row_items}
                kind = 'hole+lettuce' if len(kinds) > 1 else next(iter(kinds))
                by_row[row] = _weighted_merge(row_items, kind)
        columns.append({'local_col': local_col, 'x': float(np.mean([item['x'] for item in group])), 'rows': by_row})
    return columns

def _merge_extra_columns_to_cam0_contract(columns: List[dict]) -> Tuple[List[dict], int]:
    merged = list(columns)
    merge_count = 0
    while len(merged) > CAM0_MAX_VISIBLE_COLS:
        gaps = np.diff([column['x'] for column in merged])
        pair_index = int(np.argmin(gaps))
        left = merged[pair_index]
        right = merged[pair_index + 1]
        rows: Dict[str, dict] = {}
        for row in ('t', 'b'):
            items = [column['rows'][row] for column in (left, right) if row in column['rows']]
            if not items:
                continue
            kinds = {item['kind'] for item in items}
            kind = 'hole+lettuce' if len(kinds) > 1 else next(iter(kinds))
            rows[row] = _weighted_merge(items, kind)
        replacement = {'local_col': pair_index + 1, 'x': float(np.mean([left['x'], right['x']])), 'rows': rows}
        merged[pair_index:pair_index + 2] = [replacement]
        merge_count += 1
    for (local_col, column) in enumerate(merged, start=1):
        column['local_col'] = local_col
    return (merged, merge_count)

def _fit_row_y(columns: List[dict], row: str, fallback_ratio: float, image_h: int):
    all_points = [(float(column['rows'][row]['x']), float(column['rows'][row]['y']), str(column['rows'][row].get('kind', ''))) for column in columns if row in column['rows']]
    hole_points = [(x, y) for (x, y, kind) in all_points if 'hole' in kind]
    points = hole_points if len(hole_points) >= 2 else [(x, y) for (x, y, _) in all_points]
    if len(points) >= 2:
        xs = np.asarray([point[0] for point in points], dtype=np.float64)
        ys = np.asarray([point[1] for point in points], dtype=np.float64)
        median_y = float(np.median(ys))
        mad = float(np.median(np.abs(ys - median_y)))
        if mad > 1.0 and len(points) >= 4:
            keep = np.abs(ys - median_y) <= 3.0 * mad
            (xs, ys) = (xs[keep], ys[keep])
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0
        slope = float(np.clip(slope, -0.1, 0.1))
        intercept = float(np.median(ys - slope * xs))
        return lambda x: float(intercept + slope * x)
    fallback = fallback_ratio * image_h
    return lambda x: float(fallback)

def _visibility_from_evidence(evidence: Optional[dict]) -> Visibility:
    if evidence is None:
        return Visibility.UNKNOWN
    return Visibility.VISIBLE if 'lettuce' in evidence['kind'] else Visibility.EMPTY

def measure_cam0_bed_bbox(image: np.ndarray, columns: List[dict], pitch_reference: float) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
    (h, w) = image.shape[:2]
    observed_x = [float(column['x']) for column in columns]
    observed_y = [float(evidence['y']) for column in columns for evidence in column['rows'].values()]
    if not observed_x or not observed_y:
        return (None, {'reason': 'NO_OBSERVED_CENTER_FOR_BED_MEASUREMENT'})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    y_span = max(max(observed_y) - min(observed_y), 0.22 * h)
    profile_y1 = int(round(max(0.0, min(observed_y) - 0.45 * y_span)))
    profile_y2 = int(round(min(float(h), max(observed_y) + 0.45 * y_span)))
    edge_profile = edge_x[profile_y1:profile_y2].mean(axis=0)
    edge_profile = cv2.GaussianBlur(edge_profile.reshape(1, -1), (21, 1), 0).ravel()

    def nearest_strong_edge(expected: float) -> Tuple[int, float]:
        radius = max(20, int(round(0.34 * pitch_reference)))
        start = max(0, int(round(expected)) - radius)
        end = min(w, int(round(expected)) + radius + 1)
        if end <= start:
            return (int(np.clip(round(expected), 0, w)), 0.0)
        values = edge_profile[start:end]
        scale = max(float(values.max()), 1e-06)
        positions = np.arange(start, end, dtype=np.float32)
        distance_penalty = np.abs(positions - expected) / max(radius, 1)
        score = values / scale - 0.3 * distance_penalty
        index = int(np.argmax(score))
        return (start + index, float(values[index]))
    expected_left = min(observed_x) - 0.52 * pitch_reference
    expected_right = max(observed_x) + 0.52 * pitch_reference
    (bed_x1, left_strength) = nearest_strong_edge(expected_left)
    if expected_right >= 0.95 * w:
        (bed_x2, right_strength) = (w, float(edge_profile[-1]))
    else:
        (bed_x2, right_strength) = nearest_strong_edge(expected_right)
    top_y = [float(column['rows']['t']['y']) for column in columns if 't' in column['rows']]
    bottom_y = [float(column['rows']['b']['y']) for column in columns if 'b' in column['rows']]
    if not top_y or not bottom_y:
        return (None, {'reason': 'BED_ROW_CENTER_NOT_FOUND'})
    top_center = float(np.median(top_y))
    bottom_center = float(np.median(bottom_y))
    row_gap = max(20.0, bottom_center - top_center)
    bed_y1 = int(round(np.clip(top_center - 0.55 * row_gap, 0, h)))
    bed_y2 = int(round(np.clip(bottom_center + 0.55 * row_gap, 0, h)))
    measured_cell_pitch = (bed_x2 - bed_x1) / len(columns)
    pitch_ratio = measured_cell_pitch / max(pitch_reference, 1.0)
    if not BED_CELL_PITCH_MIN_RATIO <= pitch_ratio <= BED_CELL_PITCH_MAX_RATIO:
        return (None, {'reason': 'BED_WIDTH_INCONSISTENT_WITH_OBSERVED_COUNT', 'bed_x1': bed_x1, 'bed_x2': bed_x2, 'measured_cell_pitch': float(measured_cell_pitch), 'pitch_reference': float(pitch_reference), 'pitch_ratio': float(pitch_ratio)})
    return ((bed_x1, bed_y1, bed_x2, bed_y2), {'reason': 'OK', 'left_edge_strength': left_strength, 'right_edge_strength': right_strength, 'measured_cell_pitch': float(measured_cell_pitch), 'pitch_reference': float(pitch_reference), 'pitch_ratio': float(pitch_ratio)})

def map_cam0_v3(image: np.ndarray, seg_objects: Optional[List[dict]]=None):
    (h, w) = image.shape[:2]
    holes = detect_hole_evidence(image)
    if seg_objects is None:
        (lettuces, green_mask) = detect_lettuce_evidence(image)
        used_lettuce_seg = False
    else:
        lettuces = _lettuce_seg.segments_to_row_evidence(seg_objects, h)
        green_mask = _green_mask(image)
        used_lettuce_seg = True
    evidence_x_min = CAM0_EVIDENCE_X_MIN_RATIO * w
    for evidence_group in (holes, lettuces):
        for row in ('t', 'b'):
            evidence_group[row] = [evidence for evidence in evidence_group[row] if evidence['x'] >= evidence_x_min]
    combined = {row: holes[row] + lettuces[row] for row in ('t', 'b')}
    pitch_seed = _estimate_pitch(combined, w)
    rows = {row: _merge_row_evidence(combined[row], pitch_seed) for row in ('t', 'b')}
    pitch = _estimate_pitch(rows, w)
    columns = _merge_columns(rows, pitch)
    raw_observed_count = len(columns)
    (columns, duplicate_merge_count) = _merge_extra_columns_to_cam0_contract(columns)
    n_visible = len(columns)
    qc_flags: List[str] = []
    if not used_lettuce_seg:
        qc_flags.append('LETTUCE_SEG_NOT_PROVIDED')
    if duplicate_merge_count:
        qc_flags.append('EXTRA_EVIDENCE_DUPLICATE_MERGED')
    if not CAM0_MIN_VISIBLE_COLS <= n_visible <= CAM0_MAX_VISIBLE_COLS:
        return ([], {'status': 'CAM0_SLOT_COUNT_REVIEW', 'method': 'count_holes_and_lettuce_seg_then_split', 'n_visible': n_visible, 'raw_observed_column_count': raw_observed_count, 'duplicate_merge_count': duplicate_merge_count, 'pitch': pitch, 'hole_count': sum((len(v) for v in holes.values())), 'lettuce_peak_count': sum((len(v) for v in lettuces.values())), 'lettuce_segment_count': len(seg_objects or []), 'qc_flags': ['OBSERVED_COLUMN_COUNT_OUT_OF_RANGE'], 'hole_evidence': holes, 'lettuce_evidence': lettuces})
    gaps = np.diff([column['x'] for column in columns])
    if any((gap > GAP_REVIEW_PITCH_RATIO * pitch for gap in gaps)):
        qc_flags.append('MISSING_COLUMN_EVIDENCE_GAP')
    first_two = columns[:2]
    left_hole_support = sum((1 for column in first_two for evidence in column['rows'].values() if 'hole' in evidence['kind']))
    phase_confirmed = len(first_two) == 2 and left_hole_support >= 3
    if not phase_confirmed:
        qc_flags.append('CAM0_LEFT_PHASE_NOT_CONFIRMED')
    (bed_bbox, bed_measurement) = measure_cam0_bed_bbox(image, columns, pitch)
    if bed_bbox is None:
        return ([], {'status': 'CAM0_BED_RANGE_REVIEW', 'method': 'count_holes_and_lettuce_seg_then_measure_bed_then_split', 'n_visible': n_visible, 'raw_observed_column_count': raw_observed_count, 'duplicate_merge_count': duplicate_merge_count, 'pitch': pitch, 'hole_count': sum((len(v) for v in holes.values())), 'lettuce_peak_count': sum((len(v) for v in lettuces.values())), 'lettuce_segment_count': len(seg_objects or []), 'qc_flags': qc_flags + [bed_measurement['reason']], 'hole_evidence': holes, 'lettuce_evidence': lettuces, 'observed_column_x': [float(column['x']) for column in columns], 'bed_measurement': bed_measurement})
    (bed_x1, bed_y1, bed_x2, bed_y2) = bed_bbox
    cell_pitch = (bed_x2 - bed_x1) / n_visible
    template_x = [bed_x1 + (idx + 0.5) * cell_pitch for idx in range(n_visible)]
    top_y_fn = _fit_row_y(columns, 't', 0.28, h)
    bottom_y_fn = _fit_row_y(columns, 'b', 0.72, h)
    top_points = [(x, top_y_fn(x)) for x in template_x]
    bottom_points = [(x, bottom_y_fn(x)) for x in template_x]
    physical_cols = list(range(1, n_visible + 1)) if phase_confirmed else [None] * n_visible
    confirmed_cols = set(physical_cols) if phase_confirmed and (not qc_flags) else set()
    visibility_by_slot: Dict[Tuple[str, int], Visibility] = {}
    evidence_by_slot: Dict[Tuple[str, int], Tuple[str, float]] = {}
    for (local_col, column) in enumerate(columns, start=1):
        for row in ('t', 'b'):
            evidence = column['rows'].get(row)
            visibility_by_slot[row, local_col] = _visibility_from_evidence(evidence)
            if evidence is not None:
                evidence_by_slot[row, local_col] = (evidence['kind'], float(evidence['score']))
    out_of_fov = range(n_visible + 1, N_PHYSICAL_COLS + 1) if phase_confirmed else ()
    slots = build_slots_from_columns(top_points, bottom_points, physical_cols, visibility_by_slot=visibility_by_slot, confirmed_cols=confirmed_cols, evidence_by_slot=evidence_by_slot, out_of_fov_cols=out_of_fov, legacy_crosswalk=False)
    for slot in slots:
        if slot.local_col <= 0:
            continue
        evidence = columns[slot.local_col - 1]['rows'].get(slot.row)
        if evidence is None:
            continue
        apply_bounded_residual(slot, evidence['x'], evidence['y'], pitch=cell_pitch, max_shift_ratio=MAX_SHIFT_RATIO, reject_distance_ratio=EVIDENCE_REJECT_RATIO)
    build_cell_bboxes(slots, w, h, bed_bbox=bed_bbox)
    if not phase_confirmed:
        status = 'CAM0_PHASE_REVIEW'
    elif qc_flags:
        status = 'CAM0_SLOT_COUNT_REVIEW'
    else:
        status = 'CAM0_SLOT_MAP_OK'
    return (slots, {'status': status, 'method': 'count_holes_and_lettuce_seg_then_measure_bed_then_split', 'n_visible': n_visible, 'raw_observed_column_count': raw_observed_count, 'duplicate_merge_count': duplicate_merge_count, 'pitch': float(cell_pitch), 'bed_bbox': (int(bed_x1), int(bed_y1), int(bed_x2), int(bed_y2)), 'bed_measurement': bed_measurement, 'phase_origin_col': 1 if phase_confirmed else None, 'left_hole_support': left_hole_support, 'hole_count': sum((len(v) for v in holes.values())), 'lettuce_peak_count': sum((len(v) for v in lettuces.values())), 'lettuce_segment_count': len(seg_objects or []), 'observed_column_x': [float(column['x']) for column in columns], 'template_column_x': [float(x) for x in template_x], 'hole_evidence': holes, 'lettuce_evidence': lettuces, 'green_mask_nonzero': int(np.count_nonzero(green_mask)), 'qc_flags': qc_flags})
