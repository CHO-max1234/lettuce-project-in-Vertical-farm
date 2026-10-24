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
_core = importlib.import_module('slot_geometry')
_evidence = importlib.import_module('cam0_geometry')
_lettuce_seg = importlib.import_module('plant_segmentation')
Visibility = _core.Visibility
build_slots_from_columns = _core.build_slots_from_columns
apply_bounded_residual = _core.apply_bounded_residual
build_cell_bboxes = _core.build_cell_bboxes
CAM1_END_COL = 10
CAM1_VISIBLE_END_WITH_TOWEL = 8
CAM1_MIN_EVIDENCE_COLS = 3
CAM1_MAX_EVIDENCE_COLS = 10
ROW_DEDUP_RATIO = 0.4
COLUMN_MERGE_RATIO = 0.45
GAP_REVIEW_RATIO = 1.7
CAM1_ROW_SPLIT_RATIO = 0.45
CAM1_TARGET_Y_MIN_RATIO = 0.055
CAM1_TARGET_Y_MAX_RATIO = 0.81
CAM1_ROW_MODEL_Y_MIN_RATIO = 0.015
CAM1_ROW_MODEL_Y_MAX_RATIO = 0.82
CAM1_ROW_GAP_MIN_RATIO = 0.16
CAM1_ROW_GAP_MAX_RATIO = 0.58
CAM1_ROW_CORRIDOR_MARGIN = 0.62
CAM1_TOP_OUTER_CORRIDOR_MARGIN = 0.3
CAM1_BOTTOM_OUTER_CORRIDOR_MARGIN = 0.55
CAM1_SEG_BAND_OVERLAP_MIN = 0.5
CAM1_ROW_MODE_SEPARATION_RATIO = 0.105
MISSING_GAP_MIN_RATIO = 1.55
RIGHT_ANCHOR_MISSING_MIN_RATIO = 1.05
ANCHOR_LEFT_CENTER_MIN_PITCH = -0.1
ANCHOR_MATCH_MAX_PITCH = 0.52
T8_RESCUE_GAP_MIN_PITCH = 0.58
T8_RESCUE_GAP_MAX_PITCH = 1.42
T8_RESCUE_MIN_ENDPOINT_CLEARANCE_PITCH = 1.35
T8_RESCUE_MIN_ANCHOR_SHIFT_PITCH = 0.34

def detect_right_occlusion(image: np.ndarray) -> dict:
    (h, w) = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (hue, saturation, value) = cv2.split(hsv)
    green_dark = (hue >= 30) & (hue <= 95) & (saturation >= 35) & (value < 210)
    profile = green_dark[int(0.12 * h):int(0.72 * h)].mean(axis=0)
    binary = profile > 0.58
    binary[:int(0.55 * w)] = False
    binary[int(0.95 * w):] = False
    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    if not runs:
        return {'present': False, 'start_x': None, 'end_x': None, 'run_ratio': 0.0}
    (start, end) = max(runs, key=lambda run: run[1] - run[0])
    run_ratio = (end - start) / max(w, 1)
    panel_roi = value[int(0.15 * h):int(0.7 * h), int(0.6 * w):int(0.88 * w)]
    dark_panel_ratio = float(np.mean(panel_roi < 62)) if panel_roi.size else 0.0
    present = run_ratio > 0.11 or dark_panel_ratio > 0.26
    if present and run_ratio <= 0.11:
        (start, end) = (int(0.62 * w), int(0.86 * w))
    return {'present': bool(present), 'start_x': int(start) if present else None, 'end_x': int(end) if present else None, 'run_ratio': float(run_ratio), 'dark_panel_ratio': dark_panel_ratio}

def measure_right_endpoint(image: np.ndarray) -> Tuple[int, float]:
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    profile = edge[int(0.14 * h):int(0.78 * h)].mean(axis=0)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (21, 1), 0).ravel()
    (start, end) = (int(0.84 * w), int(0.98 * w))
    positions = np.arange(start, end, dtype=np.float32)
    values = profile[start:end]
    scale = max(float(values.max()), 1e-06)
    expected = 0.92 * w
    score = values / scale - 0.38 * np.abs(positions - expected) / max(end - start, 1)
    index = int(np.argmax(score))
    return (start + index, float(values[index]))

def _fit_two_row_centers(segments: List[dict], image_h: int) -> Tuple[float, float]:
    ys = np.asarray([float(segment['cy']) for segment in segments if CAM1_ROW_MODEL_Y_MIN_RATIO * image_h <= float(segment['cy']) <= CAM1_ROW_MODEL_Y_MAX_RATIO * image_h], dtype=np.float64)
    if len(ys) < 4:
        return (0.27 * image_h, 0.61 * image_h)
    ordered = np.sort(ys)
    mode_groups: List[List[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if float(value) - mode_groups[-1][-1] > CAM1_ROW_MODE_SEPARATION_RATIO * image_h:
            mode_groups.append([float(value)])
        else:
            mode_groups[-1].append(float(value))
    if len(mode_groups) >= 2:
        modes = [(float(np.median(group)), len(group), float(np.median(np.abs(np.asarray(group, dtype=np.float64) - np.median(group))))) for group in mode_groups]
        pairs = []
        for index in range(len(modes) - 1):
            (top_center, top_count, top_mad) = modes[index]
            (bottom_center, bottom_count, bottom_mad) = modes[index + 1]
            gap = bottom_center - top_center
            if CAM1_ROW_GAP_MIN_RATIO * image_h <= gap <= CAM1_ROW_GAP_MAX_RATIO * image_h:
                support = top_count + bottom_count
                balance = min(top_count, bottom_count)
                compactness = (top_mad + bottom_mad) / max(image_h, 1)
                score = 10.0 * support + 2.0 * balance - compactness
                pairs.append((score, top_center, bottom_center))
        if pairs:
            (_, top_center, bottom_center) = max(pairs)
            return (top_center, bottom_center)
    candidates = []
    for split_index in range(2, len(ordered) - 1):
        upper = ordered[:split_index]
        lower = ordered[split_index:]
        gap = float(lower[0] - upper[-1])
        top_center = float(np.median(upper))
        bottom_center = float(np.median(lower))
        center_gap = bottom_center - top_center
        if CAM1_ROW_GAP_MIN_RATIO * image_h <= center_gap <= CAM1_ROW_GAP_MAX_RATIO * image_h:
            upper_mad = float(np.median(np.abs(upper - top_center)))
            lower_mad = float(np.median(np.abs(lower - bottom_center)))
            score = gap / max(8.0, upper_mad + lower_mad)
            candidates.append((score, gap, top_center, bottom_center))
    if candidates:
        (_, _, top_center, bottom_center) = max(candidates)
        return (top_center, bottom_center)
    centers = np.percentile(ys, [28, 72]).astype(np.float64)
    for _ in range(16):
        labels = np.argmin(np.abs(ys[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in (0, 1):
            group = ys[labels == index]
            if len(group):
                updated[index] = float(np.median(group))
        updated.sort()
        if np.allclose(updated, centers, atol=0.5):
            centers = updated
            break
        centers = updated
    (top, bottom) = (float(centers[0]), float(centers[1]))
    gap = bottom - top
    if not CAM1_ROW_GAP_MIN_RATIO * image_h <= gap <= CAM1_ROW_GAP_MAX_RATIO * image_h:
        return (0.27 * image_h, 0.61 * image_h)
    return (top, bottom)

def _polygon_band_overlap(segment: dict, y_min: float, y_max: float) -> float:
    polygon = np.asarray(segment['polygon'], dtype=np.float32)
    (x1, y1, x2, y2) = (int(round(value)) for value in segment['bbox'])
    width = max(1, x2 - x1 + 1)
    height = max(1, y2 - y1 + 1)
    local = np.round(polygon - np.asarray([x1, y1], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [local.reshape(-1, 1, 2)], 1)
    total = int(mask.sum())
    if total <= 0:
        return 0.0
    band_y1 = int(np.clip(np.floor(y_min) - y1, 0, height))
    band_y2 = int(np.clip(np.ceil(y_max) - y1, 0, height))
    inside = int(mask[band_y1:band_y2].sum()) if band_y2 > band_y1 else 0
    return float(inside / total)

def _row_model_from_segments(segments: List[dict], image_h: int) -> Tuple[float, float, float, float]:
    (top, bottom) = _fit_two_row_centers(segments, image_h)
    gap = max(20.0, bottom - top)
    y_min = max(0.0, top - CAM1_TOP_OUTER_CORRIDOR_MARGIN * gap)
    y_max = min(float(image_h), bottom + CAM1_BOTTOM_OUTER_CORRIDOR_MARGIN * gap)
    return (top, bottom, y_min, y_max)

def filter_cam1_segments(image: np.ndarray, segments: List[dict]) -> Tuple[List[dict], List[dict]]:
    (h, w) = image.shape[:2]
    (top_center, bottom_center, y_min, y_max) = _row_model_from_segments(segments, h)
    row_gap = max(20.0, bottom_center - top_center)
    accepted: List[dict] = []
    rejected: List[dict] = []
    for segment in segments:
        cy = float(segment['cy'])
        row = 't' if abs(cy - top_center) <= abs(cy - bottom_center) else 'b'
        row_center = top_center if row == 't' else bottom_center
        close_to_row = abs(cy - row_center) <= CAM1_ROW_CORRIDOR_MARGIN * row_gap
        band_overlap = _polygon_band_overlap(segment, y_min, y_max)
        inside_x = 0.005 * w < float(segment['cx']) < 0.985 * w
        keep = y_min <= cy <= y_max and close_to_row and inside_x and (band_overlap >= CAM1_SEG_BAND_OVERLAP_MIN)
        target = accepted if keep else rejected
        copied = dict(segment)
        copied['row_hint'] = row
        copied['row_center_y'] = float(row_center)
        copied['bed_band_overlap'] = float(band_overlap)
        target.append(copied)
    return (accepted, rejected)

def _filter_and_reclassify_evidence(evidence_by_row: Dict[str, List[dict]], image_h: int, row_centers: Optional[Tuple[float, float]]=None) -> Tuple[Dict[str, List[dict]], int]:
    result: Dict[str, List[dict]] = {'t': [], 'b': []}
    rejected = 0
    (top_center, bottom_center) = row_centers or (0.27 * image_h, 0.61 * image_h)
    gap = max(20.0, bottom_center - top_center)
    y_min = top_center - CAM1_ROW_CORRIDOR_MARGIN * gap
    y_max = bottom_center + CAM1_ROW_CORRIDOR_MARGIN * gap
    for items in evidence_by_row.values():
        for item in items:
            y = float(item['y'])
            if not y_min <= y <= y_max:
                rejected += 1
                continue
            row = 't' if abs(y - top_center) <= abs(y - bottom_center) else 'b'
            result[row].append(dict(item))
    return (result, rejected)

def _segments_to_target_evidence(segments: List[dict], image_h: int) -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {'t': [], 'b': []}
    for segment in segments:
        row = segment.get('row_hint')
        if row not in result:
            row = 't' if float(segment['cy']) < 0.5 * image_h else 'b'
        result[row].append({'x': float(segment['cx']), 'y': float(segment['cy']), 'score': 100.0 * float(segment['conf']), 'kind': 'lettuce_seg', 'seg_id': int(segment['seg_id']), 'bbox': tuple(segment['bbox'])})
    return result

def _merge_row(items: List[dict], pitch: float) -> List[dict]:
    if not items:
        return []
    groups: List[List[dict]] = []
    tolerance = ROW_DEDUP_RATIO * pitch
    for item in sorted(items, key=lambda value: value['x']):
        center = np.mean([value['x'] for value in groups[-1]]) if groups else None
        if center is None or abs(item['x'] - center) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    merged = []
    for group in groups:
        kinds = {item['kind'] for item in group}
        kind = 'hole+lettuce' if len(kinds) > 1 else next(iter(kinds))
        merged.append(_evidence._weighted_merge(group, kind))
    return merged

def _merge_columns(rows: Dict[str, List[dict]], pitch: float) -> List[dict]:
    tagged = [{**item, 'row': row} for row in ('t', 'b') for item in rows[row]]
    if not tagged:
        return []
    groups: List[List[dict]] = []
    tolerance = COLUMN_MERGE_RATIO * pitch
    for item in sorted(tagged, key=lambda value: value['x']):
        center = np.mean([value['x'] for value in groups[-1]]) if groups else None
        if center is None or abs(item['x'] - center) > tolerance:
            groups.append([item])
        else:
            groups[-1].append(item)
    columns = []
    for (local_col, group) in enumerate(groups, start=1):
        by_row = {}
        for row in ('t', 'b'):
            row_items = [item for item in group if item['row'] == row]
            if row_items:
                kinds = {item['kind'] for item in row_items}
                kind = 'hole+lettuce' if len(kinds) > 1 else next(iter(kinds))
                by_row[row] = _evidence._weighted_merge(row_items, kind)
        columns.append({'local_col': local_col, 'x': float(np.mean([item['x'] for item in group])), 'rows': by_row})
    return columns

def _select_t8_rescue_column(*, base_columns: List[dict], rescue_lettuces: Dict[str, List[dict]], pitch: float, occlusion_start_x: float, endpoint_x: float, old_right_anchor_x: float) -> Optional[dict]:
    if not base_columns or not np.isfinite(pitch) or pitch <= 1.0:
        return None
    rescue_rows = {row: _merge_row(items, pitch) for (row, items) in rescue_lettuces.items()}
    rescue_columns = _merge_columns(rescue_rows, pitch)
    if not rescue_columns:
        return None
    base_last_x = float(max(base_columns, key=lambda value: value['x'])['x'])
    valid: List[dict] = []
    for column in rescue_columns:
        candidate_x = float(column['x'])
        gap_pitch = (candidate_x - base_last_x) / pitch
        anchor_shift_pitch = (candidate_x - old_right_anchor_x) / pitch
        endpoint_clearance_pitch = (endpoint_x - candidate_x) / pitch
        lettuce_rows = sum(('lettuce' in str(item.get('kind', '')) for item in column.get('rows', {}).values()))
        if candidate_x >= occlusion_start_x and T8_RESCUE_GAP_MIN_PITCH <= gap_pitch <= T8_RESCUE_GAP_MAX_PITCH and (anchor_shift_pitch >= T8_RESCUE_MIN_ANCHOR_SHIFT_PITCH) and (endpoint_clearance_pitch >= T8_RESCUE_MIN_ENDPOINT_CLEARANCE_PITCH) and (lettuce_rows >= 1):
            copied = dict(column)
            copied['rescue_gap_pitch'] = float(gap_pitch)
            copied['endpoint_clearance_pitch'] = float(endpoint_clearance_pitch)
            copied['lettuce_row_support'] = int(lettuce_rows)
            valid.append(copied)
    return valid[0] if len(valid) == 1 else None

def _visibility(evidence: Optional[dict]) -> Visibility:
    if evidence is None:
        return Visibility.UNKNOWN
    return Visibility.VISIBLE if 'lettuce' in evidence['kind'] else Visibility.EMPTY

def _build_physical_anchor_grid(*, pitch: float, right_anchor_x: float, visible_end_col: int, left_edge_hole_supported: bool, image_w: int) -> Tuple[List[int], List[float], int]:
    visible_centers = {physical_col: float(right_anchor_x - (visible_end_col - physical_col) * pitch) for physical_col in range(1, visible_end_col + 1)}
    visible_cols = [physical_col for (physical_col, center_x) in visible_centers.items() if center_x >= ANCHOR_LEFT_CENTER_MIN_PITCH * pitch and center_x <= image_w + 0.25 * pitch]
    if not visible_cols:
        visible_cols = [visible_end_col]
    start_col = min(visible_cols)
    if left_edge_hole_supported and start_col > 1:
        rescued_col = start_col - 1
        if visible_centers[rescued_col] >= -0.35 * pitch:
            start_col = rescued_col
    physical_cols = list(range(start_col, CAM1_END_COL + 1))
    template_x = [float(right_anchor_x + (physical_col - visible_end_col) * pitch) for physical_col in physical_cols]
    return (physical_cols, template_x, start_col)

def _match_columns_to_anchor_grid(columns: List[dict], physical_cols: List[int], template_x: List[float], visible_end_col: int, pitch: float) -> Tuple[Dict[int, Dict[str, dict]], int, int]:
    visible_indices = [index for (index, physical_col) in enumerate(physical_cols) if physical_col <= visible_end_col]
    matched: Dict[int, Dict[str, dict]] = {}
    unmatched = 0
    for column in columns:
        if not visible_indices:
            unmatched += 1
            continue
        grid_index = min(visible_indices, key=lambda index: abs(float(column['x']) - template_x[index]))
        distance = abs(float(column['x']) - template_x[grid_index])
        if distance > ANCHOR_MATCH_MAX_PITCH * pitch:
            unmatched += 1
            continue
        local_col = grid_index + 1
        target = matched.setdefault(local_col, {})
        for (row, item) in column['rows'].items():
            previous = target.get(row)
            if previous is None:
                target[row] = item
                continue
            previous_distance = abs(float(previous['x']) - template_x[grid_index])
            current_key = (distance, -float(item.get('score', 0.0)))
            previous_key = (previous_distance, -float(previous.get('score', 0.0)))
            if current_key < previous_key:
                target[row] = item
    inferred = sum((index + 1 not in matched for index in visible_indices))
    return (matched, unmatched, inferred)

def map_cam1_v3(image: np.ndarray, seg_objects: Optional[List[dict]]=None):
    (h, w) = image.shape[:2]
    occlusion = detect_right_occlusion(image)
    (endpoint_x, endpoint_strength) = measure_right_endpoint(image)
    if seg_objects:
        (adaptive_top, adaptive_bottom, _, _) = _row_model_from_segments(seg_objects, h)
        row_centers = (adaptive_top, adaptive_bottom)
    else:
        row_centers = None
    holes_raw = _evidence.detect_hole_evidence(image)
    (holes, rejected_hole_count) = _filter_and_reclassify_evidence(holes_raw, h, row_centers=row_centers)
    if seg_objects is None:
        (lettuces_raw, green_mask) = _evidence.detect_lettuce_evidence(image)
        (lettuces, rejected_lettuce_count) = _filter_and_reclassify_evidence(lettuces_raw, h, row_centers=row_centers)
        used_lettuce_seg = False
    else:
        lettuces = _segments_to_target_evidence(seg_objects, h)
        rejected_lettuce_count = 0
        green_mask = _evidence._green_mask(image)
        used_lettuce_seg = True
    evidence_limit = float(occlusion['start_x']) if occlusion['present'] else float(endpoint_x)
    rescue_lettuces: Dict[str, List[dict]] = {'t': [], 'b': []}
    for row in ('t', 'b'):
        holes[row] = [item for item in holes[row] if 0.0 <= item['x'] < evidence_limit]
        if occlusion['present']:
            rescue_lettuces[row] = [item for item in lettuces[row] if evidence_limit <= item['x'] < float(endpoint_x)]
        lettuces[row] = [item for item in lettuces[row] if 0.01 * w < item['x'] < evidence_limit]
    combined = {row: holes[row] + lettuces[row] for row in ('t', 'b')}
    pitch_seed = _evidence._estimate_pitch(combined, w)
    rows = {row: _merge_row(combined[row], pitch_seed) for row in ('t', 'b')}
    pitch = _evidence._estimate_pitch(rows, w)
    columns = _merge_columns(rows, pitch)
    old_right_anchor_x = evidence_limit - 0.5 * pitch
    t8_rescue = None
    if occlusion['present']:
        t8_rescue = _select_t8_rescue_column(base_columns=columns, rescue_lettuces=rescue_lettuces, pitch=pitch, occlusion_start_x=evidence_limit, endpoint_x=float(endpoint_x), old_right_anchor_x=old_right_anchor_x)
    if t8_rescue is not None:
        columns = sorted(columns + [t8_rescue], key=lambda value: value['x'])
        for (local_col, column) in enumerate(columns, start=1):
            column['local_col'] = local_col
        for row in ('t', 'b'):
            lettuces[row].extend(rescue_lettuces[row])
        right_anchor_x = float(t8_rescue['x'])
    else:
        right_anchor_x = float(old_right_anchor_x)
    raw_evidence_column_count = len(columns)
    qc_flags: List[str] = []
    if not used_lettuce_seg:
        qc_flags.append('LETTUCE_SEG_NOT_PROVIDED')
    if rejected_hole_count or rejected_lettuce_count:
        qc_flags.append('OTHER_BED_EVIDENCE_REJECTED')
    if t8_rescue is not None:
        qc_flags.append('RIGHT_T8_EVIDENCE_RESCUED')
    visible_end_col = CAM1_VISIBLE_END_WITH_TOWEL if occlusion['present'] else CAM1_END_COL
    (physical_cols, template_x, start_col) = _build_physical_anchor_grid(pitch=pitch, right_anchor_x=right_anchor_x, visible_end_col=visible_end_col, left_edge_hole_supported=any((0.0 <= float(item['x']) <= 0.28 * pitch for row in ('t', 'b') for item in holes[row])), image_w=w)
    (matched_columns, unmatched_column_count, inferred_column_count) = _match_columns_to_anchor_grid(columns, physical_cols, template_x, visible_end_col, pitch)
    visible_local_count = sum((col <= visible_end_col for col in physical_cols))
    total_local_cols = len(physical_cols)
    n_evidence = len(matched_columns)
    if inferred_column_count:
        qc_flags.append('MISSING_COLUMN_GEOMETRY_INFERRED')
    if unmatched_column_count:
        qc_flags.append('OUTLIER_COLUMN_EVIDENCE_REJECTED')
    if not CAM1_MIN_EVIDENCE_COLS <= visible_local_count <= CAM1_MAX_EVIDENCE_COLS or start_col < 1 or total_local_cols > CAM1_END_COL:
        return ([], {'status': 'CAM1_SLOT_COUNT_REVIEW', 'n_visible_evidence': n_evidence, 'n_visible_physical': visible_local_count, 'raw_evidence_column_count': raw_evidence_column_count, 'n_total_local': total_local_cols, 'towel_present': occlusion['present'], 'occlusion': occlusion, 'hole_count': sum((len(values) for values in holes.values())), 'lettuce_peak_count': sum((len(values) for values in lettuces.values())), 'lettuce_segment_count': len(seg_objects or []), 'rejected_hole_count': rejected_hole_count, 'rejected_lettuce_count': rejected_lettuce_count, 'inferred_column_count': inferred_column_count, 'hole_evidence': holes, 'lettuce_evidence': lettuces, 'qc_flags': ['OBSERVED_COLUMN_COUNT_OUT_OF_RANGE']})
    top_y_fn = _evidence._fit_row_y(columns, 't', 0.27, h)
    bottom_y_fn = _evidence._fit_row_y(columns, 'b', 0.61, h)
    bed_x1 = int(round(max(0.0, template_x[0] - 0.5 * pitch)))
    bed_x2 = int(round(min(float(w), float(endpoint_x))))
    cell_pitch = float(pitch)
    top_points = [(x, top_y_fn(x)) for x in template_x]
    bottom_points = [(x, bottom_y_fn(x)) for x in template_x]
    top_observed = [column['rows']['t']['y'] for column in columns if 't' in column['rows']]
    bottom_observed = [column['rows']['b']['y'] for column in columns if 'b' in column['rows']]
    top_center = float(np.median(top_observed)) if top_observed else 0.27 * h
    bottom_center = float(np.median(bottom_observed)) if bottom_observed else 0.61 * h
    row_gap = max(20.0, bottom_center - top_center)
    bed_y1 = int(round(np.clip(top_center - 0.55 * row_gap, 0, h)))
    bed_y2 = int(round(np.clip(bottom_center + 0.55 * row_gap, 0, h)))
    bed_bbox = (bed_x1, bed_y1, bed_x2, bed_y2)
    visibility_by_slot = {}
    evidence_by_slot = {}
    for (local_col, physical_col) in enumerate(physical_cols, start=1):
        for row in ('t', 'b'):
            if physical_col > visible_end_col:
                visibility_by_slot[row, local_col] = Visibility.OCCLUDED
                evidence_by_slot[row, local_col] = ('towel_or_panel', 1.0)
                continue
            item = matched_columns.get(local_col, {}).get(row)
            visibility_by_slot[row, local_col] = _visibility(item)
            if item is not None:
                evidence_by_slot[row, local_col] = (item['kind'], float(item['score']))
    confirmed_cols = set(physical_cols) if not qc_flags else set()
    slots = build_slots_from_columns(top_points, bottom_points, physical_cols, visibility_by_slot=visibility_by_slot, confirmed_cols=confirmed_cols, evidence_by_slot=evidence_by_slot, out_of_fov_cols=range(1, start_col), legacy_crosswalk=False)
    for slot in slots:
        if slot.local_col <= 0 or slot.physical_col is None:
            continue
        if slot.physical_col > visible_end_col:
            continue
        item = matched_columns.get(slot.local_col, {}).get(slot.row)
        if item is not None:
            apply_bounded_residual(slot, item['x'], item['y'], pitch=cell_pitch, max_shift_ratio=0.24, reject_distance_ratio=0.62)
    build_cell_bboxes(slots, w, h, bed_bbox=bed_bbox)
    blocking_flags = {'LETTUCE_SEG_NOT_PROVIDED', 'OBSERVED_COLUMN_COUNT_OUT_OF_RANGE', 'MISSING_COLUMN_EVIDENCE_GAP'}
    status = 'CAM1_SLOT_MAP_OK' if not blocking_flags.intersection(qc_flags) else 'CAM1_SLOT_COUNT_REVIEW'
    return (slots, {'status': status, 'method': 'right_anchor_physical_grid_then_match_evidence_keep_9_10_occluded', 'n_visible_evidence': n_evidence, 'n_visible_physical': visible_local_count, 'raw_evidence_column_count': raw_evidence_column_count, 'n_total_local': total_local_cols, 'start_col': start_col, 'end_col': CAM1_END_COL, 'pitch': float(cell_pitch), 'bed_bbox': bed_bbox, 'towel_present': occlusion['present'], 'occlusion': occlusion, 'endpoint_strength': endpoint_strength, 'hole_count': sum((len(values) for values in holes.values())), 'lettuce_peak_count': sum((len(values) for values in lettuces.values())), 'lettuce_segment_count': len(seg_objects or []), 'rejected_hole_count': rejected_hole_count, 'rejected_lettuce_count': rejected_lettuce_count, 'inferred_column_count': inferred_column_count, 'unmatched_column_count': unmatched_column_count, 'right_anchor_x': float(right_anchor_x), 'right_anchor_source': 't8_rescue' if t8_rescue is not None else 'ver3.3', 't8_rescue_gap_pitch': float(t8_rescue['rescue_gap_pitch']) if t8_rescue is not None else None, 't8_rescue_endpoint_clearance_pitch': float(t8_rescue['endpoint_clearance_pitch']) if t8_rescue is not None else None, 't8_rescue_row_support': int(t8_rescue['lettuce_row_support']) if t8_rescue is not None else 0, 'adaptive_top_row_y': float(row_centers[0]) if row_centers else None, 'adaptive_bottom_row_y': float(row_centers[1]) if row_centers else None, 'observed_column_x': [float(column['x']) for column in columns], 'template_column_x': [float(x) for x in template_x], 'hole_evidence': holes, 'lettuce_evidence': lettuces, 'green_mask_nonzero': int(np.count_nonzero(green_mask)), 'qc_flags': qc_flags})
