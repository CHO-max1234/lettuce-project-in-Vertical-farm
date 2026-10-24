"""Front-view geometry used by the final preprocessing adapters."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[3]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / 'slot_geometry.py'
SOURCE_ROOT = CYCLE.PREPROCESS_OUTPUT / '1. bed-seg' / 'images' / 'cam2'
OUTPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '6. bbox'
SAMPLE_RELATIVE_PATHS = (Path('260308') / 'bed08_20260308_104022_cam2_rf.jpg', Path('260403') / 'bed45_20260403_125348_cam2_rf.jpg', Path('260504') / 'bed55_20260504_023605_cam2_rf.jpg')
RAIL_ROI_MAX_X = 0.18
RAIL_MIN_LENGTH_H = 0.075
RAIL_SLOPE_MIN = -1.15
RAIL_SLOPE_MAX = -0.34
RAIL_TOP_X_RANGE = (-0.02, 0.19)
RAIL_BOTTOM_X_RANGE = (-0.16, 0.1)
TOP_ROW_EXPECTED_Y = 0.245
BOTTOM_ROW_EXPECTED_Y = 0.635
ROW_EVIDENCE_GATE_H = 0.18
ROW_SMOOTH_LIMIT_H = 0.055
PITCH_MIN_W = 0.082
PITCH_MAX_W = 0.116
PITCH_STEP_PX = 0.5
RAIL_PHASE_SHIFT_LIMIT_PITCH = 0.24
VISIBLE_CELL_MIN_IMAGE_RATIO = 0.52
MAX_PREVIEW_COLUMNS = 20
SEG_MIN_MASK_COVERAGE = 0.18
SEG_MAX_CENTER_DISTANCE = 0.94
SEG_MIN_ASSIGN_SCORE = 0.18
SEG_MERGE_RUNNERUP_COVERAGE = 0.18
SEG_MERGE_MARGIN = 0.12
BBOX_PAD_RATIO = 0.06
PALETTE = ((40, 40, 255), (255, 220, 0), (0, 170, 255), (255, 80, 0), (0, 255, 80), (255, 0, 230), (60, 210, 255), (200, 80, 255), (0, 230, 230), (120, 255, 80), (255, 140, 80), (80, 140, 255))

def load_base_module():
    module_name = 'cam2_slot_geometry'
    spec = importlib.util.spec_from_file_location(module_name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Invalid input or state in load_base_module (rejected condition: spec is None or spec.loader is None): {BASE_SCRIPT} ')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
BASE = load_base_module()

@dataclass
class RailLine:
    a: float
    b: float
    source: str
    score: float
    segment: tuple[int, int, int, int]

    def x_at(self, y: float) -> float:
        return float(self.a * y + self.b)

@dataclass
class Evidence:
    x: float
    y: float
    kind: str
    weight: float

@dataclass
class RailSlot:
    row: str
    local_col: int
    slot_name: str
    x_prior: float
    y_prior: float
    cell_polygon: np.ndarray
    cell_inside_image_ratio: float
    status: str = 'GEOMETRY_PRIOR'
    visibility: str = 'UNKNOWN'
    seg_id: Optional[int] = None
    seg_conf: float = float('nan')
    seg_center_x: float = float('nan')
    seg_center_y: float = float('nan')
    seg_area: float = float('nan')
    seg_bbox: Optional[tuple[float, float, float, float]] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    mask_coverage: float = float('nan')
    cell_coverage: float = float('nan')
    runner_up_coverage: float = float('nan')
    center_distance: float = float('nan')
    nearest_hole_distance: float = float('nan')

def all_image_holes(image: np.ndarray) -> list[object]:
    """Detect dark circular holes over the whole crop, not only the left side."""
    (height, width) = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] < BASE.DARK_VALUE_THRESHOLD) & (hsv[:, :, 1] < BASE.HOLE_MAX_MEAN_SATURATION)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(120.0, 0.00048 * height * width)
    max_area = 0.018 * height * width
    found: list[object] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.43:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        if not 0.42 <= w / max(h, 1) <= 2.4:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            continue
        cx = float(moments['m10'] / moments['m00'])
        cy = float(moments['m01'] / moments['m00'])
        if cx < 0.015 * width and (cy < 0.12 * height or cy > 0.88 * height):
            continue
        found.append(BASE.HoleEvidence(cx, cy, area, float(circularity), float(hsv[int(round(cy)), int(round(cx)), 1]), float(hsv[int(round(cy)), int(round(cx)), 2])))
    return sorted(found, key=lambda item: (item.y, item.x))

def evidence_from(holes: Sequence[object], segments: Sequence[dict]) -> list[Evidence]:
    result = [Evidence(float(h.x), float(h.y), 'hole', 1.45) for h in holes]
    result.extend((Evidence(float(s['cx']), float(s['cy']), 'seg', 0.85 + 0.25 * float(s['conf'])) for s in segments))
    return result

def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    vals = np.asarray(values, dtype=np.float64)[order]
    wts = np.asarray(weights, dtype=np.float64)[order]
    cutoff = 0.5 * float(wts.sum())
    return float(vals[min(len(vals) - 1, int(np.searchsorted(np.cumsum(wts), cutoff)))])

def estimate_rows(evidence: Sequence[Evidence], height: int) -> tuple[float, float, dict]:
    expected = {'t': TOP_ROW_EXPECTED_Y * height, 'b': BOTTOM_ROW_EXPECTED_Y * height}
    rows: dict[str, list[Evidence]] = {'t': [], 'b': []}
    for item in evidence:
        distances = {row: abs(item.y - y0) for (row, y0) in expected.items()}
        row = min(distances, key=distances.get)
        if distances[row] <= ROW_EVIDENCE_GATE_H * height:
            rows[row].append(item)
    fitted: dict[str, float] = {}
    for row in ('t', 'b'):
        items = rows[row]
        if items:
            observed = weighted_median([item.y for item in items], [item.weight for item in items])
            limit = ROW_SMOOTH_LIMIT_H * height
            fitted[row] = float(np.clip(observed, expected[row] - limit, expected[row] + limit))
        else:
            fitted[row] = float(expected[row])
    if fitted['b'] - fitted['t'] < 0.25 * height:
        fitted = expected.copy()
    return (fitted['t'], fitted['b'], {'top_evidence': len(rows['t']), 'bottom_evidence': len(rows['b'])})

def row_evidence(evidence: Sequence[Evidence], row_y: float, height: int) -> list[Evidence]:
    gate = 0.16 * height
    return [item for item in evidence if abs(item.y - row_y) <= gate]

def fit_pitch_and_shift(items: Sequence[Evidence], origin_x: float, width: int) -> tuple[float, float, float, int]:
    """Fit pitch and a bounded rail-to-first-column phase residual.

    The bounded shift is diagnostic/refinement; it cannot jump by one pitch.
    """
    if not items:
        return (0.0985 * width, 0.0, 9.0, 0)
    min_pitch = PITCH_MIN_W * width
    max_pitch = PITCH_MAX_W * width
    best: Optional[tuple[float, float, float, int, float]] = None
    for pitch in np.arange(min_pitch, max_pitch + PITCH_STEP_PX, PITCH_STEP_PX):
        for shift_ratio in np.linspace(-RAIL_PHASE_SHIFT_LIMIT_PITCH, RAIL_PHASE_SHIFT_LIMIT_PITCH, 25):
            shifted = origin_x + shift_ratio * pitch
            residuals: list[float] = []
            weights: list[float] = []
            supported = 0
            for item in items:
                k = int(round((item.x - shifted) / pitch))
                if k < 0 or k > MAX_PREVIEW_COLUMNS:
                    continue
                residual = abs(item.x - (shifted + k * pitch)) / pitch
                residuals.append(min(residual, 0.6))
                weights.append(item.weight)
                supported += int(residual <= 0.33)
            if not residuals:
                continue
            residual_array = np.asarray(residuals)
            weight_array = np.asarray(weights)
            mean_residual = float(np.average(residual_array, weights=weight_array))
            median_residual = weighted_median(residuals, weights)
            coverage = supported / max(1, len(items))
            score = median_residual + 0.45 * mean_residual + 0.22 * (1.0 - coverage) + 0.035 * abs(shift_ratio)
            candidate = (score, float(pitch), float(shift_ratio * pitch), supported, coverage)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return (0.0985 * width, 0.0, 9.0, 0)
    return (best[1], best[2], best[0], best[3])

def candidate_rail_lines(image: np.ndarray) -> list[RailLine]:
    (height, width) = image.shape[:2]
    roi_x2 = int(round(RAIL_ROI_MAX_X * width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 38, 125)[:, :roi_x2]
    min_length = max(22, int(round(RAIL_MIN_LENGTH_H * height)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360.0, threshold=max(12, min_length // 3), minLineLength=min_length, maxLineGap=max(14, int(round(0.1 * height))))
    result: list[RailLine] = []
    if lines is None:
        return result
    y_top = TOP_ROW_EXPECTED_Y * height
    y_bottom = BOTTOM_ROW_EXPECTED_Y * height
    for raw in np.asarray(lines).reshape(-1, 4):
        (x1, y1, x2, y2) = (int(v) for v in raw)
        (dy, dx) = (y2 - y1, x2 - x1)
        if abs(dy) < 2:
            continue
        a = dx / dy
        if not RAIL_SLOPE_MIN <= a <= RAIL_SLOPE_MAX:
            continue
        b = x1 - a * y1
        xt = a * y_top + b
        xb = a * y_bottom + b
        if not RAIL_TOP_X_RANGE[0] * width <= xt <= RAIL_TOP_X_RANGE[1] * width:
            continue
        if not RAIL_BOTTOM_X_RANGE[0] * width <= xb <= RAIL_BOTTOM_X_RANGE[1] * width:
            continue
        length = math.hypot(dx, dy)
        location_penalty = abs(xt - 0.04 * width) / max(0.14 * width, 1.0)
        slope_penalty = abs(a + 0.62) / 1.0
        score = 1.45 * length / height - 0.3 * location_penalty - 0.08 * slope_penalty
        result.append(RailLine(float(a), float(b), 'left_continuous_rail_hough', float(score), (x1, y1, x2, y2)))
    selected: list[RailLine] = []
    for candidate in sorted(result, key=lambda value: value.score, reverse=True):
        xt = candidate.x_at(TOP_ROW_EXPECTED_Y * height)
        xb = candidate.x_at(BOTTOM_ROW_EXPECTED_Y * height)
        if any((abs(xt - old.x_at(TOP_ROW_EXPECTED_Y * height)) < 0.008 * width and abs(xb - old.x_at(BOTTOM_ROW_EXPECTED_Y * height)) < 0.008 * width for old in selected)):
            continue
        selected.append(candidate)
        if len(selected) >= 18:
            break
    return selected

def detect_rail_and_lattice(image: np.ndarray, evidence: Sequence[Evidence], top_y: float, bottom_y: float) -> tuple[RailLine, dict]:
    (height, width) = image.shape[:2]
    top_items = row_evidence(evidence, top_y, height)
    bottom_items = row_evidence(evidence, bottom_y, height)
    candidates = candidate_rail_lines(image)
    if not candidates:
        raise RuntimeError('LEFT_CONTINUOUS_RAIL_NOT_FOUND')
    best: Optional[tuple[float, RailLine, dict]] = None
    for rail in candidates:
        (top_pitch, top_shift, top_fit, top_support) = fit_pitch_and_shift(top_items, rail.x_at(top_y), width)
        (bottom_pitch, bottom_shift, bottom_fit, bottom_support) = fit_pitch_and_shift(bottom_items, rail.x_at(bottom_y), width)
        pitch_disagreement = abs(top_pitch - bottom_pitch) / max(0.5 * (top_pitch + bottom_pitch), 1.0)
        shift_disagreement = abs(top_shift / top_pitch - bottom_shift / bottom_pitch)
        score = top_fit + bottom_fit + 0.5 * pitch_disagreement + 0.18 * shift_disagreement + 0.32 * abs(rail.a + 0.7) - 0.1 * rail.score
        meta = {'top_pitch': top_pitch, 'bottom_pitch': bottom_pitch, 'top_shift': top_shift, 'bottom_shift': bottom_shift, 'top_fit': top_fit, 'bottom_fit': bottom_fit, 'top_support': top_support, 'bottom_support': bottom_support, 'pitch_disagreement': pitch_disagreement, 'rail_candidate_count': len(candidates)}
        if best is None or score < best[0]:
            best = (score, rail, meta)
    assert best is not None
    best[2]['raw_top_shift'] = float(best[2]['top_shift'])
    best[2]['raw_bottom_shift'] = float(best[2]['bottom_shift'])
    shift_supports = np.asarray([max(1, best[2]['top_support']), max(1, best[2]['bottom_support'])], dtype=np.float64)
    common_shift = float(np.average([best[2]['top_shift'], best[2]['bottom_shift']], weights=shift_supports))
    best[2]['top_shift'] = common_shift
    best[2]['bottom_shift'] = common_shift
    best[2]['rail_phase_mode'] = 'shared_parallel_offset'
    if best[2]['pitch_disagreement'] > 0.06:
        supports = np.asarray([max(1, best[2]['top_support']), max(1, best[2]['bottom_support'])], dtype=np.float64)
        common_pitch = float(np.average([best[2]['top_pitch'], best[2]['bottom_pitch']], weights=supports))
        best[2]['top_pitch'] = common_pitch
        best[2]['bottom_pitch'] = common_pitch
        best[2]['pitch_mode'] = 'shared_after_sparse_disagreement'
    else:
        best[2]['pitch_mode'] = 'row_specific'
    best[2]['rail_lattice_score'] = float(best[0])
    return (best[1], best[2])

def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(points, dtype=np.float32))))

def clipped_polygon_area(points: np.ndarray, width: int, height: int) -> float:
    rect = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    (area, _) = cv2.intersectConvexConvex(np.asarray(points, dtype=np.float32), rect)
    return float(max(0.0, area))

def grid_point(u: float, v: float, top_origin_x: float, bottom_origin_x: float, top_pitch: float, bottom_pitch: float, top_y: float, bottom_y: float) -> np.ndarray:
    top = np.asarray([top_origin_x + u * top_pitch, top_y], dtype=np.float64)
    bottom = np.asarray([bottom_origin_x + u * bottom_pitch, bottom_y], dtype=np.float64)
    return top + v * (bottom - top)

def build_slots(image: np.ndarray, rail: RailLine, lattice: dict, top_y: float, bottom_y: float) -> list[RailSlot]:
    (height, width) = image.shape[:2]
    top_pitch = float(lattice['top_pitch'])
    bottom_pitch = float(lattice['bottom_pitch'])
    top_origin_x = rail.x_at(top_y) + float(lattice['top_shift'])
    bottom_origin_x = rail.x_at(bottom_y) + float(lattice['bottom_shift'])
    slots: list[RailSlot] = []
    empty_streak = 0
    for col in range(1, MAX_PREVIEW_COLUMNS + 1):
        u_left = col - 1.5
        u_right = col - 0.5
        column_ratios: list[float] = []
        pending: list[RailSlot] = []
        for (row, v_top, v_bottom, v_center) in (('t', -0.5, 0.5, 0.0), ('b', 0.5, 1.5, 1.0)):
            polygon = np.asarray([grid_point(u_left, v_top, top_origin_x, bottom_origin_x, top_pitch, bottom_pitch, top_y, bottom_y), grid_point(u_right, v_top, top_origin_x, bottom_origin_x, top_pitch, bottom_pitch, top_y, bottom_y), grid_point(u_right, v_bottom, top_origin_x, bottom_origin_x, top_pitch, bottom_pitch, top_y, bottom_y), grid_point(u_left, v_bottom, top_origin_x, bottom_origin_x, top_pitch, bottom_pitch, top_y, bottom_y)], dtype=np.float32)
            center = grid_point(col - 1.0, v_center, top_origin_x, bottom_origin_x, top_pitch, bottom_pitch, top_y, bottom_y)
            area = max(polygon_area(polygon), 1.0)
            inside_ratio = clipped_polygon_area(polygon, width, height) / area
            column_ratios.append(inside_ratio)
            pending.append(RailSlot(row=row, local_col=col, slot_name=f'{row}{col}', x_prior=float(center[0]), y_prior=float(center[1]), cell_polygon=polygon, cell_inside_image_ratio=float(inside_ratio)))
        column_visible = max(column_ratios) >= VISIBLE_CELL_MIN_IMAGE_RATIO if col == 1 else min(column_ratios) >= 0.48
        if column_visible:
            slots.extend(pending)
            empty_streak = 0
        else:
            empty_streak += 1
            if empty_streak >= 2 and col > 2:
                break
    return slots

def raster_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask

def assign_holes(slots: Sequence[RailSlot], holes: Sequence[object], pitch: float, row_gap: float) -> None:
    for slot in slots:
        best = float('inf')
        for hole in holes:
            distance = math.hypot((float(hole.x) - slot.x_prior) / pitch, (float(hole.y) - slot.y_prior) / row_gap)
            best = min(best, distance)
        slot.nearest_hole_distance = best
        if best <= 0.34:
            slot.visibility = 'EMPTY'
            slot.status = 'EMPTY_HOLE'

def assign_segments(image: np.ndarray, slots: list[RailSlot], segments: Sequence[dict], pitch: float, row_gap: float) -> dict:
    (height, width) = image.shape[:2]
    if not slots or not segments:
        return {'assigned': 0, 'unassigned': len(segments), 'merge_review': 0}
    cell_masks = [raster_polygon(slot.cell_polygon, width, height) for slot in slots]
    segment_masks: list[np.ndarray] = []
    for segment in segments:
        segment_masks.append(raster_polygon(segment['polygon'], width, height))
    score = np.full((len(slots), len(segments)), -1000000.0, dtype=np.float64)
    metrics: dict[tuple[int, int], tuple[float, float, float]] = {}
    per_segment_coverages: dict[int, list[float]] = {}
    for (seg_index, (segment, seg_mask)) in enumerate(zip(segments, segment_masks)):
        seg_area = max(float(seg_mask.sum()), 1.0)
        coverages: list[float] = []
        for (slot_index, (slot, cell_mask)) in enumerate(zip(slots, cell_masks)):
            intersection = float(np.count_nonzero((seg_mask > 0) & (cell_mask > 0)))
            mask_coverage = intersection / seg_area
            cell_coverage = intersection / max(float(cell_mask.sum()), 1.0)
            distance = math.hypot((float(segment['cx']) - slot.x_prior) / max(pitch, 1.0), (float(segment['cy']) - slot.y_prior) / max(row_gap, 1.0))
            center_inside = cv2.pointPolygonTest(np.asarray(slot.cell_polygon, dtype=np.float32), (float(segment['cx']), float(segment['cy'])), False) >= 0
            coverages.append(mask_coverage)
            if mask_coverage < SEG_MIN_MASK_COVERAGE and (not center_inside) or distance > SEG_MAX_CENTER_DISTANCE:
                continue
            value = 1.75 * mask_coverage + 0.42 * min(1.0, cell_coverage / 0.35) + 0.28 * float(center_inside) + 0.1 * float(segment['conf']) - 0.34 * distance
            score[slot_index, seg_index] = value
            metrics[slot_index, seg_index] = (mask_coverage, cell_coverage, distance)
        per_segment_coverages[seg_index] = sorted(coverages, reverse=True)
    (row_indices, col_indices) = linear_sum_assignment(-score)
    assigned = 0
    merge_review = 0
    for (slot_index, seg_index) in zip(row_indices, col_indices):
        if score[slot_index, seg_index] < SEG_MIN_ASSIGN_SCORE:
            continue
        slot = slots[slot_index]
        segment = segments[seg_index]
        (mask_coverage, cell_coverage, distance) = metrics[slot_index, seg_index]
        ordered = per_segment_coverages[seg_index]
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        merged = runner_up >= SEG_MERGE_RUNNERUP_COVERAGE and mask_coverage - runner_up < SEG_MERGE_MARGIN
        slot.visibility = 'VISIBLE'
        slot.status = 'SEG_MERGE_REVIEW' if merged else 'SEG_ASSIGNED'
        slot.seg_id = int(segment['seg_id'])
        slot.seg_conf = float(segment['conf'])
        slot.seg_center_x = float(segment['cx'])
        slot.seg_center_y = float(segment['cy'])
        slot.seg_area = float(segment['area'])
        slot.seg_bbox = tuple((float(v) for v in segment['bbox']))
        slot.mask_coverage = float(mask_coverage)
        slot.cell_coverage = float(cell_coverage)
        slot.runner_up_coverage = float(runner_up)
        slot.center_distance = float(distance)
        (x1, y1, x2, y2) = slot.seg_bbox
        pad_x = BBOX_PAD_RATIO * max(1.0, x2 - x1)
        pad_y = BBOX_PAD_RATIO * max(1.0, y2 - y1)
        slot.bbox = (int(round(np.clip(x1 - pad_x, 0, width - 1))), int(round(np.clip(y1 - pad_y, 0, height - 1))), int(round(np.clip(x2 + pad_x, 0, width - 1))), int(round(np.clip(y2 + pad_y, 0, height - 1))))
        assigned += 1
        merge_review += int(merged)
    return {'assigned': assigned, 'unassigned': max(0, len(segments) - assigned), 'merge_review': merge_review}

def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (scale, thickness) = (0.52, 2)
    ((tw, th), _) = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(x, 0, max(0, image.shape[1] - tw - 5)))
    y = int(np.clip(y, th + 4, image.shape[0] - 4))
    cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_overlay(image: np.ndarray, rail: RailLine, slots: Sequence[RailSlot], segments: Sequence[dict], holes: Sequence[object], meta: dict) -> np.ndarray:
    (height, width) = image.shape[:2]
    header_h = 55
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[header_h:] = image
    (y0, y1) = (0.0, float(height - 1))
    cv2.line(canvas, (int(round(rail.x_at(y0))), header_h), (int(round(rail.x_at(y1))), header_h + height - 1), (0, 90, 255), 6, cv2.LINE_AA)
    segment_lookup = {int(segment['seg_id']): segment for segment in segments}
    for slot in slots:
        color = PALETTE[(slot.local_col - 1) % len(PALETTE)]
        polygon = np.round(slot.cell_polygon + np.asarray([0, header_h])).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)
        if slot.seg_id is not None:
            segment = segment_lookup[slot.seg_id]
            seg_polygon = np.round(segment['polygon'] + np.asarray([0, header_h])).astype(np.int32)
            cv2.polylines(canvas, [seg_polygon], True, color, 3, cv2.LINE_AA)
        if slot.bbox is not None:
            (bx1, by1, bx2, by2) = slot.bbox
            cv2.rectangle(canvas, (bx1, by1 + header_h), (bx2, by2 + header_h), color, 3)
        center = (int(round(slot.x_prior)), int(round(slot.y_prior + header_h)))
        cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
        state = {'SEG_ASSIGNED': 'SEG', 'SEG_MERGE_REVIEW': 'MERGE', 'EMPTY_HOLE': 'EMPTY'}.get(slot.status, 'PRIOR')
        label_y = center[1] - 12 if slot.row == 't' else center[1] + 27
        draw_label(canvas, f'{slot.slot_name} {state}', center[0] + 7, label_y, color)
    for hole in holes:
        cv2.circle(canvas, (int(round(hole.x)), int(round(hole.y + header_h))), 4, (255, 255, 0), -1)
    top = next((slot for slot in slots if slot.slot_name == 't1'), None)
    bottom = next((slot for slot in slots if slot.slot_name == 'b1'), None)
    if top is not None and bottom is not None:
        cv2.line(canvas, (int(round(top.x_prior)), int(round(top.y_prior + header_h))), (int(round(bottom.x_prior)), int(round(bottom.y_prior + header_h))), (255, 255, 255), 4, cv2.LINE_AA)
    header = f"CAM2 rail-t1 registration | cols={meta['visible_cols']} | holes={meta['hole_count']} | seg={meta['seg_count']} | assigned={meta['assigned']} | merge={meta['merge_review']} | pitchT/B={meta['top_pitch']:.1f}/{meta['bottom_pitch']:.1f} | rail_a={meta['rail_a']:.3f} | shiftT/B={meta['top_shift']:.1f}/{meta['bottom_shift']:.1f} | QC={meta['qc']}"
    cv2.putText(canvas, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'orange=detected continuous rail | white=t1-b1 axis | thin=ownership cell | thick contour/box=lettuce-seg', (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas

def polygon_text(points: np.ndarray) -> str:
    return ';'.join((f'{float(x):.2f},{float(y):.2f}' for (x, y) in points))

def process_one(model, image_path: Path, output_root: Path) -> tuple[dict, list[dict], Path]:
    image = BASE.imread_unicode(image_path)
    segments = BASE.predict_segments(model, image)
    holes = all_image_holes(image)
    evidence = evidence_from(holes, segments)
    (top_y, bottom_y, row_meta) = estimate_rows(evidence, image.shape[0])
    (rail, lattice) = detect_rail_and_lattice(image, evidence, top_y, bottom_y)
    slots = build_slots(image, rail, lattice, top_y, bottom_y)
    pitch = 0.5 * (float(lattice['top_pitch']) + float(lattice['bottom_pitch']))
    row_gap = max(1.0, bottom_y - top_y)
    assign_holes(slots, holes, pitch, row_gap)
    assignment = assign_segments(image, slots, segments, pitch, row_gap)
    qc_flags: list[str] = []
    if abs(float(lattice['top_shift']) / lattice['top_pitch']) > 0.18:
        qc_flags.append('TOP_RAIL_PHASE_RESIDUAL')
    if abs(float(lattice['bottom_shift']) / lattice['bottom_pitch']) > 0.18:
        qc_flags.append('BOTTOM_RAIL_PHASE_RESIDUAL')
    if float(lattice['pitch_disagreement']) > 0.1:
        qc_flags.append('TOP_BOTTOM_PITCH_DISAGREE')
    if assignment['unassigned']:
        qc_flags.append('UNASSIGNED_SEG')
    if assignment['merge_review']:
        qc_flags.append('SEG_MERGE_REVIEW')
    qc = 'OK' if not qc_flags else '+'.join(qc_flags)
    visible_cols = max((slot.local_col for slot in slots), default=0)
    meta = {**row_meta, **lattice, **assignment, 'image_name': image_path.name, 'source_path': str(image_path), 'width': image.shape[1], 'height': image.shape[0], 'visible_cols': visible_cols, 'hole_count': len(holes), 'seg_count': len(segments), 'top_y': top_y, 'bottom_y': bottom_y, 'rail_a': rail.a, 'rail_b': rail.b, 'rail_source': rail.source, 'rail_segment': ','.join((str(v) for v in rail.segment)), 'qc': qc}
    overlay = draw_overlay(image, rail, slots, segments, holes, meta)
    output_path = output_root / f'{image_path.stem}.jpg'
    BASE.imwrite_unicode(output_path, overlay)
    slot_rows: list[dict] = []
    for slot in slots:
        slot_rows.append({'image_name': image_path.name, 'slot_name': slot.slot_name, 'row': slot.row, 'local_col': slot.local_col, 'x_prior': round(slot.x_prior, 3), 'y_prior': round(slot.y_prior, 3), 'cell_polygon': polygon_text(slot.cell_polygon), 'cell_inside_image_ratio': round(slot.cell_inside_image_ratio, 4), 'visibility': slot.visibility, 'status': slot.status, 'seg_id': '' if slot.seg_id is None else slot.seg_id, 'seg_conf': '' if not np.isfinite(slot.seg_conf) else round(slot.seg_conf, 5), 'seg_center_x': '' if not np.isfinite(slot.seg_center_x) else round(slot.seg_center_x, 3), 'seg_center_y': '' if not np.isfinite(slot.seg_center_y) else round(slot.seg_center_y, 3), 'seg_bbox': '' if slot.seg_bbox is None else ','.join((f'{v:.2f}' for v in slot.seg_bbox)), 'bbox': '' if slot.bbox is None else ','.join((str(v) for v in slot.bbox)), 'mask_coverage': '' if not np.isfinite(slot.mask_coverage) else round(slot.mask_coverage, 5), 'cell_coverage': '' if not np.isfinite(slot.cell_coverage) else round(slot.cell_coverage, 5), 'runner_up_coverage': '' if not np.isfinite(slot.runner_up_coverage) else round(slot.runner_up_coverage, 5), 'center_distance': '' if not np.isfinite(slot.center_distance) else round(slot.center_distance, 5), 'nearest_hole_distance': '' if not np.isfinite(slot.nearest_hole_distance) else round(slot.nearest_hole_distance, 5)})
    return (meta, slot_rows, output_path)

def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
