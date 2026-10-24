"""Top-view geometry used by the final preprocessing adapters."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[3]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import importlib
_core = importlib.import_module('slot_geometry')
LETTUCE_SEG_MODEL_PATH = CYCLE.PLANT_MODEL
LETTUCE_SEG_SHA256 = 'A2DED62BBA4D577F4EAD18907B824C587E179357C8089B72DABDE41B755D6FE8'
SEG_IMGSZ = 1280
SEG_CONF = 0.2
SEG_IOU = 0.7
SEG_MIN_AREA = 80.0
ASSIGN_MAX_DX_PITCH = 0.82
ASSIGN_MAX_DY_ROW_GAP = 0.78
ASSIGN_CELL_X_MARGIN_PITCH = 0.18
ASSIGN_CELL_Y_MARGIN_ROW_GAP = 0.12
ASSIGN_STRICT_MAX_DX_PITCH = 0.58
ASSIGN_STRICT_MAX_DY_ROW_GAP = 0.38
ASSIGN_ROW_CURVE_OUTLIER_GAP = 0.46
ASSIGN_ROW_CURVE_INLIER_GAP = 0.3
SEG_CENTER_MAX_SHIFT_PITCH = 0.2
BOUNDARY_MAX_SHIFT_PITCH = 0.18
BBOX_PAD_RATIO = 0.08
BBOX_MIN_WIDTH_PITCH = 0.56
BBOX_MIN_HEIGHT_ROW_GAP = 0.55

def load_lettuce_seg_model():
    if not LETTUCE_SEG_MODEL_PATH.exists():
        raise FileNotFoundError(f'lettuce-seg weight not found: {LETTUCE_SEG_MODEL_PATH}')
    from ultralytics import YOLO
    model = YOLO(str(LETTUCE_SEG_MODEL_PATH))
    names = {str(value).lower() for value in model.names.values()}
    if model.task != 'segment' or 'lettuce' not in names:
        raise RuntimeError(f'invalid lettuce-seg model: task={model.task}, names={model.names}')
    return model

def _polygon_center(poly: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    if moments['m00'] > 0:
        return (float(moments['m10'] / moments['m00']), float(moments['m01'] / moments['m00']))
    (x1, y1, x2, y2) = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

def predict_lettuce_segments(model, image: np.ndarray) -> List[dict]:
    result = model.predict(source=image, imgsz=SEG_IMGSZ, conf=SEG_CONF, iou=SEG_IOU, retina_masks=True, verbose=False)[0]
    if result.boxes is None or result.masks is None or result.masks.xy is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    polygons = result.masks.xy
    segments: List[dict] = []
    for (index, box) in enumerate(boxes):
        if index >= len(polygons):
            continue
        polygon = np.asarray(polygons[index], dtype=np.float32)
        if len(polygon) < 3:
            continue
        area = float(abs(cv2.contourArea(polygon)))
        if area < SEG_MIN_AREA:
            continue
        bbox = tuple((float(value) for value in box))
        (cx, cy) = _polygon_center(polygon, bbox)
        segments.append({'seg_id': len(segments) + 1, 'bbox': bbox, 'cx': cx, 'cy': cy, 'area': area, 'conf': float(confs[index]), 'cls': int(classes[index]), 'polygon': polygon})
    return segments

def segments_to_row_evidence(segments: Sequence[dict], image_h: int) -> Dict[str, List[dict]]:
    result: Dict[str, List[dict]] = {'t': [], 'b': []}
    split_y = 0.5 * image_h
    for segment in segments:
        row = 't' if float(segment['cy']) < split_y else 'b'
        result[row].append({'x': float(segment['cx']), 'y': float(segment['cy']), 'score': 100.0 * float(segment['conf']), 'kind': 'lettuce_seg', 'seg_id': int(segment['seg_id']), 'bbox': tuple(segment['bbox'])})
    return result

def _median_pitch(slots: Sequence) -> float:
    centers: Dict[int, List[float]] = {}
    for slot in slots:
        if slot.local_col > 0 and np.isfinite(slot.x_prior):
            centers.setdefault(slot.local_col, []).append(float(slot.x_prior))
    xs = [float(np.mean(centers[key])) for key in sorted(centers)]
    diffs = np.diff(xs)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0)]
    return float(np.median(diffs)) if len(diffs) else 1.0

def _median_row_gap(slots: Sequence) -> float:
    by_col: Dict[int, Dict[str, float]] = {}
    for slot in slots:
        if slot.local_col > 0 and np.isfinite(slot.y_prior):
            by_col.setdefault(slot.local_col, {})[slot.row] = float(slot.y_prior)
    gaps = [values['b'] - values['t'] for values in by_col.values() if 't' in values and 'b' in values and (values['b'] > values['t'])]
    return float(np.median(gaps)) if gaps else 1.0

def _inside_bbox(x: float, y: float, bbox: Tuple[float, float, float, float]) -> bool:
    (x1, y1, x2, y2) = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def assign_segments_to_slots(slots: Sequence, segments: Sequence[dict], *, enforce_cell_corridor: bool=False, conservative_prior_gate: bool=False) -> Dict[int, dict]:
    eligible = [slot for slot in slots if slot.local_col > 0 and slot.visibility not in (_core.Visibility.OUT_OF_FOV, _core.Visibility.OCCLUDED) and np.isfinite(slot.x_prior) and np.isfinite(slot.y_prior)]
    if not eligible or not segments:
        return {}
    pitch = max(_median_pitch(slots), 1.0)
    row_gap = max(_median_row_gap(slots), 1.0)
    cost = np.full((len(eligible), len(segments)), 1000000.0, dtype=np.float64)
    for (slot_index, slot) in enumerate(eligible):
        for (seg_index, segment) in enumerate(segments):
            dx = abs(float(segment['cx']) - float(slot.x_prior)) / pitch
            dy = abs(float(segment['cy']) - float(slot.y_prior)) / row_gap
            contains_prior = _inside_bbox(slot.x_prior, slot.y_prior, segment['bbox'])
            if enforce_cell_corridor and slot.cell_bbox is not None:
                (cell_x1, cell_y1, cell_x2, cell_y2) = slot.cell_bbox
                center_in_cell_corridor = cell_x1 - ASSIGN_CELL_X_MARGIN_PITCH * pitch <= float(segment['cx']) <= cell_x2 + ASSIGN_CELL_X_MARGIN_PITCH * pitch and cell_y1 - ASSIGN_CELL_Y_MARGIN_ROW_GAP * row_gap <= float(segment['cy']) <= cell_y2 + ASSIGN_CELL_Y_MARGIN_ROW_GAP * row_gap
                if not center_in_cell_corridor:
                    continue
                if conservative_prior_gate:
                    center_in_cell = cell_x1 <= float(segment['cx']) <= cell_x2 and cell_y1 <= float(segment['cy']) <= cell_y2
                    if not center_in_cell:
                        continue
                    if not contains_prior and (dx > ASSIGN_STRICT_MAX_DX_PITCH or dy > ASSIGN_STRICT_MAX_DY_ROW_GAP):
                        continue
            if not contains_prior and (dx > ASSIGN_MAX_DX_PITCH or dy > ASSIGN_MAX_DY_ROW_GAP):
                continue
            normalized = float(np.hypot(dx / 0.72, dy / 0.68))
            confidence_bonus = 0.08 * float(segment['conf'])
            contain_bonus = 0.18 if contains_prior else 0.0
            cost[slot_index, seg_index] = normalized - confidence_bonus - contain_bonus
    try:
        from scipy.optimize import linear_sum_assignment
        (slot_indices, seg_indices) = linear_sum_assignment(cost)
    except Exception:
        (slot_indices, seg_indices) = ([], [])
        remaining_slots = set(range(len(eligible)))
        remaining_segments = set(range(len(segments)))
        while remaining_slots and remaining_segments:
            best = min(((cost[i, j], i, j) for i in remaining_slots for j in remaining_segments), default=(1000000.0, -1, -1))
            if best[0] >= 1000000.0:
                break
            slot_indices.append(best[1])
            seg_indices.append(best[2])
            remaining_slots.remove(best[1])
            remaining_segments.remove(best[2])
    assigned: Dict[int, dict] = {}
    for (slot_index, seg_index) in zip(slot_indices, seg_indices):
        if cost[slot_index, seg_index] >= 100000.0:
            continue
        assigned[id(eligible[slot_index])] = segments[seg_index]
    return assigned

def _clamp_boundary(desired: float, base: float, pitch: float) -> float:
    limit = BOUNDARY_MAX_SHIFT_PITCH * pitch
    return float(np.clip(desired, base - limit, base + limit))

def _prune_single_row_curve_outlier(slots: Sequence, assigned: Dict[int, dict], row_gap: float) -> Tuple[Dict[int, dict], List[int]]:
    kept = dict(assigned)
    rejected_seg_ids: List[int] = []
    for row in ('t', 'b'):
        pairs = [(slot, kept[id(slot)]) for slot in slots if slot.row == row and id(slot) in kept]
        if len(pairs) < 5:
            continue
        xs = np.asarray([float(segment['cx']) for (_, segment) in pairs])
        ys = np.asarray([float(segment['cy']) for (_, segment) in pairs])
        slopes = [float((ys[j] - ys[i]) / (xs[j] - xs[i])) for i in range(len(xs)) for j in range(i + 1, len(xs)) if abs(float(xs[j] - xs[i])) > 1.0]
        if not slopes:
            continue
        slope = float(np.median(slopes))
        intercept = float(np.median(ys - slope * xs))
        residuals = np.abs(ys - (slope * xs + intercept)) / max(row_gap, 1.0)
        outliers = np.flatnonzero(residuals > ASSIGN_ROW_CURVE_OUTLIER_GAP)
        if len(outliers) != 1:
            continue
        outlier_index = int(outliers[0])
        remaining = np.delete(residuals, outlier_index)
        if len(remaining) and float(np.max(remaining)) > ASSIGN_ROW_CURVE_INLIER_GAP:
            continue
        (slot, segment) = pairs[outlier_index]
        kept.pop(id(slot), None)
        rejected_seg_ids.append(int(segment['seg_id']))
    return (kept, rejected_seg_ids)

def _row_x_regions(row_slots: Sequence, assigned: Dict[int, dict], bed_x1: float, bed_x2: float, pitch: float) -> Dict[int, Tuple[float, float]]:
    ordered = sorted(row_slots, key=lambda slot: slot.local_col)
    if not ordered:
        return {}
    boundaries = [bed_x1]
    for (left, right) in zip(ordered[:-1], ordered[1:]):
        base = 0.5 * (float(left.x) + float(right.x))
        left_seg = assigned.get(id(left))
        right_seg = assigned.get(id(right))
        desired = base
        if left_seg is not None and right_seg is not None:
            left_edge = float(left_seg['bbox'][2])
            right_edge = float(right_seg['bbox'][0])
            if left_edge < right_edge:
                desired = 0.5 * (left_edge + right_edge)
        elif left_seg is not None:
            desired = 0.5 * (float(left_seg['bbox'][2]) + float(right.x))
        elif right_seg is not None:
            desired = 0.5 * (float(left.x) + float(right_seg['bbox'][0]))
        boundaries.append(_clamp_boundary(desired, base, pitch))
    boundaries.append(bed_x2)
    return {id(slot): (boundaries[index], boundaries[index + 1]) for (index, slot) in enumerate(ordered)}

def _column_y_regions(slots: Sequence, assigned: Dict[int, dict], bed_y1: float, bed_y2: float, row_gap: float) -> Dict[int, Tuple[float, float]]:
    by_col: Dict[int, Dict[str, object]] = {}
    for slot in slots:
        if slot.local_col > 0:
            by_col.setdefault(slot.local_col, {})[slot.row] = slot
    result: Dict[int, Tuple[float, float]] = {}
    for values in by_col.values():
        if 't' not in values or 'b' not in values:
            continue
        top = values['t']
        bottom = values['b']
        base = 0.5 * (float(top.y) + float(bottom.y))
        top_seg = assigned.get(id(top))
        bottom_seg = assigned.get(id(bottom))
        desired = base
        if top_seg is not None and bottom_seg is not None:
            top_edge = float(top_seg['bbox'][3])
            bottom_edge = float(bottom_seg['bbox'][1])
            if top_edge < bottom_edge:
                desired = 0.5 * (top_edge + bottom_edge)
        desired = _clamp_boundary(desired, base, row_gap)
        result[id(top)] = (bed_y1, desired)
        result[id(bottom)] = (desired, bed_y2)
    return result

def _expand_to_minimum(low: float, high: float, center: float, minimum: float, limit_low: float, limit_high: float) -> Tuple[float, float]:
    if high - low >= minimum:
        return (low, high)
    half = 0.5 * minimum
    low = max(limit_low, center - half)
    high = min(limit_high, center + half)
    if high - low < minimum:
        low = max(limit_low, high - minimum)
        high = min(limit_high, low + minimum)
    return (low, high)

def refine_slots_with_segments(slots: Sequence, segments: Sequence[dict], img_w: int, img_h: int, *, bed_bbox: Optional[Tuple[int, int, int, int]], enforce_cell_corridor: bool=False, conservative_prior_gate: bool=False, prune_single_curve_outlier: bool=False) -> dict:
    assigned = assign_segments_to_slots(slots, segments, enforce_cell_corridor=enforce_cell_corridor, conservative_prior_gate=conservative_prior_gate)
    pitch = max(_median_pitch(slots), 1.0)
    row_gap = max(_median_row_gap(slots), 1.0)
    curve_rejected_segment_ids: List[int] = []
    if prune_single_curve_outlier:
        (assigned, curve_rejected_segment_ids) = _prune_single_row_curve_outlier(slots, assigned, row_gap)
    for slot in slots:
        if slot.local_col <= 0:
            continue
        segment = assigned.get(id(slot))
        if segment is None:
            slot.seg_status = 'NO_SEG_PRIOR_CELL'
            continue
        slot.seg_id = int(segment['seg_id'])
        slot.seg_conf = float(segment['conf'])
        slot.seg_center_x = float(segment['cx'])
        slot.seg_center_y = float(segment['cy'])
        slot.seg_area = float(segment['area'])
        slot.seg_bbox = tuple((float(value) for value in segment['bbox']))
        slot.seg_status = 'SEG_ASSIGNED'
        _core.apply_bounded_residual(slot, slot.seg_center_x, slot.seg_center_y, pitch=pitch, max_shift_ratio=SEG_CENTER_MAX_SHIFT_PITCH, reject_distance_ratio=0.8)
    _core.build_cell_bboxes(slots, img_w, img_h, bed_bbox=bed_bbox)
    if bed_bbox is None:
        (bed_x1, bed_y1, bed_x2, bed_y2) = (0.0, 0.0, float(img_w), float(img_h))
    else:
        (bed_x1, bed_y1, bed_x2, bed_y2) = (float(value) for value in bed_bbox)
    x_regions: Dict[int, Tuple[float, float]] = {}
    for row in ('t', 'b'):
        row_slots = [slot for slot in slots if slot.local_col > 0 and slot.row == row]
        x_regions.update(_row_x_regions(row_slots, assigned, bed_x1, bed_x2, pitch))
    y_regions = _column_y_regions(slots, assigned, bed_y1, bed_y2, row_gap)
    for slot in slots:
        if slot.local_col <= 0 or slot.cell_bbox is None:
            continue
        segment = assigned.get(id(slot))
        if segment is None:
            slot.bbox = slot.cell_bbox
            continue
        (limit_x1, limit_x2) = x_regions.get(id(slot), (bed_x1, bed_x2))
        (limit_y1, limit_y2) = y_regions.get(id(slot), (bed_y1, bed_y2))
        (sx1, sy1, sx2, sy2) = (float(value) for value in segment['bbox'])
        width = max(1.0, sx2 - sx1)
        height = max(1.0, sy2 - sy1)
        x1 = max(limit_x1, sx1 - BBOX_PAD_RATIO * width)
        x2 = min(limit_x2, sx2 + BBOX_PAD_RATIO * width)
        y1 = max(limit_y1, sy1 - BBOX_PAD_RATIO * height)
        y2 = min(limit_y2, sy2 + BBOX_PAD_RATIO * height)
        (x1, x2) = _expand_to_minimum(x1, x2, float(slot.x), BBOX_MIN_WIDTH_PITCH * pitch, limit_x1, limit_x2)
        (y1, y2) = _expand_to_minimum(y1, y2, float(slot.y), BBOX_MIN_HEIGHT_ROW_GAP * row_gap, limit_y1, limit_y2)
        bbox = (int(round(np.clip(x1, 0, img_w))), int(round(np.clip(y1, 0, img_h))), int(round(np.clip(x2, 0, img_w))), int(round(np.clip(y2, 0, img_h))))
        slot.bbox = bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else slot.cell_bbox
    return {'segment_count': len(segments), 'assigned_segment_count': len(assigned), 'unassigned_segment_count': max(0, len(segments) - len(assigned)), 'curve_rejected_segment_count': len(curve_rejected_segment_ids), 'curve_rejected_segment_ids': curve_rejected_segment_ids, 'assigned_by_slot_object_id': assigned}

def segment_by_id(segments: Iterable[dict]) -> Dict[int, dict]:
    return {int(segment['seg_id']): segment for segment in segments}
