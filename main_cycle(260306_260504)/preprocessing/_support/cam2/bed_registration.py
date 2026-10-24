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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
RAIL_SCRIPT = SCRIPT_DIR / 'rail_geometry.py'
SOURCE_ROOT = CYCLE.PREPROCESS_OUTPUT / '1. bed-seg' / 'images' / 'cam2'
OUTPUT_ROOT = CYCLE.PREPROCESS_OUTPUT / '6. bbox'
REF_WIDTH = 2439
REF_HEIGHT = 318
REF_FRONT_RAIL_Y = 284.0
REF_RAIL_A = -0.695
REF_RAIL_B = 191.7
WIDTH_WARN_PX = 100
N_PHYSICAL_COLS = 12
TOP_X_MEASURED = np.asarray([331.5, 569.3, 796.9, 1022.7, 1249.1, 1474.6, 1699.6, 1922.1, 2153.6, 2382.1], dtype=np.float64)
BOTTOM_X_MEASURED = np.asarray([222.5, 488.5, 751.8, 1004.0, 1252.4, 1497.7, 1749.1, 2005.3, 2261.6], dtype=np.float64)
TOP_Y_LINE = (-0.00745289015, 118.527049)
BOTTOM_Y_LINE = (-0.0067701688, 252.083181)

def load_rail_module():
    spec = importlib.util.spec_from_file_location('cam2_rail_geometry', RAIL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Invalid input or state in load_rail_module (rejected condition: spec is None or spec.loader is None): {RAIL_SCRIPT} ')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
RAIL = load_rail_module()
BASE = RAIL.BASE

@dataclass
class BoardFit:
    image_name: str
    width: int
    height: int
    width_delta_px: int
    width_scale: float
    front_rail_y: float
    rail_a: float
    rail_b: float
    rail_front_x: float
    rail_candidate_count: int
    visible_cols: int
    seg_count: int
    assigned: int
    merge_review: int
    qc: str

def extend_centres(measured: np.ndarray, count: int) -> np.ndarray:
    """Retain measured centres and linearly extrapolate only out-of-view ones."""
    measured = np.asarray(measured, dtype=np.float64)
    cols = np.arange(1, len(measured) + 1, dtype=np.float64)
    (slope, intercept) = np.polyfit(cols, measured, 1)
    result = np.empty(count, dtype=np.float64)
    result[:len(measured)] = measured
    for index in range(len(measured), count):
        result[index] = slope * (index + 1) + intercept
    return result

def make_reference_board() -> tuple[np.ndarray, np.ndarray]:
    top_x = extend_centres(TOP_X_MEASURED, N_PHYSICAL_COLS)
    bottom_x = extend_centres(BOTTOM_X_MEASURED, N_PHYSICAL_COLS)
    top_y = TOP_Y_LINE[0] * top_x + TOP_Y_LINE[1]
    bottom_y = BOTTOM_Y_LINE[0] * bottom_x + BOTTOM_Y_LINE[1]
    return (np.column_stack([top_x, top_y]), np.column_stack([bottom_x, bottom_y]))
(REF_TOP, REF_BOTTOM) = make_reference_board()

def detect_front_rail_y(image: np.ndarray) -> tuple[float, float]:
    """Find the long lower rail by horizontal edge continuity, not dark holes."""
    (height, width) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    (x1, x2) = (int(round(0.08 * width)), int(round(0.95 * width)))
    row_score = np.mean(grad_y[:, x1:x2], axis=1)
    y1 = int(round(0.55 * height))
    y2 = max(y1 + 1, int(round(0.91 * height)))
    local = row_score[y1:y2]
    if local.size == 0:
        raise RuntimeError('FRONT_RAIL_SEARCH_RANGE_EMPTY')
    y = int(np.argmax(local)) + y1
    return (float(y), float(row_score[y]))

def choose_left_rail(image: np.ndarray, front_y: float):
    """Select one continuous rail; no hole or SEG evidence participates."""
    candidates = RAIL.candidate_rail_lines(image)
    if not candidates:
        raise RuntimeError('LEFT_CONTINUOUS_RAIL_NOT_FOUND')
    width = image.shape[1]

    def cost(candidate) -> float:
        front_cross = abs(candidate.x_at(front_y)) / max(0.035 * width, 1.0)
        slope_cost = abs(candidate.a - REF_RAIL_A) / 0.24
        return 1.35 * front_cross + slope_cost - 0.18 * candidate.score
    return (min(candidates, key=cost), len(candidates))

def transfer_point(point: np.ndarray, rail, front_y: float, width_scale: float) -> np.ndarray:
    """Attach one reference point to the same place on the movable board."""
    (x_ref, y_ref) = (float(v) for v in point)
    y_target = y_ref + (front_y - REF_FRONT_RAIL_Y)
    ref_rail_x = REF_RAIL_A * y_ref + REF_RAIL_B
    target_rail_x = rail.x_at(y_target)
    x_target = target_rail_x + width_scale * (x_ref - ref_rail_x)
    return np.asarray([x_target, y_target], dtype=np.float64)

def transfer_board(rail, front_y: float, width_scale: float) -> tuple[np.ndarray, np.ndarray]:
    top = np.asarray([transfer_point(p, rail, front_y, width_scale) for p in REF_TOP])
    bottom = np.asarray([transfer_point(p, rail, front_y, width_scale) for p in REF_BOTTOM])
    return (top, bottom)

def boundary_lines(top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(top)
    top_bound = np.empty((count + 1, 2), dtype=np.float64)
    bottom_bound = np.empty((count + 1, 2), dtype=np.float64)
    top_bound[1:count] = 0.5 * (top[:-1] + top[1:])
    bottom_bound[1:count] = 0.5 * (bottom[:-1] + bottom[1:])
    top_bound[0] = top[0] - 0.5 * (top[1] - top[0])
    bottom_bound[0] = bottom[0] - 0.5 * (bottom[1] - bottom[0])
    top_bound[count] = top[-1] + 0.5 * (top[-1] - top[-2])
    bottom_bound[count] = bottom[-1] + 0.5 * (bottom[-1] - bottom[-2])
    return (top_bound, bottom_bound)

def line_point(top_point: np.ndarray, bottom_point: np.ndarray, v: float) -> np.ndarray:
    return top_point + v * (bottom_point - top_point)

def make_slots(image: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> list:
    (height, width) = image.shape[:2]
    (top_bound, bottom_bound) = boundary_lines(top, bottom)
    slots: list = []
    for index in range(N_PHYSICAL_COLS):
        for (row, v1, v2, center) in (('t', -0.5, 0.5, top[index]), ('b', 0.5, 1.5, bottom[index])):
            polygon = np.asarray([line_point(top_bound[index], bottom_bound[index], v1), line_point(top_bound[index + 1], bottom_bound[index + 1], v1), line_point(top_bound[index + 1], bottom_bound[index + 1], v2), line_point(top_bound[index], bottom_bound[index], v2)], dtype=np.float32)
            area = max(RAIL.polygon_area(polygon), 1.0)
            inside = RAIL.clipped_polygon_area(polygon, width, height) / area
            if inside < 0.15:
                continue
            slots.append(RAIL.RailSlot(row=row, local_col=index + 1, slot_name=f'{row}{index + 1}', x_prior=float(center[0]), y_prior=float(center[1]), cell_polygon=polygon, cell_inside_image_ratio=float(inside), status='BOARD_PRIOR'))
    return slots

def draw_overlay(image: np.ndarray, rail, front_y: float, top: np.ndarray, bottom: np.ndarray, slots: Sequence, segments: Sequence[dict], fit: BoardFit) -> np.ndarray:
    (height, width) = image.shape[:2]
    header_h = 66
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[header_h:] = image
    cv2.line(canvas, (int(round(rail.x_at(0))), header_h), (int(round(rail.x_at(height - 1))), header_h + height - 1), (0, 105, 255), 6, cv2.LINE_AA)
    cv2.line(canvas, (0, int(round(front_y)) + header_h), (width - 1, int(round(front_y)) + header_h), (255, 190, 0), 3, cv2.LINE_AA)
    for (index, (t_point, b_point)) in enumerate(zip(top, bottom), start=1):
        p1 = (int(round(t_point[0])), int(round(t_point[1] + header_h)))
        p2 = (int(round(b_point[0])), int(round(b_point[1] + header_h)))
        cv2.line(canvas, p1, p2, (150, 55, 195), 4, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, (255, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 5, (255, 255, 80), -1, cv2.LINE_AA)
        if -40 <= t_point[0] <= width + 40:
            RAIL.draw_label(canvas, f't{index}', p1[0] + 5, p1[1] - 8, (255, 80, 255))
        if -40 <= b_point[0] <= width + 40:
            RAIL.draw_label(canvas, f'b{index}', p2[0] + 5, p2[1] + 22, (255, 255, 80))
    segment_lookup = {int(segment['seg_id']): segment for segment in segments}
    for slot in slots:
        poly = np.round(slot.cell_polygon + np.asarray([0, header_h])).astype(np.int32)
        cv2.polylines(canvas, [poly], True, (165, 165, 165), 1, cv2.LINE_AA)
        if slot.seg_id is None:
            continue
        color = RAIL.PALETTE[(slot.local_col - 1) % len(RAIL.PALETTE)]
        seg = segment_lookup[slot.seg_id]
        seg_poly = np.round(seg['polygon'] + np.asarray([0, header_h])).astype(np.int32)
        cv2.polylines(canvas, [seg_poly], True, color, 3, cv2.LINE_AA)
        if slot.bbox is not None:
            (x1, y1, x2, y2) = slot.bbox
            cv2.rectangle(canvas, (x1, y1 + header_h), (x2, y2 + header_h), color, 3)
    header1 = f'CAM2 movable-board registration | cols={fit.visible_cols} | seg={fit.seg_count} | assigned={fit.assigned} | merge={fit.merge_review} | QC={fit.qc}'
    header2 = f'width={fit.width} ref={REF_WIDTH} delta={fit.width_delta_px}px scale={fit.width_scale:.5f} | rail_a={fit.rail_a:.3f} | front_y={fit.front_rail_y:.1f} | orange=left rail, blue=front rail, purple=fixed empty-bed centre board'
    cv2.putText(canvas, header1, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, header2, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
    return canvas

def slot_record(image_name: str, slot) -> dict:
    return {'image_name': image_name, 'slot_name': slot.slot_name, 'x_prior': round(slot.x_prior, 3), 'y_prior': round(slot.y_prior, 3), 'inside_ratio': round(slot.cell_inside_image_ratio, 4), 'status': slot.status, 'seg_id': '' if slot.seg_id is None else slot.seg_id, 'seg_center_x': '' if not np.isfinite(slot.seg_center_x) else round(slot.seg_center_x, 3), 'seg_center_y': '' if not np.isfinite(slot.seg_center_y) else round(slot.seg_center_y, 3), 'center_distance': '' if not np.isfinite(slot.center_distance) else round(slot.center_distance, 5), 'bbox': '' if slot.bbox is None else ','.join((str(v) for v in slot.bbox))}

def process_one(model, image_path: Path, use_seg: bool, output_root: Path):
    image = BASE.imread_unicode(image_path)
    (height, width) = image.shape[:2]
    width_delta = abs(width - REF_WIDTH)
    width_scale = width / REF_WIDTH
    (front_y, front_score) = detect_front_rail_y(image)
    (rail, candidate_count) = choose_left_rail(image, front_y)
    (top, bottom) = transfer_board(rail, front_y, width_scale)
    slots = make_slots(image, top, bottom)
    segments = BASE.predict_segments(model, image) if use_seg else []
    top_pitch = np.median(np.linalg.norm(np.diff(top, axis=0), axis=1))
    bottom_pitch = np.median(np.linalg.norm(np.diff(bottom, axis=0), axis=1))
    pitch = float(0.5 * (top_pitch + bottom_pitch))
    row_gap = float(np.median(np.linalg.norm(bottom - top, axis=1)))
    assignment = RAIL.assign_segments(image, slots, segments, pitch, row_gap)
    qc_flags: list[str] = []
    if width_delta > WIDTH_WARN_PX:
        qc_flags.append('WIDTH_SCALE_REVIEW')
    rail_front_x = float(rail.x_at(front_y))
    if abs(rail_front_x) > 0.035 * width:
        qc_flags.append('RAIL_FRONT_CROSS_REVIEW')
    if front_score < 20.0:
        qc_flags.append('FRONT_RAIL_WEAK')
    if assignment['unassigned']:
        qc_flags.append('UNASSIGNED_SEG')
    if assignment['merge_review']:
        qc_flags.append('SEG_MERGE_REVIEW')
    visible_cols = max((slot.local_col for slot in slots), default=0)
    fit = BoardFit(image_name=image_path.name, width=width, height=height, width_delta_px=width_delta, width_scale=width_scale, front_rail_y=front_y, rail_a=float(rail.a), rail_b=float(rail.b), rail_front_x=rail_front_x, rail_candidate_count=candidate_count, visible_cols=visible_cols, seg_count=len(segments), assigned=int(assignment['assigned']), merge_review=int(assignment['merge_review']), qc='OK' if not qc_flags else '+'.join(qc_flags))
    overlay = draw_overlay(image, rail, front_y, top, bottom, slots, segments, fit)
    output_path = output_root / f'{image_path.stem}_board.jpg'
    BASE.imwrite_unicode(output_path, overlay)
    return (fit, [slot_record(image_path.name, slot) for slot in slots], output_path)

def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
