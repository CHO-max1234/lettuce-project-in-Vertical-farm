from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
'CAM2 name matching ver1: rail is the physical t1-b1 centreline.\n\nContract\n--------\n1. Load a fixed 12-column board measured once from an empty bed.\n2. Detect only the continuous left rail and the lower front rail per image.\n3. The left rail itself is t1-b1; every measured hole line starts at t2-b2.\n4. Transfer the whole board.  Runtime dark-hole detections never move it.\n5. lettuce-seg may assign/recentre a crop bbox inside the board ownership cells,\n   but cannot change phase, pitch, or slot names.\n\nThe 30 images are processed one at a time to keep memory bounded.\n'
import argparse
import csv
import importlib.util
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
from tqdm import tqdm
SCRIPT_DIR = Path(__file__).resolve().parent
from pipeline_paths import CAM2_REGISTRATION, BRIGHT_IMAGES, NAME_ROOT
REGISTRATION_SCRIPT = CAM2_REGISTRATION
INPUT_ROOT = BRIGHT_IMAGES / 'cam2'
OUTPUT_ROOT = NAME_ROOT / 'cam2'
VER = 'CAM2_VER1'
WIDTH_WARN_PX = 100
N_PHYSICAL_COLS = 12

def load_preview_module():
    spec = importlib.util.spec_from_file_location('cam2_bed_registration', REGISTRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f'Invalid input or state in load_preview_module (rejected condition: spec is None or spec.loader is None): {REGISTRATION_SCRIPT} ')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
PREVIEW = load_preview_module()
RAIL = PREVIEW.RAIL
BASE = PREVIEW.BASE

def make_ver1_reference_board() -> tuple[np.ndarray, np.ndarray]:
    """Prepend the rail centreline and shift preview names by exactly one."""
    top_y = float(PREVIEW.REF_TOP[0, 1])
    bottom_y = float(PREVIEW.REF_BOTTOM[0, 1])
    rail_top = np.asarray([PREVIEW.REF_RAIL_A * top_y + PREVIEW.REF_RAIL_B, top_y], dtype=np.float64)
    rail_bottom = np.asarray([PREVIEW.REF_RAIL_A * bottom_y + PREVIEW.REF_RAIL_B, bottom_y], dtype=np.float64)
    top = np.vstack([rail_top, PREVIEW.REF_TOP[:N_PHYSICAL_COLS - 1]])
    bottom = np.vstack([rail_bottom, PREVIEW.REF_BOTTOM[:N_PHYSICAL_COLS - 1]])
    return (top, bottom)
(REF_TOP, REF_BOTTOM) = make_ver1_reference_board()

@dataclass
class SummaryRow:
    image_name: str
    source_path: str
    output_path: str
    width: int
    height: int
    width_delta_px: int
    width_scale: float
    front_rail_y: float | str
    front_rail_mode: str
    rail_a: float | str
    rail_b: float | str
    rail_front_x: float | str
    rail_candidate_count: int
    visible_cols: int
    seg_count: int
    assigned: int
    unassigned: int
    merge_review: int
    qc: str
    error: str

def transfer_point(point: np.ndarray, rail, front_y: float, width_scale: float) -> np.ndarray:
    (x_ref, y_ref) = (float(v) for v in point)
    y_target = y_ref + (front_y - PREVIEW.REF_FRONT_RAIL_Y)
    ref_rail_x = PREVIEW.REF_RAIL_A * y_ref + PREVIEW.REF_RAIL_B
    target_rail_x = rail.x_at(y_target)
    x_target = target_rail_x + width_scale * (x_ref - ref_rail_x)
    return np.asarray([x_target, y_target], dtype=np.float64)

def transfer_board(rail, front_y: float, width_scale: float) -> tuple[np.ndarray, np.ndarray]:
    top = np.asarray([transfer_point(point, rail, front_y, width_scale) for point in REF_TOP])
    bottom = np.asarray([transfer_point(point, rail, front_y, width_scale) for point in REF_BOTTOM])
    return (top, bottom)

def detect_front_rail_y(image: np.ndarray) -> tuple[float, float, str]:
    """Detect the rail top edge and reject a near-bottom secondary edge.

    Dense late-stage leaves can make the lower edge/background transition
    stronger than the rail's upper edge.  Only that specific near-bottom case
    activates the row-mean brightness-drop fallback.
    """
    (y, score) = PREVIEW.detect_front_rail_y(image)
    (height, width) = image.shape[:2]
    if y <= 0.9 * height:
        return (y, score, 'SOBEL_CONTINUITY')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (x1, x2) = (int(round(0.08 * width)), int(round(0.95 * width)))
    row_mean = np.mean(gray[:, x1:x2], axis=1)
    smooth = np.convolve(row_mean, np.ones(7, dtype=np.float64) / 7.0, mode='same')
    lo = max(6, int(round(0.55 * height)))
    hi = min(height - 6, int(round(0.88 * height)))
    candidates = range(lo, hi)
    fallback_y = max(candidates, key=lambda row: smooth[row - 5] - smooth[row + 5])
    brightness_drop = float(smooth[fallback_y - 5] - smooth[fallback_y + 5])
    if brightness_drop >= 8.0:
        return (float(fallback_y), brightness_drop, 'UPPER_EDGE_FALLBACK')
    return (y, score, 'SOBEL_CONTINUITY_NEAR_BOTTOM')

def make_slots(image: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> list:
    """Build two ownership cells per physical column from the fixed board."""
    (height, width) = image.shape[:2]
    (top_bound, bottom_bound) = PREVIEW.boundary_lines(top, bottom)
    slots: list = []
    for index in range(N_PHYSICAL_COLS):
        for (row, v1, v2, center) in (('t', -0.5, 0.5, top[index]), ('b', 0.5, 1.5, bottom[index])):
            polygon = np.asarray([PREVIEW.line_point(top_bound[index], bottom_bound[index], v1), PREVIEW.line_point(top_bound[index + 1], bottom_bound[index + 1], v1), PREVIEW.line_point(top_bound[index + 1], bottom_bound[index + 1], v2), PREVIEW.line_point(top_bound[index], bottom_bound[index], v2)], dtype=np.float32)
            area = max(RAIL.polygon_area(polygon), 1.0)
            inside = RAIL.clipped_polygon_area(polygon, width, height) / area
            if inside < 0.15:
                continue
            slots.append(RAIL.RailSlot(row=row, local_col=index + 1, slot_name=f'{row}{index + 1}', x_prior=float(center[0]), y_prior=float(center[1]), cell_polygon=polygon, cell_inside_image_ratio=float(inside), status='BOARD_PRIOR'))
    return slots

def label(canvas: np.ndarray, text: str, point: np.ndarray, color: tuple[int, int, int], dy: int) -> None:
    x = int(round(point[0])) + 6
    y = int(round(point[1])) + dy
    RAIL.draw_label(canvas, text, x, y, color)

def draw_overlay(image: np.ndarray, rail, front_y: float, top: np.ndarray, bottom: np.ndarray, slots: Sequence, segments: Sequence[dict], summary: SummaryRow) -> np.ndarray:
    (height, width) = image.shape[:2]
    header_h = 68
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[header_h:] = image
    cv2.line(canvas, (0, int(round(front_y)) + header_h), (width - 1, int(round(front_y)) + header_h), (255, 190, 0), 3, cv2.LINE_AA)
    segment_lookup = {int(segment['seg_id']): segment for segment in segments}
    for slot in slots:
        polygon = np.round(slot.cell_polygon + np.asarray([0, header_h])).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (155, 155, 155), 1, cv2.LINE_AA)
        if slot.seg_id is None:
            continue
        color = RAIL.PALETTE[(slot.local_col - 1) % len(RAIL.PALETTE)]
        segment = segment_lookup[slot.seg_id]
        seg_polygon = np.round(segment['polygon'] + np.asarray([0, header_h])).astype(np.int32)
        cv2.polylines(canvas, [seg_polygon], True, color, 3, cv2.LINE_AA)
        if slot.bbox is not None:
            (x1, y1, x2, y2) = slot.bbox
            cv2.rectangle(canvas, (x1, y1 + header_h), (x2, y2 + header_h), color, 3)
    for (index, (top_point, bottom_point)) in enumerate(zip(top, bottom), start=1):
        shifted_top = top_point + np.asarray([0.0, header_h])
        shifted_bottom = bottom_point + np.asarray([0.0, header_h])
        p1 = tuple(np.round(shifted_top).astype(int))
        p2 = tuple(np.round(shifted_bottom).astype(int))
        line_color = (0, 105, 255) if index == 1 else (150, 55, 195)
        thickness = 7 if index == 1 else 4
        cv2.line(canvas, p1, p2, line_color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, line_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 5, line_color, -1, cv2.LINE_AA)
        if -45 <= top_point[0] <= width + 45:
            label(canvas, f't{index}', shifted_top, line_color, -8)
        if -45 <= bottom_point[0] <= width + 45:
            label(canvas, f'b{index}', shifted_bottom, line_color, 24)
    header1 = f'{VER} | RAIL_IS_T1_B1 | cols={summary.visible_cols} | seg={summary.seg_count} | assigned={summary.assigned} | merge={summary.merge_review} | QC={summary.qc}'
    header2 = f'width={summary.width} ref={PREVIEW.REF_WIDTH} delta={summary.width_delta_px}px scale={summary.width_scale:.5f} | rail_a={summary.rail_a} | front_y={summary.front_rail_y} | orange=t1-b1, purple=t2+ fixed board, thick contour/box=lettuce-seg'
    cv2.putText(canvas, header1, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, header2, (8, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (215, 215, 215), 1, cv2.LINE_AA)
    return canvas

def draw_failure(image: np.ndarray, image_name: str, error: str) -> np.ndarray:
    header_h = 68
    (height, width) = image.shape[:2]
    canvas = np.zeros((height + header_h, width, 3), dtype=np.uint8)
    canvas[header_h:] = image
    cv2.putText(canvas, f'{VER} | REVIEW | {image_name}', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (0, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, error[:220], (8, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 160, 255), 1, cv2.LINE_AA)
    return canvas

def slot_record(image_name: str, slot) -> dict:
    return {'image_name': image_name, 'slot_name': slot.slot_name, 'row': slot.row, 'local_col': slot.local_col, 'x_prior': round(slot.x_prior, 3), 'y_prior': round(slot.y_prior, 3), 'inside_ratio': round(slot.cell_inside_image_ratio, 4), 'status': slot.status, 'seg_id': '' if slot.seg_id is None else slot.seg_id, 'seg_conf': '' if not np.isfinite(slot.seg_conf) else round(slot.seg_conf, 5), 'seg_center_x': '' if not np.isfinite(slot.seg_center_x) else round(slot.seg_center_x, 3), 'seg_center_y': '' if not np.isfinite(slot.seg_center_y) else round(slot.seg_center_y, 3), 'mask_coverage': '' if not np.isfinite(slot.mask_coverage) else round(slot.mask_coverage, 5), 'runner_up_coverage': '' if not np.isfinite(slot.runner_up_coverage) else round(slot.runner_up_coverage, 5), 'center_distance': '' if not np.isfinite(slot.center_distance) else round(slot.center_distance, 5), 'bbox': '' if slot.bbox is None else ','.join((str(value) for value in slot.bbox))}

def process_one(model, image_path: Path, output_root: Path) -> tuple[SummaryRow, list[dict]]:
    image = BASE.imread_unicode(image_path)
    (height, width) = image.shape[:2]
    output_path = output_root / f'{image_path.stem}_ver1.jpg'
    try:
        width_delta = abs(width - PREVIEW.REF_WIDTH)
        width_scale = width / PREVIEW.REF_WIDTH
        (front_y, front_score, front_mode) = detect_front_rail_y(image)
        (rail, candidate_count) = PREVIEW.choose_left_rail(image, front_y)
        (top, bottom) = transfer_board(rail, front_y, width_scale)
        slots = make_slots(image, top, bottom)
        segments = BASE.predict_segments(model, image)
        top_pitch = float(np.median(np.linalg.norm(np.diff(top, axis=0), axis=1)))
        bottom_pitch = float(np.median(np.linalg.norm(np.diff(bottom, axis=0), axis=1)))
        pitch = 0.5 * (top_pitch + bottom_pitch)
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
        if front_mode == 'SOBEL_CONTINUITY_NEAR_BOTTOM':
            qc_flags.append('FRONT_RAIL_NEAR_BOTTOM_REVIEW')
        if assignment['unassigned']:
            qc_flags.append('UNASSIGNED_SEG')
        if assignment['merge_review']:
            qc_flags.append('SEG_MERGE_REVIEW')
        visible_cols = max((slot.local_col for slot in slots), default=0)
        summary = SummaryRow(image_name=image_path.name, source_path=str(image_path), output_path=str(output_path), width=width, height=height, width_delta_px=width_delta, width_scale=width_scale, front_rail_y=round(front_y, 3), front_rail_mode=front_mode, rail_a=round(float(rail.a), 6), rail_b=round(float(rail.b), 3), rail_front_x=round(rail_front_x, 3), rail_candidate_count=candidate_count, visible_cols=visible_cols, seg_count=len(segments), assigned=int(assignment['assigned']), unassigned=int(assignment['unassigned']), merge_review=int(assignment['merge_review']), qc='OK' if not qc_flags else '+'.join(qc_flags), error='')
        BASE.imwrite_unicode(output_path, draw_overlay(image, rail, front_y, top, bottom, slots, segments, summary))
        return (summary, [slot_record(image_path.name, slot) for slot in slots])
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        summary = SummaryRow(image_name=image_path.name, source_path=str(image_path), output_path=str(output_path), width=width, height=height, width_delta_px=abs(width - PREVIEW.REF_WIDTH), width_scale=width / PREVIEW.REF_WIDTH, front_rail_y='', front_rail_mode='', rail_a='', rail_b='', rail_front_x='', rail_candidate_count=0, visible_cols=0, seg_count=0, assigned=0, unassigned=0, merge_review=0, qc='PROCESS_REVIEW', error=error)
        BASE.imwrite_unicode(output_path, draw_failure(image, image_path.name, error))
        traceback.print_exc()
        return (summary, [])

def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CAM2 name matching ver1, rail=t1-b1')
    parser.add_argument('--input-root', type=Path, default=INPUT_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image_paths = sorted((p for p in args.input_root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and ('_cam2' in p.stem.lower())))
    if not image_paths:
        raise RuntimeError(f'Invalid input or state in main (rejected condition: not image_paths): {len(image_paths)} {args.input_root} ')
    if BASE.sha256_file(BASE.MODEL_PATH) != BASE.MODEL_SHA256:
        raise RuntimeError('Invalid input or state in main (rejected condition: BASE.sha256_file(BASE.MODEL_PATH) != BASE.MODEL_SHA256)')
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = BASE.YOLO(str(BASE.MODEL_PATH))
    summaries: list[dict] = []
    slot_rows: list[dict] = []
    for image_path in tqdm(image_paths, desc=VER, unit='image'):
        (summary, rows) = process_one(model, image_path, args.output_root)
        summaries.append(asdict(summary))
        slot_rows.extend(rows)
    write_csv(args.output_root / 'cam2_ver1_summary.csv', summaries)
    write_csv(args.output_root / 'cam2_ver1_slots.csv', slot_rows)
    review_count = sum((row['qc'] != 'OK' for row in summaries))
    process_failures = sum((row['qc'] == 'PROCESS_REVIEW' for row in summaries))
    print(f'images={len(image_paths)} overlays={len(summaries)} slot_rows={len(slot_rows)} review={review_count} process_failures={process_failures} output={args.output_root}')
if __name__ == '__main__':
    main()
