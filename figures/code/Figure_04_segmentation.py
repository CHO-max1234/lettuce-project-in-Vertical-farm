from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from ultralytics import YOLO


# =============================================================================
# 0. 경로: 정식 ver1 전의 3장 맛보기(preview_3)
# =============================================================================
SOURCE_ROOT = Path(
    r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브"
    r"\RGB_정면\2. ROI_BOX\2작기\0. ROI BOX 원본\ver2_cam2"
)
OUTPUT_ROOT = Path(
    r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브"
    r"\전처리,모델 성능개선\260820_0826_1.상추이름 매칭통일_개선"
    r"\2. cam2_test\preview_3_image"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / 'main_cycle(260306_260504)' / 'artifacts' / 'segmentation' / 'plant.pt'
MODEL_SHA256 = "A2DED62BBA4D577F4EAD18907B824C587E179357C8089B72DABDE41B755D6FE8"

SAMPLE_RELATIVE_PATHS = (
    Path("260308") / "bed08_20260308_104022_cam2_rf.jpg",
    Path("260403") / "bed45_20260403_125348_cam2_rf.jpg",
    Path("260504") / "bed55_20260504_023605_cam2_rf.jpg",
)

# 정식 ver1에서는 지정된 testimages 30장만 회귀 표본으로 사용한다.
FUTURE_TESTIMAGES_ROOT = Path(
    r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브"
    r"\전처리,모델 성능개선\260820_0826_1.상추이름 매칭통일_개선"
    r"\2. cam2_test\testimages"
)


# =============================================================================
# 1. CAM2 preview 파라미터
#    - 열 수 6/7/12 고정 없음
#    - matched_count==4 통과 gate 없음
#    - crop 폭 기반 고정 fallback 없음
#    - lettuce-seg 개수로 열 수를 정하지 않음
# =============================================================================
SEG_IMGSZ = 1280
SEG_CONF = 0.20
SEG_IOU = 0.70
SEG_MIN_AREA = 80.0

LEFT_HOLE_SEARCH_RATIO = 0.32
DARK_VALUE_THRESHOLD = 75
HOLE_MIN_CIRCULARITY = 0.48
HOLE_MAX_MEAN_SATURATION = 115.0
PAIR_MIN_DX_RATIO = 0.065
PAIR_MAX_DX_RATIO = 0.16
PAIR_MAX_DY_RATIO = 0.045
TOP_BOTTOM_SPLIT_RATIO = 0.58
RIGHT_FULL_CELL_MARGIN = 0.45
BOTTOM_PITCH_FROM_TOP = 1.06
BOTTOM_X_SHIFT_RATIO = -0.045

ASSIGN_MAX_DX_PITCH = 0.56
ASSIGN_MAX_DY_ROW_GAP = 0.62
SEG_CENTER_MAX_SHIFT_PITCH = 0.20
BBOX_PAD_RATIO = 0.08
BBOX_MIN_WIDTH_PITCH = 0.54
BBOX_MIN_HEIGHT_ROW_GAP = 0.52

PALETTE = (
    (40, 40, 255),
    (255, 220, 0),
    (0, 170, 255),
    (255, 80, 0),
    (0, 255, 80),
    (255, 0, 230),
    (60, 210, 255),
    (200, 80, 255),
    (0, 230, 230),
    (120, 255, 80),
    (255, 140, 80),
    (80, 140, 255),
)


@dataclass
class HoleEvidence:
    x: float
    y: float
    area: float
    circularity: float
    mean_saturation: float
    mean_value: float


@dataclass
class Slot:
    slot_name: str
    row: str
    local_col: int
    x_prior: float
    y_prior: float
    x: float
    y: float
    visibility: str = "UNKNOWN"
    evidence_kind: str = "geometry_prior"
    cell_bbox: Optional[tuple[int, int, int, int]] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    seg_id: Optional[int] = None
    seg_conf: float = float("nan")
    seg_center_x: float = float("nan")
    seg_center_y: float = float("nan")
    seg_area: float = float("nan")
    seg_bbox: Optional[tuple[float, float, float, float]] = None
    seg_status: str = "NO_SEG_PRIOR_CELL"


def imread_unicode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽지 못했습니다: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"이미지 인코딩 실패: {path}")
    encoded.tofile(str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def predict_segments(model: YOLO, image: np.ndarray) -> list[dict]:
    result = model.predict(
        source=image,
        imgsz=SEG_IMGSZ,
        conf=SEG_CONF,
        iou=SEG_IOU,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.boxes is None or result.masks is None or result.masks.xy is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    polygons = result.masks.xy
    segments: list[dict] = []
    for index, box in enumerate(boxes):
        if index >= len(polygons):
            continue
        polygon = np.asarray(polygons[index], dtype=np.float32)
        if polygon.ndim != 2 or len(polygon) < 3:
            continue
        area = float(abs(cv2.contourArea(polygon)))
        if area < SEG_MIN_AREA:
            continue
        contour = polygon.reshape(-1, 1, 2)
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
        segments.append(
            {
                "seg_id": len(segments) + 1,
                "bbox": tuple(float(value) for value in box),
                "cx": cx,
                "cy": cy,
                "area": area,
                "conf": float(confs[index]),
                "polygon": polygon,
            }
        )
    return segments


def detect_dark_holes(image: np.ndarray) -> list[HoleEvidence]:
    """왼쪽 bed 영역의 저채도·저명도 원형 구멍 후보만 찾는다."""
    height, width = image.shape[:2]
    roi_width = int(round(width * LEFT_HOLE_SEARCH_RATIO))
    roi = image[:, :roi_width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    mask = (value < DARK_VALUE_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 세로 crop이 큰 중·후기 영상에서는 같은 실제 구멍의 상대 면적이 작아진다.
    # 절대 최소값과 완만한 영상비 기준을 함께 써서 정상 구멍을 보존한다.
    min_area = max(150.0, 0.00065 * height * width)
    max_area = 0.030 * height * width
    found: list[HoleEvidence] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < HOLE_MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if not 0.35 <= aspect <= 2.8:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        local_mask = np.zeros(value.shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [contour], -1, 255, -1)
        pixels = hsv[local_mask > 0]
        mean_saturation = float(pixels[:, 1].mean())
        mean_value = float(pixels[:, 2].mean())
        if mean_saturation > HOLE_MAX_MEAN_SATURATION:
            continue
        found.append(
            HoleEvidence(cx, cy, area, circularity, mean_saturation, mean_value)
        )
    return sorted(found, key=lambda item: (item.y, item.x))


def select_row_pair(
    candidates: Iterable[HoleEvidence],
    width: int,
    height: int,
    row: str,
) -> Optional[tuple[HoleEvidence, HoleEvidence]]:
    items = sorted([
        item
        for item in candidates
        if item.x <= 0.24 * width
        and (
            (0.14 * height <= item.y < TOP_BOTTOM_SPLIT_RATIO * height)
            if row == "t"
            else (TOP_BOTTOM_SPLIT_RATIO * height <= item.y <= 0.96 * height)
        )
    ], key=lambda item: item.x)
    best: Optional[tuple[float, HoleEvidence, HoleEvidence]] = None
    for left_index, left in enumerate(items):
        for right in items[left_index + 1 :]:
            if right.x <= left.x:
                continue
            dx = right.x - left.x
            dy = abs(right.y - left.y)
            if not PAIR_MIN_DX_RATIO * width <= dx <= PAIR_MAX_DX_RATIO * width:
                continue
            if dy > PAIR_MAX_DY_RATIO * height:
                continue
            pitch_penalty = abs(dx / width - 0.098) * 20.0
            level_penalty = dy / height * 18.0
            left_penalty = left.x / width * 0.5
            quality_bonus = 0.15 * (left.circularity + right.circularity)
            score = pitch_penalty + level_penalty + left_penalty - quality_bonus
            if best is None or score < best[0]:
                best = (score, left, right)
    return None if best is None else (best[1], best[2])


def pick_bottom_single(
    candidates: Iterable[HoleEvidence],
    width: int,
    height: int,
    top_second_x: float,
) -> Optional[HoleEvidence]:
    predicted_b2 = top_second_x + BOTTOM_X_SHIFT_RATIO * width
    items = [
        item
        for item in candidates
        if item.y >= TOP_BOTTOM_SPLIT_RATIO * height and item.x <= 0.30 * width
    ]
    if not items:
        return None
    return min(items, key=lambda item: abs(item.x - predicted_b2))


def fit_line(points: Iterable[tuple[float, float]], fallback_y: float):
    data = np.asarray(list(points), dtype=np.float64)
    if len(data) < 2 or np.ptp(data[:, 0]) < 1.0:
        return lambda x: float(fallback_y)
    slope, intercept = np.polyfit(data[:, 0], data[:, 1], 1)
    return lambda x: float(slope * x + intercept)


def build_variable_lattice(
    image: np.ndarray,
    holes: list[HoleEvidence],
) -> tuple[list[Slot], dict]:
    """왼쪽 실제 두 빈 열로 시작하고, 영상 경계까지 열 수를 동적으로 센다."""
    height, width = image.shape[:2]
    top_pair = select_row_pair(holes, width, height, "t")
    bottom_pair = select_row_pair(holes, width, height, "b")
    qc_flags: list[str] = []
    if top_pair is None:
        return [], {
            "status": "CAM2_LEFT_ANCHOR_REVIEW",
            "qc_flags": ["TOP_LEFT_TWO_HOLES_NOT_FOUND"],
        }

    top_1, top_2 = top_pair
    top_pitch = float(top_2.x - top_1.x)
    top_y = fit_line(((top_1.x, top_1.y), (top_2.x, top_2.y)), top_1.y)

    if bottom_pair is not None:
        bottom_1, bottom_2 = bottom_pair
        bottom_pitch = float(bottom_2.x - bottom_1.x)
        bottom_y = fit_line(
            ((bottom_1.x, bottom_1.y), (bottom_2.x, bottom_2.y)), bottom_1.y
        )
        bottom_anchor_mode = "two_holes"
    else:
        bottom_2 = pick_bottom_single(holes, width, height, top_2.x)
        if bottom_2 is None:
            return [], {
                "status": "CAM2_LEFT_ANCHOR_REVIEW",
                "qc_flags": ["BOTTOM_LEFT_HOLE_NOT_FOUND"],
            }
        bottom_pitch = top_pitch * BOTTOM_PITCH_FROM_TOP
        bottom_1 = HoleEvidence(
            x=bottom_2.x - bottom_pitch,
            y=bottom_2.y,
            area=bottom_2.area,
            circularity=bottom_2.circularity,
            mean_saturation=bottom_2.mean_saturation,
            mean_value=bottom_2.mean_value,
        )
        bottom_y = lambda x: float(bottom_2.y)
        bottom_anchor_mode = "b2_plus_top_pitch_prior"
        qc_flags.append("BOTTOM_FIRST_HOLE_INFERRED")

    mean_origin = 0.5 * (top_1.x + bottom_1.x)
    mean_pitch = 0.5 * (top_pitch + bottom_pitch)
    right_limit = width - RIGHT_FULL_CELL_MARGIN * mean_pitch
    visible_count = max(1, int(math.floor((right_limit - mean_origin) / mean_pitch)) + 1)

    slots: list[Slot] = []
    for local_col in range(1, visible_count + 1):
        top_x = top_1.x + (local_col - 1) * top_pitch
        bottom_x = bottom_1.x + (local_col - 1) * bottom_pitch
        for row, x, y in (
            ("t", top_x, top_y(top_x)),
            ("b", bottom_x, bottom_y(bottom_x)),
        ):
            slots.append(
                Slot(
                    slot_name=f"{row}{local_col}",
                    row=row,
                    local_col=local_col,
                    x_prior=float(x),
                    y_prior=float(y),
                    x=float(x),
                    y=float(y),
                )
            )

    return slots, {
        "status": "CAM2_SLOT_MAP_OK" if not qc_flags else "CAM2_SLOT_MAP_REVIEW",
        "qc_flags": qc_flags,
        "visible_count": visible_count,
        "top_pitch": top_pitch,
        "bottom_pitch": bottom_pitch,
        "mean_pitch": mean_pitch,
        "top_anchor": ((top_1.x, top_1.y), (top_2.x, top_2.y)),
        "bottom_anchor": ((bottom_1.x, bottom_1.y), (bottom_2.x, bottom_2.y)),
        "bottom_anchor_mode": bottom_anchor_mode,
    }


def nearest_hole_for_slot(
    slot: Slot,
    holes: Iterable[HoleEvidence],
    pitch: float,
    row_gap: float,
) -> Optional[HoleEvidence]:
    candidates = []
    for hole in holes:
        dx = abs(hole.x - slot.x_prior) / max(pitch, 1.0)
        dy = abs(hole.y - slot.y_prior) / max(row_gap, 1.0)
        if dx <= 0.30 and dy <= 0.28:
            candidates.append((math.hypot(dx, dy), hole))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def build_cell_bboxes(slots: list[Slot], width: int, height: int) -> None:
    by_row = {
        row: sorted((slot for slot in slots if slot.row == row), key=lambda slot: slot.local_col)
        for row in ("t", "b")
    }
    by_col = {
        col: {slot.row: slot for slot in slots if slot.local_col == col}
        for col in sorted({slot.local_col for slot in slots})
    }
    for row, row_slots in by_row.items():
        xs = [slot.x_prior for slot in row_slots]
        boundaries = [0.0] + [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])] + [float(width)]
        for index, slot in enumerate(row_slots):
            pair = by_col[slot.local_col]
            split_y = 0.5 * (pair["t"].y_prior + pair["b"].y_prior)
            y1, y2 = (0.0, split_y) if row == "t" else (split_y, float(height))
            slot.cell_bbox = (
                int(round(np.clip(boundaries[index], 0, width))),
                int(round(np.clip(y1, 0, height))),
                int(round(np.clip(boundaries[index + 1], 0, width))),
                int(round(np.clip(y2, 0, height))),
            )
            slot.bbox = slot.cell_bbox


def assign_segments(
    slots: list[Slot],
    segments: list[dict],
    holes: list[HoleEvidence],
    pitch: float,
) -> dict:
    if not slots:
        return {"assigned": 0, "unassigned": len(segments)}
    row_gaps = [
        b.y_prior - t.y_prior
        for t in slots
        for b in slots
        if t.row == "t" and b.row == "b" and t.local_col == b.local_col
    ]
    row_gap = max(1.0, float(np.median(row_gaps)))

    for slot in slots:
        hole = nearest_hole_for_slot(slot, holes, pitch, row_gap)
        if hole is not None:
            slot.visibility = "EMPTY"
            slot.evidence_kind = "hole"

    cost = np.full((len(slots), len(segments)), 1e6, dtype=np.float64)
    for slot_index, slot in enumerate(slots):
        for seg_index, segment in enumerate(segments):
            dx = abs(segment["cx"] - slot.x_prior) / max(pitch, 1.0)
            dy = abs(segment["cy"] - slot.y_prior) / row_gap
            expected_row = "t" if segment["cy"] < 0.5 * (
                slots[0].y_prior + next(s.y_prior for s in slots if s.row == "b")
            ) else "b"
            if expected_row != slot.row:
                continue
            if dx > ASSIGN_MAX_DX_PITCH or dy > ASSIGN_MAX_DY_ROW_GAP:
                continue
            contains_prior = (
                segment["bbox"][0] <= slot.x_prior <= segment["bbox"][2]
                and segment["bbox"][1] <= slot.y_prior <= segment["bbox"][3]
            )
            cost[slot_index, seg_index] = math.hypot(dx / 0.56, dy / 0.62) - (
                0.16 if contains_prior else 0.0
            ) - 0.06 * segment["conf"]

    row_indices, col_indices = linear_sum_assignment(cost)
    segment_by_slot: dict[int, dict] = {}
    for slot_index, seg_index in zip(row_indices, col_indices):
        if cost[slot_index, seg_index] >= 1e5:
            continue
        slot = slots[slot_index]
        segment = segments[seg_index]
        segment_by_slot[id(slot)] = segment
        slot.visibility = "VISIBLE"
        slot.evidence_kind = "lettuce_seg"
        slot.seg_id = int(segment["seg_id"])
        slot.seg_conf = float(segment["conf"])
        slot.seg_center_x = float(segment["cx"])
        slot.seg_center_y = float(segment["cy"])
        slot.seg_area = float(segment["area"])
        slot.seg_bbox = tuple(segment["bbox"])
        slot.seg_status = "SEG_ASSIGNED"
        dx = float(np.clip(segment["cx"] - slot.x_prior, -SEG_CENTER_MAX_SHIFT_PITCH * pitch, SEG_CENTER_MAX_SHIFT_PITCH * pitch))
        dy = float(np.clip(segment["cy"] - slot.y_prior, -0.20 * row_gap, 0.20 * row_gap))
        slot.x = slot.x_prior + dx
        slot.y = slot.y_prior + dy

    build_cell_bboxes(slots, max(slot.cell_bbox[2] if slot.cell_bbox else 0 for slot in slots), max(slot.cell_bbox[3] if slot.cell_bbox else 0 for slot in slots))
    for slot in slots:
        segment = segment_by_slot.get(id(slot))
        if segment is None or slot.cell_bbox is None:
            continue
        cx1, cy1, cx2, cy2 = slot.cell_bbox
        sx1, sy1, sx2, sy2 = segment["bbox"]
        sw, sh = max(1.0, sx2 - sx1), max(1.0, sy2 - sy1)
        x1 = max(cx1, sx1 - BBOX_PAD_RATIO * sw)
        x2 = min(cx2, sx2 + BBOX_PAD_RATIO * sw)
        y1 = max(cy1, sy1 - BBOX_PAD_RATIO * sh)
        y2 = min(cy2, sy2 + BBOX_PAD_RATIO * sh)
        min_w = BBOX_MIN_WIDTH_PITCH * pitch
        min_h = BBOX_MIN_HEIGHT_ROW_GAP * row_gap
        if x2 - x1 < min_w:
            x1 = max(cx1, slot.x - 0.5 * min_w)
            x2 = min(cx2, slot.x + 0.5 * min_w)
        if y2 - y1 < min_h:
            y1 = max(cy1, slot.y - 0.5 * min_h)
            y2 = min(cy2, slot.y + 0.5 * min_h)
        slot.bbox = tuple(int(round(value)) for value in (x1, y1, x2, y2))

    return {
        "assigned": len(segment_by_slot),
        "unassigned": max(0, len(segments) - len(segment_by_slot)),
    }


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.62, 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(x, 0, max(0, image.shape[1] - text_w - 8)))
    y = int(np.clip(y, text_h + 5, image.shape[0] - baseline - 3))
    cv2.rectangle(image, (x, y - text_h - 5), (x + text_w + 7, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    image: np.ndarray,
    slots: list[Slot],
    segments: list[dict],
    holes: list[HoleEvidence],
    meta: dict,
    assignment: dict,
) -> np.ndarray:
    header_h = 48
    canvas = np.zeros((image.shape[0] + header_h, image.shape[1], 3), dtype=np.uint8)
    canvas[header_h:] = image
    cv2.rectangle(canvas, (1, header_h + 1), (image.shape[1] - 2, canvas.shape[0] - 2), (0, 255, 255), 2)

    segment_lookup = {segment["seg_id"]: segment for segment in segments}
    for slot in slots:
        color = PALETTE[(slot.local_col - 1) % len(PALETTE)]
        if slot.cell_bbox is not None:
            x1, y1, x2, y2 = slot.cell_bbox
            cv2.rectangle(canvas, (x1, y1 + header_h), (x2, y2 + header_h), (95, 95, 95), 1)
        if slot.seg_id is not None and slot.seg_id in segment_lookup:
            polygon = np.round(segment_lookup[slot.seg_id]["polygon"]).astype(np.int32)
            polygon[:, 1] += header_h
            cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True, color, 3, cv2.LINE_AA)
        if slot.bbox is not None:
            x1, y1, x2, y2 = slot.bbox
            cv2.rectangle(canvas, (x1, y1 + header_h), (x2, y2 + header_h), color, 3)
        center = (int(round(slot.x)), int(round(slot.y + header_h)))
        cv2.circle(canvas, center, 7, color, 3)
        state = "SEG" if slot.seg_status == "SEG_ASSIGNED" else "PRIOR"
        label_y = int(round(slot.y + header_h - 14 if slot.row == "t" else slot.y + header_h + 29))
        draw_label(canvas, f"{slot.slot_name} {state}", int(round(slot.x + 8)), label_y, color)

    for hole in holes:
        cv2.circle(canvas, (int(round(hole.x)), int(round(hole.y + header_h))), 4, (255, 255, 0), -1)

    qc = ",".join(meta.get("qc_flags", [])) or "OK"
    header = (
        f"CAM2 preview_3 | {meta['status']} | observed_cols={meta.get('visible_count', 0)} "
        f"| holes={len(holes)} | seg={len(segments)} | assigned={assignment['assigned']} "
        f"| pitch={meta.get('mean_pitch', float('nan')):.1f} | QC={qc}"
    )
    cv2.putText(canvas, header, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def serialize_bbox(value):
    return "" if value is None else ",".join(str(v) for v in value)


def process_one(model: YOLO, image_path: Path, output_root: Path) -> tuple[dict, list[dict], Path]:
    image = imread_unicode(image_path)
    holes = detect_dark_holes(image)
    segments = predict_segments(model, image)
    slots, meta = build_variable_lattice(image, holes)
    if slots:
        build_cell_bboxes(slots, image.shape[1], image.shape[0])
        assignment = assign_segments(slots, segments, holes, float(meta["mean_pitch"]))
    else:
        assignment = {"assigned": 0, "unassigned": len(segments)}
    overlay = draw_overlay(image, slots, segments, holes, meta, assignment)
    output_path = output_root / image_path.name
    imwrite_unicode(output_path, overlay)

    summary = {
        "image_name": image_path.name,
        "source_path": str(image_path),
        "overlay_path": str(output_path),
        "status": meta["status"],
        "visible_column_count": meta.get("visible_count", 0),
        "slot_count": len(slots),
        "hole_candidate_count": len(holes),
        "lettuce_segment_count": len(segments),
        "assigned_segment_count": assignment["assigned"],
        "unassigned_segment_count": assignment["unassigned"],
        "top_pitch": meta.get("top_pitch", ""),
        "bottom_pitch": meta.get("bottom_pitch", ""),
        "mean_pitch": meta.get("mean_pitch", ""),
        "bottom_anchor_mode": meta.get("bottom_anchor_mode", ""),
        "qc_flags": "|".join(meta.get("qc_flags", [])),
    }
    slot_rows = []
    for slot in slots:
        row = asdict(slot)
        row.update(
            image_name=image_path.name,
            source_path=str(image_path),
            cell_bbox=serialize_bbox(slot.cell_bbox),
            bbox=serialize_bbox(slot.bbox),
            seg_bbox=serialize_bbox(slot.seg_bbox),
        )
        slot_rows.append(row)
    return summary, slot_rows, output_path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAM2 variable-column preview on three fixed images")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    actual_hash = sha256_file(args.model_path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"lettuce-seg hash mismatch: {actual_hash}")
    image_paths = [args.source_root / relative for relative in SAMPLE_RELATIVE_PATHS]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preview sample missing: {missing}")

    model = YOLO(str(args.model_path))
    summaries: list[dict] = []
    slot_rows: list[dict] = []
    outputs: list[Path] = []
    for image_path in tqdm(image_paths, desc="CAM2 preview_3", unit="image"):
        summary, rows, output = process_one(model, image_path, args.output_root)
        summaries.append(summary)
        slot_rows.extend(rows)
        outputs.append(output)

    write_csv(args.output_root / "cam2_preview3_summary.csv", summaries)
    write_csv(args.output_root / "cam2_preview3_slots.csv", slot_rows)
    print(f"code_mode=preview_3 (next iteration is ver1)")
    print(f"future_testimages={FUTURE_TESTIMAGES_ROOT}")
    print(f"summary_rows={len(summaries)} slot_rows={len(slot_rows)}")
    for output in outputs:
        print(f"overlay={output}")


if __name__ == "__main__":
    main()
