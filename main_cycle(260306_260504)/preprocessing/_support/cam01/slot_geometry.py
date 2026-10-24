"""Top-view geometry used by the final preprocessing adapters."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[3]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
N_PHYSICAL_COLS = 12
N_ROWS = 2
LEGACY_SKIP_LEFT = 2

class Visibility(str, Enum):
    VISIBLE = 'VISIBLE'
    OCCLUDED = 'OCCLUDED'
    EMPTY = 'EMPTY'
    OUT_OF_FOV = 'OUT_OF_FOV'
    UNKNOWN = 'UNKNOWN'

class RegStatus(str, Enum):
    CONFIRMED = 'CONFIRMED'
    AMBIGUOUS = 'AMBIGUOUS'
    REVIEW = 'REVIEW'

@dataclass
class Slot:
    canonical_id: Optional[str]
    legacy_id: Optional[str]
    row: str
    col: int
    local_col: int
    physical_col: Optional[int]
    x: float
    y: float
    x_prior: float
    y_prior: float
    visibility: Visibility
    reg_status: RegStatus
    evidence_kind: str = 'template'
    evidence_score: float = 0.0
    residual_dx: float = 0.0
    residual_dy: float = 0.0
    cell_bbox: Optional[Tuple[int, int, int, int]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    seg_id: Optional[int] = None
    seg_conf: float = float('nan')
    seg_center_x: float = float('nan')
    seg_center_y: float = float('nan')
    seg_area: float = float('nan')
    seg_bbox: Optional[Tuple[float, float, float, float]] = None
    seg_status: str = 'NOT_RUN'

    def to_dict(self) -> dict:
        data = asdict(self)
        data['visibility'] = self.visibility.value
        data['reg_status'] = self.reg_status.value
        return data

def canonical_to_legacy(col: int) -> Optional[int]:
    legacy = col - LEGACY_SKIP_LEFT
    return legacy if legacy >= 1 else None

def _validated_points(points: Sequence[Tuple[float, float]], name: str) -> List[Tuple[float, float]]:
    result = [(float(x), float(y)) for (x, y) in points]
    if not result:
        raise ValueError(f'{name} is empty')
    if any((not np.isfinite(x) or not np.isfinite(y) for (x, y) in result)):
        raise ValueError(f'{name} contains non-finite coordinates')
    return result

def build_slots_from_columns(top_points: Sequence[Tuple[float, float]], bottom_points: Sequence[Tuple[float, float]], physical_cols: Sequence[Optional[int]], *, visibility_by_col: Optional[Dict[int, Visibility]]=None, visibility_by_slot: Optional[Dict[Tuple[str, int], Visibility]]=None, confirmed_cols: Optional[set]=None, evidence_by_slot: Optional[Dict[Tuple[str, int], Tuple[str, float]]]=None, out_of_fov_cols: Sequence[int]=(), legacy_crosswalk: bool=False) -> List[Slot]:
    top = _validated_points(top_points, 'top_points')
    bottom = _validated_points(bottom_points, 'bottom_points')
    physical = list(physical_cols)
    if not len(top) == len(bottom) == len(physical):
        raise ValueError('top_points, bottom_points, physical_cols length mismatch')
    visibility_by_col = visibility_by_col or {}
    visibility_by_slot = visibility_by_slot or {}
    confirmed_cols = confirmed_cols or set()
    evidence_by_slot = evidence_by_slot or {}
    slots: List[Slot] = []
    for (local_col, (top_xy, bottom_xy, physical_col_value)) in enumerate(zip(top, bottom, physical), start=1):
        physical_col = None if physical_col_value is None else int(physical_col_value)
        if physical_col is not None and (not 1 <= physical_col <= N_PHYSICAL_COLS):
            raise ValueError(f'physical_col out of range: {physical_col}')
        for (row, (x, y)) in (('t', top_xy), ('b', bottom_xy)):
            vis = visibility_by_slot.get((row, local_col), visibility_by_col.get(local_col, Visibility.UNKNOWN))
            if physical_col is None:
                canonical_id = None
                legacy_id = None
                reg_status = RegStatus.REVIEW
                compat_col = local_col
            else:
                canonical_id = f'{row}{physical_col}'
                legacy = canonical_to_legacy(physical_col) if legacy_crosswalk else None
                legacy_id = f'{row}{legacy}' if legacy is not None else None
                reg_status = RegStatus.CONFIRMED if physical_col in confirmed_cols else RegStatus.REVIEW
                compat_col = physical_col
            (kind, score) = evidence_by_slot.get((row, local_col), ('template', 0.0))
            slots.append(Slot(canonical_id=canonical_id, legacy_id=legacy_id, row=row, col=int(compat_col), local_col=int(local_col), physical_col=physical_col, x=x, y=y, x_prior=x, y_prior=y, visibility=vis, reg_status=reg_status, evidence_kind=str(kind), evidence_score=float(score)))
    for physical_col in sorted(set((int(c) for c in out_of_fov_cols))):
        if not 1 <= physical_col <= N_PHYSICAL_COLS:
            continue
        for row in ('t', 'b'):
            slots.append(Slot(canonical_id=f'{row}{physical_col}', legacy_id=None, row=row, col=physical_col, local_col=0, physical_col=physical_col, x=float('nan'), y=float('nan'), x_prior=float('nan'), y_prior=float('nan'), visibility=Visibility.OUT_OF_FOV, reg_status=RegStatus.REVIEW, evidence_kind='out_of_fov'))
    return slots

def apply_bounded_residual(slot: Slot, evidence_x: Optional[float], evidence_y: Optional[float], *, pitch: float, max_shift_ratio: float=0.22, reject_distance_ratio: float=0.55) -> Slot:
    if evidence_x is None or evidence_y is None:
        return slot
    if not np.isfinite(evidence_x) or not np.isfinite(evidence_y):
        return slot
    dx_raw = float(evidence_x - slot.x_prior)
    dy_raw = float(evidence_y - slot.y_prior)
    if float(np.hypot(dx_raw, dy_raw)) > reject_distance_ratio * pitch:
        slot.reg_status = RegStatus.REVIEW
        return slot
    max_shift = max_shift_ratio * pitch
    dx = float(np.clip(dx_raw, -max_shift, max_shift))
    dy = float(np.clip(dy_raw, -max_shift, max_shift))
    slot.x = slot.x_prior + dx
    slot.y = slot.y_prior + dy
    slot.residual_dx = dx
    slot.residual_dy = dy
    return slot

def build_cell_bboxes(slots: Sequence[Slot], img_w: int, img_h: int, *, bed_bbox: Optional[Tuple[int, int, int, int]]=None, outer_margin_ratio: float=0.55) -> List[Slot]:
    by_col: Dict[int, Dict[str, Slot]] = {}
    for slot in slots:
        if slot.local_col <= 0 or not np.isfinite(slot.x) or (not np.isfinite(slot.y)):
            continue
        by_col.setdefault(slot.local_col, {})[slot.row] = slot
    ordered = [c for c in sorted(by_col) if 't' in by_col[c] and 'b' in by_col[c]]
    if not ordered:
        return list(slots)
    centers = [0.5 * (by_col[c]['t'].x + by_col[c]['b'].x) for c in ordered]
    if bed_bbox is None:
        (bed_x1, bed_y1, bed_x2, bed_y2) = (0.0, 0.0, float(img_w), float(img_h))
    else:
        (bx1, by1, bx2, by2) = bed_bbox
        bed_x1 = float(np.clip(bx1, 0, img_w))
        bed_y1 = float(np.clip(by1, 0, img_h))
        bed_x2 = float(np.clip(bx2, bed_x1, img_w))
        bed_y2 = float(np.clip(by2, bed_y1, img_h))
    if len(centers) == 1:
        left_edges = [bed_x1]
        right_edges = [bed_x2]
    else:
        mids = [0.5 * (a + b) for (a, b) in zip(centers[:-1], centers[1:])]
        if bed_bbox is None:
            first_pitch = centers[1] - centers[0]
            last_pitch = centers[-1] - centers[-2]
            bed_x1 = centers[0] - outer_margin_ratio * first_pitch
            bed_x2 = centers[-1] + outer_margin_ratio * last_pitch
        left_edges = [bed_x1] + mids
        right_edges = mids + [bed_x2]
    for (idx, local_col) in enumerate(ordered):
        top = by_col[local_col]['t']
        bottom = by_col[local_col]['b']
        x1 = int(round(np.clip(left_edges[idx], 0, img_w)))
        x2 = int(round(np.clip(right_edges[idx], 0, img_w)))
        row_mid = 0.5 * (top.y + bottom.y)
        row_gap = max(1.0, bottom.y - top.y)
        if bed_bbox is None:
            bed_y1 = top.y - outer_margin_ratio * row_gap
            bed_y2 = bottom.y + outer_margin_ratio * row_gap
        top_y1 = int(round(np.clip(bed_y1, 0, img_h)))
        mid_y = int(round(np.clip(row_mid, 0, img_h)))
        bot_y2 = int(round(np.clip(bed_y2, 0, img_h)))
        top.cell_bbox = (x1, top_y1, x2, mid_y)
        bottom.cell_bbox = (x1, mid_y, x2, bot_y2)
        top.bbox = top.cell_bbox
        bottom.bbox = bottom.cell_bbox
    return list(slots)

def imread_unicode(path):
    try:
        import cv2
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path, img) -> bool:
    try:
        import cv2
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        (ok, encoded) = cv2.imencode(path.suffix or '.jpg', img)
        if not ok:
            return False
        encoded.tofile(str(path))
        return True
    except Exception:
        return False
