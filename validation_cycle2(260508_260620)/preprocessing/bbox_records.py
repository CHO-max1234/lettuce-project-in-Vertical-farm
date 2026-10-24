"""Target-preserving bounding-box records and optional diagnostic overlays."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CYCLE.BBOX_OUTPUT
TEST_ROOT = DATA_ROOT / 'testimages'
UPSTREAM_ROOT = DATA_ROOT / 'test_results'
OUTPUT_ROOT = DATA_ROOT / 'testimages_bbox'
SAMPLE_MANIFEST = UPSTREAM_ROOT / 'sample_manifest.csv'
SLOT_MANIFEST = UPSTREAM_ROOT / 'slot_manifest.csv'
FRAME_MANIFEST = UPSTREAM_ROOT / 'frame_summary.csv'
SEGMENT_MANIFEST = UPSTREAM_ROOT / 'segment_manifest.csv'
SEG_MARGIN_RATIO = 0.1
MIN_MARGIN_PX = 12
MIN_CELL_SCALE = 1.1
MAX_CELL_SCALE = 1.45
NEIGHBOR_DISTANCE_RATIO = 0.85
NEIGHBOR_DISTANCE_MARGIN = 0.1
MAX_ROW_GAP_DISTANCE = 0.65
MAX_ANCHOR_MASK_DISTANCE_RATIO = 0.2
OUTWARD_ROW_DISTANCE_RATIO = 0.2
PRIOR_MASK_DISTANCE_RATIO = 0.12
VERSION = 'S3_bbox_context_target_preserved_v1'
import csv
import hashlib
import json
import math
import cv2
import numpy as np
cv2.setNumThreads(1)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    fields = list(dict.fromkeys((k for r in rows for k in r)))
    temp = path.with_suffix('.writing')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)

def number(row, key):
    v = float(row[key])
    if not math.isfinite(v):
        raise ValueError(f'Invalid input or state in number (not math.isfinite(v)): {key} ')
    return v

def imread(path):
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f'Invalid input or state in imread (result is None): {path} ')
    return result

def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    (ok, data) = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Invalid input or state in imwrite (not ok): {path} ')
    tmp = path.with_suffix('.writing')
    data.tofile(str(tmp))
    tmp.replace(path)

def centered_bounds(center, size, limit):
    size = min(limit, max(1, int(math.ceil(size))))
    low = int(round(center - size / 2))
    low = max(0, min(limit - size, low))
    return (low, low + size)

def row_y_at(slots, row, x):
    points = [(number(r, 'x_prior'), number(r, 'y_prior')) for r in slots if r['row'] == row]
    (xs, ys) = np.array(points, dtype=float).T
    if len(points) < 2 or np.ptp(xs) < 1:
        return float(np.median(ys))
    (slope, intercept) = np.polyfit(xs, ys, 1)
    return float(slope * x + intercept)

def assess_target(slot, neighbors):
    """SEGの中心・所属行・maskの中心支持を検査する。失敗時も元データは残す。"""
    if not slot.get('seg_id'):
        return dict(status='NO_SEG_CENTER_LOCKED', reasons=['NO_ASSIGNED_SEG'], nearest_slot='', anchor_mask_distance='', row_gap_distance='', prior_mask_distance='', outward_row_distance='')
    (sx, sy) = (number(slot, 'seg_center_x'), number(slot, 'seg_center_y'))
    (ax, ay) = (number(slot, 'slot_center_x'), number(slot, 'slot_center_y'))
    (cw, ch) = (number(slot, 'cell_width'), number(slot, 'cell_height'))
    reasons = []
    distance = lambda r: math.hypot((sx - number(r, 'slot_center_x')) / cw, (sy - number(r, 'slot_center_y')) / ch)
    nearest = min(neighbors, key=distance)
    (own_distance, nearest_distance) = (distance(slot), distance(nearest))
    if nearest['slot_name'] != slot['slot_name'] and nearest_distance < own_distance * NEIGHBOR_DISTANCE_RATIO and (own_distance - nearest_distance > NEIGHBOR_DISTANCE_MARGIN):
        reasons.append('SEG_CENTER_NEARER_OTHER_SLOT')
    y_own = row_y_at(neighbors, slot['row'], sx)
    other_row = 'b' if slot['row'] == 't' else 't'
    y_other = row_y_at(neighbors, other_row, sx)
    gap = max(1.0, abs(y_other - y_own))
    row_gap_distance = abs(sy - y_own) / gap
    if row_gap_distance > MAX_ROW_GAP_DISTANCE:
        reasons.append('SEG_CENTER_OFF_TARGET_ROW')
    polygon = np.asarray(json.loads(slot['seg_polygon']), dtype=np.float32).reshape(-1, 2)
    signed = cv2.pointPolygonTest(polygon, (ax, ay), True)
    mask_distance = max(0.0, -signed)
    if mask_distance > max(8.0, MAX_ANCHOR_MASK_DISTANCE_RATIO * min(cw, ch)):
        reasons.append('SEG_MASK_DOES_NOT_SUPPORT_SLOT_ANCHOR')
    (px, py) = (number(slot, 'x_prior'), number(slot, 'y_prior'))
    prior_distance = max(0.0, -cv2.pointPolygonTest(polygon, (px, py), True))
    outward = (y_own - sy if slot['row'] == 't' else sy - y_own) / gap
    if outward > OUTWARD_ROW_DISTANCE_RATIO and prior_distance > max(8.0, PRIOR_MASK_DISTANCE_RATIO * min(cw, ch)):
        reasons.append('OUTER_BED_SEG_WITHOUT_SLOT_PRIOR_SUPPORT')
    if slot.get('visibility') == 'EMPTY':
        reasons.append('UPSTREAM_HOLE_AND_SEG_EVIDENCE_CONFLICT')
    return dict(status='REVIEW_TARGET_CENTER_LOCKED' if reasons else 'TARGET_GEOMETRY_SUPPORTED', reasons=reasons, nearest_slot=nearest['slot_name'], anchor_mask_distance=round(mask_distance, 4), row_gap_distance=round(row_gap_distance, 4), prior_mask_distance=round(prior_distance, 4), outward_row_distance=round(outward, 4))

def build_bbox_record(slot, neighbors):
    row = dict(slot)
    (width, height) = (int(row['source_width']), int(row['source_height']))
    (cw, ch) = (number(row, 'cell_width'), number(row, 'cell_height'))
    if cw <= 0 or ch <= 0:
        raise ValueError('Invalid input or state in build_bbox_record (cw <= 0 or ch <= 0)')
    guard = assess_target(row, neighbors)
    candidate = bool(row.get('seg_id'))
    use_seg = guard['status'] == 'TARGET_GEOMETRY_SUPPORTED'
    if use_seg:
        (ax, ay) = (number(row, 'seg_center_x'), number(row, 'seg_center_y'))
        center_source = 'ASSIGNED_SEG_CENTER'
    elif not candidate:
        (ax, ay) = (number(row, 'slot_center_x'), number(row, 'slot_center_y'))
        center_source = 'APPROVED_SLOT_CENTER_LOCKED'
    else:
        (ax, ay) = (number(row, 'x_prior'), number(row, 'y_prior'))
        center_source = 'SLOT_PRIOR_LOCKED'
    row.update(bbox_version=VERSION, bbox_input_record_id=row['record_id'], target_confirmation=guard['status'], target_guard_reasons='|'.join(guard['reasons']), nearest_slot_to_seg_center=guard['nearest_slot'], anchor_mask_distance_px=guard['anchor_mask_distance'], seg_row_gap_distance=guard['row_gap_distance'], prior_mask_distance_px=guard['prior_mask_distance'], seg_outward_row_distance=guard['outward_row_distance'], bbox_center_source=center_source, anchor_x=ax, anchor_y=ay, assigned_seg_preserved=candidate, seg_center_adopted=use_seg, mask_cleanup_applied=False, crop_kind='RGB_CONTEXT_BEFORE_FRAGMENT_FILTER', bbox_crop_path='', bbox_crop_sha256='', bbox_crop_status='PENDING', frame_summary_path=str(FRAME_MANIFEST), slot_manifest_path=str(SLOT_MANIFEST), segment_manifest_path=str(SEGMENT_MANIFEST), bbox_qc_flags='', upstream_qc='|'.join((v for v in (row.get('qc_flags', ''), guard['status'], '|'.join(guard['reasons'])) if v)), bbox_handoff_status='CONTEXT_READY_MASK_QC_PENDING' if use_seg else 'CONTEXT_READY_TARGET_REVIEW')
    if not (0 <= ax < width and 0 <= ay < height):
        row.update(bbox_crop_status='OUT_OF_FOV_CENTER_LOCKED', bbox_qc_flags='ANCHOR_OUTSIDE_SOURCE', bbox_handoff_status='OUT_OF_FOV_REVIEW')
        return row
    bbox_qc = []
    if row.get('bed_identity_status') != 'CONFIRMED':
        bbox_qc.append('OCR_IDENTITY_REVIEW')
    if use_seg:
        seg = [float(v) for v in json.loads(row['seg_bbox'])]
        (sw, sh) = (max(1.0, seg[2] - seg[0]), max(1.0, seg[3] - seg[1]))
        (padx, pady) = (max(MIN_MARGIN_PX, sw * SEG_MARGIN_RATIO), max(MIN_MARGIN_PX, sh * SEG_MARGIN_RATIO))
        wanted_w = max(cw * MIN_CELL_SCALE, 2 * max(ax - seg[0], seg[2] - ax) + 2 * padx)
        wanted_h = max(ch * MIN_CELL_SCALE, 2 * max(ay - seg[1], seg[3] - ay) + 2 * pady)
        if wanted_w > cw * MAX_CELL_SCALE or wanted_h > ch * MAX_CELL_SCALE:
            bbox_qc.append('OVERSIZE_CONTEXT_REVIEW_TARGET_PRESERVED')
    else:
        (wanted_w, wanted_h) = (cw * MAX_CELL_SCALE, ch * MAX_CELL_SCALE)
        if candidate:
            bbox_qc.append('ASSIGNED_SEG_NOT_ADOPTED_CENTER_LOCKED')
    (x1, x2) = centered_bounds(ax, wanted_w, width)
    (y1, y2) = centered_bounds(ay, wanted_h, height)
    if use_seg:
        x1 = min(x1, max(0, math.floor(seg[0])))
        y1 = min(y1, max(0, math.floor(seg[1])))
        x2 = max(x2, min(width, math.ceil(seg[2])))
        y2 = max(y2, min(height, math.ceil(seg[3])))
    clipped = x1 == 0 or y1 == 0 or x2 == width or (y2 == height)
    if clipped:
        bbox_qc.append('SOURCE_EDGE_CONTEXT')
    inside_neighbors = [r['slot_name'] for r in neighbors if r['slot_name'] != row['slot_name'] and x1 <= number(r, 'slot_center_x') < x2 and (y1 <= number(r, 'slot_center_y') < y2)]
    if inside_neighbors:
        bbox_qc.append('OTHER_SLOT_CENTER_IN_CONTEXT')
    retention = ''
    if candidate:
        sb = [float(v) for v in json.loads(row['seg_bbox'])]
        intersection = max(0.0, min(x2, sb[2]) - max(x1, sb[0])) * max(0.0, min(y2, sb[3]) - max(y1, sb[1]))
        area = max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
        retention = round(intersection / area, 8)
        if use_seg and retention < 0.999999:
            raise ValueError('Invalid input or state in build_bbox_record (use_seg and retention < 0.999999)')
    row.update(search_x1=x1, search_y1=y1, search_x2=x2, search_y2=y2, crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2, crop_width=x2 - x1, crop_height=y2 - y1, is_crop_clipped=clipped, target_bbox_retention=retention, neighbor_centers_in_crop='|'.join(inside_neighbors), neighbor_center_count_in_crop=len(inside_neighbors), bbox_qc_flags='|'.join(bbox_qc), bbox_crop_status='READY', neighbors_complete=False, neighbor_inventory_status='ALL_EXPORTED_UPSTREAM_SLOTS', expected_neighbor_slot_count=len(neighbors), downstream_target_policy='CHECK_SLOT_ANCHOR_AND_WHOLE_BED_CONTEXT')
    row['upstream_qc'] += '|' + row['bbox_qc_flags'] if row['bbox_qc_flags'] else ''
    return row

def color(record):
    if record['target_confirmation'] == 'TARGET_GEOMETRY_SUPPORTED':
        return (40, 220, 40)
    if record['target_confirmation'] == 'NO_SEG_CENTER_LOCKED':
        return (0, 220, 255)
    return (50, 70, 255)

def draw_overlay(image, records, sample):
    (h, w) = image.shape[:2]
    header = max(54, round(w * 0.05))
    canvas = np.zeros((h + header, w, 3), dtype=np.uint8)
    canvas[header:] = image
    supported = sum((r['target_confirmation'] == 'TARGET_GEOMETRY_SUPPORTED' for r in records))
    title = f"{sample['camera'].upper()} BBOX | slots={len(records)} | SEG center={supported} | RGB BEFORE CLEANUP"
    if not records:
        title += ' | NO_SLOT_GEOMETRY - REVIEW'
    scale = min(0.85, w / 1500)
    cv2.putText(canvas, title, (8, round(header * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, round(w / 1300)), cv2.LINE_AA)
    cv2.putText(canvas, 'Green: SEG supported | Yellow: no SEG, center locked | Red: candidate review, center locked', (8, round(header * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.66, (255, 255, 255), 1, cv2.LINE_AA)
    for r in records:
        if 'search_x1' not in r:
            continue
        ink = color(r)
        (x1, y1, x2, y2) = [int(r['search_' + k]) for k in ('x1', 'y1', 'x2', 'y2')]
        cv2.rectangle(canvas, (x1, y1 + header), (x2 - 1, y2 - 1 + header), ink, max(1, round(w / 1000)))
        (ax, ay) = (round(r['anchor_x']), round(r['anchor_y']) + header)
        cv2.drawMarker(canvas, (ax, ay), ink, cv2.MARKER_CROSS, max(8, round(w * 0.008)), 2)
        text = r['slot_name'] + (' SEG' if r['seg_center_adopted'] else ' LOCK')
        tx = max(0, min(w - 100, ax + 5))
        ty = max(header + 14, min(h + header - 5, ay - 8))
        ((tw, th), base) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(canvas, (tx, ty - th - base), (min(w - 1, tx + tw + 4), ty + base), (0, 0, 0), -1)
        cv2.putText(canvas, text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)
    return canvas
