"""Select the owned plant mask and save contextual RGB crops with explicit mask provenance. Existing physical slots, camera-selection policy and REVIEW states are preserved."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from pathlib import Path
from pipeline_paths import BBOX_MANIFEST, CROP_ROOT, BRIGHT_IMAGES
INPUT_PATH = str(BBOX_MANIFEST)
OUTPUT_DIR = str(CROP_ROOT)
SOURCE_ROOT = str(BRIGHT_IMAGES)
MODEL_PATH = str(CYCLE.PLANT_MODEL)
EXPECTED_MODEL_SHA256 = 'a2ded62bba4d577f4ead18907b824c587e179357c8089b72dabde41b755d6fe8'
CASE_RECOVERY_DEFAULT = True
import argparse, csv, hashlib, json, math, os, re, sqlite3, sys, time, datetime
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)
VERSION = 'bbox_to_crop_rgb_only_camera_selection_v4_case_recovery'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
FIELDS = ['record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'source_path', 'source_sha256', 'source_coordinate_space', 'anchor_x', 'anchor_y', 'search_box', 'cell_width', 'cell_height', 'mapping_version', 'upstream_qc', 'metadata_warnings', 'status', 'reasons', 'has_crop', 'auto_qc_pass', 'training_eligible', 'crop_path', 'mask_path', 'crop_sha256', 'mask_sha256', 'crop_source_box', 'selected_route', 'source_width', 'source_height', 'mask_pixels', 'mm_per_px', 'scale_verified', 'scale_version', 'scale_ready', 'scale_source', 'scale_qc', 'local_status', 'whole_status', 'context_iou', 'cleanup', 'overlap_slots', 'error', 'input_record_json', 'run_signature', 'photometric_policy', 'track_id', 'pair_id', 'season_id']
from pipeline_contracts import PROVENANCE_FIELDS
from mask_record import pack as pack_region, unpack as unpack_region, FIELDS as REGION_FIELDS
from pipeline_events import report_error
import camera_selection
FIELDS = list(dict.fromkeys(FIELDS + PROVENANCE_FIELDS + REGION_FIELDS + camera_selection.METRICS + ['candidate_available']))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def jsontext(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def load_image(path):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('Invalid input or state in load_image (rejected condition: im is None)')
    return im

class SourceImageCache:
    """Core SourceImageCache calculations."""

    def __init__(self):
        self.key = None
        self.sha256 = None
        self.image = None

    def get(self, path, expected_sha256):
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_size, getattr(stat, 'st_mtime_ns', None), getattr(stat, 'st_ctime_ns', None))
        if key != self.key:
            current = digest(path)
            if current != expected_sha256:
                raise ValueError('Invalid input or state in get (rejected condition: current != expected_sha256)')
            image = load_image(path)
            self.key = key
            self.sha256 = current
            self.image = image
        if self.sha256 != expected_sha256:
            raise ValueError('Invalid input or state in get (rejected condition: self.sha256 != expected_sha256)')
        return self.image

def prune_scratch_rows(db, token):
    for table in ('inputs', 'neighbors', 'final_crops'):
        db.execute('DELETE FROM ' + table + ' WHERE run<>?', (token,))
    db.commit()

def number(row, *names, default=None):
    for n in names:
        try:
            v = float(row.get(n, ''))
            if math.isfinite(v):
                return v
        except (ValueError, TypeError):
            pass
    return default

def first(row, *names, default=''):
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip().lower() not in ('', 'none', 'nan', 'null'):
            return str(v).strip()
    return default

def yes(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y')

def resolve_path(value, base, root=None):
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    for parent in ([Path(root)] if root else []) + [Path(base)]:
        q = (parent / p).resolve()
        if q.exists():
            return q
    return ((Path(root) if root else Path(base)) / p).resolve()

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def normalize(row, base, source_root=None):
    r = dict(row)
    warnings = []
    bbox_only = yes(r.get('_bbox_only')) or first(r, 'source_coordinate_space') == 'bbox_local'
    value = first(r, 'source_path', 'bed_source_path', 'bbox_path', 'image_path')
    if not value:
        raise ValueError('Invalid input or state in normalize (rejected condition: not value)')
    source = resolve_path(value, base, source_root)
    if first(r, 'source_coordinate_space') not in ('', 'bed_pixels', 'bed_crop', 'bbox_local'):
        raise ValueError("Invalid input or state in normalize (rejected condition: first(r, 'source_coordinate_space') not in ('', 'bed_pixels', 'bed_crop', 'bbox_local'))")
    match = re.search('(bed\\d+)_(\\d{8})_(\\d{6})_(cam[012])', source.stem, re.I)
    camera = first(r, 'camera', default=match[4].lower() if match else '')
    bed = first(r, 'bed_id', default=match[1] if match else '')
    date = first(r, 'capture_date', default=match[2] if match else '')
    stamp = first(r, 'capture_time', default=match[3] if match else '')
    slot = first(r, 'physical_slot', 'canonical_id', 'slot_name')
    if not slot:
        m = re.search('__([tb]\\d+)$', source.stem, re.I)
        slot = m[1].lower() if m else ''
    if not camera or not bed or (not date) or (not stamp) or (not slot):
        raise ValueError('Invalid input or state in normalize (rejected condition: not camera or not bed or (not date) or (not stamp) or (not slot))')
    date = str(date).removesuffix('.0').zfill(8)
    stamp = str(stamp).removesuffix('.0').zfill(6)
    datetime.datetime.strptime(date + stamp, '%Y%m%d%H%M%S')
    if camera not in ('cam0', 'cam1', 'cam2') or not re.fullmatch('[tb][1-9][0-9]*', slot):
        raise ValueError("Invalid input or state in normalize (rejected condition: camera not in ('cam0', 'cam1', 'cam2') or not re.fullmatch('[tb][1-9][0-9]*', slot))")
    if match and (bed.lower(), date, stamp, camera) != (match[1].lower(), match[2], match[3], match[4].lower()):
        raise ValueError('Invalid input or state in normalize (rejected condition: match and (bed.lower(), date, stamp, camera) != (match[1].lower(), match[2], match[3], match[4].lower()))')
    ax = ay = None
    for (xkey, ykey) in (('anchor_x', 'anchor_y'), ('seg_center_x', 'seg_center_y'), ('slot_center_x', 'slot_center_y'), ('slot_prior_x', 'slot_prior_y'), ('x_prior', 'y_prior')):
        (x, y) = (number(r, xkey), number(r, ykey))
        if x is not None and y is not None:
            (ax, ay) = (x, y)
            break
    cw = number(r, 'cell_width')
    ch = number(r, 'cell_height')
    box = [number(r, 'search_' + k) for k in ('x1', 'y1', 'x2', 'y2')]
    if all((v is not None for v in box)):
        box = [int(v) for v in box]
    else:
        box = None
    if not bbox_only and (ax is None or ay is None or cw is None or (ch is None)):
        raise ValueError('Invalid input or state in normalize (rejected condition: not bbox_only and (ax is None or ay is None or cw is None or (ch is None)))')
    unavailable = any((t in first(r, 'visibility').upper() for t in ('OUT_OF_FOV', 'OCCLUDED'))) or first(r, 'occupancy_human').upper() == 'EMPTY'
    if not bbox_only and (cw <= 0 or ch <= 0) and (not unavailable):
        raise ValueError('Cell dimensions must be positive')
    if not bbox_only and box is None:
        warnings.append('SEARCH_DERIVED_FROM_ANCHOR_1.45_CELL')
    if bbox_only:
        warnings.append('BBOX_ONLY_NO_PARENT_CONTEXT')
    if not yes(r.get('neighbors_complete')):
        warnings.append('NEIGHBORS_NOT_CERTIFIED_COMPLETE')
    if not first(r, 'mapping_version'):
        warnings.append('MAPPING_VERSION_MISSING')
    key = jsontext([str(source).casefold(), camera, bed, date, stamp, slot])
    return {'record_id': hashlib.sha256(key.encode()).hexdigest()[:24], 'camera': camera, 'bed_id': bed, 'capture_date': date, 'capture_time': stamp, 'physical_slot': slot, 'source_path': str(source), 'anchor_x': ax, 'anchor_y': ay, 'cell_width': cw, 'cell_height': ch, 'search_box': box, 'source_coordinate_space': 'bbox_local' if bbox_only else 'bed_pixels', 'mapping_version': first(r, 'mapping_version'), 'metadata_warnings': warnings, 'upstream_qc': first(r, 'upstream_qc', 'qc_flags', 'image_qc', 'map_status'), 'occupancy_human': first(r, 'occupancy_human'), 'visibility': first(r, 'visibility'), 'mm_per_px': number(r, 'mm_per_px'), 'scale_verified': yes(r.get('scale_verified')), 'scale_ready': yes(r.get('scale_ready')), 'scale_source': first(r, 'scale_source'), 'scale_qc': first(r, 'scale_qc'), 'scale_version': first(r, 'scale_version'), 'track_id': first(r, 'track_id'), 'pair_id': first(r, 'pair_id'), 'season_id': first(r, 'season_id'), 'input_record_json': r, **{k: r.get(k, '') for k in PROVENANCE_FIELDS}}

def input_rows(path, root=None):
    path = Path(path)
    if path.is_dir():
        manifests = [path / n for n in ('bbox_manifest.csv', 'crop_manifest.csv') if (path / n).is_file()]
        if manifests:
            if len(manifests) > 1:
                raise ValueError('Invalid input or state in input_rows (rejected condition: len(manifests) > 1)')
            yield from input_rows(manifests[0], root)
            return
        for f in sorted(path.rglob('*')):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                if any((x in ('crop_final', 'crops', 'masks') for x in f.relative_to(path).parts[:-1])):
                    continue
                yield normalize({'bbox_path': str(f), '_bbox_only': True}, path, root)
    else:
        if path.suffix.lower() != '.csv':
            raise ValueError("Invalid input or state in input_rows (rejected condition: path.suffix.lower() != '.csv')")
        for (i, r) in enumerate(csv_rows(path), 2):
            try:
                yield normalize(r, path.parent, root)
            except Exception as e:
                raise ValueError(f'{path.name}:{i}: {e}') from e

def bounds(r, w, h):
    if r['source_coordinate_space'] == 'bbox_local':
        ax = r['anchor_x'] if r['anchor_x'] is not None else (w - 1) / 2
        ay = r['anchor_y'] if r['anchor_y'] is not None else (h - 1) / 2
        cw = r['cell_width'] or w
        ch = r['cell_height'] or h
        box = [0, 0, w, h]
    else:
        (ax, ay, cw, ch) = (r[k] for k in ('anchor_x', 'anchor_y', 'cell_width', 'cell_height'))
        box = r['search_box'] or [max(0, round(ax - 0.725 * cw)), max(0, round(ay - 0.725 * ch)), min(w, round(ax + 0.725 * cw)), min(h, round(ay + 0.725 * ch))]
    if not (0 <= ax < w and 0 <= ay < h):
        raise ValueError('Invalid input or state in bounds (rejected condition: not (0 <= ax < w and 0 <= ay < h))')
    (x0, y0, x1, y1) = box
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Invalid input or state in bounds (rejected condition: not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h))')
    if not (x0 <= ax < x1 and y0 <= ay < y1):
        raise ValueError('Invalid input or state in bounds (rejected condition: not (x0 <= ax < x1 and y0 <= ay < y1))')
    return ([ax, ay], cw, ch, box)

def core_coverage(mask, box, point, radius):
    (x, y) = (point[0] - box[0], point[1] - box[1])
    x0 = max(0, int(x - radius))
    x1 = min(mask.shape[1], int(x + radius) + 1)
    y0 = max(0, int(y - radius))
    y1 = min(mask.shape[0], int(y + radius) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    (yy, xx) = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return float(np.sum(mask[y0:y1, x0:x1] & disk) / max(1, disk.sum()))

@dataclass
class PreparedMeasurement:
    """Core PreparedMeasurement calculations."""
    boxes: Any
    masks: Any
    box: tuple
    source_shape: tuple
    items: list

def prepare_measurement(pred, box, source):
    if isinstance(pred, PreparedMeasurement):
        return pred
    prepared = PreparedMeasurement(getattr(pred, 'boxes', None), pred.masks, tuple(box), source.shape[:2], [])
    if pred.masks is None:
        return prepared
    (x0, y0, x1, y1) = box
    im = source[y0:y1, x0:x1]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 100) & (hsv[:, :, 1] >= 35)
    polygons = getattr(pred.masks, 'xy', None)
    for (i, arr) in enumerate(pred.masks.data.cpu().numpy()):
        mask = arr > 0.5
        if mask.shape != im.shape[:2]:
            raise ValueError('Invalid input or state in prepare_measurement (rejected condition: mask.shape != im.shape[:2])')
        if not mask.any():
            continue
        mask_u8 = mask.astype(np.uint8)
        (contours, _) = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        m = cv2.moments(cnt)
        if polygons is not None:
            if len(polygons[i]) < 3:
                continue
            m = cv2.moments(np.asarray(polygons[i], dtype=np.float32).reshape(-1, 1, 2))
            if m['m00'] <= 0:
                xyxy = pred.boxes.xyxy[i].cpu().numpy()
                m = {'m00': 1.0, 'm10': float((xyxy[0] + xyxy[2]) / 2), 'm01': float((xyxy[1] + xyxy[3]) / 2)}
        if m['m00'] <= 0:
            continue
        area = int(mask.sum())
        (_, _, stats, _) = cv2.connectedComponentsWithStats(mask_u8, 8)
        significant = sum((int(a) >= max(32, 0.03 * area) for a in stats[1:, cv2.CC_STAT_AREA]))
        (ys, xs) = np.where(mask)
        static = {'index': i, 'area': area, 'conf': float(pred.boxes.conf[i]), 'green_fraction': float(np.sum(mask & green) / area), 'edge_touch': bool(xs.min() == 0 or ys.min() == 0 or xs.max() == mask.shape[1] - 1 or (ys.max() == mask.shape[0] - 1)), 'fragmented': significant > 1}
        prepared.items.append((static, mask, m['m10'] / m['m00'] + x0, m['m01'] / m['m00'] + y0))
    return prepared

def measure(pred, box, source, anchor, cw, ch, neighbors):
    prepared = prepare_measurement(pred, box, source)
    if prepared.box != tuple(box) or prepared.source_shape != source.shape[:2]:
        raise ValueError('Invalid input or state in measure (rejected condition: prepared.box != tuple(box) or prepared.source_shape != source.shape[:2])')
    candidates = []
    masks = {}
    radius = max(8, 0.12 * min(cw, ch))
    for (static, mask, cx, cy) in prepared.items:
        distance = math.hypot((cx - anchor[0]) / cw, (cy - anchor[1]) / ch)
        if distance > 1.25:
            continue
        candidates.append({**static, 'center_distance_cell': distance, 'core_coverage': core_coverage(mask, box, anchor, radius), 'neighbor_claims': [name for (name, point) in neighbors if core_coverage(mask, box, point, radius) >= 0.5]})
        masks[static['index']] = mask
    return (candidates, masks)

def overlap(a, ba, b, bb):
    x0 = max(ba[0], bb[0])
    y0 = max(ba[1], bb[1])
    x1 = min(ba[2], bb[2])
    y1 = min(ba[3], bb[3])
    inter = int((a[y0 - ba[1]:y1 - ba[1], x0 - ba[0]:x1 - ba[0]] & b[y0 - bb[1]:y1 - bb[1], x0 - bb[0]:x1 - bb[0]]).sum()) if x1 > x0 and y1 > y0 else 0
    if not inter:
        return (0.0, 0, 0.0)
    return (inter / max(1, int(a.sum() + b.sum()) - inter), inter, inter / max(1, min(int(a.sum()), int(b.sum()))))

def fit_mask(mask, box, target_box):
    out = np.zeros((target_box[3] - target_box[1], target_box[2] - target_box[0]), bool)
    x0 = max(box[0], target_box[0])
    y0 = max(box[1], target_box[1])
    x1 = min(box[2], target_box[2])
    y1 = min(box[3], target_box[3])
    if x1 > x0 and y1 > y0:
        out[y0 - target_box[1]:y1 - target_box[1], x0 - target_box[0]:x1 - target_box[0]] = mask[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]]
    return out

def refine_color(image, seed, radius=24):
    (ys, xs) = np.where(seed)
    pad = radius + 3
    x0 = max(0, xs.min() - pad)
    x1 = min(seed.shape[1], xs.max() + pad + 1)
    y0 = max(0, ys.min() - pad)
    y1 = min(seed.shape[0], ys.max() + pad + 1)
    s = seed[y0:y1, x0:x1].astype(np.uint8)
    allowed = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2))
    core = cv2.erode(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if core.sum() < 20:
        return seed
    gc = np.full(s.shape, cv2.GC_BGD, np.uint8)
    gc[allowed > 0] = cv2.GC_PR_BGD
    gc[s > 0] = cv2.GC_PR_FGD
    gc[core > 0] = cv2.GC_FGD
    gc[[0, -1], :] = cv2.GC_BGD
    gc[:, [0, -1]] = cv2.GC_BGD
    cv2.setRNGSeed(20260909)
    cv2.grabCut(image[y0:y1, x0:x1].copy(), gc, None, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    out = np.zeros(seed.shape, bool)
    out[y0:y1, x0:x1] = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return out if out.any() else seed

def predict(model, image, size, device, whole=False):
    return model.predict(image, imgsz=size, conf=0.2, iou=0.7, retina_masks=True, rect=whole, verbose=False, device=device)[0]

def upper_subset_proposal(local, whole, iou):
    if not local or not whole or iou is None or (min(local['area'], whole['area']) <= 0):
        return False
    inter = round(iou * (local['area'] + whole['area']) / (1 + iou))
    extra = (local['area'] - inter) / local['area']
    missing = (whole['area'] - inter) / whole['area']
    return iou >= 0.75 and extra <= 0.02 and (missing >= 0.1) and (1 < whole['area'] / local['area'] <= 1.4) and (whole['conf'] - local['conf'] >= 0.08) and (whole['conf'] >= 0.75) and (whole.get('core_coverage', 0) >= 0.8) and (not whole.get('neighbor_claims')) and (not whole.get('fragmented')) and (not whole.get('edge_touch'))

def repair_upper_canvas(local, local_box, whole, whole_box, anchor_y):
    base = fit_mask(local, local_box, whole_box)
    split = max(0, min(base.shape[0], int(round(anchor_y)) - whole_box[1]))
    repaired = base.copy()
    repaired[:split] |= whole[:split]
    return (repaired, int((repaired & ~base).sum()))

def raw_recovery(record, mask, chosen_box, source, neighbors, anchor, cw, ch, box, ld, wd, model, args, raw_info):
    enabled = args.raw_context or CASE_RECOVERY_DEFAULT
    blocked = {'REVIEW_AMBIGUOUS', 'REVIEW_NEIGHBOR_CLAIM', 'REVIEW_IDENTITY_CONFLICT'}
    automatic_scope = record['camera'] == 'cam2' and record['physical_slot'].startswith('t')
    clean_agreement = record.get('selected_route') == 'local' and record.get('status') == 'CANDIDATE_CONTEXT_AGREEMENT'
    partial_shape = ld.status == 'REVIEW_SHAPE' or wd.status == 'REVIEW_SHAPE' or (record.get('cleanup', {}).get('changed', False) and (ld.status != 'CANDIDATE' or wd.status != 'CANDIDATE'))
    if not enabled or not raw_info or (not automatic_scope) or (ld.status in blocked) or (wd.status in blocked) or (not args.raw_context and clean_agreement) or (mask is not None and box[1] > 2 and (not partial_shape)):
        return (mask, chosen_box, source, neighbors, anchor)
    (rawpath, bedbox) = raw_info
    raw = load_image(rawpath)
    (ox, oy, ex, ey) = bedbox
    (h, w) = source.shape[:2]
    if ex - ox != w or ey - oy != h or (not (0 <= ox < ex <= raw.shape[1] and 0 <= oy < ey <= raw.shape[0])):
        raise ValueError('Invalid input or state in raw_recovery (rejected condition: ex - ox != w or ey - oy != h or (not (0 <= ox < ex <= raw.shape[1] and 0 <= oy < ey <= raw.shape[0])))')
    ra = [anchor[0] + ox, anchor[1] + oy]
    rn = [(n, [pt[0] + ox, pt[1] + oy]) for (n, pt) in neighbors]
    top = [box[0] + ox, max(0, box[1] + oy - round(0.5 * ch)), box[2] + ox, box[3] + oy]
    variants = [('raw_top_halfcell', top, args.local_size, False)]
    if mask is None or partial_shape:
        variants.append(('raw_context25', [ox, max(0, oy - round(0.25 * h)), ex, min(raw.shape[0], ey + round(0.1 * h))], args.whole_size, True))
    attempts = []
    for (kind, rb, size, whole_input) in variants:
        pred = predict(model, raw[rb[1]:rb[3], rb[0]:rb[2]], size, args.device, whole_input)
        (cs, ms) = measure(pred, rb, raw, ra, cw, ch, rn)
        decision = select_target(cs)
        attempts.append({'route': kind, 'status': decision.status, 'input_box_raw': rb})
        if decision.status != 'CANDIDATE' or decision.selected_index is None:
            continue
        proposed = ms[decision.selected_index]
        translated = [rb[0] - ox, rb[1] - oy, rb[2] - ox, rb[3] - oy]
        agreement = overlap(mask, chosen_box, proposed, translated)[0] if mask is not None else None
        if mask is not None:
            yy = np.where(proposed)[0] + rb[1]
            extra = int((yy < oy).sum())
            ratio = float(proposed.sum() / max(1, mask.sum()))
            if agreement < 0.5 or not 0.5 <= ratio <= 2.0:
                continue
            if kind == 'raw_top_halfcell' and (box[1] > 2 or oy <= 0 or extra < 128 or (extra / max(1, int(proposed.sum())) < 0.01)):
                continue
        if kind == 'raw_context25' and (args.color_refine or CASE_RECOVERY_DEFAULT):
            proposed = refine_color(raw[rb[1]:rb[3], rb[0]:rb[2]], proposed, 24)
            record['reasons'].append('COLOR_BOUNDARY_24_ON_CONTEXT_RECOVERY')
        record.update(status='REVIEW_RAW_CONTEXT_RECOVERY', selected_route=kind, target_conf=float(pred.boxes.conf[decision.selected_index]), source_path=str(rawpath), source_coordinate_space='raw_camera_pixels', photometric_policy='original_rgb', anchor_x=ra[0], anchor_y=ra[1], source_width=raw.shape[1], source_height=raw.shape[0])
        record['reasons'].append('RAW_CONTEXT_RECOVERY:' + kind)
        record['cleanup']['raw_attempts'] = attempts
        return (proposed, rb, raw, rn, ra)
    record['cleanup']['raw_attempts'] = attempts
    return (mask, chosen_box, source, neighbors, anchor)

def small_neighbor_subtraction(mask, box, anchor, other, other_box, other_anchor):
    unchanged = {'changed': False, 'removed_pixels': 0}
    x0 = max(box[0], other_box[0])
    y0 = max(box[1], other_box[1])
    x1 = min(box[2], other_box[2])
    y1 = min(box[3], other_box[3])
    if x1 <= x0 or y1 <= y0:
        return (mask, unchanged)
    remove = np.zeros(mask.shape, bool)
    remove[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] = mask[y0 - box[1]:y1 - box[1], x0 - box[0]:x1 - box[0]] & other[y0 - other_box[1]:y1 - other_box[1], x0 - other_box[0]:x1 - other_box[0]]
    fraction = float(remove.sum() / max(1, mask.sum()))
    if not 0.01 <= fraction <= 0.03:
        return (mask, unchanged)
    (yy, xx) = np.ogrid[:mask.shape[0], :mask.shape[1]]
    ax = anchor[0] - box[0]
    ay = anchor[1] - box[1]
    radius = max(8, 0.1 * min(mask.shape))
    if (remove & ((xx - ax) ** 2 + (yy - ay) ** 2 <= radius ** 2)).any():
        return (mask, unchanged)
    (ry, rx) = np.where(remove)
    cx = float(rx.mean() + box[0])
    cy = float(ry.mean() + box[1])
    if math.hypot(cx - other_anchor[0], cy - other_anchor[1]) >= math.hypot(cx - anchor[0], cy - anchor[1]):
        return (mask, unchanged)
    new = mask & ~remove
    if not new.any():
        return (mask, unchanged)
    (_, _, stats, _) = cv2.connectedComponentsWithStats(new.astype(np.uint8), 8)
    if 1 - max(stats[1:, cv2.CC_STAT_AREA]) / new.sum() > 0.03:
        return (mask, unchanged)
    return (new, {'changed': True, 'removed_pixels': int(remove.sum()), 'removed_fraction': fraction})

def clean_shared_fragments(selected, raw_info=None):
    if not CASE_RECOVERY_DEFAULT:
        return selected
    snapshots = []
    for (record, mask, box, im) in selected:
        bb = list(box) if box is not None else None
        anchor = [record.get('anchor_x'), record.get('anchor_y')]
        if mask is not None and record['source_coordinate_space'] == 'raw_camera_pixels':
            (ox, oy) = raw_info[1][:2]
            bb = [bb[0] - ox, bb[1] - oy, bb[2] - ox, bb[3] - oy]
            anchor = [anchor[0] - ox, anchor[1] - oy]
        snapshots.append((record, mask, bb, anchor))
    updated = []
    for (i, (record, mask, box, im)) in enumerate(selected):
        if mask is None:
            updated.append((record, mask, box, im))
            continue
        if not (record.get('selected_route') == 'local' and record.get('local_status') == 'CANDIDATE' and (record.get('whole_status') == 'REVIEW_SHAPE')):
            updated.append((record, mask, box, im))
            continue
        (own, original, bb, anchor) = snapshots[i]
        proposals = []
        for (j, (neighbor, other, nb, na)) in enumerate(snapshots):
            if i == j or other is None:
                continue
            (candidate, detail) = small_neighbor_subtraction(original, bb, anchor, other, nb, na)
            if detail['changed']:
                proposals.append((candidate, neighbor['physical_slot'], detail))
        if len(proposals) == 1:
            (mask, slot, detail) = proposals[0]
            record.setdefault('cleanup', {})['small_neighbor_subtraction'] = {**detail, 'neighbor_slot': slot}
            record.setdefault('reasons', []).append('SMALL_NEIGHBOR_SUBTRACTION:' + slot)
            record['status'] = 'REVIEW_SMALL_NEIGHBOR_SUBTRACTION'
            record['auto_qc_pass'] = False
        updated.append((record, mask, box, im))
    return updated

def infer_one(r, source, full, neighbors, model, args, raw_info=None):
    if r['occupancy_human'].upper() == 'EMPTY' or any((t in r['visibility'].upper() for t in ('OUT_OF_FOV', 'OCCLUDED'))):
        return ({**{k: v for (k, v) in r.items() if k in FIELDS}, 'status': 'CONFIRMED_EMPTY' if r['occupancy_human'].upper() == 'EMPTY' else 'UNAVAILABLE_GEOMETRY', 'has_crop': False, 'auto_qc_pass': False, 'training_eligible': False}, None, None, None)
    (h, w) = source.shape[:2]
    (anchor, cw, ch, box) = bounds(r, w, h)
    record = {**{k: v for (k, v) in r.items() if k in FIELDS}, 'anchor_x': anchor[0], 'anchor_y': anchor[1], 'cell_width': cw, 'cell_height': ch, 'search_box': box, 'has_crop': False, 'auto_qc_pass': False, 'training_eligible': False, 'source_width': w, 'source_height': h, 'reasons': list(r['metadata_warnings']), 'photometric_policy': r.get('photometric_policy') or 'original_rgb'}
    if r['occupancy_human'].upper() == 'EMPTY':
        return ({**record, 'status': 'CONFIRMED_EMPTY'}, None, None, None)
    if any((t in r['visibility'].upper() for t in ('OUT_OF_FOV', 'OCCLUDED'))):
        return ({**record, 'status': 'UNAVAILABLE_GEOMETRY'}, None, None, None)
    prior = r['input_record_json']
    if number(prior, 'source_width', default=w) != w or number(prior, 'source_height', default=h) != h:
        raise ValueError("Invalid input or state in infer_one (rejected condition: number(prior, 'source_width', default=w) != w or number(prior, 'source_height', default=h) != h)")
    local = predict(model, source[box[1]:box[3], box[0]:box[2]], args.local_size, args.device)
    (lc, lm) = measure(local, box, source, anchor, cw, ch, neighbors)
    ld = select_target(lc)
    fb = [0, 0, w, h]
    (wc, wm) = measure(full, fb, source, anchor, cw, ch, neighbors) if full is not None else ([], {})
    wd = select_target(wc)
    iou = None
    if ld.selected_index is not None and wd.selected_index is not None:
        (iou, inter, _) = overlap(lm[ld.selected_index], box, wm[wd.selected_index], fb)
    route = choose_context(ld, wd, iou)
    if route.source == 'local' and route.status == 'CANDIDATE_CONTEXT_AGREEMENT':
        l = next((x for x in lc if x['index'] == ld.selected_index))
        q = next((x for x in wc if x['index'] == wd.selected_index))
        shape = prefer_full_shape(ld, wd, iou, local_area=l['area'], whole_area=q['area'], intersection=inter, local_conf=l['conf'], whole_conf=q['conf'])
        if shape['replace_with_whole']:
            route = RouteDecision('whole', 'REVIEW_FULL_CONTEXT_SHAPE_REPLACEMENT')
    record.update(status=route.status, local_status=ld.status, whole_status=wd.status, context_iou=iou, selected_route=route.source or '')
    mask = lm[ld.selected_index] if route.source == 'local' else wm[wd.selected_index] if route.source == 'whole' else None
    chosen_box = box if route.source == 'local' else fb
    record['target_conf'] = float(local.boxes.conf[ld.selected_index]) if route.source == 'local' else float(full.boxes.conf[wd.selected_index]) if route.source == 'whole' else 0.0
    record['cleanup'] = {}
    if mask is not None:
        (mask, cleanup) = clean_detached(mask, [anchor[0] - chosen_box[0], anchor[1] - chosen_box[1]])
        record['cleanup'] = cleanup
        if cleanup['status'].startswith('REVIEW'):
            record['reasons'].append(cleanup['status'])
            record['status'] = 'REVIEW_SHAPE_CLEANUP'
    if mask is not None and (args.upper_repair or CASE_RECOVERY_DEFAULT) and (route.source == 'local') and (ld.status == 'CANDIDATE') and (wd.status == 'CANDIDATE'):
        l = next((x for x in lc if x['index'] == ld.selected_index))
        q = next((x for x in wc if x['index'] == wd.selected_index))
        if upper_subset_proposal(l, q, iou):
            (repaired, added) = repair_upper_canvas(mask, box, wm[wd.selected_index], fb, anchor[1])
            if added:
                mask = repaired
                chosen_box = fb
                record['cleanup']['upper_repair'] = {'added_pixels': added, 'below_anchor_unchanged': True}
                record['reasons'].append('UPPER_REPAIR_FULL_CANVAS')
                record['status'] = 'REVIEW_UPPER_REPAIR'
                record['selected_route'] = 'local_upper_repair'
    (mask, chosen_box, source, neighbors, anchor) = raw_recovery(record, mask, chosen_box, source, neighbors, anchor, cw, ch, box, ld, wd, model, args, raw_info)
    if mask is None:
        return (record, None, None, None)
    if args.color_refine and record['selected_route'] != 'raw_context25':
        mask = refine_color(source[chosen_box[1]:chosen_box[3], chosen_box[0]:chosen_box[2]], mask)
        record['reasons'].append('COLOR_REFINE_OPT_IN')
        record['status'] = 'REVIEW_COLOR_REFINEMENT'
    claims = [name for (name, pt) in neighbors if core_coverage(mask, chosen_box, pt, max(8, 0.12 * min(cw, ch))) >= 0.5]
    if claims:
        record['reasons'].append('NEIGHBOR_CORE:' + ','.join(claims))
        record['status'] = 'REVIEW_NEIGHBOR_CLAIM'
    if record['reasons'] or any((x in r['upstream_qc'].upper() for x in ('REVIEW', 'MERGE', 'AMBIG', 'FAIL'))):
        if record['status'].startswith('CANDIDATE'):
            record['status'] = 'REVIEW_METADATA_OR_UPSTREAM'
    record['auto_qc_pass'] = record['status'] == 'CANDIDATE_CONTEXT_AGREEMENT'
    return (record, mask, chosen_box, source)

def save_png(path, im):
    (ok, b) = cv2.imencode('.png', im)
    if not ok:
        raise ValueError('PNG encoding failed')
    temp = path.with_suffix('.writing')
    b.tofile(str(temp))
    os.replace(temp, path)

def prepare_candidate(record, mask, box):
    record.update(camera_selection.measure(record, mask, box), candidate_available=mask is not None, has_crop=False)
    if mask is None:
        return record
    (ys, xs) = np.where(mask)
    x0 = max(0, int(xs.min()) - 14)
    x1 = min(mask.shape[1], int(xs.max()) + 15)
    y0 = max(0, int(ys.min()) - 14)
    y1 = min(mask.shape[0], int(ys.max()) + 15)
    m = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    record.update(pack_region(m))
    record.update(crop_source_box=[box[0] + x0, box[1] + y0, box[0] + x1, box[1] + y1], mask_pixels=int((m > 0).sum()))
    return record

def crop_output_path(record, out):
    original = record.get('input_record_json') or {}
    if isinstance(original, str):
        original = json.loads(original)
    bed_source = original.get('source_path') or record.get('source_path')
    stem = Path(bed_source).stem if bed_source else f"{record['bed_id']}_{record['capture_date']}_{record['capture_time']}_{record['camera']}"
    slot = record['physical_slot']
    if not re.fullmatch('[tb]\\d+', slot):
        raise ValueError("Invalid input or state in crop_output_path (rejected condition: not re.fullmatch('[tb]\\\\d+', slot))")
    return Path(out) / 'crops' / record['bed_id'] / record['capture_date'] / f'{stem}_{slot}.png'

def retire_superseded_crops(db, token, out):
    current = {}
    active = set()
    obsolete = {}
    for (pair, view, payload) in db.execute('SELECT pair,view,payload FROM final_crops WHERE run=?', (token,)):
        row = json.loads(payload)
        if row.get('has_crop'):
            path = Path(row['crop_path']).resolve()
            current[pair, view] = path
            active.add(path)
    crop_root = (Path(out) / 'crops').resolve()
    for (pair, view, payload) in db.execute('SELECT pair,view,payload FROM final_crops WHERE run<>?', (token,)):
        if (pair, view) not in current:
            continue
        row = json.loads(payload)
        if not row.get('has_crop') or not row.get('crop_path'):
            continue
        path = Path(row['crop_path']).resolve()
        if path in active or not path.is_file():
            continue
        if crop_root not in path.parents:
            raise ValueError('Invalid input or state in retire_superseded_crops (rejected condition: crop_root not in path.parents)')
        obsolete.setdefault(path, set()).add(row.get('crop_sha256', ''))
    for (path, hashes) in obsolete.items():
        if digest(path) not in hashes:
            raise ValueError('Invalid input or state in retire_superseded_crops (rejected condition: digest(path) not in hashes)')
    for path in obsolete:
        path.unlink()
    return len(obsolete)

def write_prepared(record, source, out):
    if not record.get('candidate_available'):
        return record
    mask = unpack_region(record)
    (x0, y0, x1, y1) = record['crop_source_box']
    rgb = source[y0:y1, x0:x1].copy()
    if rgb.shape[:2] != mask.shape:
        raise ValueError('Invalid input or state in write_prepared (rejected condition: rgb.shape[:2] != mask.shape)')
    rgb[mask == 0] = 0
    crop = crop_output_path(record, out)
    crop.parent.mkdir(parents=True, exist_ok=True)
    save_png(crop, rgb)
    record.update(has_crop=True, crop_path=str(crop), crop_sha256=digest(crop))
    return record

def write_csv(path, rows):
    temp = path.with_suffix('.writing')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsontext(v) if isinstance(v, (dict, list, tuple)) else v for (k, v) in row.items() if k in FIELDS})
    os.replace(temp, path)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=INPUT_PATH)
    p.add_argument('--output', default=OUTPUT_DIR)
    p.add_argument('--source-root', default=SOURCE_ROOT)
    p.add_argument('--model', default=MODEL_PATH)
    p.add_argument('--model-sha256', default=EXPECTED_MODEL_SHA256)
    p.add_argument('--neighbors', help='parser: manifest. Output')
    p.add_argument('--bed-log', action='append', default=[], help='parser: 1 crop_manifest.csv; target_path→raw source/x1..y2')
    p.add_argument('--device', default='0')
    p.add_argument('--local-size', type=int, default=256)
    p.add_argument('--whole-size', type=int, default=1280)
    p.add_argument('--check', action='store_true')
    p.add_argument('--raw-context', action='store_true')
    p.add_argument('--color-refine', action='store_true')
    p.add_argument('--upper-repair', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--max-consecutive-errors', type=int, default=3)
    return p

def source_rows(db, token, source, table='inputs'):
    index = {'inputs': 'input_source', 'neighbors': 'neighbor_source'}[table]
    return [json.loads(p) for (p,) in db.execute(f'SELECT payload FROM {table} INDEXED BY {index} WHERE run=? AND source=? ORDER BY id', (token, source))]

def main(argv=None):
    args = parser().parse_args(argv)
    if not args.input:
        raise SystemExit('Validation failed in main: INPUT_PATH --input')
    if args.max_consecutive_errors < 1:
        raise ValueError('Invalid input or state in main (rejected condition: args.max_consecutive_errors < 1)')
    inp = Path(args.input).resolve()
    base = inp if inp.is_dir() else inp.parent
    if args.check:
        count = 0
        sources = set()
        warn = {}
        errors = []
        for r in input_rows(inp, args.source_root):
            count += 1
            for x in r['metadata_warnings']:
                warn[x] = warn.get(x, 0) + 1
            if r['source_path'] not in sources:
                if not Path(r['source_path']).is_file():
                    errors.append(r['source_path'])
                sources.add(r['source_path'])
        print(json.dumps({'rows': count, 'sources': len(sources), 'missing_sources': errors[:20], 'missing_source_count': len(errors), 'warnings': warn, 'inference': False, 'images_written': 0}, ensure_ascii=False, indent=2))
        return 1 if errors or count == 0 else 0
    modelpath = Path(args.model)
    if not modelpath.is_file():
        raise FileNotFoundError(modelpath)
    modelhash = digest(modelpath)
    if args.model_sha256 and modelhash.lower() != args.model_sha256.lower():
        raise ValueError('Invalid input or state in main (rejected condition: args.model_sha256 and modelhash.lower() != args.model_sha256.lower())')
    out = Path(args.output).resolve() if args.output else base.parent / 'crop_final'
    if out == base or out in base.parents:
        raise ValueError('Invalid input or state in main (rejected condition: out == base or out in base.parents)')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'crops').mkdir(exist_ok=True)
    lock = (out / '.run.lock').open('a+b')
    lock.seek(0)
    import msvcrt
    if (out / '.run.lock').stat().st_size == 0:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    config = {k: v for (k, v) in vars(args).items() if k not in ('force', 'check')}
    helpers = digest(Path(camera_selection.__file__)) + digest(Path(camera_selection.policy.__file__))
    signature = hashlib.sha256((jsontext(config) + digest(Path(__file__)) + helpers + modelhash).encode()).hexdigest()
    db = sqlite3.connect(out / 'state.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS inputs(run TEXT,id TEXT,source TEXT,payload TEXT,PRIMARY KEY(run,id))')
    db.execute('CREATE TABLE IF NOT EXISTS neighbors(run TEXT,id TEXT,source TEXT,payload TEXT,PRIMARY KEY(run,id))')
    db.execute('CREATE TABLE IF NOT EXISTS completed(run TEXT,source TEXT,fingerprint TEXT,payload TEXT,PRIMARY KEY(run,source))')
    token = hashlib.sha256((signature + str(time.time_ns())).encode()).hexdigest()
    count = 0
    for r in input_rows(inp, args.source_root):
        db.execute('INSERT INTO inputs VALUES(?,?,?,?)', (token, r['record_id'], r['source_path'], jsontext(r)))
        count += 1
        if count % 500 == 0:
            db.commit()
    if not count:
        raise ValueError('Invalid input or state in main (rejected condition: not count)')
    neighbor_input = Path(args.neighbors).resolve() if args.neighbors else inp
    neighbor_table = 'inputs' if neighbor_input.resolve() == inp.resolve() else 'neighbors'
    if neighbor_table == 'neighbors':
        for r in input_rows(neighbor_input, args.source_root):
            db.execute('INSERT INTO neighbors VALUES(?,?,?,?)', (token, r['record_id'], r['source_path'], jsontext(r)))
    db.execute('CREATE INDEX IF NOT EXISTS input_source ON inputs(run,source)')
    db.execute('CREATE INDEX IF NOT EXISTS neighbor_source ON neighbors(run,source)')
    db.commit()
    rawmap = {}
    for log in args.bed_log:
        for r in csv_rows(log):
            if first(r, 'target_path') and all((number(r, k) is not None for k in ('x1', 'y1', 'x2', 'y2'))):
                target = str(resolve_path(r['target_path'], Path(log).parent, args.source_root))
                rawmap[target] = (resolve_path(r['source_path'], Path(log).parent, args.source_root), [int(float(r[k])) for k in ('x1', 'y1', 'x2', 'y2')])
    from ultralytics import YOLO
    model = YOLO(str(modelpath))
    total = db.execute('SELECT COUNT(DISTINCT source) FROM inputs WHERE run=?', (token,)).fetchone()[0]
    sources = db.execute('SELECT DISTINCT source FROM inputs WHERE run=? ORDER BY source', (token,)).fetchall()

    def outputs_valid(records):
        try:
            for r in records:
                if not r.get('candidate_available'):
                    continue
                (x0, y0, x1, y1) = r['crop_source_box']
                if unpack_region(r).shape != (y1 - y0, x1 - x0):
                    return False
            return True
        except Exception:
            return False
    failures = 0
    consecutive = 0
    processed = set()
    query_seconds = 0.0
    frame_started = time.perf_counter()
    for (sourcepath,) in tqdm(sources, total=total, desc='bed → crop', unit='frame'):
        query_started = time.perf_counter()
        rows = source_rows(db, token, sourcepath)
        nr = rows if neighbor_table == 'inputs' else source_rows(db, token, sourcepath, 'neighbors')
        query_seconds += time.perf_counter() - query_started
        records = []
        fingerprint = ''
        try:
            sourcehash = digest(sourcepath)
            for row in rows:
                expected = first(row['input_record_json'], 'source_sha256')
                if expected and expected.lower() != sourcehash:
                    raise ValueError('Invalid input or state in main (rejected condition: expected and expected.lower() != sourcehash)')
            raw_info = rawmap.get(sourcepath)
            raw_candidates = []
            for row in rows:
                original = row['input_record_json']
                raw_path = first(original, 'raw_source_path')
                rb = [number(original, 'bed_' + axis + '_raw') for axis in ('x1', 'y1', 'x2', 'y2')]
                if raw_path:
                    if any((v is None for v in rb)):
                        if first(original, 'raw_offset_status') == 'UNAVAILABLE_EXISTING_BED' and (not args.raw_context):
                            continue
                        raise ValueError('Invalid input or state in main (rejected condition: any((v is None for v in rb)))')
                    raw_candidates.append((resolve_path(raw_path, base, args.source_root), [int(v) for v in rb]))
            if raw_candidates:
                if any((x != raw_candidates[0] for x in raw_candidates)) or (raw_info and raw_info != raw_candidates[0]):
                    raise ValueError('Invalid input or state in main (rejected condition: any((x != raw_candidates[0] for x in raw_candidates)) or (raw_info and raw_info != raw_candidates[0]))')
                raw_info = raw_candidates[0]
            rawhash = digest(raw_info[0]) if raw_info else ''
            fingerprint = hashlib.sha256((signature + sourcehash + rawhash + jsontext(raw_info[1] if raw_info else []) + jsontext(rows) + jsontext(nr)).encode()).hexdigest()
            old = db.execute('SELECT fingerprint,payload FROM completed WHERE run=? AND source=?', (signature, sourcepath)).fetchone()
            if not args.force and old and (old[0] == fingerprint):
                saved = json.loads(old[1])
                if not any((r['status'] == 'ERROR' for r in saved)) and outputs_valid(saved):
                    processed.add(sourcepath)
                    consecutive = 0
                    continue
            source = load_image(sourcepath)
            (h, w) = source.shape[:2]
            full = predict(model, source, args.whole_size, args.device, True) if rows[0]['source_coordinate_space'] != 'bbox_local' else None
            if full is not None:
                full = prepare_measurement(full, [0, 0, w, h], source)
            selected = []
            for r in rows:
                neighbors = [(n['physical_slot'], [n['anchor_x'], n['anchor_y']]) for n in nr if n['physical_slot'] != r['physical_slot'] and n['anchor_x'] is not None and (n['anchor_y'] is not None)]
                (record, mask, box, im) = infer_one(r, source, full, neighbors, model, args, raw_info)
                record['source_sha256'] = digest(record['source_path']) if record['source_path'] != sourcepath else sourcehash
                record['run_signature'] = signature
                selected.append((record, mask, box, im))
            selected = clean_shared_fragments(selected, raw_info)
            if full is not None:
                output_slots = {row['physical_slot'] for row in rows}
                extra = []
                for n in nr:
                    if n['physical_slot'] in output_slots:
                        continue
                    if n['occupancy_human'].upper() == 'EMPTY':
                        continue
                    if n['anchor_x'] is None or n['anchor_y'] is None:
                        continue
                    if not n['cell_width'] or not n['cell_height']:
                        continue
                    ns = [(v['physical_slot'], [v['anchor_x'], v['anchor_y']]) for v in nr if v['physical_slot'] != n['physical_slot'] and v['anchor_x'] is not None and (v['anchor_y'] is not None)]
                    (nc, nm) = measure(full, [0, 0, w, h], source, [n['anchor_x'], n['anchor_y']], n['cell_width'], n['cell_height'], ns)
                    nd = select_target(nc)
                    if nd.selected_index is not None:
                        extra.append((n['physical_slot'], nm[nd.selected_index]))
                for (rec, mask, box, _) in selected:
                    if mask is None:
                        continue
                    bb = list(box)
                    if rec['source_coordinate_space'] == 'raw_camera_pixels':
                        (ox, oy) = raw_info[1][:2]
                        bb = [box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy]
                    for (slot, other) in extra:
                        (_, shared, fraction) = overlap(mask, bb, other, [0, 0, w, h])
                        if shared >= 1024 and fraction >= 0.2:
                            rec['status'] = 'REVIEW_NEIGHBOR_OWNERSHIP'
                            rec['auto_qc_pass'] = False
                            rec.setdefault('overlap_slots', []).append(slot)
            for (i, (a, am, ab, _)) in enumerate(selected):
                if am is None:
                    continue
                aa = list(ab)
                if a['source_coordinate_space'] == 'raw_camera_pixels':
                    (ox, oy) = raw_info[1][:2]
                    aa = [ab[0] - ox, ab[1] - oy, ab[2] - ox, ab[3] - oy]
                for (b, bm, bb, _) in selected[i + 1:]:
                    if bm is None:
                        continue
                    bt = list(bb)
                    if b['source_coordinate_space'] == 'raw_camera_pixels':
                        (ox, oy) = raw_info[1][:2]
                        bt = [bb[0] - ox, bb[1] - oy, bb[2] - ox, bb[3] - oy]
                    (oi, shared, fraction) = overlap(am, aa, bm, bt)
                    if shared >= 1024 and fraction >= 0.2:
                        for (rec, other) in ((a, b), (b, a)):
                            rec['status'] = 'REVIEW_NEIGHBOR_OWNERSHIP'
                            rec['auto_qc_pass'] = False
                            rec.setdefault('overlap_slots', []).append(other['physical_slot'])
            for (rec, m, box, im) in selected:
                records.append(prepare_candidate(rec, m, box))
            consecutive = 0
        except Exception as e:
            report_error(sourcepath, e)
            failures += 1
            consecutive += 1
            records = [{**r, 'status': 'ERROR', 'error': repr(e), 'has_crop': False, 'auto_qc_pass': False, 'training_eligible': False, 'run_signature': signature} for r in rows]
        db.execute('INSERT OR REPLACE INTO completed VALUES(?,?,?,?)', (signature, sourcepath, fingerprint, jsontext(records)))
        db.commit()
        processed.add(sourcepath)
        if consecutive >= args.max_consecutive_errors:
            print('main: . NOT_RUN .', file=sys.stderr)
            break

    def candidate_rows():
        for (sp,) in sources:
            if sp not in processed:
                for row in source_rows(db, token, sp):
                    yield {**row, 'status': 'NOT_RUN', 'has_crop': False, 'auto_qc_pass': False, 'training_eligible': False, 'run_signature': signature}
                continue
            item = db.execute('SELECT payload FROM completed WHERE run=? AND source=?', (signature, sp)).fetchone()
            for r in json.loads(item[0]):
                yield r
    if failures:
        write_csv(out / 'crop_candidates_pending.csv', candidate_rows())
        (out / 'run_summary.json').write_text(json.dumps({'version': VERSION, 'error_frames': failures, 'state': 'needs_review'}, ensure_ascii=False), encoding='utf-8')
        db.close()
        lock.close()
        return 1
    timings = {'candidate_frames_seconds': time.perf_counter() - frame_started, 'input_query_seconds': query_seconds}
    print('main: (Input ):' + jsontext(timings), flush=True)
    print('main: CAM0/CAM1 · .', flush=True)
    phase_started = time.perf_counter()
    camera_selection.build(db, candidate_rows())
    camera_selection.export(db, out)
    timings['camera_selection_seconds'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    db.execute('CREATE TABLE IF NOT EXISTS materialized(id TEXT PRIMARY KEY,fingerprint TEXT,payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS final_crops(run TEXT,id TEXT,view TEXT,pair TEXT,payload TEXT,PRIMARY KEY(run,id),UNIQUE(run,view,pair))')
    output_rows_count = 0
    source_cache = SourceImageCache()
    for record in tqdm(candidate_rows(), desc='main: crop', unit='candidate'):
        selection = camera_selection.decision(db, record)
        if selection is None:
            continue
        record = dict(record)
        record.update(selection)
        if selection['selection_status'] == 'SELECTED_REVIEW':
            record['auto_qc_pass'] = False
        if selection['selection_status'] == 'NO_USABLE_DETECTION':
            record.update(has_crop=False, crop_path='', crop_sha256='')
        elif record.get('candidate_available'):
            fp = hashlib.sha256(jsontext(record).encode()).hexdigest()
            old = db.execute('SELECT fingerprint,payload FROM materialized WHERE id=?', (record['record_id'],)).fetchone()
            previous = json.loads(old[1]) if old else {}
            if old and old[0] == fp and Path(previous['crop_path']).is_file() and (digest(previous['crop_path']) == previous['crop_sha256']):
                record = previous
            else:
                record = write_prepared(record, source_cache.get(record['source_path'], record['source_sha256']), out)
                db.execute('INSERT OR REPLACE INTO materialized VALUES(?,?,?)', (record['record_id'], fp, jsontext(record)))
        (pair, _) = camera_selection.keys(record)
        db.execute('INSERT INTO final_crops VALUES(?,?,?,?,?)', (token, record['record_id'], record['observation_view'], pair, jsontext(record)))
        output_rows_count += 1
        if output_rows_count % 256 == 0:
            db.commit()
    db.commit()

    def final_rows():
        for (payload,) in db.execute('SELECT payload FROM final_crops WHERE run=? ORDER BY pair,view', (token,)):
            yield json.loads(payload)
    write_csv(out / 'crop_manifest.csv', final_rows())
    timings['crop_export_seconds'] = time.perf_counter() - phase_started
    summary = {'version': VERSION, 'run_signature': signature, 'model_sha256': modelhash, 'candidate_rows': count, 'rows': output_rows_count, 'frames': total, 'error_frames': failures, 'statuses': {}, 'has_crop': 0, 'auto_qc_pass': 0, 'training_eligible': 0, 'scope': 'One top observation per physical key; CAM2 preserved; all candidates retained as metrics'}
    summary['timings'] = timings
    for r in final_rows():
        s = r['status']
        summary['statuses'][s] = summary['statuses'].get(s, 0) + 1
        summary['has_crop'] += int(r.get('has_crop', False))
        summary['auto_qc_pass'] += int(r.get('auto_qc_pass', False))
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pending_path = out / 'crop_candidates_pending.csv'
    if pending_path.is_file():
        pending_path.unlink()
    retire_superseded_crops(db, token, out)
    prune_scratch_rows(db, token)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    db.close()
    lock.close()
    return 1 if failures else 0
'Select the owned plant mask and save contextual RGB crops with explicit mask provenance. Existing physical slots, camera-selection policy and REVIEW states are preserved.'
from dataclasses import dataclass, field
from typing import Sequence, Mapping, Any
import math

@dataclass(frozen=True)
class GuardConfig:
    max_center_distance_cell: float = 0.55
    min_core_coverage: float = 0.2
    min_green_fraction: float = 0.2
    ambiguity_margin: float = 0.15

@dataclass(frozen=True)
class TargetDecision:
    selected_index: int | None
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    training_eligible: bool = False

def select_target(candidates: Sequence[Mapping[str, Any]], *, identity_conflict: bool=False, physical_identity_verified: bool=False, morphology_verified: bool=False, config: GuardConfig=GuardConfig()) -> TargetDecision:
    if identity_conflict:
        return TargetDecision(None, 'REVIEW_IDENTITY_CONFLICT', ('TEMPORAL_OR_PHYSICAL_NAME_CONFLICT',))
    if not candidates:
        return TargetDecision(None, 'NO_DETECTION')
    valid = []
    for c in candidates:
        try:
            values = [float(c[k]) for k in ['center_distance_cell', 'core_coverage', 'green_fraction']]
        except (KeyError, ValueError, TypeError):
            continue
        if not all((math.isfinite(v) for v in values)):
            continue
        (d, core, green) = values
        if d < 0 or not (0 <= core <= 1 and 0 <= green <= 1):
            continue
        if d <= config.max_center_distance_cell and core >= config.min_core_coverage and (green >= config.min_green_fraction):
            valid.append(c)
    if not valid:
        return TargetDecision(None, 'NO_SUPPORTED_TARGET', ('NO_CANDIDATE_SUPPORTS_ANCHOR',))
    valid.sort(key=lambda c: (float(c['center_distance_cell']), int(c['index'])))
    if len(valid) > 1 and float(valid[1]['center_distance_cell']) - float(valid[0]['center_distance_cell']) < config.ambiguity_margin:
        return TargetDecision(None, 'REVIEW_AMBIGUOUS', ('MULTIPLE_SUPPORTED_CANDIDATES',))
    best = valid[0]
    index = int(best['index'])
    if best.get('neighbor_claims'):
        return TargetDecision(None, 'REVIEW_NEIGHBOR_CLAIM', tuple((str(n) for n in best['neighbor_claims'])))
    reasons = []
    if best.get('edge_touch'):
        reasons.append('SEARCH_EDGE_CONTACT')
    if best.get('fragmented'):
        reasons.append('DISCONNECTED_OR_PARTIAL_SHAPE')
    if reasons:
        return TargetDecision(index, 'REVIEW_SHAPE', tuple(reasons))
    verified = physical_identity_verified and morphology_verified
    return TargetDecision(index, 'VERIFIED_TARGET' if verified else 'CANDIDATE', (), verified)
'Select the owned plant mask and save contextual RGB crops with explicit mask provenance. Existing physical slots, camera-selection policy and REVIEW states are preserved.'
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class RouteDecision:
    source: str | None
    status: str
    reasons: tuple[str, ...] = ()
    training_eligible: bool = False

def choose_context(local: TargetDecision, whole: TargetDecision, mask_iou: float | None, *, identity_conflict=False, agreement_threshold=0.65):
    if identity_conflict:
        return RouteDecision(None, 'REVIEW_IDENTITY_CONFLICT')
    clean = lambda d: d.selected_index is not None and d.status in ('CANDIDATE', 'VERIFIED_TARGET')
    (lc, wc) = (clean(local), clean(whole))
    if lc and wc:
        if mask_iou is None or not math.isfinite(mask_iou) or (not 0 <= mask_iou <= 1):
            return RouteDecision(None, 'REVIEW_INVALID_COMPARISON')
        if mask_iou >= agreement_threshold:
            return RouteDecision('local', 'CANDIDATE_CONTEXT_AGREEMENT')
        return RouteDecision('local', 'REVIEW_CONTEXT_DISAGREEMENT', ('CROP_AND_FULL_BED_MASKS_DIFFER',))
    blockers = {'REVIEW_AMBIGUOUS', 'REVIEW_NEIGHBOR_CLAIM', 'REVIEW_IDENTITY_CONFLICT'}
    if local.status in blockers or whole.status in blockers:
        return RouteDecision(None, 'REVIEW_CANDIDATE_CONFLICT', (local.status, whole.status))
    if lc:
        return RouteDecision('local', 'CANDIDATE_LOCAL_ONLY', ('FULL_BED_UNCONFIRMED',))
    if wc:
        return RouteDecision('whole', 'REVIEW_FULL_CONTEXT_RECOVERY', ('CROP_INPUT_UNCONFIRMED',))
    if local.selected_index is not None:
        return RouteDecision('local', 'REVIEW_SHAPE')
    if whole.selected_index is not None:
        return RouteDecision('whole', 'REVIEW_SHAPE')
    return RouteDecision(None, 'NO_SUPPORTED_TARGET', (local.status, whole.status))
'Select the owned plant mask and save contextual RGB crops with explicit mask provenance. Existing physical slots, camera-selection policy and REVIEW states are preserved.'

def prefer_full_shape(local, whole, iou, *, local_area, whole_area, intersection, local_conf, whole_conf):
    detail = {'replace_with_whole': False, 'status': 'KEEP_EXISTING_ROUTE', 'training_eligible': False}
    if local.status != 'CANDIDATE' or whole.status != 'CANDIDATE' or iou is None:
        return detail
    unsupported = (local_area - intersection) / max(1, local_area)
    missing = (whole_area - intersection) / max(1, whole_area)
    detail.update(local_unsupported_share=unsupported, whole_missing_from_local_share=missing, confidence_gain=whole_conf - local_conf)
    if iou >= 0.75 and unsupported >= 0.04 and (missing >= 0.06) and (whole_conf - local_conf >= 0.05):
        detail.update(replace_with_whole=True, status='REVIEW_FULL_CONTEXT_SHAPE_REPLACEMENT')
    return detail
'Select the owned plant mask and save contextual RGB crops with explicit mask provenance. Existing physical slots, camera-selection policy and REVIEW states are preserved.'
import cv2, numpy as np, math

def clean_detached(mask, anchor_local, *, max_removed_share=0.05):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('2D mask required')
    if not mask.any():
        return (mask.copy(), {'status': 'EMPTY_MASK', 'changed': False})
    (n, labels, stats, centers) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_label = 1 + int(np.argmax(areas))
    main = labels == main_label
    removed = int(mask.sum() - main.sum())
    share = removed / float(mask.sum())
    detail = {'status': 'NO_DETACHED_COMPONENT', 'changed': False, 'component_count': n - 1, 'removed_pixels': 0, 'proposed_removed_pixels': removed, 'proposed_removed_share': share, 'main_component_share': float(main.sum() / mask.sum()), 'training_eligible': False}
    if n <= 2:
        return (mask.copy(), detail)
    if share > max_removed_share:
        return (mask.copy(), {**detail, 'status': 'REVIEW_LARGE_SEPARATE_COMPONENT'})
    (x, y) = anchor_local
    if not all((math.isfinite(float(v)) for v in (x, y))):
        return (mask.copy(), {**detail, 'status': 'REVIEW_INVALID_ANCHOR'})
    distance = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
    (ix, iy) = (int(round(x)), int(round(y)))
    if not (0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]) or distance[iy, ix] > max(8, 0.05 * min(mask.shape)):
        return (mask.copy(), {**detail, 'status': 'REVIEW_ANCHOR_NOT_ON_MAIN'})
    return (main, {**detail, 'status': 'REVIEW_DETACHED_COMPONENT_REMOVAL', 'changed': True, 'removed_pixels': removed})
if __name__ == '__main__':
    raise SystemExit(main())
