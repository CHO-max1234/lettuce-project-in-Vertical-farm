"""Frozen whole-frame and local segmentation candidates, cache and final record serialization."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from runtime import yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = CYCLE.MAIN_PREPROCESS
DATA_ROOT = CYCLE.BBOX_OUTPUT
SAMPLE_MANIFEST = DATA_ROOT / 'test_results' / 'sample_manifest.csv'
SLOT_MANIFEST = DATA_ROOT / 'test_results' / 'slot_manifest.csv'
BBOX_MANIFEST = DATA_ROOT / 'testimages_bbox' / 'bbox_manifest.csv'
OUTPUT_ROOT = DATA_ROOT / 'testimages_bbox'
CACHE_ROOT = OUTPUT_ROOT / 'ver1' / '_inference_cache'
VISUAL_REFERENCE = CODE_ROOT / '연습용/261001_30장_독립시각참조.csv'
MODEL_PATH = CYCLE.PLANT_MODEL
MODEL_SHA256 = 'a2ded62bba4d577f4ead18907b824c587e179357c8089b72dabde41b755d6fe8'
WHOLE_SIZE = 1280
LOCAL_SIZE = 512
DEVICE = yolo_device()
SAVE_WORKERS = 4
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from types import SimpleNamespace
os.environ['YOLO_AUTOINSTALL'] = 'false'
sys.dont_write_bytecode = True
import cv2
import numpy as np
cv2.setNumThreads(1)

def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.writing')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def write_csv(path, rows):
    fields = list(dict.fromkeys((key for row in rows for key in row))) or ['status']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else clean(v) for (key, v) in row.items()})

def load_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Invalid input or state in load_image (image is None): {path} ')
    return image

def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    (ok, encoded) = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Invalid input or state in save_image (not ok): {path} ')
    encoded.tofile(path)

class Predictor:

    def __init__(self):
        self.model = None
        self.hits = 0
        self.misses = 0
        self.whole_calls = 0
        self.local_calls = 0

    def __call__(self, ignored_model, image, size, device, whole=False):
        if whole:
            self.whole_calls += 1
        else:
            self.local_calls += 1
        key = hashlib.sha256(image.tobytes() + str((image.shape, size, whole, MODEL_SHA256, 0.2, 0.7)).encode()).hexdigest()
        path = CACHE_ROOT / (key + '.npz')
        import torch
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                shape = tuple((int(v) for v in data['shape']))
                masks = np.unpackbits(data['packed'], bitorder='little', count=int(np.prod(shape))).reshape(shape).astype(np.float32)
                points = data['points']
                ends = data['ends']
                polygons = np.split(points, ends[:-1]) if len(ends) else []
                boxes = SimpleNamespace(conf=torch.from_numpy(data['conf']), xyxy=torch.from_numpy(data['boxes']))
            self.hits += 1
            return SimpleNamespace(boxes=boxes, masks=SimpleNamespace(data=torch.from_numpy(masks), xy=polygons) if len(masks) else None)
        if self.model is None:
            from ultralytics import YOLO
            if digest(MODEL_PATH) != MODEL_SHA256:
                raise ValueError('Invalid input or state in __call__ (digest(MODEL_PATH) != MODEL_SHA256)')
            self.model = YOLO(str(MODEL_PATH))
        pred = self.model.predict(image, imgsz=size, conf=0.2, iou=0.7, retina_masks=True, rect=whole, verbose=False, device=device)[0]
        masks = pred.masks.data.cpu().numpy() > 0.5 if pred.masks is not None else np.empty((0, *image.shape[:2]), dtype=bool)
        polygons = pred.masks.xy if pred.masks is not None else []
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.writing.npz')
        np.savez_compressed(temp, packed=np.packbits(masks, bitorder='little'), shape=np.array(masks.shape), boxes=pred.boxes.xyxy.cpu().numpy(), conf=pred.boxes.conf.cpu().numpy(), points=np.concatenate(polygons) if polygons else np.empty((0, 2), dtype=np.float32), ends=np.cumsum([len(p) for p in polygons]))
        temp.replace(path)
        self.misses += 1
        return pred

def candidates(pred, source, box, origin):
    result = []
    if pred.masks is None:
        return result
    (x1, y1, x2, y2) = box
    for (index, mask) in enumerate(pred.masks.data.cpu().numpy() > 0.5):
        if mask.shape != (y2 - y1, x2 - x1):
            raise ValueError('Invalid input or state in candidates (mask.shape != (y2 - y1, x2 - x1))')
        if not mask.any():
            continue
        contour = np.asarray(pred.masks.xy[index], dtype=np.float32)
        moment = cv2.moments(contour.reshape(-1, 1, 2)) if len(contour) >= 3 else {}
        (ys, xs) = np.where(mask)
        cx = moment['m10'] / moment['m00'] if moment.get('m00', 0) else float(xs.mean())
        cy = moment['m01'] / moment['m00'] if moment.get('m00', 0) else float(ys.mean())
        global_mask = np.zeros(source.shape[:2], dtype=bool)
        global_mask[y1:y2, x1:x2] = mask
        result.append(dict(mask=global_mask, cx=cx + x1, cy=cy + y1, conf=float(pred.boxes.conf[index]), index=index, box=[int(xs.min() + x1), int(ys.min() + y1), int(xs.max() + x1 + 1), int(ys.max() + y1 + 1)], origin=origin, prediction_box=box))
    return result

def centered_box(row, source, scale):
    (h, w) = source.shape[:2]
    (ax, ay) = (float(row['eval_anchor_x']), float(row['eval_anchor_y']))
    (cw, ch) = (float(row['cell_width']), float(row['cell_height']))
    if not (0 <= ax < w and 0 <= ay < h):
        return None
    (bw, bh) = (min(w, max(1, math.ceil(scale * cw))), min(h, max(1, math.ceil(scale * ch))))
    x1 = min(w - bw, max(0, round(ax - bw / 2)))
    y1 = min(h - bh, max(0, round(ay - bh / 2)))
    return [x1, y1, x1 + bw, y1 + bh]

def export_frame(sample, rows, source, decisions, out, version, reference):
    camera = sample['camera']
    stem = Path(sample['image_name']).stem
    (h, w) = source.shape[:2]
    canvas = source.copy()
    records = []
    for row in rows:
        decision = decisions[row['record_id']]
        mask = decision.get('mask')
        record = {**row, 'experiment_version': version, 'selection_status': decision['status'], 'selection_reasons': decision.get('reasons', []), 'selected_route': decision.get('route', ''), 'has_crop': False, 'training_eligible': False, 'production_verified': False, 'bbox_crop_path': '', 'masked_crop_path': '', 'diagnostic_mask_path': '', 'mask_pixels': '', 'final_bbox': '', 'area_if_unavailable': '', 'visual_occupancy': reference.get(row['record_id'], {}).get('visual_occupancy', 'NOT_REVIEWED')}
        for field in ('bbox_crop_path', 'bbox_crop_sha256', 'bbox_crop_status', 'crop_kind', 'crop_x1', 'crop_y1', 'crop_x2', 'crop_y2', 'crop_width', 'crop_height', 'anchor_x', 'anchor_y', 'bbox_center_source', 'bbox_version', 'search_x1', 'search_y1', 'search_x2', 'search_y2', 'is_crop_clipped', 'mask_cleanup_applied', 'bbox_handoff_status', 'bbox_qc_flags', 'target_confirmation', 'target_guard_reasons', 'target_bbox_retention', 'neighbor_centers_in_crop', 'neighbor_center_count_in_crop', 'downstream_target_policy', 'visibility', 'bbox', 'seg_center_adopted'):
            record['upstream_' + field] = row.get(field, '')
        record.update(bbox_crop_sha256='', bbox_crop_status='NOT_SAVED_' + decision['status'], crop_kind='RAW_RGB_TARGET_BODY_CONTEXT', crop_x1='', crop_y1='', crop_x2='', crop_y2='', crop_width='', crop_height='', target_anchor_x=float(row.get('eval_anchor_x', row['anchor_x'])), target_anchor_y=float(row.get('eval_anchor_y', row['anchor_y'])), crop_anchor_x='', crop_anchor_y='', mask_sha256='', search_x1='', search_y1='', search_x2='', search_y2='', is_crop_clipped='', source_boundary_contact='', bbox='', mask_cleanup_applied=False, seg_center_adopted=False, mask_cleanup_policy='DETACHED_COMPONENTS_MAX_5PCT_PRESERVE_LARGE_LEAVES', bbox_version=version + '_final_target_body', bbox_center_source='REGISTERED_PHYSICAL_ANCHOR', bbox_handoff_status='TARGET_UNAVAILABLE_' + decision['status'], target_confirmation=decision['status'], target_guard_reasons='|'.join(decision.get('reasons', [])), target_bbox_retention='', neighbor_centers_in_crop='', neighbor_center_count_in_crop='', visibility='OUT_OF_FOV' if decision['status'] == 'OUT_OF_FOV' else 'UNKNOWN', downstream_target_policy='SKIP_UNAVAILABLE_TARGET_NO_NEW_CANDIDATE', handoff_schema_version='FINAL_TARGET_BODY_V1', crop_coordinate_space='bed_pixels', mask_coordinate_space='crop_local_pixels', seg_metadata_scope='UPSTREAM_GEOMETRY_REFERENCE_ONLY')
        record.update(anchor_x=record['target_anchor_x'], anchor_y=record['target_anchor_y'])
        final_qc = list(decision.get('reasons', []))
        if row.get('anchor_registration_review'):
            final_qc.append('ANCHOR_REGISTRATION_REVIEW')
        if row.get('bed_identity_status') != 'CONFIRMED':
            final_qc.append('OCR_IDENTITY_REVIEW')
        if decision.get('review_required'):
            final_qc.append('TARGET_OWNERSHIP_REVIEW')
        record['bbox_qc_flags'] = '|'.join(dict.fromkeys(final_qc))
        for (key, value) in decision.items():
            if key not in ('mask', 'baseline_record', 'status', 'reasons', 'route'):
                record['decision_' + key] = value
        if mask is not None and mask.any():
            (ys, xs) = np.where(mask)
            margin = max(6, round(0.04 * min(float(row['cell_width']), float(row['cell_height']))))
            box = [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin), min(w, int(xs.max() + 1) + margin), min(h, int(ys.max() + 1) + margin)]
            (x1, y1, x2, y2) = box
            crop = source[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2]
            masked = crop.copy()
            masked[~crop_mask] = 0
            name = stem + '__' + row['physical_slot'] + '.png'
            rgb_path = out / camera / 'crops' / name
            masked_path = out / camera / 'masked_crops' / name
            mask_path = out / camera / 'masks' / name
            save_image(rgb_path, crop)
            save_image(masked_path, masked)
            save_image(mask_path, crop_mask.astype(np.uint8) * 255)
            record.update(has_crop=True, bbox_crop_path=str(rgb_path), masked_crop_path=str(masked_path), diagnostic_mask_path=str(mask_path), mask_pixels=int(mask.sum()), final_bbox=box, bbox=box, bbox_crop_sha256=digest(rgb_path), mask_sha256=digest(mask_path), source_boundary_contact=bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or (ys.max() == h - 1)))
            record.update(bbox_crop_status='SAVED_TARGET_CONTEXT', crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2, crop_width=x2 - x1, crop_height=y2 - y1, crop_anchor_x=record['target_anchor_x'] - x1, crop_anchor_y=record['target_anchor_y'] - y1, search_x1=x1, search_y1=y1, search_x2=x2, search_y2=y2, is_crop_clipped=bool(x1 == 0 or y1 == 0 or x2 == w or (y2 == h)), mask_cleanup_applied=int(decision.get('removed_detached_pixels', 0) or 0) + int(decision.get('ownership_removed_pixels', 0) or 0) > 0, mask_cleanup_policy='DETACHED_MAX_5PCT_AND_TWO_SIDED_CONNECTED_OWNERSHIP_BODY_PROTECTED', bbox_handoff_status='TARGET_BODY_READY_SAMPLE_REVIEW_PENDING', target_confirmation='TARGET_BODY_SUPPORTED', target_bbox_retention=1.0, target_retention_definition='SELECTED_SOURCE_MASK_ONLY', visibility='VISIBLE', downstream_target_policy='USE_FINAL_TARGET_ANCHOR_AND_MASK_NO_REASSIGNMENT')
            neighbors_in_crop = [other['physical_slot'] for other in rows if other['record_id'] != row['record_id'] and x1 <= float(other.get('eval_anchor_x', other['anchor_x'])) < x2 and (y1 <= float(other.get('eval_anchor_y', other['anchor_y'])) < y2)]
            record.update(neighbor_centers_in_crop='|'.join(neighbors_in_crop), neighbor_center_count_in_crop=len(neighbors_in_crop))
            if record['is_crop_clipped']:
                final_qc.append('SOURCE_EDGE_CONTEXT')
            if record['source_boundary_contact']:
                final_qc.append('SELECTED_MASK_TOUCHES_SOURCE_BORDER')
            record['bbox_qc_flags'] = '|'.join(dict.fromkeys(final_qc))
            (contours, _) = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (30, 230, 30), max(1, round(w / 1000)))
            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (30, 230, 30), max(1, round(w / 1000)))
        ax = float(row.get('eval_anchor_x', row['anchor_x']))
        ay = float(row.get('eval_anchor_y', row['anchor_y']))
        color = (20, 230, 20) if record['has_crop'] else (0, 200, 255)
        if 0 <= ax < w and 0 <= ay < h:
            point = (round(ax), round(ay))
            cv2.drawMarker(canvas, point, color, cv2.MARKER_CROSS, max(8, round(w * 0.008)), max(1, round(w / 1200)))
            label = row['physical_slot'] + (' TARGET' if record['has_crop'] else ' HOLD')
            scale = max(0.35, w / 2500)
            cv2.putText(canvas, label, (point[0] + 4, max(14, point[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, (point[0] + 4, max(14, point[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        records.append(record)
    header = np.zeros((max(50, round(w * 0.045)), w, 3), dtype=np.uint8)
    text = f"{camera.upper()} {version} | {stem} | targets={sum((r['has_crop'] for r in records))}/{len(records)}"
    cv2.putText(header, text, (8, round(header.shape[0] * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 2500), (255, 255, 255), 1, cv2.LINE_AA)
    overlay_path = out / camera / 'overlays' / (stem + '_overlay.jpg')
    save_image(overlay_path, np.vstack([header, canvas]))
    cols = sorted({int(r['physical_slot'][1:]) for r in rows})
    (tile_w, tile_h) = (190, 235)
    sheet = np.full((70 + 2 * tile_h, max(1, len(cols)) * tile_w, 3), 24, dtype=np.uint8)
    cv2.putText(sheet, stem + ' | ' + version, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for record in records:
        ix = cols.index(int(record['physical_slot'][1:]))
        iy = 0 if record['row'] == 't' else 1
        (x, y) = (ix * tile_w, 65 + iy * tile_h)
        color = (30, 220, 30) if record['has_crop'] else (0, 200, 255)
        cv2.putText(sheet, record['physical_slot'] + ' ' + ('TARGET' if record['has_crop'] else 'HOLD'), (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if record['has_crop']:
            im = load_image(record['masked_crop_path'])
            ratio = min((tile_w - 10) / im.shape[1], (tile_h - 45) / im.shape[0])
            resized = cv2.resize(im, (max(1, round(im.shape[1] * ratio)), max(1, round(im.shape[0] * ratio))))
            xx = x + (tile_w - resized.shape[1]) // 2
            sheet[y + 28:y + 28 + resized.shape[0], xx:xx + resized.shape[1]] = resized
        else:
            label = record['selection_status'][:24]
            cv2.putText(sheet, label, (x + 4, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
    sheet_path = out / camera / 'sheets' / (stem + '_crops.jpg')
    save_image(sheet_path, sheet)
    return (records, dict(camera=camera, image_name=sample['image_name'], slot_rows=len(rows), crop_count=sum((r['has_crop'] for r in records)), overlay_path=str(overlay_path), sheet_path=str(sheet_path)))

def score(records):
    counts = Counter()
    for record in records:
        counts[record['visual_occupancy'], bool(record['has_crop'])] += 1
    return {str(key): value for (key, value) in counts.items()}
