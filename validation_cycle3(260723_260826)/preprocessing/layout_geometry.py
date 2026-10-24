"""Layout geometry calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
BASELINE_ROOT = DATA_ROOT / '4. 자리배치와 bbox/자리배치_ver1/ver1'
OUTPUT_ROOT = BASELINE_ROOT.parent / 'ver2'
CAM01_MODULE = CYCLE.PROJECT_ROOT / 'preprocessing/top_geometry.py'
CAM2_MODULE = CYCLE.PROJECT_ROOT / 'preprocessing/front_geometry_base.py'
VERSION = 'S4_VER2_CURRENT_PLATE_CONSTRAINED_ROWS'
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def module(name, path):
    spec = resolved_spec(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj
A = module('s4_ver2_common_contract', CODE_ROOT / 'layout_records.py')

def decode_row(row):
    out = dict(row)
    for (k, v) in out.items():
        if not isinstance(v, str):
            continue
        if v == 'True':
            out[k] = True
        elif v == 'False':
            out[k] = False
        elif v.startswith(('[', '{')):
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    for k in ('col', 'local_col', 'reference_local_col', 'source_width', 'source_height', 'seg_id'):
        if out.get(k) not in ('', None):
            out[k] = int(out[k])
        elif k == 'seg_id':
            out[k] = None
    for k in ('x_prior', 'y_prior', 'slot_center_x', 'slot_center_y', 'anchor_x', 'anchor_y', 'cell_width', 'cell_height', 'seg_center_x', 'seg_center_y', 'seg_conf', 'seg_area', 'eval_anchor_x', 'eval_anchor_y'):
        if out.get(k) not in ('', None):
            out[k] = float(out[k])
        elif k.startswith('seg_'):
            out[k] = None
    return out

def load_inputs(fixed_development_sample=True):
    samples = json.loads((BASELINE_ROOT / 'sample_manifest.json').read_text(encoding='utf-8'))
    frames = {r['image_name']: decode_row(r) for r in A.read_csv(BASELINE_ROOT / 'frame_manifest.csv')}
    slots = defaultdict(list)
    segments = defaultdict(list)
    for r in A.read_csv(BASELINE_ROOT / 'slot_manifest.csv'):
        row = decode_row(r)
        slots[Path(row['source_path']).name].append(row)
    for r in A.read_csv(BASELINE_ROOT / 'segment_manifest.csv'):
        segments[r['image_name']].append(dict(seg_id=int(r['seg_id']), polygon=np.asarray(json.loads(r['polygon']), np.float32), bbox=tuple(json.loads(r['bbox'])), cx=float(r['center_x']), cy=float(r['center_y']), area=float(r['area']), conf=float(r['confidence']), cls=None))
    if fixed_development_sample and (len(samples) != 30 or Counter((r['camera'] for r in samples)) != Counter(cam0=10, cam1=10, cam2=10)):
        raise ValueError('Invalid input or execution contract')
    if not samples or len({s['frame_id'] for s in samples}) != len(samples):
        raise ValueError('Invalid input or execution contract')
    for sample in samples:
        sample['geometry_details'] = frames[sample['image_name']]['geometry_details']
        sample['baseline_frame'] = frames[sample['image_name']]
    return (samples, frames, slots, segments)

def usable(row):
    return row.get('slot_geometry_usable') is True

def hold_geometry(row, reason):
    row.update(slot_geometry_usable=False, cell_crop_usable=False, cell_geometry_ready=False, bbox_handoff_status='HOLD_SLOT_GEOMETRY', anchor_registration_status='UNKNOWN_REGISTRATION', anchor_registration_review=True)
    reasons = row.setdefault('geometry_review_reasons', [])
    if reason not in reasons:
        reasons.append(reason)

def preserve_contract(rows, baseline, sample):
    original = {r['physical_slot']: r for r in baseline}
    existing = {r['physical_slot']: r for r in rows}
    if len(existing) != len(rows):
        raise ValueError('Invalid input or execution contract')
    for (name, old) in original.items():
        if name not in existing:
            row = dict(old, slot_geometry_usable=False, geometry_review_reasons=['MODULE_SLOT_UNAVAILABLE'], slot_origin='PRESERVED_UNAVAILABLE_BASELINE')
            rows.append(row)
    immutable = ('record_id', 'camera', 'physical_slot', 'slot_name', 'source_path', 'source_sha256', 'source_width', 'source_height', 'ocr_current_path', 'ocr_action', 'ocr_status', 'bed_identity_status', 'identity_verified', 'label_source', 'pair_qc', 'mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_image_key', 'raw_source_path', 'raw_crop_bbox', 'source_camera', 'native_width', 'native_height')
    geometry = ('x_prior', 'y_prior', 'slot_center_x', 'slot_center_y', 'anchor_x', 'anchor_y', 'cell_bbox', 'cell_polygon', 'cell_x1', 'cell_y1', 'cell_x2', 'cell_y2', 'cell_width', 'cell_height', 'bbox', 'seg_id', 'seg_polygon', 'seg_bbox', 'seg_center_x', 'seg_center_y', 'seg_conf', 'seg_area', 'visibility', 'anchor_registration_status')
    for r in rows:
        name = r['physical_slot']
        old = original.get(name)
        if old:
            for k in immutable:
                if k in old:
                    r[k] = old[k]
            for k in geometry:
                if k in old:
                    r.setdefault('upstream_' + k, old[k])
        else:
            for key in list(r):
                if key.startswith('upstream_'):
                    r.pop(key)
            for key in ('bbox', 'seg_id', 'seg_polygon', 'seg_bbox', 'seg_center_x', 'seg_center_y', 'seg_conf', 'seg_area', 'mask_coverage', 'runner_up_coverage'):
                r[key] = None
            r.update({k: sample.get(k) for k in ('camera', 'source_sha256', 'source_width', 'source_height', 'source_camera', 'raw_source_path', 'raw_crop_bbox', 'native_width', 'native_height', 'ocr_current_path', 'ocr_action', 'ocr_status', 'bed_identity_status', 'identity_verified', 'label_source', 'pair_qc', 'mm_per_px', 'scale_source', 'scale_ready', 'scale_verified', 'scale_image_key')})
            r['source_path'] = sample['input_path']
            r['slot_name'] = name
            namespace = r.get('slot_namespace_version', 'image_left_to_right_v1')
            r['record_id'] = hashlib.sha256(f"{sample['input_path']}|{namespace}|{name}".encode()).hexdigest()[:24]
            r.setdefault('slot_origin', 'NEW_PLATE_LATTICE_GEOMETRY')
        r.update(season_id='4작기', frame_id=sample['frame_id'], mapping_version=VERSION, slot_geometry_version=VERSION, source_coordinate_space='bed_pixels', seg_metadata_scope='UPSTREAM_GEOMETRY_REFERENCE_ONLY', bbox_target_selection_executed=False, target_confirmation='BODY_SELECTION_NOT_EXECUTED', auto_qc_pass=False, training_eligible=False, downstream_selector='S3_FINAL_ver2.2_INDEPENDENT_ANCHOR_BODY_OWNERSHIP', bbox_handoff_status='SLOT_GEOMETRY_READY_TARGET_SELECTION_PENDING' if usable(r) else 'HOLD_SLOT_GEOMETRY')
        r.pop('target_available', None)
        for key in ('bbox', 'seg_id', 'seg_polygon', 'seg_bbox', 'seg_center_x', 'seg_center_y', 'seg_conf', 'seg_area', 'mask_coverage', 'runner_up_coverage'):
            if key in r:
                r.setdefault('upstream_' + key, r[key])
                r[key] = None
        if old and 'seg_status' in old:
            r['upstream_seg_status'] = old['seg_status']
        r['seg_status'] = 'NO_BODY_SELECTION_EXECUTED'
        if 'anchor_x' not in r:
            r['anchor_x'] = r['x_prior']
        if 'anchor_y' not in r:
            r['anchor_y'] = r['y_prior']
        (r['eval_anchor_x'], r['eval_anchor_y']) = (float(r['anchor_x']), float(r['anchor_y']))
        if usable(r):
            r['anchor_registration_status'] = 'REGISTERED_VER2_PLATE_ROW'
            r['anchor_registration_source'] = r.get('geometry_source', 'VER2_INDEPENDENT_PLATE_AND_HOLE_EVIDENCE')
            r['anchor_registration_review'] = r.get('geometry_review_required', False)
        else:
            r['anchor_registration_status'] = 'UNKNOWN_REGISTRATION'
            r['anchor_registration_review'] = True
        r['eval_hole_visible'] = bool(r.get('direct_hole_support', False))
        r['eval_hole_evidence'] = 'VER2_OBSERVED_HOLE' if r['eval_hole_visible'] else 'UNKNOWN'
        r['eval_hole_strength'] = 'OBSERVED' if r['eval_hole_visible'] else 'UNKNOWN'
        r['eval_hole_x'] = r.get('geometry_hole_x', r['anchor_x']) if r['eval_hole_visible'] else None
        r['eval_hole_y'] = r.get('geometry_hole_y', r['anchor_y']) if r['eval_hole_visible'] else None
        r['eval_hole_score'] = None
        (dx, dy) = sample['raw_crop_bbox'][:2]
        (r['raw_anchor_x'], r['raw_anchor_y']) = (r['anchor_x'] + dx, r['anchor_y'] + dy)
        if 'cell_bbox' in r:
            (x1, y1, x2, y2) = r['cell_bbox']
            r.update(cell_x1=x1, cell_y1=y1, cell_x2=x2, cell_y2=y2, cell_width=x2 - x1, cell_height=y2 - y1, cell_shape_valid=x2 > x1 and y2 > y1)
            r['cell_geometry_ready'] = usable(r) and r['cell_shape_valid']
            if not r['cell_shape_valid']:
                hold_geometry(r, 'INVALID_CELL_SHAPE')
        if not usable(r):
            r.update(cell_geometry_ready=False, cell_crop_usable=False)
    return rows

def structural_checks(rows, sample):
    issues = []
    (w, h) = (sample['source_width'], sample['source_height'])
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid input or execution contract')
    for r in rows:
        if r['physical_slot'] != r['row'] + str(r['col']):
            raise ValueError('Invalid input or execution contract')
        if not 1 <= int(r['col']) <= 12:
            raise ValueError('Invalid input or execution contract')
        for key in ('anchor_x', 'anchor_y'):
            if not math.isfinite(float(r[key])):
                raise ValueError('Invalid input or execution contract')
        if usable(r) and (not (0 <= r['anchor_x'] < w and 0 <= r['anchor_y'] < h)):
            hold_geometry(r, 'ANCHOR_OUT_OF_ROI')
    for row in ('t', 'b'):
        group = sorted([r for r in rows if r['row'] == row and usable(r)], key=lambda r: r['col'])
        if any((b['anchor_x'] <= a['anchor_x'] for (a, b) in zip(group, group[1:]))):
            issues.append('NONMONOTONIC_' + row.upper())
            for r in group:
                hold_geometry(r, 'NONMONOTONIC_ROW')
    by = {(r['row'], r['col']): r for r in rows}
    for col in sorted({r['col'] for r in rows}):
        (t, b) = (by.get(('t', col)), by.get(('b', col)))
        if t and b and usable(t) and usable(b) and (b['anchor_y'] <= t['anchor_y']):
            issues.append('ROW_ORDER_CONFLICT_' + str(col))
            for r in (t, b):
                hold_geometry(r, 'ROW_ORDER_CONFLICT')
    return issues

def dashed_line(canvas, p, q, ink, width=1):
    (p, q) = (np.asarray(p, float), np.asarray(q, float))
    distance = np.linalg.norm(q - p)
    for t in np.arange(0, max(distance, 1), 14):
        a = p + (q - p) * min(t / max(distance, 1), 1)
        b = p + (q - p) * min((t + 7) / max(distance, 1), 1)
        cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), ink, width, cv2.LINE_AA)

def draw_overlay(image, rows, raw, sample, meta, M):
    (h, w) = image.shape[:2]
    scale = max(0.6, min(1.1, w / 2200))
    head = max(68, round(83 * scale))
    canvas = cv2.copyMakeBorder(image, head, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    offset = np.array([0, head])
    for segment in raw:
        cv2.polylines(canvas, [np.rint(segment['polygon'] + offset).astype(np.int32)], True, (165, 165, 165), 1, cv2.LINE_AA)
    occupied = []
    for (row, ink) in [('t', (255, 255, 0)), ('b', (255, 0, 255))]:
        group = sorted([r for r in rows if r['row'] == row], key=lambda r: r['col'])
        for (a, b) in zip(group, group[1:]):
            p = (a['anchor_x'], a['anchor_y'] + head)
            q = (b['anchor_x'], b['anchor_y'] + head)
            if usable(a) and usable(b):
                cv2.line(canvas, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), ink, max(2, round(3 * scale)), cv2.LINE_AA)
            else:
                dashed_line(canvas, p, q, ink, max(1, round(2 * scale)))
    for r in rows:
        valid = usable(r)
        proposal = r.get('geometry_status') == 'REVIEW' or r.get('geometry_proposal_status') == 'PROPOSED_REVIEW'
        ink = M.color(r['row'], r['col']) if valid or proposal else (150, 150, 150)
        if r.get('cell_polygon') is not None:
            poly = np.rint(np.asarray(r['cell_polygon'], float) + offset).astype(np.int32)
            cv2.polylines(canvas, [poly], True, ink, max(1, round(2 * scale)), cv2.LINE_AA)
        (x, y) = (float(r['anchor_x']), float(r['anchor_y']))
        if not (0 <= x < w and 0 <= y < h):
            continue
        p = (round(x), round(y) + head)
        direct = bool(r.get('direct_hole_support'))
        cv2.circle(canvas, p, max(4, round(7 * scale)), ink, -1 if direct else 2, cv2.LINE_AA)
        suffix = 'HOLE' if direct and valid else 'FIT' if valid else 'REVIEW' if proposal else 'HOLD'
        M.label(canvas, r['physical_slot'] + ' ' + suffix, np.asarray(p), ink, scale, head, occupied)
    canvas[:head] = 0
    count = sum((usable(r) for r in rows))
    title = f"{sample['camera'].upper()} S4 ver2 | slots={len(rows)} | geometry-supported={count} | hold={len(rows) - count} | BODY NOT SELECTED"
    reason = str(meta.get('status') or meta.get('geometry_status') or meta.get('frame_status') or 'REVIEW')
    lines = [title, f"{sample['image_name']} | {reason} | Dashed=proposal REVIEW needs check Gray=HOLD/SEG"]
    for (i, text) in enumerate(lines):
        fs = (0.7 if i == 0 else 0.51) * scale
        ((tw, _), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        if tw > w - 16:
            fs *= max(0.25, (w - 16) / tw)
        cv2.putText(canvas, text, (8, round(head * (0.38 if i == 0 else 0.8))), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)
    return canvas

def metrics(rows, prefer_anchor=True):
    by = {(r['row'], r['col']): r for r in rows}
    gaps = []
    for col in sorted({r['col'] for r in rows}):
        if ('t', col) in by and ('b', col) in by:
            key = 'anchor_y' if prefer_anchor else 'y_prior'
            gaps.append(float(by['b', col].get(key, by['b', col]['y_prior'])) - float(by['t', col].get(key, by['t', col]['y_prior'])))
    return dict(slot_rows=len(rows), geometry_usable=sum((usable(r) for r in rows)), gap_min=min(gaps, default=None), gap_max=max(gaps, default=None), direct_hole_support=sum((bool(r.get('direct_hole_support')) for r in rows)))

@contextmanager
def run_lock(root):
    import msvcrt
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.run.lock').open('a+b') as f:
        if f.seek(0, 2) == 0:
            f.write(b'0')
            f.flush()
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
