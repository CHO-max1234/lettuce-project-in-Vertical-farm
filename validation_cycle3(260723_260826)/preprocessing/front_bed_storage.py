"""Front bed storage calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
COMMON_FILE = CODE_ROOT / 'front_bed_policy.py'
import importlib.util
spec = resolved_spec('front_common', COMMON_FILE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
import argparse, hashlib, json, sqlite3, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
from tqdm import tqdm
KNOWN_REVIEWS = {}

def save_one(row, policy):
    row = dict(row)
    (selected, status) = c.select_target(row['detections'], row['image_width'], row['image_height'], policy['confidence'])
    row.update(status=status, selected=selected, actual_bed_number=None, human_target_qc='NOT_REVIEWED', ocr_status='NOT_RUN')
    if row['source_path'].lower() in KNOWN_REVIEWS:
        row.update(status='REVIEW_KNOWN_LABEL_TARGET_MISMATCH', human_target_qc='MODEL_DISAGREES_WITH_USER_LABEL', reason=KNOWN_REVIEWS[row['source_path'].lower()]['reason'])
        return row
    if selected is None:
        return row
    (verified, im) = c.read_source(row)
    if im is None:
        return {**row, 'status': 'ERROR_SAVE_SOURCE_READ', 'reason': verified.get('reason')}
    assert verified['source_sha256'] == row['source_sha256'], "Input contract failed: verified['source_sha256'] == row['source_sha256']"
    cropbox = c.expanded_box(selected['bbox'], im.shape[1], im.shape[0], row['date'], policy['crop_profiles'])
    (x1, y1, x2, y2) = cropbox
    crop = im[y1:y2, x1:x2]
    if not crop.size:
        return {**row, 'status': 'ERROR_EMPTY_CROP'}
    target = Path(row['target_path'])
    root = (c.OUTPUT_ROOT / 'cam2').resolve()
    resolved_target = target.resolve()
    assert resolved_target.is_relative_to(root), 'Input contract failed: resolved_target.is_relative_to(root)'
    assert target.name.endswith('_cam2.jpg') and target.parent.name == row['date'][2:]
    target.parent.mkdir(parents=True, exist_ok=True)
    (ok, data) = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    if target.exists():
        assert c.sha(target) == hashlib.sha256(data.tobytes()).hexdigest(), 'Input contract failed: c.sha(target) == hashlib.sha256(data.tobytes()).hexdigest()'
    else:
        partial = target.with_suffix('.jpg.partial')
        partial.write_bytes(data.tobytes())
        partial.replace(target)
    return {**row, 'status': 'SAVED_CANDIDATE', 'detected_bbox': selected['bbox'], 'detected_polygon': selected['polygon'], 'crop_bbox': cropbox, 'crop_profile': c.crop_profile_key(row['date']), 'output_path': str(target), 'output_sha256': c.sha(target), 'output_width': x2 - x1, 'output_height': y2 - y1, 'confidence': selected['confidence']}

def summarize(records):
    by_date = []
    groups = defaultdict(list)
    for r in records:
        groups[r['date'], r['source_camera']].append(r)
    for ((date, sourcecam), group) in sorted(groups.items()):
        saved = sum((r['status'] == 'SAVED_CANDIDATE' for r in group))
        by_date.append(dict(date=date, source_camera=sourcecam, corrected_camera='CAM2', total=len(group), saved=saved, not_saved=len(group) - saved, candidate_save_rate=saved / len(group), statuses=dict(Counter((r['status'] for r in group)))))
    by_regime = {}
    for name in ['early_CAM2', 'late_source_CAM0_as_CAM2']:
        group = [r for r in records if c.regime(r) == name]
        saved = sum((r['status'] == 'SAVED_CANDIDATE' for r in group))
        by_regime[name] = dict(total=len(group), saved=saved, not_saved=len(group) - saved, candidate_save_rate=saved / len(group), statuses=dict(Counter((r['status'] for r in group))))
    return (by_date, by_regime)
