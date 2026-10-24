"""Saved-mask serialization and crop output functions used by the final whole-cycle crop stage."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
import sys
CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_ROOT.parent))
from runtime import CROP_FOLDER, moved_path, moved_tree
DATA_ROOT = CYCLE.PREPROCESS_OUTPUT
UPSTREAM_ROOT = DATA_ROOT / '4. 자리배치'
INPUT_ROOT = DATA_ROOT / CROP_FOLDER / 'testimages'
OUTPUT_ROOT = DATA_ROOT / CROP_FOLDER / 'samples'
BRIGHT_ROOT = DATA_ROOT / '3. 밝기보정과 scale'
MASK_DATABASE = UPSTREAM_ROOT / '_whole_run' / 'state.sqlite3'
FROZEN_READER_FILE = CYCLE.PREPROCESS_CODE / '5. Build_plant_bboxes.py'
LEGACY_SAMPLE_ROOT = UPSTREAM_ROOT / 'testimages_bbox' / 'ver2.2'
SAMPLE_RUN = False
EXPECTED_IMAGES = 30
WORKERS = 3
CHUNK_FRAMES = 6
VERSION = 'S3_FROZEN_MASK_TO_CROP_SAMPLES_V1'
import csv
import datetime
import hashlib
import json
import os
sys.dont_write_bytecode = True
import cv2
import numpy as np
cv2.setNumThreads(1)
csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
HELD_STATUSES = {'NO_SLOT_GEOMETRY_REVIEW', 'INVALID_CELL_GEOMETRY_REVIEW'}
SOURCE_FIELDS = ('source_path', 'source_sha256', 'source_width', 'source_height', 'camera')
ARTIFACTS = (('crop_path', 'crop_sha256'), ('rgb_crop_path', 'rgb_crop_sha256'), ('mask_path', 'mask_sha256'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def yes(value):
    text = str(value).strip().lower()
    if text in ('true', '1'):
        return True
    if text in ('false', '0', ''):
        return False
    raise ValueError('Invalid input or state in yes')

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def token(value):
    return hashlib.sha256(encoded(value).encode('utf-8')).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')

def read_json(path):
    return moved_tree(json.loads(Path(moved_path(path)).read_text(encoding='utf-8-sig')))

def csv_rows(path):
    with Path(moved_path(path)).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            yield moved_tree(row)

def read_image(path, expected_hash=None):
    path = moved_path(path)
    content = Path(path).read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if expected_hash:
        require(actual_hash == expected_hash, 'Validation condition failed in read_image: actual_hash == expected_hash')
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    require(image is not None, 'Validation condition failed in read_image: image is not None')
    return (image, actual_hash)

def file_snapshot(paths):
    return {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}

def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def save_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))
