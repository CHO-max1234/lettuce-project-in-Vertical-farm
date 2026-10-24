"""Processing utilities for pipeline paths."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from pathlib import Path
import importlib.util
import os
import re
import sys
os.environ['YOLO_AUTOINSTALL'] = 'false'
import json

def load_profile():
    name = os.environ.get('LETTUCE_PROFILE', '')
    if not name:
        return {}
    path = Path(name).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    allowed = {'mode', 'output_root', 'season_id', 'raw_roots', 'bed_roots', 'ocr_mode', 'bbox_save_crops', 'allow_unknown_raw_offset'}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f'Unknown profile settings: {sorted(unknown)}')
    if data.get('mode') != 'reuse_bed':
        raise ValueError('Profile mode must be reuse_bed')
    if data.get('ocr_mode') != 'manifest_only':
        raise ValueError('Existing bed input requires manifest_only OCR')
    if set(data.get('bed_roots', {})) != {'cam0', 'cam1', 'cam2'}:
        raise ValueError('bed_roots requires cam0, cam1 and cam2 paths')
    if not isinstance(data.get('raw_roots'), list) or not data['raw_roots']:
        raise ValueError('raw_roots must list source directories')
    for key in ('bbox_save_crops', 'allow_unknown_raw_offset'):
        if not isinstance(data.get(key), bool):
            raise ValueError(f'{key} must be true or false')
    for value in [data.get('output_root', ''), *data['raw_roots'], *data['bed_roots'].values()]:
        if not value or not Path(value).is_absolute():
            raise ValueError(f'An absolute path is required: {value}')
    if any((os.environ.get(key) for key in ('LETTUCE_INPUT_ROOT', 'LETTUCE_OUTPUT_ROOT', 'LETTUCE_SEASON_ID'))):
        raise ValueError('Do not combine a profile with LETTUCE_* source/output overrides')
    return data
PROFILE = load_profile()
REUSE_BED = PROFILE.get('mode') == 'reuse_bed'
RAW_ROOTS = [Path(p).resolve() for p in PROFILE.get('raw_roots', [])]
EXTERNAL_BED_ROOTS = {k: Path(p).resolve() for (k, p) in PROFILE.get('bed_roots', {}).items()}
BBOX_SAVE_CROPS = PROFILE.get('bbox_save_crops', True)
ALLOW_UNKNOWN_RAW_OFFSET = PROFILE.get('allow_unknown_raw_offset', False)
CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CYCLE.DATA_ROOT
OUTPUT_ROOT = Path(PROFILE.get('output_root') or os.environ.get('LETTUCE_OUTPUT_ROOT', str(CYCLE.PREPROCESS_OUTPUT))).resolve()
INPUT_ROOT = RAW_ROOTS[0] if REUSE_BED else Path(os.environ.get('LETTUCE_INPUT_ROOT', str(CYCLE.RAW_IMAGE_ROOT))).resolve()
SEASON_ID = PROFILE.get('season_id') or os.environ.get('LETTUCE_SEASON_ID', CYCLE.SEASON_ID)
DEVICE = os.environ.get('LETTUCE_DEVICE', CYCLE.DEVICE)
if DEVICE == 'auto':
    import torch
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'cuda':
    DEVICE = '0'
MANAGEMENT = OUTPUT_ROOT / '0. 실행관리'
BED_ROOT = OUTPUT_ROOT / '1. bed-seg'
BED_IMAGES = BED_ROOT / 'images'
OCR_ROOT = OUTPUT_ROOT / '2. OCR'
OCR_IMAGES = BED_IMAGES
OCR_LOG = OCR_ROOT / 'records'
SCALE_ROOT = OUTPUT_ROOT / '3. scale'
BRIGHT_ROOT = OUTPUT_ROOT / '4. 밝기보정'
BRIGHT_IMAGES = BRIGHT_ROOT / 'corrected'
NAME_ROOT = OUTPUT_ROOT / '5. 이름 배정'
BBOX_ROOT = OUTPUT_ROOT / '6. bbox'
BBOX_OVERLAY_LIMIT = 30
BBOX_OVERLAY_SEED = 260911
BBOX_MANIFEST = BBOX_ROOT / 'bbox_manifest.csv'
CROP_ROOT = OUTPUT_ROOT / '7. lettuce-crop'
FEATURE_ROOT = OUTPUT_ROOT / '8. feature'
SUPPORT_ROOT = CODE_ROOT / '_support'
FROZEN_CAM01 = SUPPORT_ROOT / 'cam01'
CAM2_REGISTRATION = SUPPORT_ROOT / 'cam2' / 'bed_registration.py'
MODEL_ROOT = CYCLE.ARTIFACT_ROOT / 'segmentation'
BED_MODELS = {'cam01': CYCLE.BED_TOP_MODEL, 'cam2': CYCLE.BED_FRONT_MODEL}
LETTUCE_MODEL = CYCLE.PLANT_MODEL
LETTUCE_SHA256 = 'a2ded62bba4d577f4ead18907b824c587e179357c8089b72dabde41b755d6fe8'
STAGE_FILES = {'bed01': '1. Segment_beds_top.py', 'bed2': '1. Segment_beds_front.py', 'ocr': '2. Read_bed_numbers.py', 'scale': '3. Estimate_scale.py', 'bright': '4. Normalize_brightness.py', 'name0': '5. Assign_slots_cam0.py', 'name1': '5. Assign_slots_cam1.py', 'name2': '5. Assign_slots_cam2.py', 'bbox01': '6. Build_bboxes_top.py', 'bbox2': '6. Build_bboxes_front.py', 'crop': '7. Extract_plant_crops.py', 'feature': '8. Extract_features.py', 'growth_qc': '9. Assess_growth.py'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
NAME_RE = re.compile('^(bed\\d+)_(\\d{8})_(\\d{6})_(cam[012])(?P<suffix>(?:_[a-z0-9]+)*)$', re.I)

def identity(path):
    match = NAME_RE.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f'Cannot parse source image name: {path}')
    (bed, date, stamp, camera) = (v.lower() for v in match.group(1, 2, 3, 4))
    return dict(bed_id=bed, capture_date=date, capture_time=stamp, camera=camera, suffix=match.group('suffix'), key=f'{date}_{stamp}_{bed}')

def bootstrap():
    if str(FROZEN_CAM01) not in sys.path:
        sys.path.insert(0, str(FROZEN_CAM01))

def load_stage(key):
    bootstrap()
    name = 'integration_' + key
    if name in sys.modules:
        return sys.modules[name]
    path = CODE_ROOT / STAGE_FILES[key]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
