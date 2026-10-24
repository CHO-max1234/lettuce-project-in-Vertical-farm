"""User paths and runtime settings for the development cultivation cycle.

Edit this block before running pipeline/1. Run_cycle.py. Environment variables
provide optional overrides for unattended runs. Imports do not create folders.
"""
from pathlib import Path
import os
import sys

# -------------------------- USER PATHS --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('LETTUCE_DATA_ROOT', PROJECT_ROOT / 'data')).resolve()
ARTIFACT_ROOT = Path(os.environ.get('LETTUCE_ARTIFACT_ROOT', PROJECT_ROOT / 'artifacts')).resolve()
OUTPUT_ROOT = Path(os.environ.get('LETTUCE_RUN_ROOT', PROJECT_ROOT / 'runs' / 'cycle2')).resolve()
RAW_IMAGE_ROOT = DATA_ROOT / 'raw_images'
PREPROCESSED_INPUT = Path(os.environ.get('LETTUCE_PREPROCESSED_INPUT', DATA_ROOT / 'preprocessed')).resolve()
PREPROCESS_OUTPUT = OUTPUT_ROOT / 'preprocessing'
ENVIRONMENT_FILE = Path(os.environ.get('LETTUCE_ENVIRONMENT_FILE', DATA_ROOT / 'environment.xlsx')).resolve()
GROWTH_FILE = Path(os.environ.get('LETTUCE_GROWTH_FILE', DATA_ROOT / 'growth.xlsx')).resolve()
STAGE_MODEL = ARTIFACT_ROOT / 'cnn' / 'stage.pt'
A_MODEL = ARTIFACT_ROOT / 'a_step'
C_MODELS = ARTIFACT_ROOT / 'c_step'  # Frozen legacy16 ensemble; unavailable environmental inputs stay NaN.
PCA_FILE = ARTIFACT_ROOT / 'pca128_train_only.npz'
BACKBONE_FILE = ARTIFACT_ROOT / 'efficientnet_b0.pth'
HARVEST_RULE = ARTIFACT_ROOT / 'harvest_rule.json'
BED_TOP_MODEL = ARTIFACT_ROOT / 'segmentation' / 'bed_top.pt'
BED_FRONT_MODEL = ARTIFACT_ROOT / 'segmentation' / 'bed_front.pt'
PLANT_MODEL = ARTIFACT_ROOT / 'segmentation' / 'plant.pt'
OCR_MODEL_DIR = Path(os.environ.get('LETTUCE_OCR_MODELS', ARTIFACT_ROOT / 'ocr'))

# Existing preprocessing products retain their field names and storage layout.
CROP_ROOT = Path(os.environ.get('LETTUCE_CROP_ROOT', PREPROCESSED_INPUT / '7. lettuce-crop' / 'crops')).resolve()
# Optional exact old root recorded in existing manifests; source records are never rewritten.
LEGACY_PREPROCESSED_ROOT = os.environ.get('LETTUCE_LEGACY_PREPROCESSED_ROOT', '')
# Korean names below identify original research inputs, not user-facing messages.
CROP_LIST = PREPROCESSED_INPUT / '7. lettuce-crop' / '분석가능_crop목록.csv'
CROP_MANIFEST = PREPROCESSED_INPUT / '7. lettuce-crop' / 'crop_manifest.csv'
FEATURE_FILE = PREPROCESSED_INPUT / '8. feature' / 'crop_features.xlsx'
QC_FILE = PREPROCESSED_INPUT / '9. 생육QC' / 'growth_qc.xlsx'

# ------------------------ RUNTIME OPTIONS -----------------------
PYTHON = Path(sys.executable)
DEVICE = os.environ.get('LETTUCE_DEVICE', 'auto')
BATCH_SIZE = 16
WORKERS = 4
SEASON_ID = '2작기'  # Preserve the source-data season key.
SEASON_START = '2026-03-06'
SEASON_END = '2026-05-04'
ENVIRONMENT_SCHEMA = 'manuscript12'
MODEL_ENVIRONMENT_POLICY = 'legacy16_nan_adapter'
C_MODEL_TAG = '454columns_blend50'  # Preserve the original frozen-model identity.

# Optional profile for reading pre-existing bed crops instead of raw RGB.
PREPROCESS_PROFILE = None


def preprocessing_inputs(root):
    """Return the existing crop/feature/QC layout below a chosen source root."""
    root = Path(root)
    if root.resolve() == PREPROCESSED_INPUT.resolve():
        return dict(crop_root=CROP_ROOT, crop_list=CROP_LIST, crop_manifest=CROP_MANIFEST,
                    features=FEATURE_FILE, qc=QC_FILE)
    return dict(crop_root=root / '7. lettuce-crop' / 'crops',
                crop_list=root / '7. lettuce-crop' / '분석가능_crop목록.csv',
                crop_manifest=root / '7. lettuce-crop' / 'crop_manifest.csv',
                features=root / '8. feature' / 'crop_features.xlsx',
                qc=root / '9. 생육QC' / 'growth_qc.xlsx')


def resolve_crop_path(value, crop_root=None):
    """Resolve one explicitly configured root relocation, without searching files."""
    path = Path(value)
    if not LEGACY_PREPROCESSED_ROOT:
        return path
    old_root = Path(LEGACY_PREPROCESSED_ROOT) / '7. lettuce-crop' / 'crops'
    try:
        suffix = path.relative_to(old_root)
    except ValueError:
        return path
    root = Path(crop_root) if crop_root is not None else CROP_ROOT
    mapped = (root / suffix).resolve()
    if not mapped.is_relative_to(root.resolve()):
        raise ValueError('Saved crop path escapes the configured crop root')
    return mapped
