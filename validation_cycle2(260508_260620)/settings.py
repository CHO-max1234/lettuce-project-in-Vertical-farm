"""User paths for external research cycle 3 (validation_cycle2).

Edit the paths below. Model artifacts are shared with the sibling main_cycle.
Source data and reviewed labels are read-only inputs; new runs use OUTPUT_ROOT.
"""
from pathlib import Path
import os
import sys

# --------------------------- USER PATHS ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_ROOT = Path(os.environ.get('LETTUCE_MAIN_ROOT', PROJECT_ROOT.parent / 'main_cycle(260306_260504)')).resolve()
ARTIFACT_ROOT = MAIN_ROOT / 'artifacts'
DATA_ROOT = Path(os.environ.get('LETTUCE_S3_DATA_ROOT', PROJECT_ROOT / 'data')).resolve()
OUTPUT_ROOT = Path(os.environ.get('LETTUCE_S3_OUTPUT_ROOT', PROJECT_ROOT / 'runs' / 'cycle3')).resolve()
RAW_TOP_ROOT = DATA_ROOT / 'raw' / 'top'
RAW_FRONT_ROOT = DATA_ROOT / 'raw' / 'front'
PREPROCESSED_INPUT = Path(os.environ.get('LETTUCE_S3_MODEL_INPUT_ROOT', DATA_ROOT / 'preprocessed')).resolve()
PREPROCESS_OUTPUT = Path(os.environ.get('LETTUCE_S3_PREPROCESS_ROOT', OUTPUT_ROOT / 'preprocessing')).resolve()
SURVEY_FILE = Path(os.environ.get('LETTUCE_S3_SURVEY_FILE', DATA_ROOT / 'survey.xlsx')).resolve()
LABEL_FILE = Path(os.environ.get('LETTUCE_S3_LABEL_FILE', DATA_ROOT / 'labels' / 'feature.xlsx')).resolve()
QC_INPUT = DATA_ROOT / 'qc' / 'growth_qc.csv'
QC_OUTPUT = OUTPUT_ROOT / 'qc'
MATCH_OUTPUT = OUTPUT_ROOT / 'survey_matching'
MODEL_OUTPUT = OUTPUT_ROOT / 'models'

BED_TOP_MODEL = ARTIFACT_ROOT / 'segmentation' / 'bed_top.pt'
BED_FRONT_MODEL = ARTIFACT_ROOT / 'segmentation' / 'bed_front.pt'
PLANT_MODEL = ARTIFACT_ROOT / 'segmentation' / 'plant.pt'
STAGE_MODEL = ARTIFACT_ROOT / 'cnn' / 'stage.pt'
A_MODEL = ARTIFACT_ROOT / 'a_step'
C_MODELS = ARTIFACT_ROOT / 'c_step'
PCA_FILE = ARTIFACT_ROOT / 'pca128_train_only.npz'
BACKBONE_FILE = ARTIFACT_ROOT / 'efficientnet_b0.pth'
HARVEST_RULE = ARTIFACT_ROOT / 'harvest_rule.json'
OCR_MODEL_DIR = ARTIFACT_ROOT / 'ocr'

# Existing dataset layout and field identifiers are retained for compatibility.
CROP_FOLDER = '5. BBOX to crop,feature'
BED_OUTPUT = PREPROCESS_OUTPUT / '1. ROI_BOX 원본'
OCR_OUTPUT = PREPROCESS_OUTPUT / '2. OCR 기록'
BRIGHT_OUTPUT = PREPROCESS_OUTPUT / '3. 밝기보정과 scale'
BBOX_OUTPUT = PREPROCESS_OUTPUT / '4. 자리배치'
CROP_OUTPUT = PREPROCESS_OUTPUT / CROP_FOLDER
INPUT_CROP_ROOT = PREPROCESSED_INPUT / CROP_FOLDER / 'crops'
INPUT_FEATURE_DB = PREPROCESSED_INPUT / CROP_FOLDER / '_feature_run' / 'state.sqlite3'
INPUT_CROP_DB = PREPROCESSED_INPUT / CROP_FOLDER / '_whole_run' / 'state.sqlite3'
INPUT_MASK_DB = PREPROCESSED_INPUT / '4. 자리배치' / '_whole_run' / 'state.sqlite3'

# ------------------------ RUNTIME SETTINGS ------------------------
PYTHON = Path(sys.executable)
DEVICE = os.environ.get('LETTUCE_DEVICE', 'auto')
BATCH_SIZE = 16
WORKERS = 4
SEASON_ID = '3작기'
SEASON_START = '2026-05-08'
SEASON_END = '2026-06-20'
C_MODEL_TAG = '454columns_blend50'
ENVIRONMENT_SCHEMA = 'legacy16_all_missing_external_validation'
PREPROCESS_PROFILE = None
EXPECTED_TARGETS = 32

# Shared main modules read these aliases when called through this cycle adapter.
RAW_IMAGE_ROOT = RAW_TOP_ROOT
CROP_ROOT = INPUT_CROP_ROOT
FEATURE_FILE = PREPROCESSED_INPUT / CROP_FOLDER / 'features.xlsx'
PREPROCESS_CODE = PROJECT_ROOT / 'preprocessing'
MAIN_PREPROCESS = MAIN_ROOT / 'preprocessing'
MAIN_PIPELINE = MAIN_ROOT / 'pipeline'

# Historical roots are used only to resolve saved paths and preserve frame IDs.
LEGACY_STORAGE_ROOT = r'D:\양상추연구_local\newfolder'
RELOCATED_STORAGE_ROOT = r'E:\newfolder_양상추'
SOURCE_PREPROCESS_ROOT = PREPROCESSED_INPUT
