# Lettuce Growth-State Assessment and Harvest Decision Support Using a Soft-Labelled CNN and Feature Integration in a Rotating Vertical Farm

Official code, frozen model, and evaluation repository for the manuscript **"Lettuce Growth-State Assessment and Harvest Decision Support Using a Soft-Labelled CNN and Feature Integration in a Rotating Vertical Farm"**.

This repository provides the end-to-end multi-view image preprocessing and 4-stage modeling pipeline across three commercial-scale cultivation cycles, all 42 frozen model checkpoints and transformation matrices, and standalone verification scripts that reproduce the quantitative evaluation tables and analytical figures reported in the paper.

## Quick Terminology Guide (`CNN` / `A-step` / `C-step` / `D-step` & Cycle IDs)

Because folder names, checkpoint directories, and scripts preserve the internal research stage identifiers (`A-step`, `C-step`, `D-step`), the table below maps those codebase identifiers to the model roles and cultivation cycle numbers used in the manuscript:

| Code / Folder Identifier | Manuscript Model Role | Description |
| :--- | :--- | :--- |
| **`CNN`** (`artifacts/cnn/`) | **Soft-Labelled Growth-Stage Classifier** | 5-stage heading/growth-state classification (`Stage 0`–`Stage 4`) from a single top-view plant crop using soft probability targets. |
| **`A-step`** (`artifacts/a_step/`) | **Image-Based Fresh-Weight Regression Model** | Single-timepoint multimodal regression (`WeightNet`, 5-fold ensemble) combining the current top-view RGB image with current-day morphological and CNN stage features to estimate baseline fresh weight. |
| **`C-step`** (`artifacts/c_step/`) | **Late-Fusion Fresh-Weight Regression Model** | Multi-source feature integration model (`Direct Stacking` + `Residual` CatBoost 50:50 blend) that fuses the `A-step` baseline prediction with longitudinal growth-history trajectories (3d/7d rates, PCA-128 embedding shifts) and cultivation environment features. |

## Repository Structure

```text
.
├── main_cycle(260306_260504)/          # Manuscript Cycle 1 — Development cycle (6 Mar – 4 May 2026; internal ID: 2작기)
│   ├── settings.py                     # Central path and runtime configuration
│   ├── requirements.txt                # Verified Python package versions (Python >= 3.10)
│   ├── artifacts/                      # 42 frozen checkpoints (CNN, A-step, C-step, D-step, YOLO, OCR)
│   ├── preprocessing/                  # Stages 1–9: bed segmentation to paired-view daily growth QC
│   ├── model/                          # Stages 1–4: CNN -> A-step (image reg.) -> C-step (late fusion) -> D-step (harvest)
│   └── pipeline/                       # Integrated cycle runner (1. Run_cycle.py) and contract guards
├── validation_cycle2(260508_260620)/   # Manuscript Cycle 2 — External validation 1 (8 May – 20 Jun 2026; internal ID: 3작기)
│   ├── preprocessing/                  # Cycle-specific registration, feature extraction, and gap-aware QC
│   ├── model/                          # Frozen-model transfer evaluation (CNN -> A-step -> C-step -> D-step)
│   └── pipeline/                       # Storage and shared-module adapters
├── validation_cycle3(260723_260826)/   # Manuscript Cycle 3 — External validation 2 (23 Jul – 26 Aug 2026; internal ID: 4작기)
│   ├── preprocessing/                  # Confirmed OCR ledger and summer-cycle preprocessing/QC
│   └── model/                          # Frozen-model transfer evaluation (CNN -> A-step -> C-step)
├── figures/                            # Manuscript (Fig. 1–15) and supplementary (Fig. S1–S7) figures
│   └── code/                           # Analytical figure scripts (Figure_04.py – Figure_S07.py) & daily env CSV
└── evaluation/                         # Quantitative table verification (Tables 3–9, Table S4)
    ├── code/                           # Metric recalculation scripts (Table_03.py – Table_09.py, Table_S04.py)
    └── data/                           # Minimal reference & prediction CSVs for standalone table verification
```

## Data and Code Availability

To align with the manuscript's **Data Availability Statement**, the materials in this project are partitioned into publicly bundled verification assets and external raw cultivation archives:

### 1. Publicly Included in This Repository (No External Data Required)

* **Complete Pipeline & Evaluation Source Code:** Full preprocessing (`preprocessing/`), model inference (`model/`), pipeline execution (`pipeline/`), figure plotting (`figures/code/`), and metric recalculation (`evaluation/code/`) scripts across all three cultivation cycles.
* **Frozen Model Checkpoints and Feature Transformers (`main_cycle(260306_260504)/artifacts/`, 42 files):**
  * View-specific bed ROI segmentation (`top_bed/best.pt`, `front_bed/best.pt`) and plant instance segmentation (`lettuce/best.pt`) weights.
  * **`CNN` (Growth-Stage Classifier):** 5-stage soft-labelled EfficientNet-B0 heading classifier (`cnn/stage.pt`, 384×384 input).
  * **`A-step` (Image-Based Fresh-Weight Regression):** 5-fold multimodal regression bundles (`a_step/fold_0`–`fold_4`, combining 256×256 EfficientNet-B0 image embeddings with 25 morphological and 8 CNN stage features).
  * **`C-step` (Late-Fusion Fresh-Weight Regression):** 6 longitudinal/environmental CatBoost regressors (`c_step/direct_seed*.cbm`, `c_step/residual_seed*.cbm`), frozen ImageNet EfficientNet-B0 backbone (`c_step/efficientnet_b0.pth`), and train-only 128-D PCA transformation matrix (`c_step/pca128_train_only.npz`).
  * **OCR:**  local EasyOCR detector/recognizer weights (`ocr/`).

* **Minimal Evaluation & Reference Datasets (`evaluation/data/` and `figures/code/Figure_S03_daily.csv`):** De-identified reference tables containing plant/capture identifiers, ground-truth labels, saved model predictions, and daily environmental summaries that reproduce the manuscript tables directly without raw RGB images or GPU inference:
  * `Table_03_*.csv` & `Table_S04_*.csv`: Slot assignment / top–front pairing counts ($58,018 / 64,596 = 89.8\%$ assigned; $28,953 / 32,847 = 88.1\%$ paired across 858 unique plant slots) and segmentation training/validation logs.
  * `Table_04_predictions.csv`: Locked test set ($n = 668$ images across 19 beds) with human primary stage, soft-label targets, and predicted stage probabilities for Hard vs. Soft CNN comparison.
  * `Table_05_predictions.csv`, `Table_06_predictions.csv`, `Table_07_predictions.csv`: Locked test set ($n = 268$ plants in Cultivation Cycle 1; 51 true harvest-positive plants defined as human stage 4 and fresh weight $\ge 120\text{ g}$) for `A-step` image-based regression models, `C-step` late-fusion models (50:50 blend of `Direct Stacking` + `Residual`), and `D-step` harvest decision rules.
  * `Table_08_cases.csv` & `Table_08_daily_history.csv`: 4 operational showcase cases and 176 daily trajectory rows for 7-day growth-rate QC and  harvest decision verification.
  * `Table_09_*.csv`: External validation prediction records for Cultivation Cycle 2 ($n = 31$ evaluable of 32 observations) and Cultivation Cycle 3 ($n = 85$ evaluable of 89 top-view observations).
  * `Figure_S03_daily.csv`: 46-day daily environmental summary (DAT 13–58) for reproducing `Figure S3`.

### 2. Reproducibility Scope: Saved-Prediction Verification vs. Raw-Data Execution

| Verification Level | Input Source | Included in Repo? | Command / Entrypoint |
| :--- | :--- | :--- | :--- |
| **Model Artifact & Checksum Audit** | `artifacts/manifest.json` + 42 weight/config files | **Yes** | `python "pipeline/1. Run_cycle.py" --check-models` |
| **Quantitative Table Recalculation (`Tables 3–9`, `Table S4`)** | `evaluation/data/*.csv` | **Yes** | `python evaluation/code/Table_04.py` (etc.) |
| **Daily Environmental Feature Computation (`manuscript12`)** | Daily sensor CSV (`Figure_S03_daily.csv` or custom CSV) | **Yes** | `python "model/3. Fuse_growth_history.py" --environment-only ...` |
| **Full Preprocessing & Inference (`Stages 1–9` $\rightarrow$ `CNN` $\rightarrow$ `A` $\rightarrow$ `C` $\rightarrow$ `D`)** | Multi-camera raw RGB archive + sensor/survey workbooks (`DATA_ROOT`) | **External** | `python "pipeline/1. Run_cycle.py" --start preprocessing` |
| **Composite Photo Figure Panels (`Fig. 4, 5, 9, 14, 15, S2`)** | Raw/cropped plant image directories + intermediate manifests | **External** | `python figures/code/Figure_04.py` (etc.; rendered PNGs included in `figures/`) |

* **Note on Evaluation vs. Retraining:** The scripts in `evaluation/code/` and `figures/code/` deterministically verify metrics and regenerate plots from the frozen evaluation predictions without retraining models, re-fitting PCA, or re-tuning validation thresholds.
* **Note on C-Step Environmental Schema Guard (`manuscript12` vs. `legacy16`):** Following manuscript revision, `model/3. Fuse_growth_history.py` computes the 12 manuscript environmental variables (`manuscript12`, using capped GDD without heuristic stress penalties). Because the bundled historical `C-step` CatBoost checkpoints (`454columns_blend50`) were trained with the earlier 16-variable environmental schema (`legacy16`), `pipeline/model_contract.py` explicitly flags `c_step.compatible: false` during full-pipeline execution rather than silently zero-filling or renaming columns.

### 3. External Research Data Not Hosted on GitHub

Due to repository storage constraints (~hundreds of gigabytes of multi-camera imagery), the following raw datasets are not hosted directly on GitHub but are archived by the authors and available upon reasonable request for non-commercial academic research:

* Full-resolution raw RGB image archives captured by `CAM0` (top-left), `CAM1` (top-right + OCR plate), and `CAM2` (angled front) across Cultivation Cycles 1–3.
* Intermediate full-bed crops, binary mask archives, and manual annotation split manifests used during initial YOLO and CNN training.
* Raw minute-level environmental sensor logs and destructive fresh-weight harvest workbooks (`DATA_ROOT`).

## System and Pipeline Overview

### 1. Multi-View Rotating Vertical Farm Geometry

Each cultivation bed holds 24 lettuce plants arranged in two rows of 12 physical slots (`t1`–`t12` and `b1`–`b12`). As beds circulate continuously through the vertical farm, images are captured at fixed camera stations:

* **Top views (`CAM0`, `CAM1`):** Vertical overhead cameras covering physical columns 1–7 (`CAM0`) and columns 3–12 (`CAM1`, which also captures the right-side bed-number plate for OCR). Overlapping columns (3–7) are resolved per plant track using a deterministic mask-quality and temporal consistency policy (`preprocessing/camera_selection.py`).
* **Front view (`CAM2`):** Angled lateral camera (~45°) registered to the 12-column physical board (`rail = t1-b1`) to extract plant height (`front_height_cm`) and paired top–front volume proxy (`area_cm2 * front_height_cm`).

### 2. Nine-Stage Preprocessing Pipeline (`preprocessing/`)

1. **Bed Segmentation (`1. Segment_beds_top.py`, `1. Segment_beds_front.py`):** Crops bed regions from raw RGB frames using view-specific YOLO segmentation checkpoints.
2. **Bed-Number OCR (`2. Read_bed_numbers.py`):** Reads the bed number plate on `CAM1` via EasyOCR and assigns synchronized `bed_id` keys across `CAM0`, `CAM1`, and `CAM2` frames captured at the exact same timestamp.
3. **Pixel Scale Estimation (`3. Estimate_scale.py`):** Fits ellipses to nominal 50-mm vacant planting holes to estimate local `mm_per_px` scale, with same-group angle-indexed median fallback.
4. **Brightness Normalization (`4. Normalize_brightness.py`):** Applies a monotone RGB common-gain tone curve that normalizes illumination without altering pixel geometry.
5. **Physical Slot Assignment (`5. Assign_slots_cam0.py`, `5. Assign_slots_cam1.py`, `5. Assign_slots_cam2.py`):** Registers detected plant masks and vacant holes onto the canonical 12-column bed coordinate system (`physical_12col_v1`).
6. **Bounding-Box Construction (`6. Build_bboxes_top.py`, `6. Build_bboxes_front.py`):** Generates per-slot search corridors and bounding-box manifests.
7. **Contextual Crop and Mask Extraction (`7. Extract_plant_crops.py`):** Resolves target plant ownership against neighboring slots, packs explicit binary masks, and selects one primary top-view observation per `(date, time, bed, slot)`.
8. **Morphological and Radiometric Feature Extraction (`8. Extract_features.py`):** Computes contour, grayscale, radial, and upper/lower structural features from explicit masks, plus front-view height and volume proxy.
9. **Paired-View Daily Growth QC (`9. Assess_growth.py`):** Aggregates daily paired observations, evaluates 7-day causal growth trajectories (`increase`, `normal`, `slowing`, `plateau`, `decrease`, `insufficient`), and exports `growth_qc.xlsx`.

### 3. Four-Stage Model Inference Chain (`model/`)

1. **`CNN` — Soft-Labelled Growth-Stage Classifier (`1. Predict_stage.py`):** Evaluates the frozen 5-stage soft-labelled EfficientNet-B0 classifier (`stage.pt`, 384×384 input) on eligible top-view crops (`crop_integrity.py`) to produce stage probabilities `cnn_p0`–`cnn_p4`, expected continuous stage, and Shannon entropy.
2. **`A-step` — Image-Based Fresh-Weight Regression Model (`2. Predict_weight.py`):** Runs the frozen 5-fold single-timepoint multimodal regression network (`WeightNet`, fusing a 256×256 EfficientNet-B0 image branch with 25 current-day morphological features and 8 CNN stage features) to output baseline image-based fresh-weight predictions (`pred_weight_g`) and 128-D fold embeddings.
3. **`C-step` — Late-Fusion Fresh-Weight Regression Model (`3. Fuse_growth_history.py`):** Performs multi-source feature integration by combining the `A-step` baseline weight prediction, current morphological/CNN features, causal past-only longitudinal trajectories (3-day/7-day slopes, AUCs, stage-transition intervals, and PCA-128 embedding shifts), and daily/7-day environmental variables using frozen Direct Stacking and Residual CatBoost ensembles (50:50 blend).
4. **`D-step` — Harvest Decision Support Model (`4. Decide_harvest.py`):** Applies the validation-frozen dual threshold rule ($p_4 \ge 0.30$ from `A-step` fold-averaged stage probability and `C-step` late-fusion predicted weight $\ge 120.0\text{ g}$) to classify harvest readiness while reporting 7-day growth QC as an auxiliary operational flag.

## Quick Start

### 1. Environment Setup

Python 3.10 or later is required (verified with Python 3.10.20):

```bash
pip install -r "main_cycle(260306_260504)/requirements.txt"
```

### 2. Verify Bundled Model Checksums

```bash
cd "main_cycle(260306_260504)"
python "pipeline/1. Run_cycle.py" --check-models
```

### 3. Recalculate Manuscript Evaluation Tables

Run any table verification script directly from the repository root (outputs are written to a user-specified directory without modifying reference data):

```bash
python evaluation/code/Table_04.py --output-dir runs/table_check
python evaluation/code/Table_05.py --output-dir runs/table_check
python evaluation/code/Table_06.py --output-dir runs/table_check
python evaluation/code/Table_07.py --output-dir runs/table_check
python evaluation/code/Table_08.py --output-dir runs/table_check
python evaluation/code/Table_09.py --output-dir runs/table_check
```

### 4. Run the Cycle Pipeline (With External Raw Data Mounted)

Configure `DATA_ROOT`, `ARTIFACT_ROOT`, and `OUTPUT_ROOT` in `main_cycle(260306_260504)/settings.py` (or via environment variables `LETTUCE_DATA_ROOT`, `LETTUCE_ARTIFACT_ROOT`, `LETTUCE_RUN_ROOT`), then run:

```bash
cd "main_cycle(260306_260504)"
python "pipeline/1. Run_cycle.py" --check
python "pipeline/1. Run_cycle.py"
```

## About the Author

**Yeon-jin Cho (조연진), Ph.D.**  
Agricultural Researcher in Smart Farming & Plant Phenotyping

* **Research Background:** Ph.D. in agricultural multimodal imaging (RGB, depth, and thermal sensing for crop stress detection), with applied research experience in computer vision, longitudinal plant phenotyping, and data-driven cultivation pipelines.
* **Research Focus:** Developing reproducible, field-ready computer vision and decision-support systems tailored to non-ideal commercial cultivation environments such as rotating vertical farms.
