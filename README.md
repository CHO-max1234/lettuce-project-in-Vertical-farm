# Lettuce Growth-State Assessment and Harvest Decision Support Using a Soft-Labelled CNN and Feature Integration in a Rotating Vertical Farm

Code, trained models, and evaluation materials supporting the manuscript **"Lettuce Growth-State Assessment and Harvest Decision Support Using a Soft-Labelled CNN and Feature Integration in a Rotating Vertical Farm"**.

This repository contains the final multi-view preprocessing and frozen-model inference workflows for three cultivation cycles, trained models with required settings and feature transforms, and scripts for regenerating available figures and recalculating numerical summaries from saved evaluation data. Full-data execution requires external cultivation records and reviewed inputs.

## Quick Terminology Guide (`CNN` / `A-step` / `C-step`  & Cycle IDs)

The table maps the retained code identifiers to their model roles. Cultivation-cycle IDs are mapped in the repository structure below.

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
│   ├── requirements.txt                # Model-workflow package versions
│   ├── artifacts/                      # Shared models, feature transforms, configuration and fixed harvest rule
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
├── figures/                            # Manuscript (Fig. 1–11) and supplementary (Fig. S1–S11) figures
│   └── code/                           # Figure scripts (Figure_04.py – Figure_S11.py) and daily environment CSV
└── evaluation/                         # Quantitative table verification (Tables 3–9, Table S4)
    ├── code/                           # Metric recalculation scripts (Table_03.py – Table_09.py, Table_S04.py)
    └── data/                           # Minimal reference & prediction CSVs for standalone table verification
```

## Data and Code Availability

To align with the manuscript's **Data Availability Statement**, the materials in this project are partitioned into publicly bundled verification assets and external raw cultivation archives:

### 1. Publicly Included Materials

* **Final Pipeline & Evaluation Source Code:** Final preprocessing (`preprocessing/`), inference (`model/`), execution (`pipeline/`), figure (`figures/code/`) and table-calculation (`evaluation/code/`) scripts. Cycle-specific inputs and reviewed checkpoints are described in each cycle guide.
* **Frozen Models and Required Configuration (`main_cycle(260306_260504)/artifacts/`):**
  * View-specific bed ROI segmentation (`segmentation/bed_top.pt`, `segmentation/bed_front.pt`) and plant instance segmentation (`segmentation/plant.pt`) weights.
  * **`CNN` (Growth-Stage Classifier):** 5-stage soft-labelled EfficientNet-B0 heading classifier (`cnn/stage.pt`, 384×384 input).
  * **`A-step` (Image-Based Fresh-Weight Regression):** 5-fold multimodal regression bundles (`a_step/fold_1/`–`a_step/fold_5/`, combining 256×256 EfficientNet-B0 image embeddings with 25 morphological and 8 CNN stage features).
  * **`C-step` (Late-Fusion Fresh-Weight Regression):** 6 longitudinal/environmental CatBoost regressors (`c_step/direct_stacking/seed_*.cbm`, `c_step/residual/seed_*.cbm`), frozen ImageNet EfficientNet-B0 backbone (`efficientnet_b0.pth`), and train-only 128-D PCA transformation matrix (`pca128_train_only.npz`).
  * **OCR:** Local EasyOCR detector/recognizer weights (`ocr/`).
  * Keep all checkpoints paired with their feature-order and preprocessing files. Cycle 3 also supplies a distinct front-view bed model in `validation_cycle3(260723_260826)/artifacts/`.

* **Evaluation Data (`evaluation/data/`) and Figure Data (`figures/code/Figure_S04_daily.csv`):** Retained inputs support standalone table calculations and the environmental plot without the full RGB archive or GPU inference:
  * `Table_03_*.csv` and `Table_S04_*`: preprocessing inventories, segmentation logs and configuration metadata.
  * `Table_04_predictions.csv`: saved hard/soft-label stage-classifier predictions and reference labels.
  * `Table_05_predictions.csv`, `Table_06_predictions.csv`, `Table_07_predictions.csv`: saved weight predictions, stage probabilities and harvest-rule inputs.
  * `Table_08_cases.csv` and `Table_08_daily_history.csv`: retained example cases and daily histories for projected-area QC calculations.
  * `Table_09_*.csv`: saved external-validation predictions.
  * `Figure_S04_daily.csv`: daily summary for regenerating Figure S4.

Saved-prediction recalculation does not repeat training, original sampling or manual case selection. Some descriptive entries and archived training details cannot be independently reconstructed from these retained files; see the [table guide](evaluation/README.md).

### 2. Reproducibility Scope: Saved Results and New Inference

All example commands below run from the repository root.

| Operation | Inputs | Entrypoint |
| :--- | :--- | :--- |
| Check bundled model files | Included artifacts and manifest | `python "main_cycle(260306_260504)/pipeline/1. Run_cycle.py" --check-models` |
| Recalculate a numerical table | Included saved evaluation data | `python "evaluation/code/Table_04.py" --output-dir "runs/Table_04"` |
| Regenerate Figure S4 | Included daily summary | `python "figures/code/Figure_S04.py" --output "runs/FigS4.png"` |
| Run preprocessing and inference | External images, sensor/survey data and required reviewed records | See the cycle-specific guides linked below |
| Regenerate photo panels or a website capture | Corresponding source images, manifests or website build | See the [figure guide](figures/README.md) |

* **Environmental missing values:** Model input names, order and weights are preserved. Observations are used only where their definitions and units match the expected predictors. Missing or incompatible predictors remain `NaN`; they are not substituted with another quantity, zero-filled or interpolated. The saved fusion models handle these missing inputs. Environmental joins use the completed previous day and `[D-7, D)` history; plant histories respect physical identity and capture-time cutoffs.
* **Cycle-specific handling:** Manuscript Cycle 1 uses compatible observations. Cycle 2 excludes environmental observations under its validation protocol. Cycle 3 derives supported temperature/humidity stress features and leaves unsupported terms missing. New inputs or missing-data conditions can change predictions; archived evaluation tables retain their saved predictions.
* **Raw-data and review boundaries:** The default runners start from saved preprocessing products. The final cycle's preprocessing branch starts from reviewed OCR, preserved layout and calibration records and stops for survey/label review before model evaluation. It does not reconstruct human decisions automatically. Missing predictions remain unavailable.
* **Figure scope:** Numerical regeneration does not guarantee identical photograph selection, fonts, annotations or website capture layout. Some figures require separate source data; supplied photographs and diagrams are retained assets.

### 3. External Research Data Not Hosted on GitHub

The following full research datasets are not hosted directly on GitHub. They are archived by the author and available upon reasonable request for non-commercial academic research:

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

### 3. Model Inference and Harvest Decision Rule (`model/`)

1. **`CNN` — Soft-Labelled Growth-Stage Classifier (`1. Predict_stage.py`):** Evaluates the frozen 5-stage soft-labelled EfficientNet-B0 classifier (`stage.pt`, 384×384 input) on eligible top-view crops (`crop_integrity.py`) to produce stage probabilities `cnn_p0`–`cnn_p4`, expected continuous stage, and Shannon entropy.
2. **`A-step` — Image-Based Fresh-Weight Regression Model (`2. Predict_weight.py`):** Runs the frozen 5-fold single-timepoint multimodal regression network (`WeightNet`, fusing a 256×256 EfficientNet-B0 image branch with 25 current-day morphological features and 8 CNN stage features) to output baseline image-based fresh-weight predictions (`pred_weight_g`) and 128-D fold embeddings.
3. **`C-step` — Late-Fusion Fresh-Weight Regression Model (`3. Fuse_growth_history.py`):** Performs multi-source feature integration by combining the `A-step` baseline weight prediction, current morphological/CNN features, causal past-only longitudinal trajectories (3-day/7-day slopes, AUCs, stage-transition intervals, and PCA-128 embedding shifts), and daily/7-day environmental variables using frozen Direct Stacking and Residual CatBoost ensembles (50:50 blend).
4. **`D-step` — Fixed Harvest Decision Rule (`4. Decide_harvest.py`):** Applies the validation-frozen dual threshold rule ($p_4 \ge 0.30$ from `A-step` fold-averaged stage probability and `C-step` late-fusion predicted weight $\ge 120.0\text{ g}$) to classify harvest readiness while reporting 7-day growth QC as an auxiliary operational flag.

The main and first external-validation workflows run CNN → A-step → C-step → the fixed harvest rule. The final external-validation workflow runs CNN/A/C and a separate front-view assessment; it does not run D-step.

## Quick Start

### 1. Install and Configure Paths

Use Python 3.10 on Windows. Processing locks use Windows APIs; other operating systems have not been verified. Install the shared model-workflow dependencies from the repository root, using a PyTorch build compatible with the CPU or CUDA runtime:

```text
python -m pip install -r requirements.txt
```

Keep the three cycle directories together. Edit the path block at the top of each cycle's `settings.py`, including actual input locations, shared model roots and a separate output root. For moved datasets, configure the saved-root mappings while retaining source identities, sheet names and hashes.

* [Main / manuscript Cycle 1](main_cycle(260306_260504)/README.md): saved crop list/manifest, RGB crops, features, growth QC and sensor/survey workbooks.
* [External validation / manuscript Cycle 2](validation_cycle2(260508_260620)/README.md): finalized crop/mask/feature databases, crops, completion records and the reviewed label ledger.
* [External validation / manuscript Cycle 3](validation_cycle3(260723_260826)/README.md): preserved crops/masks, reviewed target/history workbooks, sensor/survey inputs and required preprocessing checkpoints.

### 2. Check Model Files and Inputs

The runners support `--check-models` and `--check`. For example:

```text
python "main_cycle(260306_260504)/pipeline/1. Run_cycle.py" --check-models
python "main_cycle(260306_260504)/pipeline/1. Run_cycle.py" --check
```

These checks do not execute the inference or preprocessing batch.

### 3. Recalculate Tables and Regenerate Figures

Saved-data table calculations can run without the full image archive. Choose a new output directory for each table run:

```text
python "evaluation/code/Table_04.py" --output-dir "runs/Table_04"
python "evaluation/code/Table_05.py" --output-dir "runs/Table_05"
python "evaluation/code/Table_06.py" --output-dir "runs/Table_06"
python "evaluation/code/Table_07.py" --output-dir "runs/Table_07"
python "evaluation/code/Table_08.py" --output-dir "runs/Table_08"
python "evaluation/code/Table_09.py" --output-dir "runs/Table_09"
```

See the [table guide](evaluation/README.md) for the retained-data limits. Use the separate plotting environment in the [figure guide](figures/README.md); Figure S4 uses the included daily CSV, while other scripts may need external images, logs, workbooks or the original website build.

### 4. Run a Cycle

After configuring inputs and passing the corresponding checks, run from the repository root:

```text
python "main_cycle(260306_260504)/pipeline/1. Run_cycle.py" --output "runs/main"
python "validation_cycle2(260508_260620)/pipeline/1. Run_cycle.py" --output "runs/cycle3"
python "validation_cycle3(260723_260826)/pipeline/1. Run_cycle.py" --output "runs/cycle4"
```

The default uses saved preprocessing products. Main and the first external-validation runner produce `predictions.xlsx`; the final external-validation runner produces `predictions_top89.csv` and an optional separate `predictions_front26.csv`. The runners pass newly completed outputs to their dependent stages. Use `--device cpu` when needed.

For preprocessing, follow the additional input requirements and `--start preprocessing` commands in the respective cycle guide. The final cycle stops with `PREPROCESSING_COMPLETE_REVIEW_REQUIRED` before model evaluation; reviewed survey matches and labels must be supplied separately.

## Applicability and Interpretation

The models and fixed harvest rule were developed for crisphead lettuce in the studied rotating vertical farm. Performance may vary with cultivar, cultivation conditions, camera setup, illumination, image geometry and data quality. Check physical-plant/camera registration and evaluate accuracy and harvest thresholds on the intended dataset before interpreting predictions. Growth QC is an image-based indicator, not a direct biomass-growth measurement or a diagnosis of crop condition.

## About the Author

**Yeon-jin Cho (조연진), Ph.D.**  
Agricultural Researcher in Smart Farming & Plant Phenotyping

* **Research Background:** Ph.D. in agricultural multimodal imaging (RGB, depth, and thermal sensing for crop stress detection), with applied research experience in computer vision, longitudinal plant phenotyping, and data-driven cultivation pipelines.
* **Research Focus:** Developing reproducible, field-ready computer vision and decision-support systems tailored to non-ideal commercial cultivation environments such as rotating vertical farms.

