# External validation: research cycle 4

23 July-26 August 2026. Install the repository-root [requirements](../requirements.txt).
Keep both sibling cycle directories: `MAIN_ROOT` supplies the shared models/code,
and `THIRD_ROOT` supplies cycle-3 preprocessing functions. The additional
front-view bed model and its inference policy are in this directory's `artifacts/`.

## Inputs and paths

Edit the user paths in `settings.py`; `LETTUCE_S4_*` variables provide optional
overrides.

| Setting | Required input |
| --- | --- |
| `SOURCE_ROOT` | Preserved preprocessing layout, RGB crops and finalized crop/mask database |
| `TARGET_FILE` | Reviewed top-view target workbook, with `생육매칭` and `과거Feature` sheets |
| `SOURCE115` | Reviewed survey workbook containing all targets, including the separate front-view group |
| `GROWTH_FILE` | Original destructive-survey workbook |
| `ENVIRONMENT_FILE` | Original minute sensor workbook; retain distinct raw/converted sensor columns |
| `MAIN_ROOT`, `THIRD_ROOT` | Sibling common-model and preprocessing directories |
| `OUTPUT_ROOT` | New outputs, separate from source data and models |

The adapter retains the study's 89 top-view targets and separate 26 front-view
targets. Preserve the reviewed row identities, physical slots, camera routing
and complete history sheet. Weights and human stage labels are used only for
evaluation. Saved source paths are mapped in memory; databases must be finalized
and checkpointed before immutable reading.

## Model execution

From this cycle directory:

```text
python "pipeline/1. Run_cycle.py" --check-models
python "pipeline/1. Run_cycle.py" --check
python "pipeline/1. Run_cycle.py" --output "C:/lettuce_outputs/cycle4"
```

Order: environment processing -> crop/mask/identity checks and label separation
-> standalone CNN -> A-step prediction/evaluation -> causal history and environment
join -> C-step prediction/evaluation -> separate front-view CNN assessment.
Use `--skip-front` to omit the front assessment or `--device cpu` when needed.
Models remain frozen; D-step is not run for this cycle.

Each `run_<timestamp>/` contains `execution.json`, `predictions_top89.csv`,
optional `predictions_front26.csv`, and `models/` with stage outputs and metrics.
New environment, A and evaluation runs are passed explicitly to C.

## Preprocessing from reviewed checkpoints

The preprocessing branch requires reviewed OCR, renamed bed crops and ROI
manifests, a preserved base-layout database, scale/reference records and the
review/calibration ledgers listed in `settings.py` and the numbered stages.
It does not reconstruct historical human decisions from raw images.

```text
python "pipeline/1. Run_cycle.py" --start preprocessing --check
python "pipeline/1. Run_cycle.py" --start preprocessing --output "C:/lettuce_outputs/cycle4_preprocessing"
```

This runs reviewed OCR loading -> brightness -> scale -> geometry -> plant crops
-> features -> calibration, then stops with `PREPROCESSING_COMPLETE_REVIEW_REQUIRED`.
Review the survey matches and configure the reviewed input workbooks before
starting model evaluation. Separate numbered bed-segmentation entry points are
provided; new bed crops require renewed OCR/identity review.

## Environmental and evaluation scope

Supported temperature/humidity stress features are calculated; unsupported
environmental terms remain `NaN`. Inputs use the completed previous day and
`[D-7, D)` history without zero fill or interpolation. Plant histories respect
the physical identity and capture-time cutoff. Missing predictions, identity
review and unverified scale remain explicit.

Front-view results are reported separately from top-view A/C results. Growth
QC is an image-based indicator and restarts across camera, resolution or
observation-gap boundaries. See the [repository guide](../README.md) for broader
application limits and the separate figure/table workflows.
