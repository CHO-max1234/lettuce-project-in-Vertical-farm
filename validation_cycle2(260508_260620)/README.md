# External validation: research cycle 3

8 May-20 June 2026. Install the repository-root [requirements](../requirements.txt).
Keep this directory beside `main_cycle(260306_260504)`; `MAIN_ROOT` connects the
shared code and bundled models. The runner uses frozen models and the fixed
harvest rule without training or threshold selection.

## Inputs and paths

Set the following paths at the top of `settings.py`:

- `PREPROCESSED_INPUT`: the original preprocessing layout, finalized slot/mask,
  crop and feature SQLite databases, crop images and completion records.
- `LABEL_FILE`: the reviewed `feature.xlsx` ledger, with the original `feature`
  and `미매칭_조사` sheets. Measured weights and human stage labels are evaluation
  references, not predictor inputs.
- `MAIN_ROOT`: shared model/code directory; model files are not duplicated here.
- `OUTPUT_ROOT`: a new location separate from source data and models.

Saved database paths are mapped in memory using the source-root settings. Keep
the original plant slots, camera assignments and field identifiers. Source
databases must be finalized and checkpointed before immutable reading. The
adapter expects the study's reviewed target cohort; changing paths alone does
not adapt it to a different experiment.

## Execution

From this cycle directory:

```text
python "pipeline/1. Run_cycle.py" --check-models
python "pipeline/1. Run_cycle.py" --check
python "pipeline/1. Run_cycle.py" --output "C:/lettuce_outputs/cycle3"
```

Order: target CNN/A prediction -> causal-history preparation and inference ->
C-step fusion -> D-step harvest decision -> target evaluation. Each execution
creates `run_<timestamp>/` containing `execution.json`, model outputs and
`predictions.xlsx`. History-only observations do not enter target metrics.
Use `--device cpu` if needed.

For raw-image processing, additionally configure `RAW_TOP_ROOT`, `RAW_FRONT_ROOT`
and `SURVEY_FILE`, then run:

```text
python "pipeline/1. Run_cycle.py" --start preprocessing --check
python "pipeline/1. Run_cycle.py" --start preprocessing --output "C:/lettuce_outputs/cycle3_raw"
```

This runs bed segmentation, OCR, brightness/scale correction, slot and mask
processing, crops, features and growth QC before the model chain. A reviewed
label ledger must already be supplied. Survey matching produces candidates
with blank human-label cells; it does not replace reviewed labels.

## Missing data and interpretation

Environmental observations are excluded under this cycle's validation protocol;
the corresponding model inputs remain `NaN`. Model columns and order are kept
unchanged. Histories use the same physical plant and permitted camera at or
before each prediction timestamp. Unresolved identities, unavailable images and
missing predictions remain explicit and are not treated as negative harvest
decisions. Growth QC restarts across observation gaps or camera/scale changes.

See the [repository guide](../README.md) for application scope, models and the
separate figure/table calculations.
