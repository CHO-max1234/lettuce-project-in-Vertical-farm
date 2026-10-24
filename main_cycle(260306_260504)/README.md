# Main development cycle

Research cycle 2, 6 March-4 May 2026. Install the repository-root
[requirements](../requirements.txt) and keep the bundled [models](artifacts/README.md)
in place. No training is performed by the integrated runner.

## Inputs and paths

Edit the user paths in `settings.py` before running.

| Setting | Required input |
| --- | --- |
| `PREPROCESSED_INPUT` | Existing crop list/manifest, feature workbook and growth-QC workbook in the declared source layout |
| `CROP_ROOT` | RGB plant crops referenced by those records |
| `ENVIRONMENT_FILE` | Minute sensor workbook with date, time, temperature and humidity; EC, water temperature and day/night labels are optional |
| `GROWTH_FILE` | Survey date, bed ID, plant number and measured fresh weight |
| `ARTIFACT_ROOT` | Included model checkpoints, preprocessing settings, PCA and harvest rule |
| `OUTPUT_ROOT` | New outputs, separate from source data and models |

If manifests record an old absolute crop root, set `LEGACY_PREPROCESSED_ROOT`
and the actual `CROP_ROOT`. The prefix is mapped in memory; filenames, identities
and image hashes remain checked. Individual data paths also have `LETTUCE_*`
environment overrides declared in `settings.py`.

## Execution

From this cycle directory:

```text
python "pipeline/1. Run_cycle.py" --check-models
python "pipeline/1. Run_cycle.py" --check
python "pipeline/1. Run_cycle.py" --output "C:/lettuce_outputs/main"
```

Order: survey/environment processing -> saved feature and mask checks -> CNN
stage prediction -> A-step weight prediction -> C-step history fusion -> D-step
harvest decision -> `predictions.xlsx`. The runner passes each newly completed
model run to the next stage. D uses the fixed stage-probability and weight rule
stored in `artifacts/harvest_rule.json`.

To start from raw images, also configure `RAW_IMAGE_ROOT` and run:

```text
python "pipeline/1. Run_cycle.py" --start preprocessing --check
python "pipeline/1. Run_cycle.py" --start preprocessing --output "C:/lettuce_outputs/main_raw"
```

This runs the numbered preprocessing stages before the model chain. Source
geometry, slot/camera registration and required source records must be valid.
Use `--device cpu` if needed. The default saved-product run records stage status
in `00_input/execution_state.json`; reuse requires unchanged inputs and code.

## Environmental and output handling

Compatible environmental observations are joined from the completed previous
day and `[D-7, D)` history. Unavailable or incompatible model predictors remain
`NaN`. Missing daily columns, nonnumeric values and invalid numeric ranges are
treated as missing; date identity errors remain errors. No interpolation, zero
fill or substitution of one environmental quantity for another is performed.
Without recorded daytime labels, the daytime water-air temperature gap remains
missing.

The saved models retain their original input order and weights. Missing-input
conditions can change predictions, so a new inference run need not reproduce
the archived table values. Missing predictions remain unavailable. Outputs
include stage predictions, model inputs, completion records, available-weight
metrics and the final workbook. See the [repository guide](../README.md) for
application scope and the separate figure/table workflows.
