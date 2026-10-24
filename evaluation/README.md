# Table calculations from saved results

`code/` recalculates available numerical summaries using the supplied files in
`data/`. Paths are repository-relative. These scripts require NumPy and pandas
from the repository-root `requirements.txt`; they do not run training or new
image/model inference.

## Run

From this directory, choose a new output directory:

```text
python "code/Table_09.py" --output-dir "outputs/Table_09"
```

The same form applies to the other `Table_*.py` scripts. Tables 3, 8, 9 and S4
accept `--data-dir`; Tables 4-7 accept `--input` to use another compatible CSV.
Use `--help` for the individual options. Outputs include calculated summaries
and source hashes; unavailable values remain unavailable.

| Scripts | Available calculation |
| --- | --- |
| `Table_03.py` | Saved preprocessing inventories and matched-observation counts |
| `Table_04.py` | Stage-classification metrics from saved predictions |
| `Table_05.py` | A-step weight-prediction metrics |
| `Table_06.py` | Baseline and fusion weight-prediction metrics |
| `Table_07.py` | Fixed stage/weight harvest-rule calculations |
| `Table_08.py` | Retained example cases, area history and growth QC |
| `Table_09.py` | External-cycle metrics and fixed-rule calculations |
| `Table_S04.py` | Retained segmentation training logs and configuration metadata |

## Interpretation and limits

Metrics from saved predictions are distinct from new inference results. Changes
to data, missing predictors, preprocessing or the evaluation cohort can change
them. Original training, image-based segmentation evaluation, sampling decisions
and manual case selection are not repeated by these scripts. Archived logs and
case summaries support only the quantities retained in the included data; some
manuscript entries cannot be independently reconstructed from these files.

Tables without a numerical script are not regenerated here. Cycle-4 harvest
calculations apply the fixed rule to saved predictions and are separate from
the cycle-4 CNN/A/C pipeline, which does not run D-step. Research cycles 2/3/4
correspond to manuscript cycles 1/2/3. Keep the declared target/history separation
and unavailable-row handling when interpreting the calculated metrics.
