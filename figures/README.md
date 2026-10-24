# Figure reproduction

Figure assets are in this directory; executable scripts are in `code/`, named
`Figure_<number>.py` or `Figure_S<number>.py`. The supplied photographs, diagrams
and manuscript screen capture are retained image assets. Numerical plots can
be regenerated with the corresponding scripts and inputs.

## Environment and inputs

Use a separate plotting environment and install this directory's `requirements.txt`.
Run Figure 4 in the model environment with the bundled main
plant-segmentation checkpoint. Figure 13 requires Node.js, Playwright with
Chromium, and the original English website build as a separate input.

Configure input paths at the top of each script or use its command-line options
shown by `--help`. Most scripts require external feature workbooks, saved
predictions, training logs, QC records or source images. An old absolute path in
a script is not an included dataset. Supplementary Figure 3 uses the included
`code/Figure_S03_daily.csv` and can run without the full research dataset.

## Examples

Run from this directory in the appropriate environment:

```text
python "code/Figure_S03.py" --output "outputs/FigS3.png" --report "outputs/FigS3.json"
python "code/Figure_07.py" --input "C:/research/features.xlsx" --output-dir "outputs/Figure_07"
```

The second example requires the actual feature workbook at the specified path.
For a new Figure 13 capture, provide the website build:

```text
node "code/Figure_13.mjs" --site "C:/research/website/dist" --output-dir "outputs/Figure_13"
```

Use a new output directory and retain the supplied figure assets. Figure 4's
rail/segmentation modules, `Figure_14_storage.py` and `history_checks.py` are
required helpers and should remain beside the scripts.

## Scope

Regenerating a plot from saved data does not repeat model training, recover
unavailable source photographs or reconstruct all original preprocessing.
Layout, font rendering, annotation placement and a new website capture may
differ from the supplied images. Input identities, missing values and review
states are retained; source-data availability determines what can be regenerated.
