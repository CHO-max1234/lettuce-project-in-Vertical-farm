# Figure reproduction

Figure assets are in this directory; executable scripts are in `code/`, named
`Figure_<number>.py` or `Figure_S<number>.py`. The supplied photographs, diagrams
and manuscript screen capture are retained image assets. Numerical plots can
be regenerated with the corresponding scripts and inputs.

## Current manuscript numbering and assets

Updated on 2026-10-28 from the author's current local `figures` directory.
The current manuscript uses Figures 1-11 and Supplementary Figures S1-S11.
Filenames are retained from that directory. The following table selects the
current manuscript versions where more than one unit or diagram version is present.

| Figure | Current supplied asset | Entry point or retained source |
| --- | --- | --- |
| 1 | [Figure 1.png](Figure%201.png) | Retained image |
| 2 | [Figure 2.png](Figure%202.png) | Latest diagram source: [figures2_v2.drawio](figures2_v2.drawio) |
| 3 | [Figure 3_A.jpg](Figure%203_A.jpg), [Figure 3_B.jpg](Figure%203_B.jpg), [Figure 3_C.jpg](Figure%203_C.jpg) | Retained photographs A-C |
| 4 | [Figure 4.jpg](Figure%204.jpg) | [code/Figure_04.py](code/Figure_04.py) |
| 5 | [Figure 5.png](Figure%205.png) | [code/Figure_05.py](code/Figure_05.py) |
| 6 | [Figure 6.png](Figure%206.png) | [code/Figure_06.py](code/Figure_06.py) |
| 7 | [Figure 7_mm2 ver.png](Figure%207_mm2%20ver.png) | [code/Figure_07_mm2 ver.py](code/Figure_07_mm2%20ver.py) |
| 8 | [Figure 8_mm2 ver.png](Figure%208_mm2%20ver.png) | [code/Figure_08_mm2 ver.py](code/Figure_08_mm2%20ver.py) |
| 9 | [Figure 9.png](Figure%209.png) | [code/Figure_09.py](code/Figure_09.py) |
| 10 | [Figure_10_final.png](Figure_10_final.png) | [code/Figure_10.py](code/Figure_10.py); final display labels differ |
| 11 | [Figure 11.png](Figure%2011.png) | [code/Figure_11.py](code/Figure_11.py) |
| S1 | [FigS1.png](FigS1.png) | [code/Figure_S01.py](code/Figure_S01.py) |
| S2 | [FigS2_learning_curves.png](FigS2_learning_curves.png) | [code/Figure_S02.py](code/Figure_S02.py) |
| S3 | [FigS3_error_examples.png](FigS3_error_examples.png) | [code/Figure_S03.py](code/Figure_S03.py) |
| S4 | [FigS4_.png](FigS4_.png) | [code/Figure_S04.py](code/Figure_S04.py) and included [code/Figure_S04_daily.csv](code/Figure_S04_daily.csv) |
| S5 | [FigS5_mm2.png](FigS5_mm2.png) | [code/Figure_S05.py](code/Figure_S05.py); see unit note below |
| S6 | [FigS6_.png](FigS6_.png) | [code/Figure_S06.py](code/Figure_S06.py) |
| S7 | [FigS7_.png](FigS7_.png) | [code/Figure_S07.py](code/Figure_S07.py) |
| S8 | [FigS8.png](FigS8.png) | [code/Figure_S08.py](code/Figure_S08.py) |
| S9 | [FigS9.png](FigS9.png) | [code/Figure_S09.mjs](code/Figure_S09.mjs) |
| S10 | [FigS10_mm2 ver.png](FigS10_mm2%20ver.png) | [code/Figure_S10_mm2 ver.py](code/Figure_S10_mm2%20ver.py) |
| S11 | [FigS11_mm2 ver.png](FigS11_mm2%20ver.png) | [code/Figure_S11_mm2 ver.py](code/Figure_S11_mm2%20ver.py) |

Original-unit alternatives and the older `figures2.drawio` are retained because
they remain in the local source directory. Use the mm² variants listed above for
the current manuscript, rather than choosing a version from its date or pixel size.
Obsolete remote figure numbers 12-15 and the superseded supplementary filenames
have been replaced by the current directory structure; earlier files remain in Git history.

Two supplied-image/code differences are retained explicitly:

- `Figure_10_final.png` has revised human-readable feature labels. The retained
  `Figure_10.py` writes `Figure 10.png` using the saved feature identifiers; it
  should not be described as an exact rendering of the supplied final labels.
- `FigS5_mm2.png` is the supplied mm²/day version. The retained `Figure_S05.py`
  computes and labels cm²/day. It requires a display conversion of ×100 and a
  corresponding axis-label change to reproduce that mm²/day presentation.

This synchronization checks file identity and current numbering, not a fresh
execution of every external-data-dependent figure script. Internal validation
records, backups, caches and generated-run directories are not published here.

## Environment and inputs

Use a separate plotting environment and install this directory's `requirements.txt`.
Run Figure 4 in the model environment with the bundled main
plant-segmentation checkpoint. FigS9 requires Node.js, Playwright with
Chromium, and the original English website build as a separate input.

Configure input paths at the top of each script or use its command-line options
shown by `--help`. Most scripts require external feature workbooks, saved
predictions, training logs, QC records or source images. An old absolute path in
a script is not an included dataset. Supplementary Figure 4 uses the included
`code/Figure_S04_daily.csv` and can run without the full research dataset.

## Examples

Run from this directory in the appropriate environment:

```text
python "code/Figure_S04.py" --output "outputs/FigS4.png" --report "outputs/FigS4.json"
python "code/Figure_07.py" --input "C:/research/features.xlsx" --output-dir "outputs/Figure_07"
```

The second example requires the actual feature workbook at the specified path.
For a new FigS9 capture, provide the website build:

```text
node "code/Figure_S09.mjs" --site "C:/research/website/dist" --output-dir "outputs/Figure_S09"
```

Use a new output directory and retain the supplied figure assets. Figure 4's
rail/segmentation modules, `Figure_S10_storage.py` and `history_checks.py` are
required helpers and should remain beside the scripts.

## Scope

Regenerating a plot from saved data does not repeat model training, recover
unavailable source photographs or reconstruct all original preprocessing.
Layout, font rendering, annotation placement and a new website capture may
differ from the supplied images. Input identities, missing values and review
states are retained; source-data availability determines what can be regenerated.
