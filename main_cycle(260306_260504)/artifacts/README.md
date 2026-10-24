# Model artifacts

Shared model files for the main and external-validation workflows. Runtime
locations are declared in each cycle's `settings.py`. `manifest.json` records
the supplied files, sizes and SHA-256 hashes.

| Location | Contents |
| --- | --- |
| `segmentation/` | Top/front bed and lettuce segmentation checkpoints |
| `cnn/stage.pt` | Standalone stage-classification checkpoint |
| `a_step/` | Fold-specific stage/weight checkpoints, numeric preprocessing, ordered features and selection metadata |
| `c_step/direct_stacking/`, `c_step/residual/` | Fusion ensembles and ordered feature metadata |
| `pca128_train_only.npz`, `efficientnet_b0.pth` | Frozen PCA and its image-feature backbone |
| `harvest_rule.json` | Fixed stage-probability and predicted-weight rule |
| `ocr/` | EasyOCR detector and English recognizer |

Keep checkpoints paired with their preprocessing and feature-order files.
The standalone stage CNN and A-step fold-specific stage models serve different
roles. Cycle 4 uses its own additional front-view bed checkpoint from
`validation_cycle3(260723_260826)/artifacts/`.

From the main cycle directory, run:

```text
python "pipeline/1. Run_cycle.py" --check-models
```

This verifies file integrity and the declared connections without running image
inference. Missing or incompatible environmental predictors remain `NaN` while
the saved model input order is preserved. Predictions depend on the supplied
data and missing-input conditions. The original model metadata may contain old
source paths; runtime paths are resolved through `settings.py`.
