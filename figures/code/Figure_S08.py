"""Reproduce manuscript Figure S8 from saved predictions; no refitting."""
from pathlib import Path

ROOT = Path(r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\3. C-step direct&residual regression")

OUTPUT = ROOT / "4. 학습,평가,비교,시각화"

SUPPORT = OUTPUT / "_검증자료" / "publication_current_2nd"

PREDICTIONS = OUTPUT / "current_predictions.csv"

METRICS = OUTPUT / "current_full_metrics.csv"

METADATA = ROOT / "3. 학습용 데이터 생성" / "00_common" / "all_rows_metadata.csv"

STAGES = ROOT / "1. 데이터 생성 및 병합" / "05. 최종병합 및 검증" / "C_step_2nd_wide_1288x454.csv"

SELECTION = OUTPUT / "selection_before_test.json"

DPI = 600

MODELS = [
    ("Baseline CNN", "A_step", "cnn_only", "#62676D", "o"),
    ("Direct", "454columns", "direct", "#0072B2", "s"),
    ("Residual", "454columns", "residual", "#009E73", "^"),
    ("50:50 blend", "454columns", "blend50", "#D55E00", "D"),
]

import hashlib

import json

import math

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib.ticker import MultipleLocator

from PIL import Image

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics_for(frame):
    y = frame.y_true.to_numpy(float)
    error = frame.prediction.to_numpy(float) - y
    denominator = np.sum((y - np.mean(y)) ** 2)
    return dict(n=len(y), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error ** 2))),
                r2=float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
                bias=float(np.mean(error)))

def read_and_verify():
    sources = [PREDICTIONS, METRICS, METADATA, STAGES, SELECTION]
    hashes = {str(p): sha256(p) for p in sources}
    all_predictions = pd.read_csv(PREDICTIONS)
    saved_metrics = pd.read_csv(METRICS)
    metadata = pd.read_csv(METADATA)
    stage = pd.read_csv(STAGES, usecols=["manifest_row_id", "cnn_pred_stage", "split", "fresh_weight_g"])
    for frame in (all_predictions, metadata, stage):
        ids = frame.manifest_row_id.to_numpy(float)
        assert np.isfinite(ids).all() and np.equal(ids, np.floor(ids)).all()
        frame["manifest_row_id"] = ids.astype(int)
    assert metadata.manifest_row_id.is_unique and stage.manifest_row_id.is_unique
    test_meta = metadata.loc[metadata.split.eq("test")].sort_values("manifest_row_id")
    assert len(test_meta) == 268
    test_stage = stage.loc[stage.split.eq("test")].sort_values("manifest_row_id")
    np.testing.assert_array_equal(test_meta.manifest_row_id, test_stage.manifest_row_id)
    np.testing.assert_allclose(test_meta.fresh_weight_g, test_stage.fresh_weight_g, rtol=0, atol=1e-10)
    assert test_stage.cnn_pred_stage.isin(range(5)).all()
    groups, summary = [], []
    for label, variant, mode, _, _ in MODELS:
        frame = all_predictions.loc[all_predictions.variant.eq(variant) &
                                    all_predictions['mode'].eq(mode) &
                                    all_predictions.split.eq("test")].sort_values("manifest_row_id").copy()
        assert frame.manifest_row_id.is_unique
        np.testing.assert_array_equal(frame.manifest_row_id, test_meta.manifest_row_id)
        assert np.isfinite(frame[["y_true", "prediction", "error_g"]].to_numpy()).all()
        np.testing.assert_allclose(frame.y_true, test_meta.fresh_weight_g, rtol=0, atol=1e-10)
        np.testing.assert_allclose(frame.error_g, frame.prediction - frame.y_true, rtol=0, atol=1e-10)
        if mode == "cnn_only":
            np.testing.assert_allclose(frame.prediction, test_meta.pred_weight_cnn, rtol=0, atol=1e-10)
        frame = frame.merge(test_meta[["manifest_row_id", "bed_name", "stable_plant_key"]],
                            on="manifest_row_id", how="left", validate="one_to_one")
        frame = frame.merge(test_stage[["manifest_row_id", "cnn_pred_stage"]],
                            on="manifest_row_id", how="left", validate="one_to_one")
        frame["cnn_pred_stage"] = frame.cnn_pred_stage.astype(int)
        frame["Model"] = label
        observed_metrics = metrics_for(frame)
        stored = saved_metrics.loc[saved_metrics.variant.eq(variant) &
                                   saved_metrics['mode'].eq(mode) & saved_metrics.split.eq("test")]
        assert len(stored) == 1
        for key in ("n", "mae", "rmse", "r2", "bias"):
            np.testing.assert_allclose(observed_metrics[key], stored.iloc[0][key], rtol=0, atol=1e-10)
        summary.append(dict(Model=label, **observed_metrics))
        groups.append(frame)
    long = pd.concat(groups, ignore_index=True)
    # The published Direct is pure Direct. Only use the stacking predictions here
    # to verify the already-frozen blend, never to replace the published Direct.
    stacking = all_predictions.loc[all_predictions.variant.eq("454columns") &
                                   all_predictions['mode'].eq("direct_stacking") &
                                   all_predictions.split.eq("test")].sort_values("manifest_row_id")
    assert stacking.manifest_row_id.is_unique
    np.testing.assert_array_equal(stacking.manifest_row_id, test_meta.manifest_row_id)
    residual = groups[2].sort_values("manifest_row_id").prediction.to_numpy()
    blend = groups[3].sort_values("manifest_row_id").prediction.to_numpy()
    np.testing.assert_allclose(blend, 0.5 * stacking.prediction.to_numpy() + 0.5 * residual,
                               rtol=0, atol=1e-10)
    stage_rows = []
    for label, *_ in MODELS:
        for s in range(5):
            part = long.loc[long.Model.eq(label) & long.cnn_pred_stage.eq(s)]
            stage_rows.append(dict(Model=label, cnn_pred_stage=s, n_beds=part.bed_name.nunique(),
                                   **metrics_for(part)))
    return long, pd.DataFrame(summary), pd.DataFrame(stage_rows), hashes

def set_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })

def save_figure(fig, filename):
    path = OUTPUT / OUTPUT_NAME
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    with Image.open(path) as im:
        return dict(file=path.name, width_px=im.width, height_px=im.height,
                    dpi=list(im.info.get("dpi", [])), sha256=sha256(path))

def scatter_figure(long):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), sharex=True, sharey=True)
    extrema = long[["y_true", "prediction"]].to_numpy()
    low = math.floor(min(0.0, float(extrema.min())) / 25) * 25
    high = math.ceil(float(extrema.max()) / 25) * 25
    if high == extrema.max():
        high += 25
    for i, (ax, (label, _, _, color, marker)) in enumerate(zip(axes.flat, MODELS)):
        part = long.loc[long.Model.eq(label)]
        ax.plot([low, high], [low, high], color="#525252", lw=0.85, ls=(0, (4, 3)), zorder=1)
        ax.scatter(part.y_true, part.prediction, s=14, color=color, alpha=0.56,
                   edgecolors="none", marker=marker, zorder=2)
        ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.text(0.045, 0.95, f"({chr(97+i)})  {label}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, weight="bold")
        ax.tick_params(direction="out", length=3, labelbottom=True, labelleft=True)
    fig.supxlabel("Observed fresh weight (g)", y=0.028, fontsize=10)
    fig.supylabel("Predicted fresh weight (g)", x=0.025, fontsize=10)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.98, wspace=0.17, hspace=0.18)
    return save_figure(fig, "Figure_01_current_2nd_observed_predicted.png")

OUTPUT_NAME = 'FigS8.png'

def output_directory():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'reproduced')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / OUTPUT_NAME).exists():
        raise FileExistsError('Choose another --output-dir to preserve the existing figure')
    return args.output_dir

def main():
    global OUTPUT
    OUTPUT = output_directory()
    long,summary,stage_metrics,hashes=read_and_verify()
    set_style()
    image=scatter_figure(long)
    long.to_csv(OUTPUT/(Path(OUTPUT_NAME).stem+'_observations.csv'),index=False,encoding='utf-8-sig')
    stage_metrics.to_csv(OUTPUT/(Path(OUTPUT_NAME).stem+'_stage_metrics.csv'),index=False,encoding='utf-8-sig')
    assert all(sha256(Path(p))==h for p,h in hashes.items()), 'Source changed'
    (OUTPUT/(Path(OUTPUT_NAME).stem+'.json')).write_text(json.dumps(dict(status='REPRODUCED',source_hashes=hashes,observations_per_model=268,stage_counts=stage_metrics.iloc[:5][['cnn_pred_stage','n']].to_dict('records'),metrics=summary.to_dict('records'),image=image,model_inference=False),indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
