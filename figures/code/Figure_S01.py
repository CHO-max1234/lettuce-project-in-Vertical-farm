"""Reproduce FigS1 with the original QC, aggregation and camera weighting."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


# -----------------------------------------------------------------------------
# Paths and analysis settings
# -----------------------------------------------------------------------------
FILE_PATH = Path(r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\정면 윗면 통합분석\6. late fusion을 위한 이미지 전처리\1. cam0-cam1 병합\base파일\2. cam01_merged_B_Hardlabel.xlsx")
SHEET_NAME = "all_merged"
OUTPUT_DIR = Path("outputs/lettuce_publication_figures")
TRANSPLANT_DATE = pd.Timestamp("2026-03-06")
MIN_DAILY_N = 30




# -----------------------------------------------------------------------------
# Columns
# -----------------------------------------------------------------------------
MAIN_TOP = {
    "Projected area": "area_cm2",
    "Perimeter": "perimeter_px",
    "Circularity": "circularity",
    "Angular CV": "angular_cv",
}

MAIN_FRONT = {
    "Projected area": "area_cm2_front",
    "Perimeter": "perimeter_px_front",
    "Circularity": "circularity_front",
    "Angular CV": "angular_cv_front",
}

FEATURE_GROUPS = {
    "Size": [
        "area_cm2", "perimeter_px",
    ],
    "Overall shape": [
        "circularity", "solidity", "curvature", "aspect_ratio",
        "bottom_flatness",
    ],
    "Upper-lower\nstructure": [
        "upper_lower_area_ratio", "upper_circularity", "upper_edge_density",
    ],
    "Radial-central\nstructure": [
        "radial_max_min_ratio", "inner_ratio_30", "inner_ratio_50",
        "angular_cv",
    ],
    "Ridge\nstructure": [
        "ridge_circularity", "ridge_std_r", "ridge_mean_r",
    ],
    "Appearance-\ntexture": [
        "brightness_mean", "dark_ratio", "grad_std",
    ],
}

FEATURE_LABELS = {
    "area_cm2": "Projected area",
    "perimeter_px": "Perimeter",
    "circularity": "Circularity",
    "solidity": "Solidity",
    "curvature": "Curvature",
    "aspect_ratio": "Aspect ratio",
    "bottom_flatness": "Bottom flatness",
    "upper_lower_area_ratio": "Upper/lower area ratio",
    "upper_circularity": "Upper circularity",
    "upper_edge_density": "Upper edge density",
    "radial_max_min_ratio": "Radial max/min ratio",
    "inner_ratio_30": "Inner ratio (30%)",
    "inner_ratio_50": "Inner ratio (50%)",
    "angular_cv": "Angular CV",
    "ridge_circularity": "Ridge circularity",
    "ridge_std_r": "Ridge radial SD",
    "ridge_mean_r": "Ridge mean radius",
    "brightness_mean": "Mean brightness",
    "dark_ratio": "Dark ratio",
    "grad_std": "Gradient SD",
}

HEATMAP_FEATURES = [
    feature
    for group_features in FEATURE_GROUPS.values()
    for feature in group_features
]

USECOLS = [
    "cam_source", "lettuce_key", "date", "qc_status", "crop_reason",
    "assigned", "size_match_csv", "coord_in_bounds", "load_ok",
    "cnn_pred_stage", "cnn_inference_ok",
    *sorted(set(MAIN_TOP.values())),
    *sorted(set(MAIN_FRONT.values())),
    *sorted(set(HEATMAP_FEATURES)),
]



def main():
    global FILE_PATH, OUTPUT_DIR
    import argparse, hashlib, json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=FILE_PATH)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced')
    args=parser.parse_args()
    FILE_PATH=args.input;OUTPUT_DIR=args.output_dir
    if not FILE_PATH.is_file():raise FileNotFoundError(FILE_PATH)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    if (OUTPUT_DIR/'FigS1.png').exists():raise FileExistsError('Choose another output directory')
    source_hash=hashlib.sha256(FILE_PATH.read_bytes()).hexdigest()
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, usecols=USECOLS)
    df["date_dt"] = pd.to_datetime(
        df["date"].astype("Int64").astype(str), format="%y%m%d", errors="coerce"
    )
    df["daf"] = (df["date_dt"] - TRANSPLANT_DATE).dt.days

    qc_mask = (
        df["assigned"].eq(True)
        & df["qc_status"].eq("assigned")
        & df["crop_reason"].eq("ok")
        & df["size_match_csv"].eq(True)
        & df["coord_in_bounds"].eq(True)
        & df["load_ok"].eq(True)
        & df["date_dt"].notna()
    )
    df = df.loc[qc_mask].copy()

    numeric_cols = sorted(
        set(MAIN_TOP.values())
        | set(MAIN_FRONT.values())
        | set(HEATMAP_FEATURES)
        | {"cnn_pred_stage"}
    )
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)


    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    heat_source = df[
        df["cnn_inference_ok"].eq(True)
        & df["cnn_pred_stage"].isin([0, 1, 2, 3, 4])
    ].copy()

    group_keys = ["cam_source", "lettuce_key", "date_dt", "daf"]
    feature_medians = heat_source.groupby(group_keys)[HEATMAP_FEATURES].median()
    stage_modes = heat_source.groupby(group_keys)["cnn_pred_stage"].agg(
        lambda s: int(s.mode().iloc[0])
    )
    heat_plant_day = feature_medians.join(stage_modes).reset_index()

    # Standardize stage medians within each top-view camera, then average the two
    # standardized matrices so CAM0 and CAM1 contribute equally.
    z_matrices = []
    for camera in ["cam0", "cam1"]:
        stage_median = (
            heat_plant_day[heat_plant_day["cam_source"] == camera]
            .groupby("cnn_pred_stage")[HEATMAP_FEATURES]
            .median()
            .reindex([0, 1, 2, 3, 4])
            .T
        )
        row_mean = stage_median.mean(axis=1)
        row_sd = stage_median.std(axis=1, ddof=0).replace(0, np.nan)
        z_matrices.append(stage_median.sub(row_mean, axis=0).div(row_sd, axis=0))

    heat_z = pd.concat(z_matrices, keys=["cam0", "cam1"]).groupby(level=1).mean()
    heat_z = heat_z.loc[HEATMAP_FEATURES]
    heat_z.columns = [f"Stage {stage}" for stage in heat_z.columns]
    heat_z.index = [FEATURE_LABELS[x] for x in heat_z.index]

    group_colors = {
        "Size": "#DCEAF4",
        "Overall shape": "#E7E3F3",
        "Upper-lower\nstructure": "#DDEEDF",
        "Radial-central\nstructure": "#F3E7D7",
        "Ridge\nstructure": "#E8E8E8",
        "Appearance-\ntexture": "#F2DEDF",
    }

    fig = plt.figure(figsize=(8.6, 8.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 2.35, 5.40], wspace=0.02)
    ax_group = fig.add_subplot(gs[0, 0])
    ax_labels = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[0, 2])

    limit = max(2.0, float(np.nanmax(np.abs(heat_z.to_numpy()))))
    sns.heatmap(
        heat_z,
        ax=ax_heat,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.45,
        linecolor="white",
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8},
        yticklabels=False,
        cbar_kws={"label": "Row z-score", "shrink": 0.72, "pad": 0.03},
    )
    ax_heat.set_xlabel("CNN stage")
    ax_heat.set_ylabel("")
    ax_heat.set_title(
        "Stage-wise patterns of handcrafted image features",
        loc="left", fontsize=11, pad=12
    )
    ax_heat.tick_params(axis="x", rotation=0, length=0)
    ax_heat.tick_params(axis="y", length=0)

    n_features = len(HEATMAP_FEATURES)
    ax_group.set_xlim(0, 1)
    ax_group.set_ylim(n_features, 0)
    ax_group.axis("off")

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(n_features, 0)
    ax_labels.axis("off")
    for row_idx, feature_label in enumerate(heat_z.index):
        ax_labels.text(
            0.98, row_idx + 0.5, feature_label,
            ha="right", va="center", fontsize=8.5, color="#333333"
        )

    start = 0
    boundaries = []
    for group_name, features in FEATURE_GROUPS.items():
        end = start + len(features)
        rect = patches.Rectangle(
            (0.05, start), 0.90, end - start,
            facecolor=group_colors[group_name], edgecolor="white", linewidth=1.0
        )
        ax_group.add_patch(rect)
        ax_group.text(
            0.50, (start + end) / 2, group_name,
            ha="center", va="center", fontsize=8.5, color="#333333"
        )
        if end < n_features:
            boundaries.append(end)
        start = end

    for boundary in boundaries:
        ax_labels.axhline(boundary, color="#555555", linewidth=1.0)
        ax_heat.axhline(boundary, color="#555555", linewidth=1.0)

    fig.subplots_adjust(left=0.025, right=0.965, top=0.94, bottom=0.07)

    heat_png = OUTPUT_DIR / "FigS1.png"
    fig.savefig(heat_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


    heat_z.to_csv(OUTPUT_DIR/'FigS1_z_scores.csv',encoding='utf-8-sig')
    assert hashlib.sha256(FILE_PATH.read_bytes()).hexdigest()==source_hash
    (OUTPUT_DIR/'FigS1.json').write_text(json.dumps(dict(status='REPRODUCED',source=str(FILE_PATH),source_sha256=source_hash,qc_rows=len(df),model_inference=False,aggregation='Original collected code; no changes',cohort_matches_manuscript_verified=False,plant_day_camera_rows=len(heat_plant_day),stage_counts=pd.crosstab(heat_plant_day.cam_source,heat_plant_day.cnn_pred_stage).to_dict(),row_z_score='Stage medians within each camera; ddof=0; average CAM0 and CAM1 z matrices equally'),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
