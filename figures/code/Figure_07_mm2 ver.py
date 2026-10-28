"""Reproduce Figure 7_mm2 ver with the original QC, aggregation and camera weighting."""
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
    if (OUTPUT_DIR/'Figure 7_mm2 ver.png').exists():raise FileExistsError('Choose another output directory')
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


    # First collapse repeated observations within each camera, plant and day.
    top_camera_day = (
        df.groupby(["cam_source", "lettuce_key", "date_dt", "daf"], as_index=False)[
            list(MAIN_TOP.values())
        ]
        .median()
    )

    # Then combine CAM0 and CAM1 into one Top-view value per matched plant-day.
    top_plant_day = (
        top_camera_day.groupby(["lettuce_key", "date_dt", "daf"], as_index=False)[
            list(MAIN_TOP.values())
        ]
        .mean()
    )

    # CAM2 measurements are stored in *_front columns and can repeat after merging.
    front_plant_day = (
        df.dropna(subset=list(MAIN_FRONT.values()), how="all")
        .groupby(["lettuce_key", "date_dt", "daf"], as_index=False)[
            list(MAIN_FRONT.values())
        ]
        .median()
    )


    def make_long_view(data, mapping, view_name):
        parts = []
        for feature_name, column_name in mapping.items():
            part = data[["lettuce_key", "date_dt", "daf", column_name]].copy()
            part = part.rename(columns={column_name: "value"})
            part["feature"] = feature_name
            part["view"] = view_name
            parts.append(part)
        return pd.concat(parts, ignore_index=True)


    main_long = pd.concat(
        [
            make_long_view(top_plant_day, MAIN_TOP, "Top view"),
            make_long_view(front_plant_day, MAIN_FRONT, "Front view"),
        ],
        ignore_index=True,
    )
    main_long = main_long.dropna(subset=["value"])

    main_summary = (
        main_long.groupby(["feature", "view", "daf"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
    )
    main_summary["sd"] = main_summary["sd"].fillna(0)
    # Convert only panel (a); source measurements remain in cm2.
    area_rows = main_summary["feature"].eq("Projected area")
    main_summary.loc[area_rows, ["mean", "sd"]] *= 100.0


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

    colors = {"Top view": "#0072B2", "Front view": "#D55E00"}
    panel_info = [
        ("Projected area", "Projected area (mm$^2$)"),
        ("Perimeter", "Perimeter (px)"),
        ("Circularity", "Circularity"),
        ("Angular CV", "Angular CV"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True)
    axes = axes.ravel()

    for panel_idx, (ax, (feature_name, y_label)) in enumerate(zip(axes, panel_info)):
        for view_name in ["Top view", "Front view"]:
            d = main_summary[
                (main_summary["feature"] == feature_name)
                & (main_summary["view"] == view_name)
                & (main_summary["n"] >= MIN_DAILY_N)
            ].sort_values("daf")

            x = d["daf"].to_numpy(dtype=float)
            y = d["mean"].to_numpy(dtype=float)
            sd = d["sd"].to_numpy(dtype=float)
            lower = y - sd
            upper = y + sd

            if feature_name in ["Projected area", "Perimeter", "Angular CV"]:
                lower = np.maximum(lower, 0)
            elif feature_name == "Circularity":
                lower = np.maximum(lower, 0)
                upper = np.minimum(upper, 1)

            ax.fill_between(
                x, lower, upper, color=colors[view_name], alpha=0.14,
                linewidth=0, zorder=1
            )
            ax.plot(
                x, y, color=colors[view_name], linewidth=1.8,
                label=view_name, zorder=2
            )

        ax.set_title(f"({chr(97 + panel_idx)}) {feature_name}", loc="left", pad=6)
        ax.set_ylabel(y_label)
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(colors="#333333", length=3)
        ax.set_xlim(0, 60)
        ax.set_xticks(np.arange(0, 61, 10))

    for ax in axes[2:]:
        ax.set_xlabel("Days after transplanting")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, ["Top view (CAM0 + CAM1)", "Front view (CAM2)"],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
        frameon=False, handlelength=2.8, columnspacing=1.8
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.10, top=0.90, wspace=0.30, hspace=0.34)

    main_png = OUTPUT_DIR / "Figure 7_mm2 ver.png"
    fig.savefig(main_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


    main_summary.to_csv(OUTPUT_DIR/'Figure 7_mm2 ver_daily_summary.csv',index=False,encoding='utf-8-sig')
    assert hashlib.sha256(FILE_PATH.read_bytes()).hexdigest()==source_hash
    (OUTPUT_DIR/'Figure 7_mm2 ver.json').write_text(json.dumps(dict(status='REPRODUCED',source=str(FILE_PATH),source_sha256=source_hash,qc_rows=len(df),model_inference=False,aggregation='Original QC and aggregation; projected-area mean and SD converted from cm2 to mm2 (x100)',cohort_matches_manuscript_verified=False,plant_day_counts=main_summary.groupby(['feature','view'],as_index=False)['n'].sum().to_dict('records'),center='Mean',shade='Sample SD; lower bound clipped at 0, circularity upper bound clipped at 1',minimum_daily_n=MIN_DAILY_N),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
