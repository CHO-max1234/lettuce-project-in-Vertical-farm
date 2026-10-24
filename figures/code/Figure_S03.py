"""Reproduce Fig. S3 from the saved daily environmental descriptors.

Usage: python Figure_S03.py
       python Figure_S03.py --check-only
The adjacent CSV is a selected, unchanged export of environment_daily_causal.csv.
No environmental penalty, model fitting, interpolation, or sensor correction is used.
"""
from pathlib import Path

# Input and presentation settings.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "Figure_S03_daily.csv"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "reproduced" / "FigS3.png"
TRANSPLANT_DATE = "2026-03-06"
FIRST_SENSOR_DATE = "2026-03-19"
LAST_PLOTTED_DATE = "2026-05-03"
BASE_TEMPERATURE = 5.0
CAP_TEMPERATURE = 23.0
LOW_COVERAGE_HOURS = 20
DPI = 600
COLORS = ("#35567F", "#C96C36", "#8465A5", "#2D8586")
REQUIRED_COLUMNS = [
    "date", "temp_mean", "capped_gdd_day", "heat_degree_hours_28",
    "vpd_hours_ratio_1_8", "water_air_gap_day", "valid_stress_hour_count",
]

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def digest(path):
    """Hash the small read-only plot input."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_daily(source):
    """Select the observed sensor days and validate their original definitions."""
    frame = pd.read_csv(source, usecols=REQUIRED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate sensor dates; one row per date is required.")
    frame = frame.sort_values("date").loc[
        lambda x: x["date"].between(FIRST_SENSOR_DATE, LAST_PLOTTED_DATE)
    ].copy()
    expected = pd.date_range(FIRST_SENSOR_DATE, LAST_PLOTTED_DATE)
    if not np.array_equal(frame["date"].to_numpy(), expected.to_numpy()):
        raise ValueError("Missing sensor days; this figure does not impute days.")
    values = frame[REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite plot values; do not replace with zero.")
    hours = frame["valid_stress_hour_count"].to_numpy(float)
    if not ((hours == np.floor(hours)) & (hours > 0) & (hours <= 24)).all():
        raise ValueError("Invalid number of valid hourly observations.")
    if not frame["vpd_hours_ratio_1_8"].between(0, 1).all():
        raise ValueError("The VPD source column must be a ratio between zero and one.")
    expected_gdd = np.maximum(np.minimum(frame["temp_mean"], CAP_TEMPERATURE)
                              - BASE_TEMPERATURE, 0)
    np.testing.assert_allclose(frame["capped_gdd_day"], expected_gdd,
                               atol=1e-9, rtol=0)
    frame["DAT"] = (frame["date"] - pd.Timestamp(TRANSPLANT_DATE)).dt.days
    frame["capped_gdd_cum"] = frame["capped_gdd_day"].cumsum()
    frame["vpd_percent"] = frame["vpd_hours_ratio_1_8"] * 100
    frame["low_coverage"] = frame["valid_stress_hour_count"] < LOW_COVERAGE_HOURS
    return frame.reset_index(drop=True)


def plot_daily(frame, output):
    """Draw the four source-based panels and preserve low-coverage markers."""
    font = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font, "font.size": 9.5, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.15), sharex=True)
    fig.subplots_adjust(left=.105, right=.975, top=.94, bottom=.135,
                        wspace=.30, hspace=.35)
    specs = [
        ("(a)  Cumulative capped GDD", "capped_gdd_cum", "GDD (°C·d)", (0, 800), np.arange(0, 801, 100)),
        ("(b)  Heat above 28 °C", "heat_degree_hours_28", r"Degree hours (°C·h d$^{-1}$)", (-5, 160), np.arange(0, 151, 25)),
        ("(c)  VPD above 1.8 kPa", "vpd_percent", "Valid hourly observations (%)", (-5, 105), np.arange(0, 101, 25)),
        ("(d)  Water-air temperature gap", "water_air_gap_day", "Reported difference (°C)", (-21, 24), [-20, -10, 0, 10, 20]),
    ]
    for index, (ax, color, spec) in enumerate(zip(axes.flat, COLORS, specs)):
        title, column, ylabel, ylim, yticks = spec
        x, y = frame["DAT"], frame[column]
        ax.set_title(title, loc="left", pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E6EBF1", linewidth=.7)
        ax.plot(x, y, color=color, linewidth=1.1, zorder=2)
        low = frame["low_coverage"] if index in (1, 2) else pd.Series(False, index=frame.index)
        ax.scatter(x[~low], y[~low], s=17, color=color, edgecolors=color,
                   linewidths=.3, zorder=3)
        if low.any():
            ax.scatter(x[low], y[low], s=29, facecolors="white", edgecolors=color,
                       linewidths=1.0, zorder=4)
        ax.set(ylabel=ylabel, ylim=ylim, yticks=yticks, xlim=(12, 59),
               xticks=[13, 20, 30, 40, 50, 58])
        if index == 3:
            ax.axhline(0, color="#8FA0B0", linewidth=.85, linestyle=(0, (2.5, 3)))
    for ax in axes[1]:
        ax.set_xlabel("Days after transplanting (DAT)")
    fig.text(.105, .035,
             "○  <20 valid hourly observations in panels b and c; cumulative GDD begins at the first sensor day (DAT 13).",
             color="#647489", fontsize=7.8, ha="left", va="bottom")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in axes.flat:
        artists += [ax.title, ax.xaxis.label, ax.yaxis.label]
        artists += ax.get_xticklabels() + ax.get_yticklabels()
    bounds = fig.bbox
    for artist in artists:
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > bounds.width + .5 or box.y1 > bounds.height + .5:
            raise ValueError(f"Text extends outside the figure: {artist.get_text()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, metadata={"Software": "Saved daily environment descriptors; Fig. S3"})
    plt.close(fig)
    return font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, help="Optional JSON validation record.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    before = digest(args.input)
    frame = load_daily(args.input)
    report = {
        "source": str(args.input.resolve()), "source_sha256": before,
        "date_start": frame["date"].min().strftime("%Y-%m-%d"),
        "date_end": frame["date"].max().strftime("%Y-%m-%d"),
        "days": len(frame), "DAT": [int(frame["DAT"].min()), int(frame["DAT"].max())],
        "last_cumulative_capped_gdd": float(frame["capped_gdd_cum"].iloc[-1]),
        "low_coverage_days": frame.loc[frame["low_coverage"], "date"].dt.strftime("%Y-%m-%d").tolist(),
        "gdd_definition": "max(min(daily_mean_temperature, 23) - 5, 0); cumulative from first sensor date",
        "environmental_penalty_applied": False, "interpolation": False,
        "model_inference": False, "sensor_anomaly_correction": False,
        "water_air_gap": "saved signed water_air_gap_day, without taking its absolute value",
    }
    if not args.check_only:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite explicitly: {args.output}")
        report["font"] = plot_daily(frame, args.output)
        report["output"] = str(args.output.resolve())
        report["output_sha256"] = digest(args.output)
    if digest(args.input) != before:
        raise RuntimeError("The source CSV changed during reproduction.")
    report["source_unchanged"] = True
    if args.report and not args.check_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
