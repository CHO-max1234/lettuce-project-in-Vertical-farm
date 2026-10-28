"""Reproduce Figure S6 using fixed training cohort and existing PCA embeddings."""
from pathlib import Path

DATA_ROOT = Path(r"D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\3. C-step direct&residual regression")

INPUT_ROOT = DATA_ROOT / "1. 데이터 생성 및 병합"

OUTPUT_DIR = DATA_ROOT / "2. 기본 데이터 시각화"

MANIFEST = INPUT_ROOT / "01. 기준자료/current_manifest.csv"

HISTORY = INPUT_ROOT / "04. 과거정보/history_daily_model_features.csv"

EMBEDDINGS = INPUT_ROOT / "03. Embedding/common_embedding_all_images.npy"

WIDE = INPUT_ROOT / "05. 최종병합 및 검증/C_step_1st_wide_344columns.csv"

SCHEMA = INPUT_ROOT / "05. 최종병합 및 검증/C_step_1st_schema_344columns.csv"

LEGACY_CODE = Path(__file__).resolve().with_name('history_checks.py')

DPI = 600

TRANSPLANT_DATE = "2026-03-06"

WIDTH_IN = 7.2

BLUE, ORANGE = "#0072B2", "#D55E00"

METRICS = ["area_cm2_slope_3d", "area_cm2_slope_7d",
           "embedding_cosine_change_3d", "embedding_cosine_change_7d"]

import argparse

import hashlib

import importlib.util

import json

import platform

import sys

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from tqdm import tqdm

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")

def calculate_one(days, capture, area, vectors, index, window):
    """Calendar window [date(T)-N days, T], using observed endpoints only."""
    left = np.searchsorted(days, days[index] - window, side="left")
    candidates = np.arange(left, index + 1)
    assert np.all(capture[candidates] <= capture[index])
    result = {}
    for kind, valid in [("area", np.isfinite(area[candidates])),
                        ("embedding", np.isfinite(vectors[candidates]).all(axis=1))]:
        ix = candidates[valid]
        gap = int(days[ix[-1]] - days[ix[0]]) if len(ix) > 1 else 0
        value = np.nan
        if gap > 0:
            first, last = ix[0], ix[-1]
            if kind == "area":
                value = float((area[last] - area[first]) / gap)
            else:
                a, b = vectors[first], vectors[last]
                norm = float(np.linalg.norm(a) * np.linalg.norm(b))
                value = float(1 - np.dot(a, b) / norm) if norm > 0 else np.nan
        metric = f"area_cm2_slope_{window}d" if kind == "area" else f"embedding_cosine_change_{window}d"
        result[metric] = value
        result[f"{kind}_n_obs_{window}d"] = len(ix)
        result[f"{kind}_gap_days_{window}d"] = gap if gap > 0 else np.nan
        result[f"{kind}_first_date_{window}d"] = str(np.datetime64(int(days[ix[0]]), "D")) if len(ix) else ""
        result[f"{kind}_last_date_{window}d"] = str(np.datetime64(int(days[ix[-1]]), "D")) if len(ix) else ""
    return result

def array_inputs(group, embedding):
    days = group.history_date.to_numpy(dtype="datetime64[D]").astype(np.int64)
    capture = group.capture_ts.to_numpy(dtype="datetime64[ns]")
    area = group.area_cm2.to_numpy(dtype=np.float64)
    vectors = np.asarray(embedding[group.image_index.to_numpy(dtype=int)], dtype=np.float64)
    return days, capture, area, vectors

def derive_daily(history, cohort, embedding):
    rows = []
    groups = history[history.manifest_row_id.isin(cohort.manifest_row_id)].groupby("manifest_row_id", sort=True)
    owner = cohort.set_index("manifest_row_id")
    for row_id, group in tqdm(groups, total=len(cohort), desc="Daily 3d/7d features", mininterval=5):
        group = group.sort_values(["history_date", "capture_ts"], kind="stable")
        days, capture, area, vectors = array_inputs(group, embedding)
        for i, (_, observation) in enumerate(group.iterrows()):
            values = dict(manifest_row_id=int(row_id), history_key=owner.loc[row_id, "history_key"],
                          stable_plant_key=owner.loc[row_id, "stable_plant_key"], split="train",
                          date=observation.history_date, capture_ts=observation.capture_ts,
                          owner_prediction_cutoff=owner.loc[row_id, "prediction_cutoff"],
                          image_index=int(observation.image_index))
            for window in [3, 7]:
                values.update(calculate_one(days, capture, area, vectors, i, window))
            rows.append(values)
    daily = pd.DataFrame(rows)
    assert not daily.duplicated(["history_key", "date"]).any()
    assert daily.capture_ts.le(daily.owner_prediction_cutoff).all()
    assert daily.split.eq("train").all()
    return daily

def verify_features(history, train, embedding, daily, wide):
    """All 801 train endpoints against delivered CSV, plus legacy prefix checks."""
    ref = wide.set_index("manifest_row_id")
    endpoint_errors = {metric: 0.0 for metric in METRICS}
    for row_id, group in history[history.manifest_row_id.isin(train.manifest_row_id)].groupby("manifest_row_id"):
        group = group.sort_values(["history_date", "capture_ts"])
        values = {}
        args = array_inputs(group, embedding)
        for window in [3, 7]:
            values.update(calculate_one(*args, len(group) - 1, window))
        for metric in METRICS:
            a, b = values[metric], ref.loc[row_id, metric]
            assert np.isclose(a, b, rtol=1e-9, atol=2e-8, equal_nan=True), (row_id, metric, a, b)
            if pd.notna(a):
                endpoint_errors[metric] = max(endpoint_errors[metric], abs(a - b))
    # Independently evaluate existing helpers on multiple earlier prefixes.
    spec = importlib.util.spec_from_file_location("cstep_legacy_plot_check", LEGACY_CODE)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    checked = 0
    emb_cols = [f"emb_{i:03d}" for i in range(128)]
    selected_ids = sorted(daily.manifest_row_id.unique())
    for row_id in np.asarray(selected_ids)[np.linspace(0, len(selected_ids) - 1, 12, dtype=int)]:
        group = history[history.manifest_row_id.eq(row_id)].sort_values(["history_date", "capture_ts"]).copy()
        group = pd.concat([group, pd.DataFrame(np.asarray(embedding[group.image_index.to_numpy(int)], dtype=float), index=group.index, columns=emb_cols)], axis=1)
        observed = daily[daily.manifest_row_id.eq(row_id)].set_index("date")
        for k in sorted(set([0, len(group) // 2, len(group) - 1])):
            prefix = group.iloc[:k + 1]
            date = prefix.history_date.iloc[-1]
            for window in [3, 7]:
                for metric, expected in [(f"area_cm2_slope_{window}d", legacy.window_slope(prefix, "area_cm2", date, window)),
                                         (f"embedding_cosine_change_{window}d", legacy._embedding_window_change(prefix, date, window)[0])]:
                    actual = observed.loc[date, metric]
                    assert np.isclose(actual, expected, rtol=1e-9, atol=2e-8, equal_nan=True), (row_id, date, metric)
                    checked += 1
    return dict(endpoint_rows=len(train), endpoint_scalar_checks=len(train) * len(METRICS),
                max_absolute_endpoint_errors=endpoint_errors, legacy_prefix_scalar_checks=checked,
                no_future_observations=True, no_duplicate_plant_dates=True)

def summarize(daily):
    rows = []
    for date, group in daily.groupby("date", sort=True):
        row = dict(date=date, observed_plants=len(group))
        for metric in METRICS:
            v = group[metric].dropna()
            row.update({f"{metric}_n": len(v), f"{metric}_median": v.median(),
                        f"{metric}_q25": v.quantile(.25), f"{metric}_q75": v.quantile(.75),
                        f"{metric}_min": v.min(), f"{metric}_max": v.max()})
        rows.append(row)
    return pd.DataFrame(rows)

def paper_style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.linewidth": .7, "lines.linewidth": 1.35,
                         "xtick.major.width": .65, "ytick.major.width": .65,
                         "xtick.major.size": 3, "ytick.major.size": 3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white",
                         "axes.unicode_minus": True})

def decorate(ax, panel, dates):
    ax.text(-.17, 1.04, f"({panel})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_xlabel("Days after transplanting (DAT)")
    dat = (pd.DatetimeIndex(dates) - pd.Timestamp(TRANSPLANT_DATE)).days
    lower, upper = 10 * (dat.min() // 10), 10 * int(np.ceil(dat.max() / 10))
    ax.set_xticks(np.arange(lower, upper + 1, 10))
    ax.set_xlim(lower, upper)
    ax.tick_params(direction="out")
    ax.grid(axis="y", color="#E4E4E4", linewidth=.45)
    ax.set_axisbelow(True)

def export_png(fig, target):
    # No title is allowed, including individual panel titles.
    assert not fig._suptitle
    assert all(not ax.get_title() for ax in fig.axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    for ax in fig.axes:
        bounds = ax.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= box.x1 and bounds.y1 <= box.y1, "Clipped figure labels"
    fig.savefig(target, dpi=DPI, metadata={"Software": "C-step 344-column descriptive figures"})
    plt.close(fig)

def display_summary(summary):
    """Interpolate summary curves only; keep raw features/counts unchanged."""
    dates = pd.date_range(summary.date.min(), summary.date.max(), freq="D")
    data = summary.set_index("date").reindex(dates)
    data.index.name = "date"
    data["dat"] = (dates - pd.Timestamp(TRANSPLANT_DATE)).days
    for metric in METRICS:
        data[f"{metric}_observed"] = data[f"{metric}_median"].notna()
        for suffix in ["q25", "median", "q75"]:
            source = data[f"{metric}_{suffix}"]
            shown = source.interpolate(method="time", limit_area="inside")
            assert np.array_equal(shown[source.notna()], source.dropna())
            data[f"{metric}_{suffix}_display"] = shown
        interpolated = data[f"{metric}_median"].isna() & data[f"{metric}_median_display"].notna()
        data[f"{metric}_interpolated_display"] = interpolated
        assert data.loc[interpolated, f"{metric}_n"].fillna(0).eq(0).all()
    return data

def draw_history(summary, kind, target):
    data = display_summary(summary)
    dates, dat = data.index, data.dat.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.2), sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.20, top=.80, wspace=.30)
    bounds = []
    for ax, window, color, panel in zip(axes, [3, 7], [BLUE, ORANGE], "ab"):
        metric = f"area_cm2_slope_{window}d" if kind == "area" else f"embedding_cosine_change_{window}d"
        q25, med, q75 = (data[f"{metric}_{suffix}_display"].to_numpy(float) for suffix in ["q25", "median", "q75"])
        ax.fill_between(dat, q25, q75, facecolor=color, alpha=.19, linewidth=0)
        ax.plot(dat, med, color=color)
        observed = data[f"{metric}_observed"].to_numpy(bool)
        ax.plot(dat[observed], med[observed], linestyle="none", color=color, marker="o", markersize=2, markeredgewidth=0)
        ax.axhline(0, color="#6B6B6B", linewidth=.6, linestyle=(0, (3, 3)), zorder=0)
        decorate(ax, panel, dates)
        # A window label in the legend identifies the feature without a panel title.
        ax.legend(handles=[Line2D([0], [0], color=color, marker="o", markersize=3, label=f"{window}-day window")],
                  loc="upper right", frameon=False, borderaxespad=.2)
        bounds.extend(q25[np.isfinite(q25)].tolist() + q75[np.isfinite(q75)].tolist())
    low, high = min(bounds), max(bounds)
    margin = (high - low) * .13
    axes[0].set_ylim(min(0, low - margin) if kind == "area" else 0, high + margin)
    axes[0].set_ylabel("Projected-area change rate\n(cm²/day)" if kind == "area" else "Embedding cosine distance")
    fig.legend(handles=[Line2D([0], [0], color="#4A4A4A", lw=1.4, label="Median"),
                        Patch(facecolor="#929292", alpha=.3, linewidth=0, label="Interquartile range")],
               loc="upper center", bbox_to_anchor=(.56, .985), ncol=2, frameon=False, handlelength=1.7)
    export_png(fig, target)

OUTPUT_NAME = 'FigS6.png'

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
    out=output_directory()
    sources=[MANIFEST,HISTORY,EMBEDDINGS,WIDE,SCHEMA,LEGACY_CODE]
    hashes={str(p):sha(p) for p in sources}
    manifest=pd.read_csv(MANIFEST)
    manifest['prediction_cutoff']=pd.to_datetime(manifest.prediction_cutoff)
    train=manifest[manifest.split.eq('train')].copy()
    assert len(train)==801 and manifest.groupby('history_key').split.nunique().max()==1
    cohort=train.sort_values(['prediction_cutoff','manifest_row_id'],kind='stable').drop_duplicates('history_key',keep='last')
    assert cohort.stable_plant_key.is_unique and len(cohort)==798
    cols=['manifest_row_id','owner_split','history_key','history_date','capture_ts','area_cm2','image_index','prediction_cutoff']
    history=pd.read_csv(HISTORY,usecols=cols)
    history=history[history.manifest_row_id.isin(train.manifest_row_id)].copy()
    for col in ['history_date','capture_ts','prediction_cutoff']:history[col]=pd.to_datetime(history[col])
    assert history.owner_split.eq('train').all() and history.capture_ts.le(history.prediction_cutoff).all()
    assert history.history_date.eq(history.capture_ts.dt.normalize()).all()
    assert not history.duplicated(['manifest_row_id','history_date']).any()
    owner=train.set_index('manifest_row_id')
    assert history.history_key.eq(history.manifest_row_id.map(owner.history_key)).all()
    assert history.prediction_cutoff.eq(history.manifest_row_id.map(owner.prediction_cutoff)).all()
    embedding=np.load(EMBEDDINGS,mmap_mode='r')
    assert embedding.shape[1]==128
    wide=pd.read_csv(WIDE)
    assert wide.shape==(1288,344)
    daily=derive_daily(history,cohort,embedding)
    checks=verify_features(history,train,embedding,daily,wide)
    summary=summarize(daily)
    for window in [3,7]:assert daily[f'embedding_cosine_change_{window}d'].dropna().between(-1e-12,2+1e-12).all()
    save_csv(summary,out/(Path(OUTPUT_NAME).stem+'_daily_summary.csv'))
    save_csv(cohort,out/(Path(OUTPUT_NAME).stem+'_cohort.csv'))
    paper_style()
    draw_history(summary,'embedding',out/OUTPUT_NAME)
    assert all(sha(Path(p))==h for p,h in hashes.items()), 'Source changed'
    report=dict(status='REPRODUCED',source_hashes=hashes,unique_train_plants=len(cohort),plant_date_rows=len(daily),observed_dates=daily.date.nunique(),checks=checks,model_inference=False,pca_refitted=False,display_interpolation='Linear internal gaps of summary only; observed dots and raw values unchanged',output_sha256=sha(out/OUTPUT_NAME))
    (out/(Path(OUTPUT_NAME).stem+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':
    main()
