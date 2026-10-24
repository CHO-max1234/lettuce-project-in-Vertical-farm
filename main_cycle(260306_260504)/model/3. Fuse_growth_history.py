"""Fuse image history with frozen legacy16 models. Calculate manuscript12 environmental features, retain equivalent fields and pass unavailable model inputs as NaN. Paths are configured in settings.py."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
from model_contract import (ENVIRONMENT_FEATURES, MODEL_ENVIRONMENT_FEATURES,
                            align_model_environment, environment_contract, component_contracts)
from pathlib import Path
DATA_ROOT = CYCLE.DATA_ROOT
A_ROOT = CYCLE.ARTIFACT_ROOT
A_OUTPUT = CYCLE.OUTPUT_ROOT / '02_A-step'
A_MODEL = CYCLE.A_MODEL
A_RUN = None
OUTPUT_DIR = CYCLE.OUTPUT_ROOT / '04_C-step'
MODEL_ROOT = CYCLE.C_MODELS
GEN = CYCLE.DATA_ROOT
PCA_FILE = CYCLE.PCA_FILE
ENV_BASE = CYCLE.OUTPUT_ROOT / '03_environment' / 'environment_daily_from_raw.csv'
SUPPORTED_SEASON = CYCLE.SEASON_ID
SEASON_START = CYCLE.SEASON_START
SEASON_END = CYCLE.SEASON_END
MODEL_ROLE = 'operational_ensemble_not_oof'
BATCH_SIZE = CYCLE.BATCH_SIZE
ROW_CHUNK = 256
IMAGE_WORKERS = CYCLE.WORKERS
DEVICE = CYCLE.DEVICE
USE_AMP = True
import argparse, csv, hashlib, io, json, os, re, sqlite3, time, traceback, warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
EMB_COLS = [f'emb_{i:03d}' for i in range(128)]
IMAGE_NAME = re.compile('^(bed\\d+)_(20\\d{6})_(\\d{6})_(cam[012])_r(?:f)?_([bt]\\d+)(?:_p\\d+)?\\.(?:png|jpg|jpeg)$', re.I)
RAW_GEOMETRY = 'blur_score area_cm2 perimeter_px circularity solidity curvature aspect_ratio bottom_flatness upper_area_px lower_area_px upper_lower_area_ratio upper_circularity upper_edge_density radial_min radial_max_min_ratio brightness_mean dark_ratio grad_std inner_ratio_30 inner_ratio_50 ridge_circularity ridge_std_r ridge_mean_r angular_cv'.split()
EXTRA = ['c_step_status', 'c_step_error', 'c_step_prediction_role', 'c_step_model', 'c_step_pred_weight_g', 'c_step_stacking_g', 'c_step_residual_g', 'c_step_residual_correction_g', 'c_step_negative_prediction', 'c_step_history_n_days', 'c_step_history_first_date', 'c_step_history_last_timestamp', 'c_step_env_source_date', 'c_step_env_7d_observed_days', 'c_step_env_missing_reason', 'c_step_missing_feature_count']

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def js(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    os.replace(tmp, path)

def require(ok, message):
    if not ok:
        raise ValueError(message)
ARTIFACT_HASHES = {}

def parse_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    text = str(value).strip()
    if not text:
        return pd.NaT
    if text.endswith('.0'):
        text = text[:-2]
    if re.fullmatch('\\d{6}', text):
        return pd.to_datetime('20' + text, format='%Y%m%d', errors='coerce')
    if re.fullmatch('\\d{8}', text):
        return pd.to_datetime(text, format='%Y%m%d', errors='coerce')
    out = pd.to_datetime(text, errors='coerce')
    return out.normalize() if pd.notna(out) else pd.NaT

def fmt_date(value: object) -> str:
    ts = parse_date(value)
    return ts.strftime('%Y-%m-%d') if pd.notna(ts) else ''

def safe_num(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan

def date_days(a: pd.Timestamp, b: pd.Timestamp) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float((a - b).days)

def latest_nonnull(history: pd.DataFrame, col: str) -> float:
    if col not in history.columns or history.empty:
        return np.nan
    s = pd.to_numeric(history[col], errors='coerce').dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def mean_nonnull(history: pd.DataFrame, col: str) -> float:
    if col not in history.columns or history.empty:
        return np.nan
    s = pd.to_numeric(history[col], errors='coerce').dropna()
    return float(s.mean()) if len(s) else np.nan

def window_delta(history: pd.DataFrame, col: str, cutoff: pd.Timestamp, days: int) -> float:
    if col not in history.columns or history.empty or pd.isna(cutoff):
        return np.nan
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].copy()
    h = h[pd.to_numeric(h[col], errors='coerce').notna()]
    if h['history_date'].nunique() < 2:
        return np.nan
    first = h.iloc[0]
    last = h.iloc[-1]
    return float(safe_num(last[col]) - safe_num(first[col]))

def window_slope(history: pd.DataFrame, col: str, cutoff: pd.Timestamp, days: int) -> float:
    if col not in history.columns or history.empty or pd.isna(cutoff):
        return np.nan
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].copy()
    h = h[pd.to_numeric(h[col], errors='coerce').notna()]
    if h['history_date'].nunique() < 2:
        return np.nan
    first = h.iloc[0]
    last = h.iloc[-1]
    gap = (last['history_date'] - first['history_date']).days
    if gap <= 0:
        return np.nan
    return float((safe_num(last[col]) - safe_num(first[col])) / gap)

def window_auc(history: pd.DataFrame, col: str, cutoff: pd.Timestamp, days: int) -> float:
    if col not in history.columns or history.empty or pd.isna(cutoff):
        return np.nan
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].copy()
    h[col] = pd.to_numeric(h[col], errors='coerce')
    h = h.dropna(subset=[col, 'history_date']).drop_duplicates('history_date', keep='last')
    if len(h) < 2:
        return np.nan
    x = (h['history_date'] - h['history_date'].min()).dt.days.astype(float).to_numpy()
    y = h[col].astype(float).to_numpy()
    return float(np.trapz(y, x))

def _embedding_columns(history: pd.DataFrame) -> list[str]:
    return sorted([c for c in history.columns if re.fullmatch('emb_\\d{3}', str(c))])

def _embedding_vector(row: pd.Series, emb_cols: list[str]) -> np.ndarray | None:
    if not emb_cols:
        return None
    values = pd.to_numeric(row[emb_cols], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    return values

def _cosine_change(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return np.nan
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    cosine = float(np.dot(vec_a, vec_b) / denom)
    return float(1.0 - cosine)

def _l2_change(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float:
    if vec_a is None or vec_b is None:
        return np.nan
    return float(np.linalg.norm(vec_b - vec_a))

def _latest_embedding(history: pd.DataFrame) -> tuple[pd.Series | None, np.ndarray | None, list[str]]:
    emb_cols = _embedding_columns(history)
    if not emb_cols or history.empty:
        return (None, None, emb_cols)
    h = history.dropna(subset=emb_cols, how='any')
    if h.empty:
        return (None, None, emb_cols)
    row = h.iloc[-1]
    return (row, _embedding_vector(row, emb_cols), emb_cols)

def _embedding_window_change(history: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> tuple[float, float, float]:
    emb_cols = _embedding_columns(history)
    if not emb_cols or history.empty or pd.isna(cutoff):
        return (np.nan, np.nan, np.nan)
    h = history[history['history_date'].ge(cutoff - pd.Timedelta(days=days))].dropna(subset=emb_cols, how='any')
    if h['history_date'].nunique() < 2:
        return (np.nan, np.nan, np.nan)
    first = h.iloc[0]
    last = h.iloc[-1]
    va = _embedding_vector(first, emb_cols)
    vb = _embedding_vector(last, emb_cols)
    gap = (last['history_date'] - first['history_date']).days
    l2 = _l2_change(va, vb)
    return (_cosine_change(va, vb), l2, l2 / gap if pd.notna(l2) and gap > 0 else np.nan)

def _embedding_between_rows(row_a: pd.Series | None, row_b: pd.Series | None, emb_cols: list[str]) -> tuple[float, float]:
    if row_a is None or row_b is None:
        return (np.nan, np.nan)
    va = _embedding_vector(row_a, emb_cols)
    vb = _embedding_vector(row_b, emb_cols)
    return (_cosine_change(va, vb), _l2_change(va, vb))

def history_tracking_features(history: pd.DataFrame, current: pd.Series) -> dict[str, object]:
    cutoff = parse_date(current.get('survey_date'))
    out: dict[str, object] = {}
    h = history.sort_values(['history_date', 'crop_filename'], na_position='last').copy()
    valid_dates = h['history_date'].dropna()
    out['history_n_obs'] = int(len(h))
    out['history_first_date'] = valid_dates.min().strftime('%Y-%m-%d') if len(valid_dates) else ''
    out['history_last_date'] = valid_dates.max().strftime('%Y-%m-%d') if len(valid_dates) else ''
    out['history_days_span'] = int((valid_dates.max() - valid_dates.min()).days) if len(valid_dates) else np.nan
    lead_days = pd.to_numeric(h.get('history_lead_days'), errors='coerce') if 'history_lead_days' in h else pd.Series(dtype=float)
    out['history_min_lead_days'] = float(lead_days.min()) if len(lead_days.dropna()) else np.nan
    out['history_max_lead_days'] = float(lead_days.max()) if len(lead_days.dropna()) else np.nan
    out['history_current_gap_days'] = float(lead_days.min()) if len(lead_days.dropna()) else np.nan
    out['history_has_current_image'] = bool((lead_days == 0).any()) if len(lead_days) else False
    out['history_key_source'] = h['source_tier'].iloc[0] if len(h) and 'source_tier' in h.columns else ''
    out['history_key_confidence'] = 'exact_bed_slot' if out['history_key_source'] in {'B_history', 'raw_direct'} else 'similar_fallback'
    stage = pd.to_numeric(h.get('cnn_pred_stage'), errors='coerce') if 'cnn_pred_stage' in h else pd.Series(index=h.index, dtype=float)
    expected = pd.to_numeric(h.get('cnn_expected_stage'), errors='coerce') if 'cnn_expected_stage' in h else pd.Series(index=h.index, dtype=float)
    h['_stage_int'] = stage.round().astype('Int64') if len(stage) else pd.Series(dtype='Int64')
    for stage_id in [1, 2, 3, 4]:
        rows = h[h['_stage_int'].eq(stage_id)]
        first = rows['history_date'].dropna().min() if len(rows) else pd.NaT
        out[f'days_since_stage{stage_id}_first'] = date_days(cutoff, first) if pd.notna(first) else np.nan
    valid_stage = h[h['_stage_int'].notna()].copy()
    if len(valid_stage):
        changes = valid_stage['_stage_int'].ne(valid_stage['_stage_int'].shift()).fillna(True)
        change_rows = valid_stage[changes]
        out['stage_transition_count'] = max(int(changes.sum()) - 1, 0)
        last_change = change_rows.iloc[-1]
        out['last_stage_change_date'] = fmt_date(last_change['history_date'])
        out['days_since_last_stage_change'] = date_days(cutoff, last_change['history_date'])
        out['days_in_current_stage'] = out['days_since_last_stage_change']
        out['cnn_stage_current'] = int(valid_stage['_stage_int'].iloc[-1])
        out['cnn_stage_min_history'] = int(valid_stage['_stage_int'].min())
        out['cnn_stage_max_history'] = int(valid_stage['_stage_int'].max())
        out['cnn_stage_transition_count'] = out['stage_transition_count']
        out['cnn_expected_stage_at_last_change'] = safe_num(last_change.get('cnn_expected_stage'))
        out['cnn_expected_stage_delta_since_last_change'] = latest_nonnull(h, 'cnn_expected_stage') - safe_num(last_change.get('cnn_expected_stage'))
    else:
        for col in ['stage_transition_count', 'last_stage_change_date', 'days_since_last_stage_change', 'days_in_current_stage', 'cnn_stage_current', 'cnn_stage_min_history', 'cnn_stage_max_history', 'cnn_stage_transition_count', 'cnn_expected_stage_at_last_change', 'cnn_expected_stage_delta_since_last_change']:
            out[col] = np.nan if col != 'last_stage_change_date' else ''
    stage3_dates = h.loc[stage.ge(3), 'history_date'].dropna()
    out['days_since_stage3'] = date_days(cutoff, stage3_dates.min()) if len(stage3_dates) else np.nan
    p4_dates = h.loc[pd.to_numeric(h.get('cnn_p4'), errors='coerce').gt(0.5), 'history_date'].dropna() if 'cnn_p4' in h else []
    out['days_since_p4_over_0_5'] = date_days(cutoff, min(p4_dates)) if len(p4_dates) else np.nan
    return out

def beforeday_features(history: pd.DataFrame, current: pd.Series) -> dict[str, object]:
    cutoff = parse_date(current.get('survey_date'))
    h = history.sort_values(['history_date', 'crop_filename'], na_position='last').copy()
    out = history_tracking_features(h, current)
    out.update({'cnn_pred_weight_current': latest_nonnull(h, 'pred_weight_cnn'), 'cnn_pred_weight_slope_3d': window_slope(h, 'pred_weight_cnn', cutoff, 3), 'cnn_pred_weight_slope_7d': window_slope(h, 'pred_weight_cnn', cutoff, 7), 'cnn_pred_weight_delta_3d': window_delta(h, 'pred_weight_cnn', cutoff, 3), 'cnn_pred_weight_delta_7d': window_delta(h, 'pred_weight_cnn', cutoff, 7)})
    (_latest_emb_row, latest_emb, _emb_cols) = _latest_embedding(h)
    out['embedding_norm_current'] = float(np.linalg.norm(latest_emb)) if latest_emb is not None else np.nan
    (cos3, l23, _per3) = _embedding_window_change(h, cutoff, 3)
    (cos7, l27, per7) = _embedding_window_change(h, cutoff, 7)
    out['embedding_cosine_change_3d'] = cos3
    out['embedding_cosine_change_7d'] = cos7
    out['embedding_l2_change_3d'] = l23
    out['embedding_l2_change_7d'] = l27
    out['embedding_change_per_day_7d'] = per7
    out['cnn_expected_stage_current'] = latest_nonnull(h, 'cnn_expected_stage')
    out['cnn_expected_stage_slope_3d'] = window_slope(h, 'cnn_expected_stage', cutoff, 3)
    out['cnn_expected_stage_slope_7d'] = window_slope(h, 'cnn_expected_stage', cutoff, 7)
    out['cnn_p4_current'] = latest_nonnull(h, 'cnn_p4')
    out['cnn_p4_delta_3d'] = window_delta(h, 'cnn_p4', cutoff, 3)
    out['cnn_p4_delta_7d'] = window_delta(h, 'cnn_p4', cutoff, 7)
    out['cnn_entropy_current'] = latest_nonnull(h, 'cnn_entropy')
    out['cnn_entropy_delta_7d'] = window_delta(h, 'cnn_entropy', cutoff, 7)
    out['area_cm2_current'] = latest_nonnull(h, 'area_cm2')
    out['area_cm2_slope_3d'] = window_slope(h, 'area_cm2', cutoff, 3)
    out['area_cm2_slope_7d'] = window_slope(h, 'area_cm2', cutoff, 7)
    out['area_cm2_auc_7d'] = window_auc(h, 'area_cm2', cutoff, 7)
    out['mask_area_px_current'] = latest_nonnull(h, 'mask_area_px')
    out['mask_area_px_slope_7d'] = window_slope(h, 'mask_area_px', cutoff, 7)
    out['perimeter_slope_7d'] = window_slope(h, 'perimeter_px', cutoff, 7)
    out['solidity_delta_7d'] = window_delta(h, 'solidity', cutoff, 7)
    out['circularity_delta_7d'] = window_delta(h, 'circularity', cutoff, 7)
    out['volume_proxy_current'] = latest_nonnull(h, 'volume_proxy')
    out['volume_proxy_slope_7d'] = window_slope(h, 'volume_proxy', cutoff, 7)
    return out

def stage_transition_features(history: pd.DataFrame, current: pd.Series) -> dict[str, object]:
    cutoff = parse_date(current.get('survey_date'))
    h = history.sort_values(['history_date', 'crop_filename'], na_position='last').copy()
    out = beforeday_features(h, current)
    stage = pd.to_numeric(h.get('cnn_pred_stage'), errors='coerce') if 'cnn_pred_stage' in h else pd.Series(index=h.index, dtype=float)
    h['_stage_int'] = stage.round().astype('Int64') if len(stage) else pd.Series(dtype='Int64')
    valid_stage = h[h['_stage_int'].notna()].copy()
    if len(valid_stage):
        changes = valid_stage['_stage_int'].ne(valid_stage['_stage_int'].shift()).fillna(True)
        change_rows = valid_stage[changes]
        last_change_row = change_rows.iloc[-1]
    else:
        last_change_row = pd.Series(dtype=object)
    for metric in ['area_cm2', 'perimeter_px', 'circularity', 'aspect_ratio', 'angular_cv']:
        out[f'{metric}_current'] = latest_nonnull(h, metric)
        out[f'{metric}_history_mean'] = mean_nonnull(h, metric)
        out[f'{metric}_history_last'] = latest_nonnull(h, metric)
        for stage_id in [0, 1, 2, 3, 4]:
            out[f'{metric}_stage{stage_id}_mean'] = mean_nonnull(h[h['_stage_int'].eq(stage_id)], metric)
        if len(valid_stage) and metric in h.columns:
            base_value = safe_num(last_change_row.get(metric))
            current_value = latest_nonnull(h, metric)
            days_since = date_days(cutoff, parse_date(last_change_row.get('history_date')))
            out[f'{metric}_delta_since_last_stage_change'] = current_value - base_value
            out[f'{metric}_slope_since_last_stage_change'] = (current_value - base_value) / days_since if days_since and days_since > 0 else np.nan
        else:
            out[f'{metric}_delta_since_last_stage_change'] = np.nan
            out[f'{metric}_slope_since_last_stage_change'] = np.nan
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            va = out.get(f'{metric}_stage{a}_mean', np.nan)
            vb = out.get(f'{metric}_stage{b}_mean', np.nan)
            out[f'{metric}_delta_stage{a}_to{b}'] = vb - va if pd.notna(va) and pd.notna(vb) else np.nan
            da = h.loc[h['_stage_int'].eq(a), 'history_date'].dropna().min()
            db = h.loc[h['_stage_int'].eq(b), 'history_date'].dropna().min()
            days = date_days(db, da) if pd.notna(da) and pd.notna(db) else np.nan
            out[f'{metric}_days_stage{a}_to{b}'] = days
            out[f'{metric}_rate_stage{a}_to{b}'] = out[f'{metric}_delta_stage{a}_to{b}'] / days if pd.notna(days) and days > 0 else np.nan
    for col in ['embedding_cosine_change_since_last_stage_change', 'embedding_l2_change_since_last_stage_change', 'embedding_cosine_change_stage0_to1', 'embedding_cosine_change_stage1_to2', 'embedding_cosine_change_stage2_to3', 'embedding_cosine_change_stage3_to4', 'embedding_l2_change_stage0_to1', 'embedding_l2_change_stage1_to2', 'embedding_l2_change_stage2_to3', 'embedding_l2_change_stage3_to4']:
        out[col] = np.nan
    (latest_row, _latest_vec, emb_cols) = _latest_embedding(h)
    if len(valid_stage) and emb_cols and (latest_row is not None):
        last_change_date = parse_date(last_change_row.get('history_date')) if len(last_change_row) else pd.NaT
        if pd.notna(last_change_date):
            after_change = h[h['history_date'].ge(last_change_date)].dropna(subset=emb_cols, how='any')
            if not after_change.empty:
                (cos, l2) = _embedding_between_rows(after_change.iloc[0], latest_row, emb_cols)
                out['embedding_cosine_change_since_last_stage_change'] = cos
                out['embedding_l2_change_since_last_stage_change'] = l2
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            rows_a = h[h['_stage_int'].eq(a)].dropna(subset=emb_cols, how='any')
            rows_b = h[h['_stage_int'].eq(b)].dropna(subset=emb_cols, how='any')
            if not rows_a.empty and (not rows_b.empty):
                (cos, l2) = _embedding_between_rows(rows_a.iloc[-1], rows_b.iloc[0], emb_cols)
                out[f'embedding_cosine_change_stage{a}_to{b}'] = cos
                out[f'embedding_l2_change_stage{a}_to{b}'] = l2
    return out
ENV_COLS = list(ENVIRONMENT_FEATURES)
RAW = ['date', 'temp_mean', 'heat_hours_28_ratio', 'heat_degree_hours_28', 'vpd_hours_ratio_1_8', 'vpd_degree_hours_1_8', 'hot_dry_load_28_1_8', 'water_air_gap_day', 'water_air_gap_day_abs', 'ec_mean', 'minute_count', 'valid_stress_hour_count']

def derive(raw):
    """Calculate capped GDD and preserve observed daily environmental values.

    Cumulative GDD begins on the first supplied sensor date. Missing days are
    not interpolated; a missing observed value propagates through the cumulative
    series. No EC, VPD, temperature or water-gap penalty is applied.
    """
    d = raw[RAW].copy().sort_values('date').reset_index(drop=True)
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    d['capped_gdd_day'] = (d['temp_mean'].clip(upper=23) - 5).clip(lower=0)
    d['capped_gdd_cum'] = d['capped_gdd_day'].cumsum(skipna=False)
    return d

def join_current(frame, daily):
    """Join exactly D-1 and the completed calendar window [D-7,D).

    An absent previous day stays unavailable. The seven-day sums use observed
    values only, with NaN when no valid value exists; missing days are not zeroes.
    """
    rows = []
    for r in frame.itertuples():
        cutoff = pd.Timestamp(r.prediction_cutoff).normalize()
        past = daily.loc[daily.date.lt(cutoff)]
        exact = daily.loc[daily.date.eq(cutoff - pd.Timedelta(days=1))]
        values = {name: np.nan for name in ENV_COLS}
        if len(exact):
            values.update({name: exact.iloc[0][name] for name in ENV_COLS if name in daily})
        window = past.loc[past.date.ge(cutoff - pd.Timedelta(days=7))]
        for (target, source) in [('past_7d_capped_gdd_sum', 'capped_gdd_day'), ('past_7d_hot_dry_load_sum', 'hot_dry_load_28_1_8')]:
            values[target] = window[source].sum(min_count=1) if len(window) else np.nan
        values.update(manifest_row_id=r.manifest_row_id, prediction_cutoff=r.prediction_cutoff, env_daily_source_date=exact.date.iloc[0] if len(exact) else pd.NaT, env_last_observed_date=past.date.max() if len(past) else pd.NaT, env_past_7d_observed_days=len(window), env_missing_reason='' if len(exact) else 'NO_COMPLETED_PREVIOUS_DAY_SENSOR', env_daily_join_source='previous_completed_day' if len(exact) else 'missing', env_lag_join_source='calendar_[D-7,D)' if len(window) else 'missing')
        rows.append(values)
    return pd.DataFrame(rows)

def load_models():
    """Load saved components only after validating the frozen legacy16 input contract."""
    from catboost import CatBoostRegressor
    loaded = {}
    sources = {}
    for (mode, (folder, metadata)) in component_contracts(MODEL_ROOT).items():
        ensemble = []
        for record in metadata['seeds']:
            path = folder / f"seed_{record['seed']}.cbm"
            digest = sha(path)
            require(digest == record['model_sha256'], f'Model checksum mismatch: {path}')
            estimator = CatBoostRegressor()
            estimator.load_model(str(path))
            require(estimator.feature_names_ == metadata['features'], f'Feature order mismatch: {path}')
            ensemble.append(estimator)
            sources[str(path)] = digest
        loaded[mode] = (metadata['features'], ensemble)
        sources[str(folder / 'ensemble.json')] = sha(folder / 'ensemble.json')
    for path in [PCA_FILE, A_MODEL / 'selected_model_bundle.json', CYCLE.BACKBONE_FILE]:
        require(path.is_file(), f'Required model artifact is missing: {path}')
        sources[str(path)] = sha(path)
    return (loaded, sources)

def predict(frame, models):
    out = {}
    base = frame['pred_weight_cnn'].to_numpy(dtype=float)
    for (mode, (names, ensemble)) in models.items():
        require(set(names) <= set(frame.columns), 'Validation condition failed in predict: set(names) <= set(frame.columns)')
        x = frame[names].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
        require(not np.isinf(x).any(), 'Validation condition failed in predict: not np.isinf(x).any()')
        values = [m.predict(x, thread_count=4) + (base if mode == 'residual' else 0) for m in ensemble]
        out[mode] = np.mean(values, axis=0)
    out['blend50'] = (out['direct_stacking'] + out['residual']) * 0.5
    require(all((np.isfinite(v).all() for v in out.values())), 'Validation condition failed in predict: all((np.isfinite(v).all() for v in out.values()))')
    return out

def choose_a_run(explicit):
    if explicit is not None:
        run = Path(explicit)
    else:
        candidates = [p for p in A_OUTPUT.glob('run_*') if (p / 'completion.json').is_file()]
        require(bool(candidates), 'Validation condition failed in choose_a_run: bool(candidates)')
        run = max(candidates, key=lambda p: p.name)
    done = read_json(run / 'completion.json')
    require(done['status'] in {'COMPLETE', 'COMPLETE_WITH_ROW_ERRORS'}, 'A-step is incomplete')
    require(done.get('prediction_role') == 'operational_ensemble_not_oof', 'Unsupported A-step prediction role')
    require(done.get('selected_model') == 'cnn_only', 'A-step model mismatch')
    require(done['model_bundle_sha256'] == sha(A_MODEL / 'selected_model_bundle.json'), 'A-step bundle mismatch')
    return (run, done)

def prepare_environment(path):
    raw = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
    require('date' in raw.columns, 'Missing environment date column')
    absent = sorted(set(RAW) - set(raw.columns))
    for name in absent:
        raw[name] = np.nan
    raw['date'] = pd.to_datetime(raw.date, errors='raise').dt.normalize()
    require(not raw.date.isna().any() and (not raw.date.duplicated().any()), 'Missing or duplicate environment dates')
    raw = raw.sort_values('date').reset_index(drop=True)
    require(raw.date.between(pd.Timestamp(SEASON_START), pd.Timestamp(SEASON_END)).all(), 'Environment dates fall outside SEASON_START/SEASON_END; configure the sensor period explicitly')
    invalid = {}
    for c in RAW[1:]:
        original = raw[c]
        numeric = pd.to_numeric(original, errors='coerce')
        numeric = numeric.where(np.isfinite(numeric), np.nan)
        if c in {'heat_hours_28_ratio', 'vpd_hours_ratio_1_8'}:
            numeric = numeric.where(numeric.between(0, 1), np.nan)
        if c in {'heat_degree_hours_28', 'vpd_degree_hours_1_8',
                 'hot_dry_load_28_1_8', 'ec_mean', 'minute_count',
                 'valid_stress_hour_count', 'water_air_gap_day_abs'}:
            numeric = numeric.where(numeric.ge(0), np.nan)
        invalid[c] = int((original.notna() & numeric.isna()).sum())
        raw[c] = numeric
    raw.attrs['missing_value_audit'] = dict(absent_base_columns=absent,
                                          invalid_cells_to_nan=invalid)
    return (raw, derive(raw))

def ingest(db, run):
    db.execute('CREATE TABLE rows (idx INTEGER PRIMARY KEY, rid TEXT UNIQUE, identity TEXT, ts TEXT, payload TEXT, feat TEXT, error TEXT)')
    path = run / '01_A_step_predictions.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        needed = set(RAW_GEOMETRY + ['roi_area_px', 'mask_pixels', 'crop_source_box', 'record_id', 'season_id', 'bed_id', 'physical_slot', 'camera', 'capture_date', 'capture_time', 'crop_path', 'crop_sha256', 'slot_namespace_version', 'selected_camera', 'selected_record_id', 'embedding_row_index', 'a_step_status', 'a_step_resolved_path', 'pred_weight_g', 'prediction_role'])
        require(needed <= set(headers), f'A-step Missing required columns: {needed - set(headers)}')
        require(not set(EXTRA) & set(headers), 'Validation condition failed in ingest: not set(EXTRA) & set(headers)')
        n = 0
        for row in tqdm(reader, desc='1/3 A-step inventory', mininterval=5):
            require(int(row['embedding_row_index']) == n, 'A-step row index changed or is non-contiguous')
            require(row['camera'] in ['cam0', 'cam1'] and row['selected_camera'] == row['camera'] and (row['selected_record_id'] == row['record_id']), 'CAM01 selection contract mismatch')
            require(row['prediction_role'] == 'operational_ensemble_not_oof', 'A-step prediction role mismatch')
            require(row['season_id'] == SUPPORTED_SEASON and row['slot_namespace_version'] == 'physical_12col_v1', 'Season or slot namespace mismatch')
            require(re.fullmatch('bed\\d+', row['bed_id']) and re.fullmatch('[bt](?:[1-9]|1[0-2])', row['physical_slot']), 'Invalid plant identifier')
            date = re.sub('\\.0$', '', row['capture_date'])
            hour = re.sub('\\.0$', '', row['capture_time']).zfill(6)
            ts = pd.to_datetime(date + hour, format='%Y%m%d%H%M%S', errors='raise')
            require(pd.Timestamp(SEASON_START) <= ts < pd.Timestamp(SEASON_END) + pd.Timedelta(days=1), 'Capture date is outside the configured cycle')
            identity = '|'.join((row[k] for k in ['season_id', 'bed_id', 'physical_slot']))
            db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?)', (n, row['record_id'], identity, str(ts), json.dumps(row, ensure_ascii=False), None, '' if row['a_step_status'] == 'OK' else 'A_STEP_UNAVAILABLE: ' + row.get('a_step_error', '')))
            n += 1
            if n % ROW_CHUNK == 0:
                db.commit()
    require(n > 0, 'No A-step input rows')
    require(db.execute('SELECT count(*) FROM (SELECT identity,ts FROM rows GROUP BY identity,ts HAVING count(*)>1)').fetchone()[0] == 0, 'Duplicate plant/capture timestamp; rows are not averaged or dropped')
    db.execute('CREATE INDEX history_idx ON rows(identity,ts)')
    db.commit()
    db.execute('ALTER TABLE rows ADD COLUMN work_payload TEXT')
    return (headers, n)

def stage_probabilities(db, run, out, n):
    probs = np.lib.format.open_memmap(out / 'stage_probabilities.npy', mode='w+', dtype='float64', shape=(n, 5))
    probs[:] = 0
    invalid = np.zeros(n, bool)
    for fold in range(1, 6):
        with (run / f'fold_{fold}' / 'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                i = int(row['embedding_row_index'])
                require(0 <= i < n and i not in seen, 'Validation condition failed in stage_probabilities: 0 <= i < n and i not in seen')
                seen.add(i)
                require(db.execute('SELECT rid FROM rows WHERE idx=?', (i,)).fetchone()[0] == row['record_id'], "Validation condition failed in stage_probabilities: db.execute('SELECT rid FROM rows WHERE idx=?', (i,)).fetchone()[0] == row['record_id']")
                p = np.array([safe_num(row[f'stage_p{k}']) for k in range(5)])
                valid = np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and (abs(p.sum() - 1) < 0.0001)
                if not valid:
                    invalid[i] = True
                else:
                    probs[i] += p / 5
            for i in set(range(n)) - seen:
                invalid[i] = True
    for i in np.flatnonzero(invalid):
        db.execute("UPDATE rows SET error=CASE WHEN error='' THEN 'A_STEP_STAGE_UNAVAILABLE' ELSE error END WHERE idx=?", (int(i),))
    db.commit()
    probs.flush()
    return probs

def resolve_history_crop(row):
    """Apply only the explicitly configured input-root relocation."""
    resolver = getattr(CYCLE, 'resolve_crop_path', None)
    return resolver(row['crop_path']) if resolver else Path(row['crop_path'])

def recover_history_only(db, probs, out, args):
    candidates = []
    for (idx, payload) in db.execute("SELECT idx,payload FROM rows WHERE error!=''"):
        r = json.loads(payload)
        if r.get('feature_status') == 'OK' and 'FEATURE_NOT_ANALYSABLE' in r.get('a_step_error', ''):
            candidates.append((idx, payload))
    if not candidates:
        return 0
    import torch
    from torch import nn
    from torchvision.models import efficientnet_b0

    class WeightNet(nn.Module):

        def __init__(self, num):
            super().__init__()
            self.backbone = efficientnet_b0(weights=None)
            self.backbone.classifier = nn.Identity()
            self.image_embedding = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(inplace=True), nn.Dropout(0.2))
            self.image_head = nn.Linear(128, 1)
            self.numeric_net = nn.Sequential(nn.Linear(num, 32), nn.ReLU(inplace=True), nn.Dropout(0.2))
            self.fusion_head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

        def forward(self, x, n):
            emb = self.image_embedding(self.backbone(x))
            return self.fusion_head(torch.cat([emb, self.numeric_net(n)], 1)).squeeze(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    bundle = read_json(A_MODEL / 'selected_model_bundle.json')
    core = bundle['core_numeric_features']
    accum = np.zeros((len(candidates), 6), np.float64)
    valid = np.ones(len(candidates), bool)
    failures = {}

    def decode(item):
        (j, (idx, payload)) = item
        r = json.loads(payload)
        r['a_step_resolved_path'] = str(resolve_history_crop(r))
        r['pred_weight_g'] = '0'
        checked = decode_image((idx, json.dumps(r)))
        if checked[3] is not None:
            return (j, None, None, None, checked[3])
        with Image.open(r['a_step_resolved_path']) as im:
            a = np.asarray(im.convert('RGB').resize((384, 384), Image.Resampling.BILINEAR), np.float32) / 255
        a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return (j, np.ascontiguousarray(a.transpose(2, 0, 1)), checked[1], r, None)
    for info in bundle['folds']:
        fold = info['fold']
        folder = A_MODEL / f'fold_{fold}'
        for artifact in info['files']:
            require(sha(A_MODEL / artifact['path']) == artifact['sha256'], "Validation condition failed in recover_history_only: sha(A_MODEL / artifact['path']) == artifact['sha256']")
        stage = efficientnet_b0(weights=None)
        stage.classifier[1] = nn.Linear(1280, 5)
        stage.load_state_dict(torch.load(folder / 'stage.pt', map_location='cpu', weights_only=True)['model_state'])
        cp = torch.load(folder / 'weight.pt', map_location='cpu', weights_only=True)
        require(cp['numeric_dim'] == 33, "Validation condition failed in recover_history_only: cp['numeric_dim'] == 33")
        weight = WeightNet(33)
        weight.load_state_dict(cp['model_state'])
        del cp
        for m in [stage, weight]:
            m.to(device, memory_format=torch.channels_last).eval()
        proc = read_json(folder / 'numeric_preprocessor.json')
        (median, mean, std) = (np.asarray(proc[k], float) for k in ['median', 'mean', 'std'])
        require(all((a.shape == (33,) and np.isfinite(a).all() for a in [median, mean, std])) and (std > 0).all(), 'Invalid A-step numeric preprocessor')
        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            for start in tqdm(range(0, len(candidates), args.batch_size), desc=f'history-only A fold {fold}', mininterval=5):
                decoded = list(pool.map(decode, enumerate(candidates[start:start + args.batch_size], start)))
                good = [r for r in decoded if r[4] is None]
                for (j, _, _, _, error) in decoded:
                    if error:
                        valid[j] = False
                        failures[j] = error
                if not good:
                    continue
                a = torch.from_numpy(np.stack([r[1] for r in good])).to(device, memory_format=torch.channels_last)
                b = torch.from_numpy(np.stack([r[2] for r in good])).to(device, memory_format=torch.channels_last)
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
                    p = torch.softmax(stage(a).float(), 1).cpu().numpy()
                q = p.astype(np.float64)
                stagevalues = np.column_stack([q @ np.arange(5), q.max(1), -(q * np.log(np.clip(q, 1e-12, 1))).sum(1), q])
                numeric = np.column_stack([np.array([[safe_num(r[3].get(c)) for c in core] for r in good]), stagevalues])
                numeric = ((np.where(np.isfinite(numeric), numeric, median) - mean) / std).astype(np.float32)
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
                    weight_pred = weight(b, torch.from_numpy(numeric).to(device)).float().cpu().numpy()
                require(np.isfinite(p).all() and np.isfinite(weight_pred).all(), 'Validation condition failed in recover_history_only: np.isfinite(p).all() and np.isfinite(weight_pred).all()')
                for (k, item) in enumerate(good):
                    accum[item[0]] += np.r_[q[k], float(weight_pred[k])] / 5
        del stage, weight
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    audit = []
    for (j, (idx, payload)) in enumerate(candidates):
        row = json.loads(payload)
        if valid[j]:
            probs[idx] = accum[j, :5]
            row['a_step_resolved_path'] = str(resolve_history_crop(row))
            row['pred_weight_g'] = max(0, float(accum[j, 5]))
            db.execute("UPDATE rows SET work_payload=?,error='' WHERE idx=?", (json.dumps(row, ensure_ascii=False), idx))
        audit.append(dict(embedding_row_index=idx, record_id=row['record_id'], history_only_recovered=bool(valid[j]), error=failures.get(j, '')))
    pd.DataFrame(audit).to_csv(out / '04_history_only_recovery.csv', index=False, encoding='utf-8-sig')
    db.commit()
    probs.flush()
    return int(valid.sum())

def decode_image(item):
    (idx, payload) = item
    row = json.loads(payload)
    try:
        path = Path(row['a_step_resolved_path'])
        require(path.is_file(), f'Crop file missing: {path}')
        m = IMAGE_NAME.fullmatch(path.name)
        require(m is not None, 'Cannot parse crop filename')
        expected = (row['bed_id'], re.sub('\\.0$', '', row['capture_date']), re.sub('\\.0$', '', row['capture_time']).zfill(6), row['camera'], row['physical_slot'])
        require(m.groups() == expected, 'Crop filename/plant identity mismatch')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['crop_sha256'], 'Crop checksum changed')
        with Image.open(io.BytesIO(raw)) as im:
            (width, height) = im.size
            box = json.loads(row['crop_source_box'])
            require(len(box) == 4, 'Invalid crop box')
            require(float(box[2]) - float(box[0]) == width and float(box[3]) - float(box[1]) == height, 'Crop box/image dimensions differ')
            arr = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        feat = {c: safe_num(row[c]) for c in RAW_GEOMETRY}
        feat['roi_area_px'] = safe_num(row['roi_area_px'])
        feat.update(crop_aspect_ratio=width / height, bbox_aspect_ratio=width / height, mask_area_px=safe_num(row['roi_area_px']), mask_fill_ratio=safe_num(row['roi_area_px']) / (width * height), seg_area=safe_num(row['mask_pixels']), pred_weight_cnn=safe_num(row['pred_weight_g']))
        require(np.isfinite(feat['pred_weight_cnn']), 'Non-finite A-step weight')
        require(not any((np.isinf(v) for v in feat.values())), 'Infinite geometry feature')
        return (idx, np.ascontiguousarray(arr.transpose(2, 0, 1)), feat, None)
    except Exception as exc:
        return (idx, None, None, f'{type(exc).__name__}: {exc}')

def infer_common_embedding(db, probs, out, n, args):
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    weights = EfficientNet_B0_Weights.DEFAULT
    ckpt = CYCLE.BACKBONE_FILE
    require(ckpt.is_file(), 'Fixed ImageNet B0 checkpoint is missing: ' + str(ckpt))
    require(sha(ckpt) == '7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934', 'Backbone checksum mismatch')
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.classifier = torch.nn.Identity()
    model = model.eval().to(device, memory_format=torch.channels_last)
    with np.load(PCA_FILE, allow_pickle=False) as z:
        mean = z['mean'].astype(np.float64)
        components = z['components'].astype(np.float64)
    require(mean.shape == (1280,) and components.shape == (128, 1280), 'PCA dimensions differ')
    emb = np.lib.format.open_memmap(out / 'common_embedding_128.npy', mode='w+', dtype='float32', shape=(n, 128))
    emb[:] = np.nan
    count = db.execute("SELECT count(*) FROM rows WHERE error='' ").fetchone()[0]
    cursor = db.execute("SELECT idx,COALESCE(work_payload,payload) FROM rows WHERE error='' ORDER BY idx")
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool, tqdm(total=count, desc='2/3 fixed B0 + frozen PCA', mininterval=5) as bar:
        while (chunk := cursor.fetchmany(args.batch_size)):
            decoded = list(pool.map(decode_image, chunk))
            good = [r for r in decoded if r[3] is None]
            if good:
                x = torch.from_numpy(np.stack([r[1] for r in good])).to(device, memory_format=torch.channels_last)
                with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == 'cuda'):
                    raw = model(x).float().cpu().numpy()
                reduced = ((raw.astype(np.float64) - mean) @ components.T).astype(np.float32)
                for ((idx, _, feat, _), vec) in zip(good, reduced):
                    require(np.isfinite(vec).all(), 'Non-finite PCA value')
                    p = probs[idx]
                    feat.update({f'cnn_p{k}': float(p[k]) for k in range(5)})
                    feat.update(cnn_pred_stage=int(p.argmax()), cnn_expected_stage=float(p @ np.arange(5)), cnn_max_prob=float(p.max()), cnn_entropy=float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()))
                    feat.update(zip(EMB_COLS, map(float, vec)))
                    emb[idx] = vec
                    db.execute('UPDATE rows SET feat=? WHERE idx=?', (json.dumps(feat, allow_nan=True), idx))
            for (idx, _, _, err) in decoded:
                if err is not None:
                    db.execute('UPDATE rows SET error=? WHERE idx=?', (err, idx))
            db.commit()
            bar.update(len(chunk))
    emb.flush()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return str(device)

def causal_history(history, cutoff):
    h = history.loc[pd.to_datetime(history.capture_ts) <= cutoff].copy()
    h['history_date'] = pd.to_datetime(h.capture_ts).dt.normalize()
    h = h.sort_values(['history_date', 'capture_ts', 'crop_filename'], kind='stable').drop_duplicates('history_date', keep='last')
    h['history_lead_days'] = (cutoff.normalize() - h.history_date).dt.days
    return (stage_transition_features(h, pd.Series({'survey_date': cutoff})), h)

def export_predictions(db, headers, models, daily, out, n):
    db.execute('CREATE TABLE results (idx INTEGER PRIMARY KEY, payload TEXT)')
    names = models['direct_stacking'][0]
    dates = [r[0] for r in db.execute('SELECT DISTINCT substr(ts,1,10) FROM rows')]
    dummy = pd.DataFrame({'manifest_row_id': range(len(dates)), 'prediction_cutoff': dates})
    env = align_model_environment(join_current(dummy, daily))
    envmap = {str(pd.Timestamp(r.prediction_cutoff).date()): r._asdict() for r in env.itertuples(index=False)}
    env.to_csv(out / '03_environment_by_capture_day.csv', index=False, encoding='utf-8-sig')
    identities = [r[0] for r in db.execute('SELECT DISTINCT identity FROM rows ORDER BY identity')]
    stats = {'rows': n, 'predicted': 0, 'unavailable': 0, 'negative_predictions': 0, 'missing_previous_day_environment': 0}
    with (out / '02_C_step_model_features.csv').open('w', encoding='utf-8-sig', newline='') as ff, tqdm(total=n, desc='3/3 causal history + CatBoost', mininterval=5) as bar:
        writer = csv.DictWriter(ff, fieldnames=['embedding_row_index', 'record_id'] + names)
        writer.writeheader()
        for identity in identities:
            rows = db.execute('SELECT idx,ts,payload,feat,error FROM rows WHERE identity=? ORDER BY ts,idx', (identity,)).fetchall()
            records = []
            for (idx, ts, payload, feat, error) in rows:
                if error or feat is None:
                    continue
                r = json.loads(payload)
                v = json.loads(feat)
                records.append(dict(v, capture_ts=pd.Timestamp(ts), crop_filename=Path(r['crop_path']).name, source_tier='raw_direct', volume_proxy=max(v['area_cm2'], 0) ** 1.5))
            history = pd.DataFrame(records)
            batch = []
            metadata = []

            def flush():
                if not batch:
                    return
                p = predict(pd.DataFrame(batch), models)
                for (j, (idx, meta)) in enumerate(metadata):
                    meta.update(c_step_stacking_g=float(p['direct_stacking'][j]), c_step_residual_g=float(p['residual'][j]), c_step_residual_correction_g=float(p['residual'][j] - batch[j]['pred_weight_cnn']), c_step_pred_weight_g=float(p['blend50'][j]), c_step_negative_prediction=bool(p['blend50'][j] < 0))
                    stats['negative_predictions'] += int(p['blend50'][j] < 0)
                    db.execute('INSERT INTO results VALUES (?,?)', (idx, json.dumps(meta, ensure_ascii=False)))
                db.commit()
                batch.clear()
                metadata.clear()
            for (idx, ts, payload, feat, error) in rows:
                source = json.loads(payload)
                meta = {c: '' for c in EXTRA}
                meta.update(c_step_prediction_role='operational_ensemble_not_oof', c_step_model=CYCLE.C_MODEL_TAG)
                if error or feat is None or source['a_step_status'] != 'OK':
                    meta.update(c_step_status='PREDICTION_UNAVAILABLE', c_step_error=error or ('A_STEP_UNAVAILABLE; usable as past observation only' if feat is not None else 'FEATURE_UNAVAILABLE'))
                    db.execute('INSERT INTO results VALUES (?,?)', (idx, json.dumps(meta, ensure_ascii=False)))
                    stats['unavailable'] += 1
                else:
                    cutoff = pd.Timestamp(ts)
                    v = json.loads(feat)
                    (hist, h) = causal_history(history, cutoff)
                    v.update(hist)
                    e = envmap[str(cutoff.date())]
                    v.update({c: e[c] for c in MODEL_ENVIRONMENT_FEATURES})
                    require(set(names) <= v.keys(), 'Validation condition failed in export_predictions: set(names) <= v.keys()')
                    require(pd.to_datetime(h.capture_ts).max() <= cutoff, 'Validation condition failed in export_predictions: pd.to_datetime(h.capture_ts).max() <= cutoff')
                    meta.update(c_step_status='OK', c_step_history_n_days=len(h), c_step_history_first_date=str(h.history_date.min().date()), c_step_history_last_timestamp=str(pd.to_datetime(h.capture_ts).max()), c_step_env_source_date='' if pd.isna(e['env_daily_source_date']) else str(pd.Timestamp(e['env_daily_source_date']).date()), c_step_env_7d_observed_days=int(e['env_past_7d_observed_days']), c_step_env_missing_reason=e['env_missing_reason'], c_step_missing_feature_count=sum((pd.isna(v[c]) for c in names)))
                    stats['missing_previous_day_environment'] += int(bool(e['env_missing_reason']))
                    writer.writerow(dict(embedding_row_index=idx, record_id=source['record_id'], **{c: v[c] for c in names}))
                    batch.append(v)
                    metadata.append((idx, meta))
                    stats['predicted'] += 1
                    if len(batch) >= ROW_CHUNK:
                        flush()
                bar.update(1)
            flush()
    db.commit()
    with (out / '01_C_step_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers + EXTRA)
        writer.writeheader()
        for (original, result) in db.execute('SELECT r.payload,p.payload FROM rows r JOIN results p ON r.idx=p.idx ORDER BY r.idx'):
            row = json.loads(original)
            row.update(json.loads(result))
            writer.writerow(row)
    require(db.execute('SELECT count(*) FROM results').fetchone()[0] == n, "Validation condition failed in export_predictions: db.execute('SELECT count(*) FROM results').fetchone()[0] == n")
    require(stats['predicted'] + stats['unavailable'] == n, "Validation condition failed in export_predictions: stats['predicted'] + stats['unavailable'] == n")
    return stats

def main():
    global A_MODEL, MODEL_ROOT, SUPPORTED_SEASON, SEASON_START, SEASON_END, MODEL_ROLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-run', type=Path, default=A_RUN)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--env-base', type=Path, default=ENV_BASE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--device', default=DEVICE, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--a-model', type=Path, default=A_MODEL)
    parser.add_argument('--model-root', type=Path, default=MODEL_ROOT)
    parser.add_argument('--season-id', default=SUPPORTED_SEASON)
    parser.add_argument('--season-start', default=SEASON_START)
    parser.add_argument('--season-end', default=SEASON_END)
    parser.add_argument('--model-role', default=MODEL_ROLE)
    parser.add_argument('--check', action='store_true', help='Validate inputs without inference or output files')
    parser.add_argument('--environment-only', action='store_true', help='Export the 12 environmental inputs; no models needed')
    args = parser.parse_args()
    require(args.batch_size > 0, 'batch-size must be positive')
    A_MODEL = args.a_model
    MODEL_ROOT = args.model_root
    SUPPORTED_SEASON = args.season_id
    SEASON_START = args.season_start
    SEASON_END = args.season_end
    MODEL_ROLE = args.model_role
    (raw, daily) = prepare_environment(args.env_base)
    if args.environment_only:
        dates = daily.date + pd.Timedelta(days=1)
        frame = pd.DataFrame({'manifest_row_id': range(len(dates)), 'prediction_cutoff': dates})
        result = join_current(frame, daily)
        out = args.output_dir / ('environment_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        if args.check:
            print(json.dumps({'status': 'READY', 'schema': 'manuscript12', 'rows': len(result), 'features': ENV_COLS}, indent=2))
            return
        out.mkdir(parents=True, exist_ok=False)
        result.to_csv(out / 'environment_features.csv', index=False, encoding='utf-8-sig')
        js(out / 'completion.json', dict(status='COMPLETE', schema='manuscript12', features=ENV_COLS, rows=len(result), source=str(args.env_base), source_sha256=sha(args.env_base), input_audit=raw.attrs.get('missing_value_audit', {}), model_inference=False))
        print('Environmental features:', out)
        return
    (models, hashes) = load_models()
    (run, done) = choose_a_run(args.a_run)
    if args.check:
        print(json.dumps({'status': 'READY', 'schema': 'manuscript12', 'a_run': str(run), 'models': {k: len(v[0]) for (k, v) in models.items()}, 'environment_days': len(daily)}, indent=2))
        return
    out = args.output_dir / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    hashes[str(args.env_base)] = sha(args.env_base)
    for path in [run / 'completion.json', run / '01_A_step_predictions.csv'] + [run / f'fold_{i}' / 'predictions.csv' for i in range(1, 6)]:
        hashes[str(path)] = sha(path)
    report = dict(status='RUNNING', script_sha256=sha(__file__), a_run=str(run), source_hashes=hashes, selected=CYCLE.C_MODEL_TAG, prediction_role='operational_ensemble_not_oof', model_scope=MODEL_ROLE, season_id=SUPPORTED_SEASON, environment_schema='manuscript12', environment_features=ENV_COLS, environment_input_audit=raw.attrs.get('missing_value_audit', {}), model_environment_adapter=environment_contract(), history_scope='Same physical plant; capture_ts<=cutoff; latest valid observation per day', environment_scope='Exactly D-1 and calendar [D-7,D); unmatched legacy inputs remain NaN; no interpolation', missing_policy='Unavailable rows and missing features remain unavailable', negative_policy='Raw model predictions retained without extra clipping')
    js(out / 'started.json', report)
    js(out / 'feature_contract.json', dict(features={k: v[0] for (k, v) in models.items()}, embedding='ImageNet EfficientNet-B0 plus frozen train-only PCA128', ensemble='Mean saved seeds per component; equal stacking/residual blend', cutoff='Actual capture timestamp; completed previous environment day', scope='Operational inference; no labels, training or OOF generation', environment_adapter=environment_contract()))
    try:
        db = sqlite3.connect(out / 'working.sqlite3')
        try:
            (headers, n) = ingest(db, run)
            require(n == done['counts']['top_rows'], 'A-step completion row count mismatch')
            probs = stage_probabilities(db, run, out, n)
            report['history_only_recovered'] = recover_history_only(db, probs, out, args)
            report['device'] = infer_common_embedding(db, probs, out, n, args)
            report['counts'] = export_predictions(db, headers, models, daily, out, n)
            report['a_step_input'] = done.get('input')
        finally:
            db.close()
        for (path, digest) in hashes.items():
            require(sha(path) == digest, f'Source changed during run: {path}')
        report['status'] = 'COMPLETE_WITH_ROW_ERRORS' if report.get('counts', {}).get('unavailable', 0) else 'COMPLETE'
        report['elapsed_seconds'] = round(time.time() - start, 2)
        js(out / 'completion.json', report)
        print('C-step complete:', out)
    except BaseException as exc:
        js(out / 'failure.json', dict(status='FAILED', error=str(exc), traceback=traceback.format_exc()))
        raise
if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*trapz.*')
    main()
