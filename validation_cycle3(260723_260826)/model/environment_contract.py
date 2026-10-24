"""Environment contract calculations used by the final cycle pipeline."""
import numpy as np
import pandas as pd
EC_PHASE1 = '2026-03-23'
EC_PHASE2 = '2026-04-14'
ENV_COLS = ['effective_gdd_cum', 'effective_gdd_day', 'gdd_loss_ratio', 'daily_env_penalty', 'past_7d_env_penalty_mean', 'heat_hours_28_ratio', 'heat_degree_hours_28', 'vpd_hours_ratio_1_8', 'vpd_degree_hours_1_8', 'hot_dry_load_28_1_8', 'past_7d_hot_dry_load_sum', 'ec_penalty', 'water_air_gap_day', 'water_air_gap_day_abs', 'is_ec_shock', 'past_7d_effective_gdd_sum', 'past_7d_gdd_loss_ratio_mean']
RAW = ['date', 'temp_mean', 'heat_hours_28_ratio', 'heat_degree_hours_28', 'vpd_hours_ratio_1_8', 'vpd_degree_hours_1_8', 'hot_dry_load_28_1_8', 'water_air_gap_day', 'water_air_gap_day_abs', 'ec_mean', 'minute_count', 'valid_stress_hour_count']

def derive(raw):
    d = raw[RAW].copy().sort_values('date').reset_index(drop=True)
    d['date'] = pd.to_datetime(d.date)
    low = np.select([d.date < pd.Timestamp(EC_PHASE1), d.date < pd.Timestamp(EC_PHASE2)], [0.8, 1.0], default=1.2)
    high = np.select([d.date < pd.Timestamp(EC_PHASE1), d.date < pd.Timestamp(EC_PHASE2)], [1.2, 1.5], default=1.8)
    d['ec_penalty'] = ((np.maximum(low - d.ec_mean, 0) + np.maximum(d.ec_mean - high, 0)) / (high - low)).clip(0, 1)
    d['is_ec_shock'] = np.nan
    for col in ['heat_degree_hours_28', 'vpd_degree_hours_1_8', 'hot_dry_load_28_1_8', 'water_air_gap_day_abs']:
        q = d[col].expanding(min_periods=1).quantile(0.95)
        d['p95_asof_' + col] = q
        n = d[col].div(q.where(q > 0)).clip(0, 1)
        n = n.mask(q.eq(0) & d[col].eq(0), 0)
        n = n.mask(q.eq(0) & d[col].gt(0), 1)
        d['norm_' + col] = n
    heat = 0.5 * d.heat_hours_28_ratio + 0.5 * d.norm_heat_degree_hours_28
    vpd = 0.5 * d.vpd_hours_ratio_1_8 + 0.5 * d.norm_vpd_degree_hours_1_8
    d['daily_env_penalty'] = (0.25 * heat + 0.25 * vpd + 0.25 * d.norm_hot_dry_load_28_1_8 + 0.15 * d.ec_penalty + 0.1 * d.norm_water_air_gap_day_abs).clip(0, 1)
    d['capped_gdd_day'] = (d.temp_mean.clip(upper=23) - 5).clip(lower=0)
    d['effective_gdd_day'] = d.capped_gdd_day * (1 - d.daily_env_penalty)
    d['effective_gdd_cum'] = d.effective_gdd_day.cumsum(skipna=False)
    cap = d.capped_gdd_day.cumsum(skipna=False)
    d['gdd_loss_ratio'] = (cap - d.effective_gdd_cum) / cap.where(cap > 0)
    return d

def join_current(f, d):
    rows = []
    for r in f.itertuples():
        cutoff = pd.Timestamp(r.prediction_cutoff).normalize()
        past = d[d.date < cutoff]
        exact = d[d.date.eq(cutoff - pd.Timedelta(days=1))]
        v = {c: np.nan for c in ENV_COLS}
        if len(exact):
            v.update({c: exact.iloc[0][c] for c in ENV_COLS if c in d})
        w = past[past.date >= cutoff - pd.Timedelta(days=7)]
        for (dest, src, agg) in [('past_7d_env_penalty_mean', 'daily_env_penalty', 'mean'), ('past_7d_hot_dry_load_sum', 'hot_dry_load_28_1_8', 'sum'), ('past_7d_effective_gdd_sum', 'effective_gdd_day', 'sum'), ('past_7d_gdd_loss_ratio_mean', 'gdd_loss_ratio', 'mean')]:
            v[dest] = w[src].mean() if len(w) and agg == 'mean' else w[src].sum(min_count=1) if len(w) else np.nan
        v.update(manifest_row_id=r.manifest_row_id, prediction_cutoff=r.prediction_cutoff, env_daily_source_date=exact.date.iloc[0] if len(exact) else pd.NaT, env_last_observed_date=past.date.max() if len(past) else pd.NaT, env_past_7d_observed_days=len(w), env_missing_reason='' if len(exact) else 'NO_COMPLETED_PREVIOUS_DAY_SENSOR', env_daily_join_source='causal_daily_base_previous_day' if len(exact) else 'missing', env_lag_join_source='calendar_[cutoff_day-7,cutoff_day)' if len(w) else 'missing')
        rows.append(v)
    return pd.DataFrame(rows)
