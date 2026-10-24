"""Compute provisional past-only area growth QC for 115 plant tracks."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
ROOT = CYCLE.PREPROCESS_OUTPUT / '6. 시각화와 QC'
INPUT = CYCLE.SOURCE115
PROTECTED89 = CYCLE.TARGET_FILE
OUTPUT = ROOT / '생육 QC'
REFERENCE = CYCLE.THIRD_ROOT / 'preprocessing/9. Assess_growth.py'
WINDOW = 7
MIN_DAYS = 3
MIN_SPAN = 2
GOOD_DAYS = 5
MAX_GAP = 3
THRESHOLD = 1.0
SLOWING = 0.5
METRICS = ['area_cm2', 'circularity', 'solidity', 'angular_cv', 'inner_ratio_50', 'brightness_mean', 'dark_ratio']
LABELS = {'GROWTH_OBSERVED': 'Area increase', 'GROWTH_SLOWING': 'Slower increase', 'PLATEAU_CANDIDATE': 'Candidate plateau', 'DECREASE_OBSERVED': 'Area decrease', 'INSUFFICIENT_DATA': 'Insufficient observations', 'SOURCE_REVIEW': 'Measurement review', 'NO_MEASUREMENT': 'No measurement'}
STATES = list(LABELS)
COLORS = ['#159A78', '#E7A33E', '#9872B7', '#D95F59', '#D4D9DF', '#F5DC96', '#F0F2F4']
FIGURES = ['01_growth_transition_top.png', '02_beddate_bedslot_consistency.png', '03_growth_qc_calendar.png', '04_size_and_appearance.png']
import argparse, hashlib, json, math, sys
from datetime import datetime
import numpy as np
import pandas as pd
import openpyxl

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if x is None or x is pd.NaT:
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, np.generic):
        x = x.item()
    if not isinstance(x, str) and pd.isna(x):
        return None
    return x

def dump(p, x):
    p.write_text(json.dumps(safe(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_sheet(w, name):
    it = w[name].iter_rows(values_only=True)
    cols = list(next(it))
    return pd.DataFrame(list(it), columns=cols)

def read_inputs():
    w = openpyxl.load_workbook(INPUT, read_only=True, data_only=True)
    current = read_sheet(w, '생육매칭')
    history = read_sheet(w, '과거Feature')
    w.close()
    require(len(current) == 115 and current.item_id.nunique() == 115, 'Invalid input or execution contract')
    require(current.groupby('view').size().to_dict() == {'front': 26, 'top': 89}, 'Invalid input or execution contract')
    require(len(history) == 1825 and (not history.duplicated(['item_id', 'capture_date']).any()), 'Invalid input or execution contract')
    for c in ['survey_date', 'image_date']:
        current[c] = pd.to_datetime(current[c])
    history['date'] = pd.to_datetime(history.capture_date, format='%Y%m%d')
    history['capture_timestamp'] = pd.to_datetime(history.capture_timestamp)
    history['target_cutoff'] = pd.to_datetime(history.target_cutoff)
    require((history.capture_timestamp <= history.target_cutoff).all(), 'Invalid input or execution contract')
    parent = current.set_index('item_id')
    for c in ['bed_id', 'physical_slot', 'survey_date', 'matching_bools', 'lead_days']:
        history[c] = history.item_id.map(parent[c])
    history['lettuce_key'] = '4작기|' + history.bed_id + '|' + history.physical_slot
    require(history.groupby('item_id').view.nunique().eq(1).all(), 'Invalid input or execution contract')
    require(history.loc[history.role.eq('CURRENT')].set_index('item_id').record_id.to_dict() == parent.record_id.to_dict(), 'Invalid input or execution contract')
    return (current, history)

def window_result(w):
    if len(w) < MIN_DAYS:
        return None
    span = (w[-1]['date'] - w[0]['date']).days
    if span < MIN_SPAN:
        return None
    (first, last) = (w[0]['area_cm2'], w[-1]['area_cm2'])
    return dict(area_growth_pct_day=100 * math.expm1(math.log(last / first) / span), area_change_pct=100 * (last / first - 1), growth_window_days=span, valid_obs_n=len(w), window_start_date=w[0]['date'], window_end_date=w[-1]['date'], window_start_area_cm2=first, window_end_area_cm2=last, window_record_ids=json.dumps([r['record_id'] for r in w]))

def temporal(h):
    rows = []
    for (item, g) in h.groupby('item_id', sort=False):
        history = []
        previous = None
        signature = None
        segment = 0
        for source in g.sort_values('date').to_dict('records'):
            r = dict(source)
            sig = (r['view'], r['camera'], r['source_camera'], r['native_width'], r['native_height'])
            reasons = []
            if previous:
                if (r['date'] - previous['date']).days > MAX_GAP:
                    reasons.append('GAP_GT_3_DAYS')
                if sig != signature:
                    reasons.append('CAMERA_SOURCE_OR_RESOLUTION_CHANGE')
            if reasons:
                segment += 1
                history = []
            r.update(temporal_segment_id=f'{item}|{segment}', segment_start_reason=';'.join(reasons) if reasons else 'FIRST_OBSERVATION' if previous is None else '', growth_basis='TOP_AREA_ONLY' if r['view'] == 'top' else 'FRONT_AREA_EXPLORATORY', scale_verified=False, identity_confirmed=False, measurement_valid=False, area_growth_pct_day=None, previous_area_growth_pct_day=None, area_change_pct=None, growth_window_days=None, valid_obs_n=0, window_start_date=None, window_end_date=None, window_start_area_cm2=None, window_end_area_cm2=None, window_record_ids='[]', growth_qc='INSUFFICIENT_DATA', timeseries_qc='INSUFFICIENT', step_area_ratio=None, step_area_change_pct=None, step_scale_ratio=None, step_pixel_area_ratio=None, abrupt_change_review=False, QC_human=None, QC_note=None)
            valid = math.isfinite(r['area_cm2']) and r['area_cm2'] > 0 and (not r['qc_hold_physical_size_v2_2'])
            r['measurement_valid'] = valid
            r['source_qc'] = 'USABLE_ESTIMATED_SCALE' if valid else 'HOLD_PHYSICAL_SCALE' if r['qc_hold_physical_size_v2_2'] else 'NO_MEASUREMENT'
            if previous and (not reasons) and previous['measurement_valid'] and valid:
                r['step_area_ratio'] = r['area_cm2'] / previous['area_cm2']
                r['step_area_change_pct'] = 100 * (r['step_area_ratio'] - 1)
                r['step_scale_ratio'] = r['mm_per_px'] / previous['mm_per_px']
                r['step_pixel_area_ratio'] = r['roi_area_px'] / previous['roi_area_px'] if previous['roi_area_px'] > 0 else None
                r['abrupt_change_review'] = r['step_area_ratio'] < 0.5 or r['step_area_ratio'] > 2
            if valid:
                history.append(r)
                history = [x for x in history if (r['date'] - x['date']).days < 2 * WINDOW]
                w = [x for x in history if (r['date'] - x['date']).days < WINDOW]
                r['valid_obs_n'] = len(w)
                result = window_result(w)
                if result:
                    r.update(result)
                    r['timeseries_qc'] = 'GOOD' if len(w) >= GOOD_DAYS else 'USABLE'
                    prev = window_result([x for x in history if WINDOW <= (r['date'] - x['date']).days < 2 * WINDOW])
                    rate = r['area_growth_pct_day']
                    old = prev['area_growth_pct_day'] if prev else None
                    r['previous_area_growth_pct_day'] = old
                    if rate < -THRESHOLD:
                        r['growth_qc'] = 'DECREASE_OBSERVED'
                    elif abs(rate) <= THRESHOLD:
                        if len(w) >= GOOD_DAYS:
                            r['growth_qc'] = 'PLATEAU_CANDIDATE'
                    elif old is not None and old > THRESHOLD and (rate < old * SLOWING):
                        r['growth_qc'] = 'GROWTH_SLOWING'
                    else:
                        r['growth_qc'] = 'GROWTH_OBSERVED'
            else:
                r['growth_qc'] = 'SOURCE_REVIEW' if r['qc_hold_physical_size_v2_2'] else 'NO_MEASUREMENT'
            r['growth_qc_ko'] = LABELS[r['growth_qc']]
            r['review_reason'] = ';'.join(filter(None, [r['source_qc'] if not valid else '', r['segment_start_reason'] if reasons else '', 'AREA_STEP_LT_HALF_OR_GT_DOUBLE' if r['abrupt_change_review'] else '']))
            r['review_required'] = bool(r['review_reason'])
            r['automatic_exclusion'] = False
            r['final_message'] = r['growth_qc_ko'] + ' · ' + ('top' if r['view'] == 'top' else 'front exploratory') + ' | unverified estimated scale'
            r['threshold_status'] = 'PROVISIONAL_AREA_RULES_NOT_BIOLOGICALLY_VALIDATED'
            rows.append(r)
            previous = r
            signature = sig
    return pd.DataFrame(rows).sort_values(['item_id', 'date']).reset_index(drop=True)

def verify_temporal(d):
    m = d.loc[d.area_growth_pct_day.notna()]
    expected = 100 * ((m.window_end_area_cm2 / m.window_start_area_cm2) ** (1 / m.growth_window_days) - 1)
    require(np.allclose(expected, m.area_growth_pct_day, rtol=1e-10, atol=1e-09), 'Invalid input or execution contract')
    require(m.growth_window_days.between(2, 6).all(), 'Invalid input or execution contract')
    lookup = d.set_index('record_id')
    for r in m.itertuples():
        w = lookup.loc[json.loads(r.window_record_ids)]
        require(w.temporal_segment_id.eq(r.temporal_segment_id).all(), 'Invalid input or execution contract')
        require(w.measurement_valid.all() and (w.date <= r.date).all() and ((r.date - w.date).dt.days < 7).all(), 'Invalid input or execution contract')
    require(d.loc[~d.measurement_valid, 'area_growth_pct_day'].isna().all(), 'Invalid input or execution contract')
    cols = ['record_id', 'growth_qc', 'area_growth_pct_day', 'valid_obs_n', 'temporal_segment_id']
    checks = 0
    for (item, g) in d.groupby('item_id'):
        prefix = g.iloc[:max(1, len(g) // 2)].copy()
        test = temporal(prefix)
        pd.testing.assert_frame_equal(test[cols].reset_index(drop=True), prefix[cols].reset_index(drop=True), check_dtype=False)
        checks += 1
    return dict(formula_rows=len(m), past_only_prefix_tracks=checks, segment_window_checks='PASS')

def table(name, frame):
    dates = [c for c in frame if pd.api.types.is_datetime64_any_dtype(frame[c])]
    rows = []
    for r in frame.itertuples(index=False, name=None):
        rows.append([(v - pd.Timestamp('1899-12-30')).total_seconds() / 86400 if isinstance(v, pd.Timestamp) and pd.notna(v) and (c in dates) else safe(v) for (c, v) in zip(frame.columns, r)])
    return dict(name=name, columns=list(frame.columns), date_columns=dates, rows=rows)

def prepare():
    require(not (OUTPUT / 'growth_qc.xlsx').exists(), 'Invalid input or execution contract')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    work = OUTPUT / '_근거'
    work.mkdir(exist_ok=True)
    protected = {str(p): sha(p) for p in [INPUT, PROTECTED89]}
    (current, h) = read_inputs()
    d = temporal(h)
    checks = verify_temporal(d)
    fields = ['item_id', 'date', 'view', 'bed_id', 'physical_slot', 'record_id', 'role', 'growth_basis'] + METRICS + ['area_growth_pct_day', 'previous_area_growth_pct_day', 'area_change_pct', 'growth_window_days', 'valid_obs_n', 'growth_qc', 'growth_qc_ko', 'timeseries_qc', 'source_qc', 'review_required', 'review_reason', 'QC_human', 'QC_note', 'final_message']
    op = d[fields].copy()
    full = d[fields + [c for c in d if c not in fields]].copy()
    latest = d.loc[d.role.eq('CURRENT')].set_index('item_id')
    added = ['growth_qc', 'growth_qc_ko', 'area_growth_pct_day', 'previous_area_growth_pct_day', 'valid_obs_n', 'growth_window_days', 'timeseries_qc', 'review_required', 'review_reason', 'QC_human', 'QC_note', 'growth_basis', 'temporal_segment_id', 'segment_start_reason']
    summary = current[['item_id', 'Stage_human', 'survey_date', 'image_date', 'bed_id', 'physical_slot', 'view', 'fresh_weight_g', 'matching_bools', 'lead_days', 'record_id', 'area_cm2', 'area_cm2_original', 'collected_crop_path', 'collected_bed_overlay_path']].copy()
    for c in added:
        summary[c] = summary.item_id.map(latest[c])
    summary['history_review_rows'] = summary.item_id.map(d.groupby('item_id').review_required.sum())
    summary['history_abrupt_steps'] = summary.item_id.map(d.groupby('item_id').abrupt_change_review.sum())
    summary['model_route'] = summary.item_id.map(current.set_index('item_id').model_route)
    params = pd.DataFrame([['window', 'D-6 through D', 'past-only'], ['scale', 'unverified', 'upstream review retained']], columns=['parameter', 'value', 'note'])
    report = dict(status='NUMERIC_GROWTH_QC_COMPUTED', targets=115, observations=len(d), source_hashes=protected, verification=checks, model_inference=False, figures_generated=False)
    spec = dict(tables=[table('현재115_QC', summary), table('operational_output', op), table('all_required_columns', full), table('기준과읽기', params)])
    dump(work / 'workbook_spec.json', spec)
    d.to_csv(OUTPUT / 'growth_qc.csv', index=False, encoding='utf-8-sig')
    dump(work / 'calculation.json', report)
    print(json.dumps(safe(report), ensure_ascii=False, indent=2))

def main():
    prepare()
    spec = json.loads((OUTPUT / '_근거/workbook_spec.json').read_text(encoding='utf-8'))
    from runtime import write_workbooks
    write_workbooks([dict(path=str(OUTPUT / 'growth_qc.xlsx'), sheets=spec['tables'])])
    report = json.loads((OUTPUT / '_근거/calculation.json').read_text(encoding='utf-8'))
    assert all((sha(p) == h for (p, h) in report['source_hashes'].items()))
    report.update(status='GROWTH_QC_COMPLETE_WITH_REVIEW', figures_generated=False)
    dump(OUTPUT / 'growth_qc_completion.json', report)
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
