"""Run frozen C components using explicit environment and causal plant histories."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
HERE = Path(__file__).resolve().parent
A_RUN = CYCLE.A_RUN
PREP = CYCLE.PREPARED_RUN
ENV_RUN = CYCLE.ENVIRONMENT_RUN
A_EVAL = CYCLE.A_EVALUATION_RUN
OUTPUT_ROOT = CYCLE.C_ROOT
BATCH_SIZE = 16
import argparse
from contextlib import closing
import importlib.util
import json
import math
import shutil
import sqlite3
import sys
import time
import traceback
from types import SimpleNamespace
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spec = resolved_spec('s4_input_c', HERE / '2. Predict_stage.py')
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)

def module():
    cm = s4.load('s4_frozen_c', s4.REFERENCE / 'model/3. Fuse_growth_history.py')
    cm.IMAGE_NAME = s4.IMAGE_NAME
    cm.SUPPORTED_SEASON = '4작기'
    cm.SEASON_START = '2026-07-23'
    cm.SEASON_END = '2026-08-25'
    env = pd.read_csv(ENV_RUN / '05_Cstep_environment_asof.csv')
    assert len(env) == 34 and (not env.prediction_cutoff.duplicated().any())
    env['prediction_cutoff'] = pd.to_datetime(env.prediction_cutoff)
    env['env_daily_source_date'] = pd.to_datetime(env.env_daily_source_date)
    lookup = env.set_index('prediction_cutoff', drop=False)

    def explicit_join(frame, unused):
        rows = []
        for r in frame.to_dict('records'):
            day = pd.Timestamp(r['prediction_cutoff']).normalize()
            assert day in lookup.index
            e = lookup.loc[day].to_dict()
            e['manifest_row_id'] = r['manifest_row_id']
            e['env_missing_reason'] = '' if pd.isna(e['env_missing_reason']) else e['env_missing_reason']
            rows.append(e)
        return pd.DataFrame(rows)
    cm.join_current = explicit_join
    return (cm, env)

def protected_sources(cm):
    (models, hashes) = cm.load_models()
    for p in [s4.TARGET_FILE, s4.SOURCE115, Path(__file__), s4.REFERENCE / 'model/3. Fuse_growth_history.py', A_RUN / 'completion.json', A_RUN / '01_A_step_predictions.csv', A_RUN / 'fold_predictions.npy', PREP / 'completion.json', A_EVAL / '01_current89_predictions_and_truth.csv']:
        hashes[str(p)] = s4.sha(p)
    for folder in [ENV_RUN]:
        for p in folder.iterdir():
            if p.is_file():
                hashes[str(p)] = s4.sha(p)
    for i in range(1, 6):
        for name in ['predictions.csv', 'embedding.npy']:
            p = A_RUN / f'fold_{i}' / name
            hashes[str(p)] = s4.sha(p)
    bundle = s4.read_json(s4.A_MODEL / 'selected_model_bundle.json')
    for fold in bundle['folds']:
        for rec in fold['files']:
            p = s4.A_MODEL / rec['path']
            assert s4.sha(p) == rec['sha256']
            hashes[str(p)] = rec['sha256']
    return (models, hashes)

def infer():
    (cm, env) = module()
    (models, hashes) = protected_sources(cm)
    (_, adone) = cm.choose_a_run(A_RUN)
    assert adone['counts']['top_rows'] == 1313
    s4.validate_prepared(PREP)
    out = OUTPUT_ROOT / ('run_' + s4.stamp())
    out.mkdir(parents=True, exist_ok=False)
    t = time.time()
    report = dict(status='RUNNING', a_run=str(A_RUN), prepared=str(PREP), environment_run=str(ENV_RUN), source_hashes=hashes, model_training=False, selected='454columns_blend50', environment_policy='Explicit frozen17 asof; no EC calendar re-derivation, pH not added', history_policy='same season/bed/physical slot, latest actual observation per day <= capture timestamp', current_targets=89, fixed_evaluation_n=85)
    s4.js(out / 'started.json', report)
    shutil.copy2(__file__, out / '실행코드_보존.py')
    s4.js(out / 'feature_contract.json', dict(features={k: v[0] for (k, v) in models.items()}, environment=cm.ENV_COLS, common_embedding='fixed ImageNet B0 256 + frozen train-only PCA128', stage_probabilities='A fivefold mean; history-only recovery from same frozen A', output='three seeds per component; blend50=(direct_stacking+residual)/2; no added clipping', current_unavailable='all four A failures stay unavailable even if useful as past observations'))
    try:
        with closing(sqlite3.connect(out / 'working.sqlite3')) as db:
            (headers, n) = cm.ingest(db, A_RUN)
            assert n == 1313
            probs = cm.stage_probabilities(db, A_RUN, out, n)
            recovered = cm.recover_history_only(db, probs, out, SimpleNamespace(device=__import__('runtime').torch_device(), batch_size=BATCH_SIZE))
            device = cm.infer_common_embedding(db, probs, out, n, SimpleNamespace(device=__import__('runtime').torch_device(), batch_size=BATCH_SIZE))
            stats = cm.export_predictions(db, headers, models, env, out, n)
        for (p, digest) in hashes.items():
            assert s4.sha(p) == digest, 'Protected source changed: ' + p
        report.update(status='COMPLETE_WITH_ROW_ERRORS' if stats['unavailable'] else 'COMPLETE', counts=stats, history_only_recovered=recovered, device=device, elapsed_seconds=round(time.time() - t, 2))
        report['output_hashes'] = {str(p): s4.sha(p) for p in out.iterdir() if p.is_file() and p.name != 'started.json'}
        s4.js(out / 'completion.json', report)
        print('C_RUN', out, flush=True)
        return out
    except BaseException as exc:
        s4.js(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise

def number_metrics(y, p):
    e = p - y
    return dict(n=len(y), MAE_g=float(np.mean(abs(e))), RMSE_g=float(np.sqrt(np.mean(e ** 2))), bias_g=float(np.mean(e)), R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), over_n=int((e > 0).sum()), under_n=int((e < 0).sum()), negative_prediction_n=int((p < 0).sum()))

def evaluate(out):
    (cm, env) = module()
    (models, _) = cm.load_models()
    done = s4.read_json(out / 'completion.json')
    for (p, h) in done['source_hashes'].items():
        assert s4.sha(p) == h, 'Protected file changed: ' + p
    a = s4.read_csv(A_RUN / '01_A_step_predictions.csv')
    c = s4.read_csv(out / '01_C_step_predictions.csv')
    assert len(c) == len(a) == 1313
    for (ar, cr) in zip(a, c):
        assert all((cr[k] == v for (k, v) in ar.items())), 'A source values changed'
    truth = pd.read_csv(A_EVAL / '01_current89_predictions_and_truth.csv')
    assert len(truth) == 89
    assert truth.record_id.tolist() == [r['record_id'] for r in c[:89]]
    current = pd.DataFrame(c[:89])
    result = truth.copy()
    for name in cm.EXTRA:
        result[name] = current[name].replace('', np.nan).to_numpy()
    for name in ['c_step_pred_weight_g', 'c_step_stacking_g', 'c_step_residual_g']:
        result[name] = pd.to_numeric(result[name])
    good = result.pred_weight_g.notna()
    assert good.sum() == 85 and np.array_equal(good, result.c_step_status == 'OK')
    assert result.loc[~good, 'c_step_pred_weight_g'].isna().all()
    for (label, col) in [('A', 'pred_weight_g'), ('C', 'c_step_pred_weight_g')]:
        result[label + '_error_g'] = result[col] - result.fresh_weight_g
        result[label + '_absolute_error_g'] = abs(result[label + '_error_g'])
    result['absolute_error_reduction_g'] = result.A_absolute_error_g - result.C_absolute_error_g
    result['prediction_change_C_minus_A_g'] = result.c_step_pred_weight_g - result.pred_weight_g
    result['overestimate_reduction_g'] = result.A_error_g.clip(lower=0) - result.C_error_g.clip(lower=0)
    metrics = []
    groups = [('ALL85', good)] + [(str(d), good & (result.survey_date == d)) for d in sorted(result.survey_date.unique())]
    cols = {'A_baseline': 'pred_weight_g', 'C_stacking': 'c_step_stacking_g', 'C_residual': 'c_step_residual_g', 'C_blend50': 'c_step_pred_weight_g'}
    for (group, mask) in groups:
        y = result.loc[mask, 'fresh_weight_g'].to_numpy(float)
        for (name, col) in cols.items():
            metrics.append(dict(group=group, model=name, **number_metrics(y, result.loc[mask, col].to_numpy(float))))
    yy = result.loc[good, 'fresh_weight_g'].to_numpy(float)
    aa = result.loc[good, 'pred_weight_g'].to_numpy(float)
    cc = result.loc[good, 'c_step_pred_weight_g'].to_numpy(float)
    paired = dict(n=85, absolute_error_improved_n=int((abs(cc - yy) < abs(aa - yy)).sum()), absolute_error_worsened_n=int((abs(cc - yy) > abs(aa - yy)).sum()), absolute_error_tied_n=int((abs(cc - yy) == abs(aa - yy)).sum()), A_overestimated_n=int((aa > yy).sum()), C_overestimated_n=int((cc > yy).sum()), A_overestimated_C_still_over_n=int(((aa > yy) & (cc > yy)).sum()), A_overestimated_C_crossed_to_under_n=int(((aa > yy) & (cc < yy)).sum()), A_overestimated_positive_error_reduced_n=int(((aa > yy) & (np.maximum(cc - yy, 0) < aa - yy)).sum()), prediction_reduced_n=int((cc < aa).sum()), mean_positive_error_A_g=float(np.maximum(aa - yy, 0).mean()), mean_positive_error_C_g=float(np.maximum(cc - yy, 0).mean()))
    ev = out / '실증평가'
    if ev.exists():
        ev = out / ('실증평가_재검산_' + s4.stamp())
    ev.mkdir(exist_ok=False)
    result.to_csv(ev / '01_current89_A_C_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(metrics).to_csv(ev / '02_metrics_same85.csv', index=False, encoding='utf-8-sig')
    s4.js(ev / '03_paired_comparison.json', paired)
    verify(out, cm, models, env, a, c)
    s4.js(ev / 'completion.json', dict(status='C_CURRENT89_EVALUATION_COMPLETE', evaluated_n=85, target_n=89, unavailable_n=4, metrics=metrics, paired=paired, model_training=False, pH_experiment=False, output_hashes={str(p): s4.sha(p) for p in ev.iterdir() if p.is_file()}))
    s4.js(OUTPUT_ROOT / 'latest_completed.json', dict(run=str(out), evaluation=str(ev)))
    print(json.dumps(dict(metrics=metrics[:4], paired=paired), ensure_ascii=False, indent=2), flush=True)
    return ev

def verify(out, cm, models, env, a, c):
    f = pd.read_csv(out / '02_C_step_model_features.csv')
    emb = np.load(out / 'common_embedding_128.npy')
    probs = np.load(out / 'stage_probabilities.npy')
    original = np.load(A_RUN / 'stage_probabilities.npy')
    assert emb.shape == (1313, 128) and probs.shape == (1313, 5)
    idx = f.embedding_row_index.to_numpy(int)
    assert [c[i]['record_id'] for i in idx] == f.record_id.tolist()
    assert np.allclose(emb[idx], f[cm.EMB_COLS], rtol=0, atol=1e-05)
    ok = np.array([r['a_step_status'] == 'OK' for r in a])
    assert np.allclose(probs[ok], original[ok], rtol=0, atol=1e-12)
    pred = cm.predict(f, models)
    for (key, col) in [('direct_stacking', 'c_step_stacking_g'), ('residual', 'c_step_residual_g'), ('blend50', 'c_step_pred_weight_g')]:
        assert np.allclose(pred[key], [float(c[i][col]) for i in idx], rtol=1e-12, atol=1e-10)
    lookup = env.set_index('prediction_cutoff')
    selected = [n for n in cm.ENV_COLS if n in f]
    for row in f.to_dict('records'):
        cr = c[int(row['embedding_row_index'])]
        day = pd.to_datetime(cr['capture_date'], format='%Y%m%d')
        expected = lookup.loc[day]
        np.testing.assert_allclose([row[n] for n in selected], [expected[n] for n in selected], rtol=0, atol=0, equal_nan=True)
    checks = 0
    with closing(sqlite3.connect((out / 'working.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        for target in c[:89]:
            if target['c_step_status'] != 'OK':
                continue
            identity = '|'.join((target[k] for k in ['season_id', 'bed_id', 'physical_slot']))
            records = []
            for (ts, payload, feat) in db.execute("SELECT ts,payload,feat FROM rows WHERE identity=? AND error='' AND feat IS NOT NULL ORDER BY ts", (identity,)):
                r = json.loads(payload)
                v = json.loads(feat)
                records.append(dict(v, capture_ts=pd.Timestamp(ts), crop_filename=Path(r['crop_path']).name, source_tier='raw_direct', volume_proxy=max(v['area_cm2'], 0) ** 1.5))
            hist = pd.DataFrame(records)
            cutoff = pd.Timestamp(target['capture_timestamp'])
            (values, h) = cm.causal_history(hist, cutoff)
            (before, _) = cm.causal_history(hist[hist.capture_ts <= cutoff], cutoff)
            future = dict(records[-1], capture_ts=cutoff + pd.Timedelta(days=1), area_cm2=1000000000000.0, pred_weight_cnn=1000000000000.0)
            (after, _) = cm.causal_history(pd.concat([hist, pd.DataFrame([future])], ignore_index=True), cutoff)
            pd.testing.assert_series_equal(pd.Series(values), pd.Series(before))
            pd.testing.assert_series_equal(pd.Series(values), pd.Series(after))
            assert len(h) == int(target['c_step_history_n_days'])
            saved = f[f.record_id == target['record_id']].iloc[0]
            for (name, value) in values.items():
                if name in f:
                    np.testing.assert_allclose(saved[name], value, rtol=1e-10, atol=1e-08, equal_nan=True)
            checks += 1
    assert checks == 85
    s4.js(out / 'verification.json', dict(status='PASS', same85=True, all_A_columns_preserved=True, model_replay_rows=len(f), current_prefix_future_invariance_checks=checks, stage_probabilities_match=True, embedding_match=True, selected_environment_columns=len(selected), explicit_environment17_preserved=True, reference_code_unchanged=True))

def main():
    global A_RUN, PREP, ENV_RUN, A_EVAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-run', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--environment-run', type=Path, required=True)
    p.add_argument('--a-evaluation', type=Path, required=True)
    p.add_argument('--evaluate-run', type=Path)
    args = p.parse_args()
    (A_RUN, PREP, ENV_RUN, A_EVAL) = (args.a_run, args.prepared, args.environment_run, args.a_evaluation)
    out = args.evaluate_run or infer()
    evaluate(out)
if __name__ == '__main__':
    main()
