"""Run the frozen five-fold A ensemble and evaluate the 89 top-view targets."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
PROJECT = CYCLE.PROJECT_ROOT
INPUT_CODE = CYCLE.PROJECT_ROOT / 'model/2. Predict_stage.py'
A_ROOT = CYCLE.A_ROOT
EVALUATION_ROOT = CYCLE.A_EVALUATION_ROOT
ENVIRONMENT_RUN = CYCLE.ENVIRONMENT_RUN
(DEVICE, BATCH_SIZE) = ('cuda', 16)
import argparse
from collections import Counter
from contextlib import closing
import importlib.util
import json
import math
import shutil
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spec = resolved_spec('s4_model_inputs', INPUT_CODE)
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)

def run_a(prepared, cnn_run):
    s4.validate_prepared(prepared)
    cnn_done = s4.read_json(cnn_run / 'completion.json')
    assert cnn_done['rows'] == 1313 and cnn_done['prepared'] == str(prepared)
    assert s4.sha(cnn_run / 'crop_features_CNN.csv') == cnn_done['output_sha256']
    mod = s4.load('s4_frozen_a', s4.REFERENCE / 'model/1. A_step embedding and regression.py')
    mod.IMAGE_NAME = s4.IMAGE_NAME
    before = set(A_ROOT.glob('run_*'))
    old = sys.argv
    try:
        sys.argv = ['frozen_a', '--input', str(cnn_run / 'crop_features_CNN.csv'), '--output-dir', str(A_ROOT), '--run', str(s4.A_MODEL), '--crop-root', str(s4.CROP_ROOT), '--device', DEVICE, '--batch-size', str(BATCH_SIZE)]
        mod.main()
    finally:
        sys.argv = old
    runs = [p for p in set(A_ROOT.glob('run_*')) - before if (p / 'completion.json').is_file()]
    assert len(runs) == 1
    out = runs[0]
    shutil.copy2(__file__, out / '실행코드_보존.py')
    s4.js(out / 'season4_contract.json', dict(prepared=str(prepared), cnn_run=str(cnn_run), filename_adapter=s4.IMAGE_NAME.pattern, source_code_sha256=s4.sha(s4.REFERENCE / 'model/1. A_step embedding and regression.py'), script_sha256=s4.sha(__file__), weight_model='cnn_only_20260918_5fold', preprocessing_unchanged=True, labels_used=False, environment_used_in_A=False, row_order='current89_then_history1224', namespace_adapter='physical_12col_v1; source namespace preserved separately', original_identity_review=True, scale_verified=False, prediction_role='operational_ensemble_not_oof'))
    s4.js(A_ROOT / 'latest_completed.json', dict(run=str(out), prepared=str(prepared), cnn_run=str(cnn_run)))
    return out

def assemble_probabilities(a_run):
    rows = s4.read_csv(a_run / '01_A_step_predictions.csv')
    assert len(rows) == 1313
    probs = np.full((1313, 5, 5), np.nan, dtype=np.float64)
    ids = [r['record_id'] for r in rows]
    for fold in range(1, 6):
        for r in s4.read_csv(a_run / f'fold_{fold}/predictions.csv'):
            i = int(r['embedding_row_index'])
            assert ids[i] == r['record_id']
            probs[i, fold - 1] = [float(r[f'stage_p{j}']) for j in range(5)]
        emb = np.load(a_run / f'fold_{fold}/embedding.npy', mmap_mode='r')
        assert emb.shape == (1313, 128)
        for (i, r) in enumerate(rows):
            expected = r['a_step_status'] == 'OK'
            assert bool(np.isfinite(emb[i]).all()) == expected
            assert bool(np.isfinite(probs[i, fold - 1]).all()) == expected
    mean = probs.mean(axis=1)
    valid = np.isfinite(mean).all(axis=1)
    assert np.allclose(mean[valid].sum(axis=1), 1, atol=0.0001)
    folds = np.load(a_run / 'fold_predictions.npy', mmap_mode='r')
    assert folds.shape == (1313, 5)
    for (i, r) in enumerate(rows):
        assert int(r['embedding_row_index']) == i
        if r['a_step_status'] == 'OK':
            assert math.isclose(float(r['pred_weight_g']), max(0, float(folds[i].astype(float).mean())), rel_tol=1e-12, abs_tol=1e-12)
        else:
            assert not r['pred_weight_g']
    np.save(a_run / 'stage_probabilities.npy', mean, allow_pickle=False)
    np.save(a_run / 'fold_stage_probabilities.npy', probs, allow_pickle=False)
    return (rows, mean, folds)

def stage_metrics(rows, prefix):
    selected = [r for r in rows if r[prefix + '_status'] == 'OK']
    if not selected:
        return dict(n=0, accuracy=None, macro_f1_012=None, macro_f1_fixed5=None)
    truth = np.array([int(r['Stage_human']) for r in selected])
    pred = np.array([int(r[prefix + '_pred_stage']) for r in selected])
    expected = np.array([float(r[prefix + '_expected_stage']) for r in selected])
    cm = confusion_matrix(truth, pred, labels=[0, 1, 2, 3, 4])
    per = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
    return dict(n=len(selected), accuracy=float(accuracy_score(truth, pred)), macro_f1_012=float(f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0)), macro_f1_fixed5=float(f1_score(truth, pred, labels=[0, 1, 2, 3, 4], average='macro', zero_division=0)), macro_definition='fixed labels0,1,2; zero_division0; predictions3/4 retained as errors', expected_stage_mae=float(np.abs(expected - truth).mean()), late_stage_false_predictions=int((pred >= 3).sum()), confusion_truth0to4_pred0to4=cm.tolist(), per_class=[dict(stage=c, precision=float(per[0][c]), recall=float(per[1][c]), f1=float(per[2][c]), support=int(per[3][c])) for c in range(3)])

def weight_metrics(rows):
    selected = [r for r in rows if r['a_step_status'] == 'OK']
    if not selected:
        return dict(n=0, MAE_g=None, RMSE_g=None, bias_g=None, R2=None)
    y = np.array([float(r['fresh_weight_g']) for r in selected])
    p = np.array([float(r['pred_weight_g']) for r in selected])
    e = p - y
    den = float(np.square(y - y.mean()).sum())
    return dict(n=len(selected), MAE_g=float(abs(e).mean()), RMSE_g=float(np.sqrt(np.square(e).mean())), bias_g=float(e.mean()), R2=float(1 - np.square(e).sum() / den) if den > 0 else None, actual_min_g=float(y.min()), actual_max_g=float(y.max()), pred_min_g=float(p.min()), pred_max_g=float(p.max()))

def evaluate(prepared, cnn_run, a_run):
    input_done = s4.validate_prepared(prepared)
    a_done = s4.read_json(a_run / 'completion.json')
    assert a_done['status'] in ['COMPLETE', 'COMPLETE_WITH_ROW_ERRORS']
    targets = s4.read_csv(prepared / 'evaluation_targets89.csv')
    cnn = s4.read_csv(cnn_run / 'crop_features_CNN.csv')
    (a, probs, folds) = assemble_probabilities(a_run)
    assert len(cnn) == len(a) == 1313
    result = []
    for (i, t) in enumerate(targets):
        (c, w) = (cnn[i], a[i])
        assert t['record_id'] == c['record_id'] == w['record_id']
        assert c['row_role'] == w['row_role'] == 'CURRENT'
        r = dict(t)
        for k in c:
            if k.startswith('cnn_') or k.startswith('crop_integrity_'):
                r[k] = c[k]
        for k in ['embedding_row_index', 'a_step_status', 'a_step_error', 'a_step_numeric_missing_count', 'a_step_model', 'prediction_role', 'pred_weight_g', 'a_step_resolved_path']:
            r[k] = w[k]
        r['a_ensemble_status'] = 'OK' if np.isfinite(probs[i]).all() else 'PREDICTION_UNAVAILABLE'
        r['a_ensemble_pred_stage'] = int(probs[i].argmax()) if r['a_ensemble_status'] == 'OK' else None
        r['a_ensemble_expected_stage'] = float(probs[i] @ np.arange(5)) if r['a_ensemble_status'] == 'OK' else None
        for j in range(5):
            r[f'a_stage_p{j}'] = float(probs[i, j]) if np.isfinite(probs[i, j]) else None
        if w['a_step_status'] == 'OK':
            r['a_raw_fold_mean_g'] = float(folds[i].astype(float).mean())
            r['weight_error_g'] = float(w['pred_weight_g']) - float(t['fresh_weight_g'])
            r['absolute_error_g'] = abs(r['weight_error_g'])
        else:
            r.update(a_raw_fold_mean_g=None, weight_error_g=None, absolute_error_g=None)
        result.append(r)
    assert len(result) == 89
    groups = [('ALL89', result), ('SAME_DAY_TIME_UNKNOWN', [r for r in result if int(r['lead_days']) == 0]), ('PRIOR_DAY_IMAGE', [r for r in result if int(r['lead_days']) > 0])]
    groups += [(date, [r for r in result if r['survey_date'][:10] == date]) for date in sorted({r['survey_date'][:10] for r in result})]
    groups += [('HUMAN_STAGE_' + str(c), [r for r in result if int(r['Stage_human']) == c]) for c in [0, 1, 2]]
    scores = {}
    summary = []
    for (name, rr) in groups:
        scores[name] = dict(target_n=len(rr), single_cnn=stage_metrics(rr, 'cnn'), a_fold_mean_stage=stage_metrics(rr, 'a_ensemble'), a_weight=weight_metrics(rr))
        ss = scores[name]
        summary.append(dict(group=name, target_n=len(rr), single_cnn_n=ss['single_cnn']['n'], single_cnn_accuracy=ss['single_cnn']['accuracy'], single_cnn_f1_012=ss['single_cnn']['macro_f1_012'], a_stage_n=ss['a_fold_mean_stage']['n'], a_stage_accuracy=ss['a_fold_mean_stage']['accuracy'], a_stage_f1_012=ss['a_fold_mean_stage']['macro_f1_012'], **ss['a_weight']))
    out = EVALUATION_ROOT / ('run_' + s4.stamp())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / '실행코드_보존.py')
    s4.write_csv(out / '01_current89_predictions_and_truth.csv', result)
    s4.write_csv(out / '02_current89_metric_summary.csv', summary)
    s4.js(out / '03_metrics.json', scores)
    original = s4.read_sheet(s4.TARGET_FILE, '생육매칭')
    for (r, t) in zip(result, original):
        assert r['record_id'] == t['record_id'] and r['item_id'] == t['item_id']
        assert int(r['Stage_human']) == int(t['Stage_human']) and float(r['fresh_weight_g']) == float(t['fresh_weight_g'])
    saved = s4.read_csv(out / '01_current89_predictions_and_truth.csv')
    for (r, t) in zip(saved, targets):
        assert all((r[k] == t[k] for k in t)), 'Input contract failed: all((r[k] == t[k] for k in t))'
    valid = [r for r in saved if r['a_step_status'] == 'OK']
    independent_errors = [float(r['pred_weight_g']) - float(r['fresh_weight_g']) for r in valid]
    assert math.isclose(sum((abs(e) for e in independent_errors)) / len(valid), scores['ALL89']['a_weight']['MAE_g'], abs_tol=1e-10)
    assert all((s4.sha(p) == h for (p, h) in input_done['source_hashes'].items()))
    verification = dict(status='PASS', rows_preserved=89, independent_workbook_truth_join=True, source_columns_and_order_preserved=True, all1313_row_index_alignment=True, fold_embedding_shapes=[1313, 128], folds=5, fold_probabilities=[1313, 5, 5], stage_probabilities=[1313, 5], failed_embeddings_and_probabilities_nan=True, raw_5fold_mean_and_frozen_zero_clip_verified=True, labels_absent_from_inference=True, history_not_counted_as_truth=True, saved_MAE_independently_recomputed=True, source_model_hashes_unchanged=True, visual_review='PENDING')
    s4.js(out / 'verification.json', verification)
    plots = []
    report = None
    done = dict(status='CNN_A_CURRENT89_EVALUATION_COMPLETE', run=str(out), prepared=str(prepared), cnn_run=str(cnn_run), a_run=str(a_run), counts=dict(input_rows=1313, current_targets=89, history_only=1224, cnn_current_ok=scores['ALL89']['single_cnn']['n'], a_current_ok=scores['ALL89']['a_weight']['n'], a_all_ok=a_done['counts']['prediction_OK']), metrics=scores['ALL89'], model_training=False, ph_experiment=False, c_step_executed=False, model_scope='frozen_season2_external_season4', prediction_role='operational_ensemble_not_oof', limitations=['identity_confirmed=False', 'scale_verified=False', 'survey time unknown', 'image-survey offset0to2days', 'current labels0to2 only', 'source preprocessing has full-track label-free selection'], verification=verification, plots=plots, report=str(report))
    done['outputs'] = {p.name: s4.sha(p) for p in out.iterdir() if p.is_file()}
    s4.js(out / 'completion.json', done)
    s4.js(EVALUATION_ROOT / 'latest_completed.json', dict(run=str(out), a_run=str(a_run), cnn_run=str(cnn_run)))
    return (out, done)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--cnn-run', type=Path)
    parser.add_argument('--a-run', type=Path)
    parser.add_argument('--environment-run', type=Path)
    global ENVIRONMENT_RUN
    args = parser.parse_args()
    ENVIRONMENT_RUN = args.environment_run or ENVIRONMENT_RUN
    environment_hashes = {str(p): s4.sha(p) for p in (ENVIRONMENT_RUN.iterdir() if ENVIRONMENT_RUN else []) if p.is_file()}
    prepared = args.prepared or s4.prepare()
    cnn_run = args.cnn_run or s4.run_cnn(prepared)
    a_run = args.a_run or run_a(prepared, cnn_run)
    (out, done) = evaluate(prepared, cnn_run, a_run)
    assert all((s4.sha(p) == v for (p, v) in environment_hashes.items())), 'Input contract failed: all((s4.sha(p) == v for (p, v) in environment_hashes.items()))'
    s4.js(out / 'environment_unchanged.json', dict(status='PASS', source_hashes=environment_hashes, environment_used_in_A=False))
    print(json.dumps(s4.clean({k: done[k] for k in ['status', 'run', 'counts', 'metrics', 'plots', 'report']}), ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
