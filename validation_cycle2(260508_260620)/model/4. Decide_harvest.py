"""Apply the frozen p4 and weight rule to an explicitly supplied C run. Evaluate current targets only; unavailable predictions remain unavailable."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
import sys
import importlib.util
sys.dont_write_bytecode = True
PROJECT = CYCLE.PROJECT_ROOT
spec = importlib.util.spec_from_file_location('season3_c', CYCLE.PROJECT_ROOT / 'model' / '3. Fuse_growth_history.py')
c3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c3)
s3 = c3.s3
C_RUN = None
OUTPUT_ROOT = CYCLE.MODEL_OUTPUT / '05_D-step'
import argparse
import math
from collections import Counter
import numpy as np

def number(value):
    if value in ('', None):
        return None
    result = float(value)
    s3.require(math.isfinite(result), 'Nonfinite evaluation value')
    return result

def metrics(name, rows):
    good = [r for r in rows if r['prediction'] is not None]
    counts = Counter((r['outcome'] for r in good))
    (tn, fp, fn, tp) = (counts[k] for k in ['TN', 'FP', 'FN', 'TP'])
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return dict(group=name, targets=len(rows), predicted=len(good), unavailable=len(rows) - len(good), coverage=len(good) / len(rows) if rows else None, positives=tp + fn, negatives=tn + fp, TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / len(good) if good else None, balanced_accuracy=(recall + specificity) / 2 if recall is not None and specificity is not None else None, recall=recall, precision=tp / (tp + fp) if tp + fp else None, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, specificity=specificity)

def outcome(truth, prediction):
    return 'PREDICTION_UNAVAILABLE' if prediction is None else ('TP' if truth else 'FP') if prediction else 'FN' if truth else 'TN'

def verify_rows(run):
    done = s3.read_json(run / 'completion.json')
    s3.require(done['c_run'] == str(C_RUN.resolve()), 'D upstream mismatch')
    for (p, digest) in done['source_hashes'].items():
        s3.require(s3.sha(p) == digest, 'D protected source changed: ' + p)
    path = run / '01_D_step_harvest_decisions.csv'
    s3.require(s3.sha(path) == done['output_sha256'], 'D output hash mismatch')
    rows = c3.read_csv(path)
    original = c3.read_csv(C_RUN / '01_C_step_predictions.csv')
    probs = np.load(C_RUN / 'stage_probabilities.npy', allow_pickle=False)
    s3.require(len(rows) == len(original) == 420 and probs.shape == (420, 5), 'D rows/probabilities mismatch')
    s3.require(len({r['record_id'] for r in rows}) == len(rows), 'Duplicate D IDs')
    for (i, (r, old)) in enumerate(zip(rows, original)):
        s3.require(all((r[k] == v for (k, v) in old.items())), 'D changed upstream value')
        s3.require(int(r['embedding_row_index']) == i, 'D row alignment failed')
        s3.require(np.allclose([float(r[f'd_step_p{k}']) for k in range(5)], probs[i], rtol=0, atol=1e-12), 'D probability source mismatch')
        weight = number(r['c_step_pred_weight_g'])
        if old['c_step_status'] != 'OK':
            s3.require(r['d_step_status'] == 'PREDICTION_UNAVAILABLE' and all((r[k] == '' for k in ['d_step_harvest_ornot', 'd_step_stage_gate', 'd_step_weight_gate'])), 'Unavailable lost')
        else:
            (stage, mass) = (int(probs[i, 4] >= 0.3), int(weight >= 120))
            s3.require(r['d_step_status'] == 'OK' and int(r['d_step_stage_gate']) == stage and (int(r['d_step_weight_gate']) == mass) and (int(r['d_step_harvest_ornot']) == stage * mass), 'Saved D gate mismatch')
    s3.require(sum((r['d_step_status'] == 'OK' for r in rows)) == done['counts']['predicted'], 'D counts mismatch')
    return rows

def evaluate(run):
    raw = verify_rows(run)
    lookup = {r['record_id']: r for r in raw}
    cdone = s3.read_json(C_RUN / 'completion.json')
    prep = s3.read_json(Path(cdone['input_preparation']) / 'completion.json')
    s3.require(s3.sha(s3.LABEL_FILE) == prep['source_hashes'][str(s3.LABEL_FILE)], 'Labels changed after model run')
    targets = c3.read_csv(Path(cdone['target_a_run']) / '01_A_step_predictions.csv')
    labels = {r['record_id']: r for r in s3.rows_xlsx(s3.LABEL_FILE, 'feature') if r['match_eligible'] is True}
    s3.require(len(targets) == 32 and len({r['record_id'] for r in targets}) == 32, 'Evaluation target mismatch')
    evaluated = []
    for target in targets:
        (r, lab) = (lookup[target['record_id']], labels[target['record_id']])
        s3.require(all((r[k] == v for (k, v) in target.items())), 'Target source changed')
        (stage, weight) = (int(lab['human_label']), float(lab['fresh_weight_g']))
        truth = int(stage == 4 and weight >= 120)
        pred = int(r['d_step_harvest_ornot']) if r['d_step_status'] == 'OK' else None
        evaluated.append(dict(sample_key=lab['sample_key'], survey_date=lab['survey_date'].strftime('%Y-%m-%d'), capture_date=r['capture_date'], lead_days=int(lab['lead_days']), human_label=stage, fresh_weight_g=weight, truth=truth, cnn_pred_stage=number(r.get('cnn_pred_stage')), five_fold_pred_stage=number(r['d_step_pred_stage']), five_fold_p4=number(r['d_step_p4']), a_step_weight_g=number(r['pred_weight_g']), c_step_weight_g=number(r['c_step_pred_weight_g']), stage_gate=number(r['d_step_stage_gate']), weight_gate=number(r['d_step_weight_gate']), prediction=pred, outcome=outcome(truth, pred), d_step_status=r['d_step_status'], d_step_error=r['d_step_error'], record_id=r['record_id'], embedding_row_index=int(r['embedding_row_index'])))
    comparison = []
    comparison.append(metrics('cycle3_ALL_MATCHED', evaluated))
    comparison.extend((metrics('3작기_' + date, [r for r in evaluated if r['survey_date'] == date]) for date in sorted({r['survey_date'] for r in evaluated})))
    comparison.extend([metrics('cycle3_same_day', [r for r in evaluated if r['lead_days'] == 0]), metrics('cycle3_lag_1_3_days', [r for r in evaluated if r['lead_days'] > 0])])
    weight_comparison = []
    for r in c3.read_csv(C_RUN / 'evaluation' / 'metrics.csv'):
        if r['group'] == 'ALL_MATCHED' and r['model'] in ['A_baseline', 'C_blend50']:
            weight_comparison.append(dict(group='cycle3_ALL_MATCHED', model='A-step' if r['model'] == 'A_baseline' else 'C-blend50', predicted=int(r['predicted']), MAE_g=float(r['MAE_g']), RMSE_g=float(r['RMSE_g']), R2=float(r['R2'])))
    return (evaluated, comparison, weight_comparison)

def workbook_specs(rows, comparison, weights):
    return [s3.sheet_spec(name, items) for (name, items) in [('Metrics', comparison), ('Target_predictions', rows), ('Weight_metrics', weights)]]

def main(argv=None):
    global C_RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-run', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    C_RUN = args.c_run.resolve()
    d = s3.load_reference('s3_frozen_d', 'model/4. Decide_harvest.py')
    common = ['--c-run', str(C_RUN), '--output-dir', str(OUTPUT_ROOT), '--season-id', CYCLE.SEASON_ID, '--selection', str(CYCLE.HARVEST_RULE), '--model-scope', 'frozen_development_model_external_cycle3_environment_missing']
    c3.call_main(d, common + ['--check'])
    if args.check:
        return 0
    before = set(OUTPUT_ROOT.glob('run_*'))
    c3.call_main(d, common)
    created = set(OUTPUT_ROOT.glob('run_*')) - before
    s3.require(len(created) == 1, 'Cannot identify the new D run')
    run = created.pop()
    (rows, comparison, weights) = evaluate(run)
    ev = run / 'evaluation'
    ev.mkdir(exist_ok=False)
    protected = [s3.LABEL_FILE, Path(__file__), C_RUN / 'evaluation' / 'metrics.csv', C_RUN / 'completion.json']
    hashes = {str(p): s3.sha(p) for p in protected}
    c3.save_csv(ev / 'target_predictions.csv', rows)
    c3.save_csv(ev / 'metrics.csv', comparison)
    c3.save_csv(ev / 'weight_metrics.csv', weights)
    s3.save_workbooks([dict(path=str(ev / 'predictions.xlsx'), sheets=workbook_specs(rows, comparison, weights))], 'd_evaluation')
    s3.require(all((s3.sha(p) == h for (p, h) in hashes.items())), 'Source changed during evaluation')
    report = dict(status='PASS', target_rows=32, history_only_rows=388, ground_truth='human_label == 4 AND fresh_weight_g >= 120', thresholds_frozen=True, model_training=False, history_only_rows_excluded_from_metrics=True, unavailable_not_negative=True, metrics=comparison, weight_metrics=weights, source_hashes=hashes, output_hashes={str(p): s3.sha(p) for p in ev.iterdir() if p.is_file()})
    s3.write_json(ev / 'completion.json', report)
    s3.write_json(OUTPUT_ROOT / 'latest_completed.json', dict(run=str(run), evaluation=str(ev)))
    print('D-step complete:', run, flush=True)
    return 0
if __name__ == '__main__':
    main()
