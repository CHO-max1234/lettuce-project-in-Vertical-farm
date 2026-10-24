"""Apply the frozen p4 and fresh-weight gates to a completed C-step run. Stage probabilities come from the A-step five-fold mean. Unavailable rows remain unavailable; growth QC does not change the decision."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
import argparse
import csv
import hashlib
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
RESULT_ROOT = CYCLE.OUTPUT_ROOT
C_RUNS = CYCLE.OUTPUT_ROOT / '04_C-step'
D_RESULT = CYCLE.OUTPUT_ROOT / '05_D-step'
OUTPUT_DIR = CYCLE.OUTPUT_ROOT / '05_D-step'
SELECTION = CYCLE.HARVEST_RULE
SUPPORTED_SEASON = CYCLE.SEASON_ID
MODEL_SCOPE = 'validated_2nd_crop_scope'
SELECTED_WEIGHT_MODEL = CYCLE.C_MODEL_TAG
C_MODEL_NAME = CYCLE.C_MODEL_TAG
ROW_PROGRESS = 10000
EXTRA = ['d_step_status', 'd_step_error', 'd_step_prediction_role', 'd_step_stage_probability_source', 'd_step_p0', 'd_step_p1', 'd_step_p2', 'd_step_p3', 'd_step_p4', 'd_step_pred_stage', 'd_step_expected_stage', 'd_step_pred_weight_g', 'd_step_stage_gate', 'd_step_weight_gate', 'd_step_harvest_ornot', 'd_step_p4_threshold', 'd_step_weight_threshold_g', 'd_step_rule']

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)

def write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    tmp.replace(path)

def choose_c_run(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [p for p in C_RUNS.glob('run_*') if (p / 'completion.json').is_file()]
    require(bool(candidates), 'Validation condition failed in choose_c_run: bool(candidates)')
    return max(candidates, key=lambda p: p.name).resolve()

def frozen_rule() -> tuple[float, float, dict]:
    selection = read_json(SELECTION)
    require(selection.get('selection_split') == 'valid' and selection.get('test_read_for_selection') is False, 'D-step selection is not frozen on validation')
    require(selection.get('target_definition') == 'actual_stage == 4 AND fresh_weight_g >= 120', 'D-step target definition changed')
    require(selection.get('weight_model') == SELECTED_WEIGHT_MODEL, 'D-step selected weight model differs')
    choice = selection.get('selected', {}).get('fusion_selected', {})
    require(choice.get('strategy') == 'fusion_selected', 'D-step selected rule missing')
    p4 = float(choice['p4_threshold'])
    grams = float(choice['weight_threshold_g'])
    require(p4 == 0.3 and grams == 120.0, 'D-step thresholds differ from the declared rule; no automatic retuning')
    return (p4, grams, selection)

def inspect_c_run(run: Path) -> tuple[dict, Path, Path, int]:
    done_path = run / 'completion.json'
    require(done_path.is_file(), f'C-step completion record missing: {done_path}')
    done = read_json(done_path)
    require(done.get('status') in {'COMPLETE', 'COMPLETE_WITH_ROW_ERRORS'}, 'C-step is incomplete')
    require(done.get('selected') == SELECTED_WEIGHT_MODEL, 'C-step selected weight model mismatch')
    require(done.get('prediction_role') == 'operational_ensemble_not_oof', 'C-step prediction role mismatch')
    counts = done.get('counts', {})
    n = counts.get('rows')
    require(isinstance(n, int) and n > 0, 'Invalid C-step completion row count')
    require(counts.get('predicted', 0) + counts.get('unavailable', 0) == n, 'C-step prediction/failure counts differ from completion')
    csv_path = run / '01_C_step_predictions.csv'
    prob_path = run / 'stage_probabilities.npy'
    require(csv_path.is_file() and prob_path.is_file(), 'C-step prediction or fold-mean probabilities unavailable')
    probs = np.load(prob_path, mmap_mode='r', allow_pickle=False)
    require(probs.shape == (n, 5) and np.issubdtype(probs.dtype, np.floating), 'C-step probability array shape differs')
    return (done, csv_path, prob_path, n)

def stage_vector(vector: np.ndarray) -> tuple[list[float] | None, str]:
    values = [float(x) for x in vector]
    if not all((math.isfinite(x) and 0 <= x <= 1 for x in values)):
        return (None, 'STAGE_PROBABILITY_INVALID')
    if abs(sum(values) - 1.0) > 0.0001:
        return (None, 'STAGE_PROBABILITY_MISSING_OR_NOT_NORMALIZED')
    return (values, '')

def make_decision(row: dict[str, str], p: list[float] | None, stage_error: str, p4_limit: float, weight_limit: float) -> dict[str, object]:
    result: dict[str, object] = {column: '' for column in EXTRA}
    result.update(d_step_prediction_role='operational_ensemble_not_oof', d_step_stage_probability_source='A-step five-fold mean via C-step stage_probabilities.npy', d_step_p4_threshold=p4_limit, d_step_weight_threshold_g=weight_limit, d_step_rule='p4 >= threshold AND C-step predicted weight >= threshold')
    if p is not None:
        for (k, value) in enumerate(p):
            result[f'd_step_p{k}'] = value
        result['d_step_pred_stage'] = int(np.argmax(p))
        result['d_step_expected_stage'] = sum((k * value for (k, value) in enumerate(p)))
    problems = []
    if row['c_step_status'] != 'OK':
        problems.append('C_STEP_UNAVAILABLE: ' + (row.get('c_step_error') or row['c_step_status']))
    if stage_error:
        problems.append(stage_error)
    try:
        weight = float(row['c_step_pred_weight_g'])
        if not math.isfinite(weight):
            raise ValueError('non-finite')
    except (ValueError, TypeError):
        weight = None
        if row['c_step_status'] == 'OK':
            problems.append('C_STEP_WEIGHT_UNAVAILABLE')
    if problems:
        result.update(d_step_status='PREDICTION_UNAVAILABLE', d_step_error='; '.join(problems))
        return result
    require(p is not None and weight is not None, 'Validation condition failed in make_decision: p is not None and weight is not None')
    stage_gate = int(p[4] >= p4_limit)
    weight_gate = int(weight >= weight_limit)
    result.update(d_step_status='OK', d_step_pred_weight_g=weight, d_step_stage_gate=stage_gate, d_step_weight_gate=weight_gate, d_step_harvest_ornot=int(stage_gate and weight_gate))
    return result

def export(run: Path, target: Path, csv_path: Path, prob_path: Path, n: int, p4_limit: float, weight_limit: float) -> dict[str, int]:
    probs = np.load(prob_path, mmap_mode='r', allow_pickle=False)
    stats = {'rows': 0, 'predicted': 0, 'unavailable': 0, 'harvest': 0, 'not_harvest': 0, 'stage_probability_unavailable': 0, 'c_step_unavailable': 0}
    with csv_path.open('r', encoding='utf-8-sig', newline='') as source, target.open('w', encoding='utf-8-sig', newline='') as output:
        reader = csv.DictReader(source)
        headers = reader.fieldnames or []
        needed = {'record_id', 'embedding_row_index', 'season_id', 'prediction_role', 'c_step_status', 'c_step_error', 'c_step_prediction_role', 'c_step_model', 'c_step_pred_weight_g'}
        require(needed <= set(headers), f'C-step Missing required columns: {sorted(needed - set(headers))}')
        require(not set(EXTRA) & set(headers), 'Validation condition failed in export: not set(EXTRA) & set(headers)')
        writer = csv.DictWriter(output, fieldnames=headers + EXTRA)
        writer.writeheader()
        for (i, row) in enumerate(reader):
            require(i < n, 'C-step CSV has extra rows')
            require(int(row['embedding_row_index']) == i, 'C-step row order/probability index mismatch')
            require(bool(row['record_id']), 'Missing C-step record_id')
            require(row['season_id'] == SUPPORTED_SEASON, "Validation condition failed in export: row['season_id'] == SUPPORTED_SEASON")
            require(row['prediction_role'] == 'operational_ensemble_not_oof' and row['c_step_prediction_role'] == 'operational_ensemble_not_oof', "Validation condition failed in export: row['prediction_role'] == 'operational_ensemble_not_oof' and row['c_step_prediction_role'] == 'operational_ensemble_not_oof'")
            require(row['c_step_model'] == C_MODEL_NAME, 'C-step row model mismatch')
            (vector, stage_error) = stage_vector(probs[i])
            decision = make_decision(row, vector, stage_error, p4_limit, weight_limit)
            writer.writerow({**row, **decision})
            stats['rows'] += 1
            if stage_error:
                stats['stage_probability_unavailable'] += 1
            if row['c_step_status'] != 'OK':
                stats['c_step_unavailable'] += 1
            if decision['d_step_status'] == 'OK':
                stats['predicted'] += 1
                stats['harvest' if decision['d_step_harvest_ornot'] else 'not_harvest'] += 1
            else:
                stats['unavailable'] += 1
            if stats['rows'] % ROW_PROGRESS == 0:
                print(f"D-step {stats['rows']:,}/{n:,}export: ", flush=True)
    require(stats['rows'] == n, 'C-step CSV has missing rows')
    require(stats['predicted'] + stats['unavailable'] == n, 'D-step status counts differ')
    require(stats['harvest'] + stats['not_harvest'] == stats['predicted'], 'D-step decision counts differ')
    return stats

def main() -> None:
    global SUPPORTED_SEASON, MODEL_SCOPE, SELECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c-run', type=Path, help='main: C-step run. Output run')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='main: D-step')
    parser.add_argument('--season-id', default=SUPPORTED_SEASON)
    parser.add_argument('--model-scope', default=MODEL_SCOPE)
    parser.add_argument('--selection', type=Path, default=SELECTION)
    parser.add_argument('--check', action='store_true', help='main: Input')
    args = parser.parse_args()
    (SUPPORTED_SEASON, MODEL_SCOPE, SELECTION) = (args.season_id, args.model_scope, args.selection)
    (p4_limit, weight_limit, frozen) = frozen_rule()
    run = choose_c_run(args.c_run)
    (done, csv_path, prob_path, n) = inspect_c_run(run)
    if args.check:
        print(json.dumps({'status': 'READY', 'c_run': str(run), 'rows': n, 'c_step_counts': done['counts'], 'p4_threshold': p4_limit, 'weight_threshold_g': weight_limit, 'stage_source': str(prob_path), 'selection': str(SELECTION)}, ensure_ascii=False, indent=2))
        return
    output_dir = args.output_dir / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = [run / 'completion.json', csv_path, prob_path, SELECTION]
    hashes = {str(path): sha256(path) for path in sources}
    info = {'status': 'RUNNING', 'c_run': str(run), 'source_hashes': hashes, 'script_sha256': sha256(Path(__file__)), 'frozen_selection_utc': frozen['frozen_utc'], 'selected_weight_model': SELECTED_WEIGHT_MODEL, 'p4_threshold': p4_limit, 'weight_threshold_g': weight_limit, 'prediction_role': 'operational_ensemble_not_oof', 'stage_source': 'C-step stage_probabilities.npy: A-step five-fold mean', 'label_policy': 'No human labels or measured weights in operational input', 'scope': MODEL_SCOPE, 'season_id': SUPPORTED_SEASON, 'c_step_input': done.get('a_step_input'), 'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output_dir / 'started.json', info)
    try:
        target = output_dir / '01_D_step_harvest_decisions.csv'
        stats = export(run, target, csv_path, prob_path, n, p4_limit, weight_limit)
        for path in sources:
            require(sha256(path) == hashes[str(path)], f'Source changed during execution: {path}')
        info.update(status='COMPLETE_WITH_ROW_ERRORS' if stats['unavailable'] else 'COMPLETE', counts=stats, output_sha256=sha256(target), completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_dir / 'completion.json', info)
        print('main: D-step :', output_dir, flush=True)
    except BaseException as exc:
        write_json(output_dir / 'failure.json', {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise
if __name__ == '__main__':
    main()
