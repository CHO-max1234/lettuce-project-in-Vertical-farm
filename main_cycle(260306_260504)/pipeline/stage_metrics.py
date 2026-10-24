"""Processing utilities for stage metrics."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import csv
import math
import os
from pathlib import Path
from final_xlsx import exact_front, label_weights, number, representative_predictions
FIELDS = ['lettuce_key', 'record_id', 'actual_fresh_weight_g', 'pred_weight_g', 'error_g', 'prediction_status', 'cnn_pred_stage', 'p4', 'harvest_ornot']
METRICS = ['stage', 'cohort', 'target_keys', 'predicted_keys', 'coverage_pct', 'MAE_g', 'RMSE_g', 'bias_g', 'R2', 'note']

def load_stage(path: Path, wanted: set[str], prediction: str, status: str) -> dict[str, dict]:
    found = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'record_id', prediction, status}
        if required - set(reader.fieldnames or []):
            raise ValueError(f'Invalid input or state in load_stage (rejected condition: required - set(reader.fieldnames or [])): {path.name} {sorted(required - set(reader.fieldnames or []))} ')
        for row in reader:
            rid = row['record_id']
            if rid in wanted:
                if rid in found:
                    raise ValueError(f'Invalid input or state in load_stage (rejected condition: rid in found): {rid} ')
                found[rid] = row
    return found

def score(actual: dict[str, float], predictions: dict[str, float], keys: set[str], stage: str, cohort: str, note: str) -> dict:
    available = [(actual[key], predictions[key]) for key in sorted(keys) if key in predictions]
    errors = [pred - truth for (truth, pred) in available]
    result = {'stage': stage, 'cohort': cohort, 'target_keys': len(keys), 'predicted_keys': len(errors), 'coverage_pct': 100 * len(errors) / len(keys) if keys else 0, 'MAE_g': None, 'RMSE_g': None, 'bias_g': None, 'R2': None, 'note': note}
    if not errors:
        return result
    result['MAE_g'] = sum((abs(error) for error in errors)) / len(errors)
    result['RMSE_g'] = math.sqrt(sum((error * error for error in errors)) / len(errors))
    result['bias_g'] = sum(errors) / len(errors)
    mean = sum((truth for (truth, _) in available)) / len(available)
    denominator = sum(((truth - mean) ** 2 for (truth, _) in available))
    result['R2'] = 1 - sum((error * error for error in errors)) / denominator if denominator > 0 else None
    return result

def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.writing')
    try:
        with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def write_available_stage(*, name: str, labels_csv: Path, front_csv: Path, gate_csv: Path, a_csv: Path, current_csv: Path, output_dir: Path, prediction_column: str, status_column: str, targets: set[str]) -> dict:
    actual = label_weights(labels_csv)
    fronts = exact_front(front_csv, gate_csv)
    (representatives, _) = representative_predictions(a_csv, targets, fronts)
    selected = load_stage(current_csv, {row['record_id'] for row in representatives.values()}, prediction_column, status_column)
    predictions = {}
    rows = []
    for key in sorted(targets):
        rep = representatives.get(key)
        rid = rep['record_id'] if rep else ''
        source = selected.get(rid, {})
        estimate = number(source.get(prediction_column)) if source.get(status_column) == 'OK' else None
        if estimate is not None:
            predictions[key] = estimate
        rows.append({'lettuce_key': key, 'record_id': rid, 'actual_fresh_weight_g': actual[key], 'pred_weight_g': estimate, 'error_g': estimate - actual[key] if estimate is not None else '', 'prediction_status': source.get(status_column, 'NO_IMAGE_OR_STAGE_ROW'), 'cnn_pred_stage': '', 'p4': '', 'harvest_ornot': ''})
    metric = score(actual, predictions, targets, name, 'all_measured_targets', 'Frozen model; no refitting')
    write_csv(output_dir / 'current_predictions.csv', FIELDS, rows)
    write_csv(output_dir / 'current_full_metrics.csv', METRICS, [metric])
    return {'stage': name, 'target_keys': len(targets), 'available_predictions': len(predictions), 'MAE_g': metric['MAE_g'], 'RMSE_g': metric['RMSE_g']}

def write_stage_metrics(*, labels_csv: Path, front_csv: Path, gate_csv: Path, a_csv: Path, c_csv: Path, d_csv: Path, a_dir: Path, c_dir: Path, d_dir: Path, targets: set[str]) -> dict:
    actual = label_weights(labels_csv)
    if not targets <= set(actual):
        raise ValueError('Invalid input or state in write_stage_metrics (rejected condition: not targets <= set(actual))')
    fronts = exact_front(front_csv, gate_csv)
    (representatives, _) = representative_predictions(a_csv, targets, fronts)
    ids = {row['record_id'] for row in representatives.values()}
    stage_files = {'A-step': (a_csv, a_dir, 'pred_weight_g', 'a_step_status'), 'C-step': (c_csv, c_dir, 'c_step_pred_weight_g', 'c_step_status'), 'D-step': (d_csv, d_dir, 'd_step_pred_weight_g', 'd_step_status')}
    source = {name: load_stage(path, ids, column, status) for (name, (path, _, column, status)) in stage_files.items()}
    predictions: dict[str, dict[str, float]] = {name: {} for name in stage_files}
    output_rows: dict[str, list[dict]] = {name: [] for name in stage_files}
    for key in sorted(targets):
        representative = representatives.get(key)
        rid = representative['record_id'] if representative else ''
        for (name, (_, _, column, status_column)) in stage_files.items():
            row = source[name].get(rid, {})
            value = number(row.get(column)) if row.get(status_column) == 'OK' else None
            if value is not None:
                predictions[name][key] = value
            output_rows[name].append({'lettuce_key': key, 'record_id': rid, 'actual_fresh_weight_g': actual[key], 'pred_weight_g': value, 'error_g': value - actual[key] if value is not None else '', 'prediction_status': row.get(status_column, 'NO_IMAGE_OR_STAGE_ROW'), 'cnn_pred_stage': row.get('d_step_pred_stage', '') if name == 'D-step' else '', 'p4': row.get('d_step_p4', '') if name == 'D-step' else '', 'harvest_ornot': row.get('d_step_harvest_ornot', '') if name == 'D-step' else ''})
    common = set.intersection(*(set(values) for values in predictions.values()))
    report = {'target_keys': len(targets), 'representative_keys': len(representatives), 'common_three_stage_keys': len(common)}
    for (name, (_, folder, _, _)) in stage_files.items():
        note = 'D-step weight is the C-step blended weight' if name == 'D-step' else 'Frozen model; no refitting'
        metrics = [score(actual, predictions[name], targets, name, 'all_measured_targets', note), score(actual, predictions[name], common, name, 'common_A_C_D', note)]
        write_csv(folder / 'current_predictions.csv', FIELDS, output_rows[name])
        write_csv(folder / 'current_full_metrics.csv', METRICS, metrics)
        report[name] = {'available_predictions': len(predictions[name]), 'MAE_g': metrics[0]['MAE_g'], 'RMSE_g': metrics[0]['RMSE_g']}
    return report
