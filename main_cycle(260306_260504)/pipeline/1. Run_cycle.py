"""Run the development cultivation cycle from preprocessing or saved products.

User paths: ../settings.py. Use --check for read-only validation, or choose
--start preprocessing to run raw-image processing before CNN/A/C/D inference.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
CODE_BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as CYCLE
from model_contract import component_contracts, inspect_artifacts
PYTHON = CYCLE.PYTHON
THIS = Path(__file__).resolve()
PREPROCESS = CYCLE.PREPROCESSED_INPUT
OUTPUT = CYCLE.OUTPUT_ROOT
ENVIRONMENT = CYCLE.ENVIRONMENT_FILE
GROWTH = CYCLE.GROWTH_FILE
SEASON_ID = CYCLE.SEASON_ID
SEASON_START = CYCLE.SEASON_START
SEASON_END = CYCLE.SEASON_END
DEVICE = CYCLE.DEVICE
MODEL_BATCH_SIZE = CYCLE.BATCH_SIZE
SCRIPTS = {'gate': CODE_BASE / 'pipeline/crop_integrity.py', 'cnn': CODE_BASE / 'model/1. Predict_stage.py', 'a': CODE_BASE / 'model/2. Predict_weight.py', 'c': CODE_BASE / 'model/3. Fuse_growth_history.py', 'd': CODE_BASE / 'model/4. Decide_harvest.py'}
PATHS = {}
FOLDERS = {}

def configure(start, output, device):
    """Use one path configuration for the runner and all subprocesses."""
    global OUTPUT, PREPROCESS, DEVICE, PATHS, FOLDERS
    OUTPUT = Path(output).resolve()
    DEVICE = device
    PREPROCESS = OUTPUT / 'preprocessing' if start == 'preprocessing' else CYCLE.PREPROCESSED_INPUT
    PATHS = {**CYCLE.preprocessing_inputs(PREPROCESS), 'environment': ENVIRONMENT, 'growth': GROWTH, 'stage_model': CYCLE.STAGE_MODEL, 'a_model': CYCLE.A_MODEL, 'c_models': CYCLE.C_MODELS, 'd_rule': CYCLE.HARVEST_RULE}
    FOLDERS = {name: OUTPUT / folder for (name, folder) in {'input': '00_input', 'cnn': '01_CNN', 'a': '02_A-step', 'raw': '03_environment', 'c': '04_C-step', 'd': '05_D-step', 'final': '06_final'}.items()}
    os.environ['LETTUCE_RUN_ROOT'] = str(OUTPUT)

def inspect_configuration(start):
    """Validate local code and required files without inference or output creation."""
    problems = []
    python_files = sorted(CODE_BASE.rglob('*.py'))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except SyntaxError as exc:
            problems.append(str(exc))
    for path in [*SCRIPTS.values(), CODE_BASE / 'preprocessing/pipeline_runner.py']:
        if not path.is_file():
            problems.append('Missing stage code: ' + str(path))
    if start == 'preprocessing' and CYCLE.PREPROCESS_PROFILE:
        problems.append('Integrated raw-image execution uses settings.py; run an existing-bed profile with preprocessing/pipeline_runner.py separately.')
    for source in (CYCLE.DATA_ROOT, CYCLE.ARTIFACT_ROOT, CYCLE.PREPROCESSED_INPUT):
        if OUTPUT == source.resolve() or OUTPUT in source.resolve().parents:
            problems.append('Output directory must not contain source data or model artifacts: ' + str(OUTPUT))
    required = {k: v for (k, v) in PATHS.items() if start == 'models' or k in {'environment', 'growth', 'stage_model', 'a_model', 'c_models', 'd_rule'}}
    if start == 'preprocessing':
        required.update(raw_images=CYCLE.RAW_IMAGE_ROOT, bed_top=CYCLE.BED_TOP_MODEL, bed_front=CYCLE.BED_FRONT_MODEL, plant=CYCLE.PLANT_MODEL)
    required.update(pca=CYCLE.PCA_FILE, backbone=CYCLE.BACKBONE_FILE)
    missing = [{'input': name, 'path': str(path)} for (name, path) in required.items() if not path.exists()]
    contracts = {}
    if CYCLE.C_MODELS.is_dir():
        try:
            components = component_contracts(CYCLE.C_MODELS)
            contracts = {mode: len(meta['features']) for (mode, (_, meta)) in components.items()}
            for (mode, (folder, meta)) in components.items():
                for record in meta['seeds']:
                    model = folder / f"seed_{record['seed']}.cbm"
                    if not model.is_file():
                        missing.append({'input': mode, 'path': str(model)})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            problems.append(str(exc))
    if CYCLE.A_MODEL.is_dir():
        bundle = CYCLE.A_MODEL / 'selected_model_bundle.json'
        if not bundle.is_file():
            missing.append({'input': 'a_bundle', 'path': str(bundle)})
        else:
            try:
                for fold in json.loads(bundle.read_text(encoding='utf-8-sig'))['folds']:
                    for item in fold['files']:
                        path = CYCLE.A_MODEL / item['path']
                        if not path.is_file():
                            missing.append({'input': 'a_fold', 'path': str(path)})
            except (ValueError, KeyError, TypeError) as exc:
                problems.append('Invalid A-step bundle: ' + str(exc))
    report = dict(status='READY' if not problems and (not missing) else 'INPUTS_REQUIRED', code_files_checked=len(python_files), code_paths_valid=not any(('Missing stage code:' in p for p in problems)), environment_schema='manuscript12', model_environment_policy=CYCLE.MODEL_ENVIRONMENT_POLICY, components=contracts, missing=missing, problems=problems, start=start, output=str(OUTPUT), model_inference=False, writes=0)
    return report

def run_preprocessing():
    """Run the existing preprocessing stages and pass their products to the model chain."""
    environment = os.environ.copy()
    if CYCLE.PREPROCESS_PROFILE:
        environment['LETTUCE_PROFILE'] = str(CYCLE.PREPROCESS_PROFILE)
    else:
        environment['LETTUCE_INPUT_ROOT'] = str(CYCLE.RAW_IMAGE_ROOT)
        environment['LETTUCE_OUTPUT_ROOT'] = str(PREPROCESS)
        environment['LETTUCE_SEASON_ID'] = SEASON_ID
    device = DEVICE
    if device == 'auto':
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    environment['LETTUCE_DEVICE'] = device
    subprocess.run([str(PYTHON), str(CODE_BASE / 'preprocessing/pipeline_runner.py'), '--run'], cwd=str(CODE_BASE), env=environment, check=True)

def save_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def signature() -> dict:
    files = {name: path for (name, path) in PATHS.items() if path.is_file()}
    files.update({name: path for (name, path) in SCRIPTS.items()})
    files.update({'wrapper': THIS, 'input_adapter': CODE_BASE / 'pipeline' / 'stage_inputs.py', 'metrics_adapter': CODE_BASE / 'pipeline' / 'stage_metrics.py', 'final_adapter': CODE_BASE / 'pipeline' / 'final_xlsx.py'})
    files['a_bundle'] = PATHS['a_model'] / 'selected_model_bundle.json'
    files['a_selection'] = PATHS['a_model'] / 'MODEL_SELECTION_FROZEN.json'
    for fold in json.loads(files['a_bundle'].read_text(encoding='utf-8-sig'))['folds']:
        for item in fold['files']:
            files['a_artifact/' + item['path']] = PATHS['a_model'] / item['path']
    for (mode, (folder, metadata)) in component_contracts(PATHS['c_models']).items():
        files[mode + '_ensemble'] = folder / 'ensemble.json'
        for record in metadata['seeds']:
            files[f"{mode}_{record['seed']}"] = folder / f"seed_{record['seed']}.cbm"
    for path in CODE_BASE.rglob('*.py'):
        files['code/' + path.relative_to(CODE_BASE).as_posix()] = path
    files['pca'] = CYCLE.PCA_FILE
    files['backbone'] = CYCLE.BACKBONE_FILE
    return {name: {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for (name, path) in files.items()}

def stage(state: dict, state_file: Path, name: str, outputs: list[Path], task):
    prior = state['stages'].get(name)
    if prior and prior['status'] == 'COMPLETE' and all((path.exists() for path in outputs)):
        print(f'[Reuse] {name}', flush=True)
        return prior['result']
    print(f'\n[Start] {name}', flush=True)
    result = task()
    if not all((path.exists() for path in outputs)):
        raise RuntimeError(f'{name} Completion file missing: {[str(p) for p in outputs if not p.exists()]}')
    state['stages'][name] = {'status': 'COMPLETE', 'result': result, 'completed_at': datetime.now().isoformat()}
    save_json(state_file, state)
    print(f'[Complete] {name}', flush=True)
    return result

def command(script: Path, *args: object) -> None:
    print('[Run]', script.name, flush=True)
    subprocess.run([str(PYTHON), '-u', str(script), *map(str, args)], cwd=str(CODE_BASE), check=True)

def new_run(parent: Path, before: set[Path]) -> Path:
    created = {path for path in parent.glob('run_*') if path.is_dir()} - before
    if len(created) != 1:
        raise RuntimeError(f'Cannot identify a single new run directory: {parent} {created}')
    run = created.pop()
    if not (run / 'completion.json').is_file():
        raise RuntimeError(f'Stage completion record missing: {run}')
    return run

def run_models() -> None:
    sys.path.insert(0, str(CODE_BASE))
    sys.path.insert(0, str(CODE_BASE / 'preprocessing'))
    sys.path.insert(0, str(THIS.parent))
    from stage_inputs import build_subset, labels_from_csv, preflight
    from stage_metrics import write_available_stage, write_stage_metrics
    from final_xlsx import write_final
    from growth_labels import build as build_growth
    from environment_features import build as build_environment
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    for script in SCRIPTS.values():
        if not script.is_file():
            raise FileNotFoundError(script)
    (targets, inspection) = preflight(PATHS)
    print(json.dumps({'status': 'READ_ONLY_PREFLIGHT_PASS', **inspection}, ensure_ascii=False, indent=2), flush=True)
    for folder in FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)
    state_file = FOLDERS['input'] / 'execution_state.json'
    source_signature = signature()
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding='utf-8'))
        if state.get('source_signature') != source_signature:
            raise RuntimeError('Inputs or code differ from the previous run; choose a new output directory')
        if state.get('status') == 'COMPLETE' and (OUTPUT / 'predictions.xlsx').is_file():
            print(f"Completed result already exists: {OUTPUT / 'predictions.xlsx'}", flush=True)
            return
    else:
        state = {'status': 'RUNNING', 'source_signature': source_signature, 'started_at': datetime.now().isoformat(), 'inspection': inspection, 'stages': {}}
        save_json(state_file, state)
    try:
        labels_csv = FOLDERS['raw'] / 'growth_labels.csv'
        stage(state, state_file, 'growth_raw', [labels_csv], lambda : build_growth(GROWTH, labels_csv))
        environment_csv = FOLDERS['raw'] / 'environment_daily_from_raw.csv'
        stage(state, state_file, 'environment_raw', [environment_csv], lambda : build_environment(ENVIRONMENT, environment_csv))
        labels = labels_from_csv(labels_csv)
        feature_subset = FOLDERS['input'] / 'model_feature_subset.xlsx'
        manifest_subset = FOLDERS['input'] / 'model_crop_manifest.csv'
        front_csv = FOLDERS['input'] / 'same_capture_front.csv'
        stage(state, state_file, 'existing_preprocess_subset', [feature_subset, manifest_subset, front_csv], lambda : build_subset(PATHS, labels, FOLDERS['input']))
        gate_csv = FOLDERS['input'] / 'crop_integrity.csv'

        def gate():
            command(SCRIPTS['gate'], '--input', manifest_subset, '--output', gate_csv)
            report = json.loads(gate_csv.with_suffix('.summary.json').read_text(encoding='utf-8'))
            if report['counts']['rows'] != state['stages']['existing_preprocess_subset']['result']['subset_rows']:
                raise ValueError("Invalid input or state in gate (rejected condition: report['counts']['rows'] != state['stages']['existing_preprocess_subset']['result']['subset_rows'])")
            return report
        stage(state, state_file, 'crop_mask_integrity', [gate_csv], gate)
        cnn_xlsx = FOLDERS['cnn'] / 'crop_features_CNN.xlsx'
        stage(state, state_file, 'cnn_inference', [cnn_xlsx], lambda : command(SCRIPTS['cnn'], '--feature', feature_subset, '--crop-root', PATHS['crop_root'], '--model', PATHS['stage_model'], '--output', cnn_xlsx, '--gate-csv', gate_csv, '--label-csv', labels_csv, '--device', DEVICE) or {'output': str(cnn_xlsx)})

        def a_step():
            before = set(FOLDERS['a'].glob('run_*'))
            command(SCRIPTS['a'], '--input', cnn_xlsx, '--output-dir', FOLDERS['a'], '--run', PATHS['a_model'], '--crop-root', PATHS['crop_root'], '--device', DEVICE, '--batch-size', MODEL_BATCH_SIZE)
            return {'run': str(new_run(FOLDERS['a'], before))}
        a_info = stage(state, state_file, 'a_step', [], a_step)
        a_run = Path(a_info['run'])
        if not (a_run / 'completion.json').is_file():
            raise FileNotFoundError(a_run / 'completion.json')
        a_csv = a_run / '01_A_step_predictions.csv'
        stage(state, state_file, 'a_current_metrics', [FOLDERS['a'] / 'current_predictions.csv', FOLDERS['a'] / 'current_full_metrics.csv'], lambda : write_available_stage(name='A-step', labels_csv=labels_csv, front_csv=front_csv, gate_csv=gate_csv, a_csv=a_csv, current_csv=a_csv, output_dir=FOLDERS['a'], prediction_column='pred_weight_g', status_column='a_step_status', targets=targets))

        def c_step():
            before = set(FOLDERS['c'].glob('run_*'))
            command(SCRIPTS['c'], '--a-run', a_run, '--output-dir', FOLDERS['c'], '--env-base', environment_csv, '--a-model', PATHS['a_model'], '--model-root', PATHS['c_models'], '--season-id', SEASON_ID, '--season-start', SEASON_START, '--season-end', SEASON_END, '--model-role', '2nd_crop_existing_preprocess_model_only', '--device', DEVICE, '--batch-size', MODEL_BATCH_SIZE)
            return {'run': str(new_run(FOLDERS['c'], before))}
        c_info = stage(state, state_file, 'c_step', [], c_step)
        c_run = Path(c_info['run'])
        if not (c_run / 'completion.json').is_file():
            raise FileNotFoundError(c_run / 'completion.json')
        c_csv = c_run / '01_C_step_predictions.csv'
        stage(state, state_file, 'c_current_metrics', [FOLDERS['c'] / 'current_predictions.csv', FOLDERS['c'] / 'current_full_metrics.csv'], lambda : write_available_stage(name='C-step', labels_csv=labels_csv, front_csv=front_csv, gate_csv=gate_csv, a_csv=a_csv, current_csv=c_csv, output_dir=FOLDERS['c'], prediction_column='c_step_pred_weight_g', status_column='c_step_status', targets=targets))

        def d_step():
            before = set(FOLDERS['d'].glob('run_*'))
            command(SCRIPTS['d'], '--c-run', c_run, '--output-dir', FOLDERS['d'], '--season-id', SEASON_ID, '--model-scope', '2nd_crop_existing_preprocess_model_only', '--selection', PATHS['d_rule'])
            return {'run': str(new_run(FOLDERS['d'], before))}
        d_info = stage(state, state_file, 'd_step', [], d_step)
        d_run = Path(d_info['run'])
        if not (d_run / 'completion.json').is_file():
            raise FileNotFoundError(d_run / 'completion.json')
        d_csv = d_run / '01_D_step_harvest_decisions.csv'

        def metrics():
            return write_stage_metrics(labels_csv=labels_csv, front_csv=front_csv, gate_csv=gate_csv, a_csv=a_csv, c_csv=c_csv, d_csv=d_csv, a_dir=FOLDERS['a'], c_dir=FOLDERS['c'], d_dir=FOLDERS['d'], targets=targets)
        stage(state, state_file, 'current_stage_metrics', [FOLDERS[name] / file for name in ('a', 'c', 'd') for file in ('current_predictions.csv', 'current_full_metrics.csv')], metrics)
        final_file = FOLDERS['final'] / 'predictions.xlsx'
        final_report = FOLDERS['final'] / 'summary.json'
        stage(state, state_file, 'final_xlsx', [final_file, final_report], lambda : write_final(labels_csv=labels_csv, qc_xlsx=PATHS['qc'], front_csv=front_csv, gate_csv=gate_csv, a_csv=a_csv, d_csv=d_csv, targets=targets, output=final_file, report_path=final_report))
        root_final = OUTPUT / 'predictions.xlsx'
        if root_final.is_file():
            backup = FOLDERS['final'] / f'previous_predictions_{datetime.now():%Y%m%d_%H%M%S}.xlsx'
            shutil.copy2(root_final, backup)
        temporary = OUTPUT / 'predictions.writing.xlsx'
        shutil.copy2(final_file, temporary)
        os.replace(temporary, root_final)
        state['status'] = 'COMPLETE'
        state['final_file'] = str(root_final)
        state['completed_at'] = datetime.now().isoformat()
        save_json(state_file, state)
        print(f'\nCycle complete: {root_final}', flush=True)
    except BaseException as exc:
        state['status'] = 'FAILED'
        state['error'] = f'{type(exc).__name__}: {exc}'
        save_json(state_file, state)
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=['models', 'preprocessing'], default='models')
    parser.add_argument('--output', type=Path, default=CYCLE.OUTPUT_ROOT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=CYCLE.DEVICE)
    parser.add_argument('--check', action='store_true', help='Read-only code and input validation')
    parser.add_argument('--check-models', action='store_true', help='Read-only bundled-model checksums and compatibility; no data needed')
    args = parser.parse_args(argv)
    configure(args.start, args.output, args.device)
    if args.check_models:
        report = inspect_artifacts(CYCLE.ARTIFACT_ROOT, CYCLE.C_MODEL_TAG)
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'FILES_VERIFIED' else 2
    report = inspect_configuration(args.start)
    if args.check:
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'READY' else 2
    if report['status'] != 'READY':
        print(json.dumps(report, indent=2))
        return 2
    if args.start == 'preprocessing':
        run_preprocessing()
    run_models()
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
