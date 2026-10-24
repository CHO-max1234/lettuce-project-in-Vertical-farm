"""Run external cycle 3 using local final stages and the shared main models."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as CYCLE
from runtime import load_module, load_shared
import argparse
import ast
import json
import os
import shutil
import subprocess
from datetime import datetime
PREPROCESSING = [('bed_top', '1. Segment_beds_top.py', []), ('bed_front', '1. Segment_beds_front.py', ['--mode', 'run']), ('ocr', '2. Read_bed_numbers.py', []), ('brightness', '3. Normalize_brightness.py', []), ('scale', '4. Estimate_scale.py', []), ('bboxes', '5. Build_plant_bboxes.py', []), ('coordinate_refinement', '6. Refine_slot_geometry.py', []), ('crops', '7. Extract_plant_crops.py', ['--run']), ('features', '8. Extract_features.py', ['--run-full']), ('growth_qc', '9. Assess_growth.py', [])]

def check_models():
    """Verify shared files and this external cycle's historical model contract."""
    inspector = load_shared('cycle3_artifact_inspector', 'pipeline/model_contract.py')
    files = inspector.inspect_artifacts(CYCLE.ARTIFACT_ROOT, CYCLE.C_MODEL_TAG)
    metadata = {}
    for mode in ('direct_stacking', 'residual'):
        path = CYCLE.C_MODELS / mode / 'ensemble.json'
        if path.is_file():
            metadata[mode] = json.loads(path.read_text(encoding='utf-8-sig'))
    rule = json.loads(CYCLE.HARVEST_RULE.read_text(encoding='utf-8-sig')) if CYCLE.HARVEST_RULE.is_file() else {}
    compatible = len(metadata) == 2 and metadata['direct_stacking']['features'] == metadata['residual']['features'] + ['pred_weight_cnn'] and (rule.get('weight_model') == CYCLE.C_MODEL_TAG) and (files.get('c_step', {}).get('packaged_model') == CYCLE.C_MODEL_TAG)
    return dict(status='READY' if files['status'] == 'FILES_VERIFIED' and compatible else 'MODELS_REQUIRED', files_status=files['status'], files_checked=files['files_checked'], missing=files['missing'], mismatched=files['mismatched'], feature_counts={mode: len(item['features']) for (mode, item) in metadata.items()}, model_identity=CYCLE.C_MODEL_TAG, environment_protocol=CYCLE.ENVIRONMENT_SCHEMA, environment_observations_used=0, model_training=False, contract_compatible=compatible, model_inference=False, writes=0)

def inspect_inputs(start):
    """Read-only checks; this does not establish full raw-batch completion."""
    problems = []
    python_files = list(PROJECT_ROOT.rglob('*.py'))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        except SyntaxError as exc:
            problems.append(str(exc))
    paths = {'reviewed_labels': CYCLE.LABEL_FILE}
    if start == 'preprocessing':
        paths.update(raw_top=CYCLE.RAW_TOP_ROOT, raw_front=CYCLE.RAW_FRONT_ROOT, survey=CYCLE.SURVEY_FILE)
    else:
        paths.update(features=CYCLE.INPUT_FEATURE_DB, crops=CYCLE.INPUT_CROP_DB, masks=CYCLE.INPUT_MASK_DB, crop_images=CYCLE.INPUT_CROP_ROOT, crop_completion=CYCLE.PREPROCESSED_INPUT / CYCLE.CROP_FOLDER / 'completion.json', feature_completion=CYCLE.PREPROCESSED_INPUT / CYCLE.CROP_FOLDER / 'feature_completion.json')
    missing = {name: str(path) for (name, path) in paths.items() if not path.exists()}
    source_roots = [CYCLE.PREPROCESSED_INPUT, CYCLE.RAW_TOP_ROOT, CYCLE.RAW_FRONT_ROOT, CYCLE.LABEL_FILE.parent, CYCLE.ARTIFACT_ROOT]
    output = CYCLE.OUTPUT_ROOT.resolve()
    for source in source_roots:
        source = source.resolve()
        if source == output or source in output.parents or output in source.parents:
            problems.append('Use a separate output root: ' + str(output))
    models = check_models()
    data = None
    if not missing and (not problems) and (start == 'models') and (models['status'] == 'READY'):
        adapter = load_module('cycle3_input_check', PROJECT_ROOT / 'model/2. Predict_stage_and_weight.py')
        (inputs, gates, candidates, unmatched, _) = adapter.prepare_data()
        data = dict(targets=len(inputs), gate_available=sum((row['model_eligible'] in (True, 'True', 'true', 1, '1') for row in gates)), identity_review=len(candidates) - len(inputs), unmatched=len(unmatched))
    return dict(status='READY' if not missing and (not problems) and (models['status'] == 'READY') else 'INPUTS_REQUIRED', start=start, code_files_checked=len(python_files), models=models, missing=missing, problems=problems, input_contract=data, writes=0, model_inference=False)

def execute(args):
    """Use one new run directory; pass newly completed A and C runs explicitly."""
    run = args.output.resolve() / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment['LETTUCE_S3_OUTPUT_ROOT'] = str(run)
    environment['LETTUCE_DEVICE'] = args.device
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.start == 'preprocessing':
        environment['LETTUCE_S3_PREPROCESS_ROOT'] = str(run / 'preprocessing')
        environment['LETTUCE_S3_MODEL_INPUT_ROOT'] = str(run / 'preprocessing')
    state = dict(status='RUNNING', start=args.start, stages={}, model_identity=CYCLE.C_MODEL_TAG, environment_protocol=CYCLE.ENVIRONMENT_SCHEMA, model_training=False)
    state_path = run / 'execution.json'

    def save():
        temporary = state_path.with_suffix('.writing')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(state_path)

    def stage(name, relative, arguments):
        print('[Run]', name, flush=True)
        subprocess.run([str(CYCLE.PYTHON), '-B', str(PROJECT_ROOT / relative), *map(str, arguments)], cwd=PROJECT_ROOT, env=environment, check=True)
        state['stages'][name] = {'status': 'COMPLETE'}
        save()

    def completed_pointer(folder, extra=None):
        data = json.loads((folder / (extra or 'latest_completed.json')).read_text(encoding='utf-8-sig'))
        path = Path(data.get('run', data.get('a_run', ''))).resolve()
        if not path.is_relative_to(run) or not (path / 'completion.json').is_file():
            raise ValueError('New completed run is missing or outside this execution: ' + str(path))
        return path
    save()
    try:
        if args.start == 'preprocessing':
            for (name, filename, arguments) in PREPROCESSING:
                stage(name, 'preprocessing/' + filename, arguments)
            stage('survey_candidates', 'model/1. Match_surveys.py', [])
        stage('cnn_and_a', 'model/2. Predict_stage_and_weight.py', ['--stage', 'cnn-a'])
        a_run = completed_pointer(run / 'models/02_A-step', 'evaluation_latest_completed.json')
        state['a_run'] = str(a_run)
        stage('c_history_and_inference', 'model/3. Fuse_growth_history.py', ['--stage', 'c', '--a-run', a_run])
        c_run = completed_pointer(run / 'models/04_C-step')
        state['c_run'] = str(c_run)
        stage('harvest_decision', 'model/4. Decide_harvest.py', ['--c-run', c_run])
        d_run = completed_pointer(run / 'models/05_D-step')
        state['d_run'] = str(d_run)
        source = d_run / 'evaluation/predictions.xlsx'
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, run / 'predictions.xlsx')
        state.update(status='COMPLETE', final_output=str(run / 'predictions.xlsx'))
        save()
        print('Cycle complete:', run / 'predictions.xlsx', flush=True)
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    return run

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', choices=['models', 'preprocessing'], default='models')
    parser.add_argument('--output', type=Path, default=CYCLE.OUTPUT_ROOT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=CYCLE.DEVICE)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-models', action='store_true')
    args = parser.parse_args(argv)
    CYCLE.OUTPUT_ROOT = args.output.resolve()
    if args.check_models:
        report = check_models()
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'READY' else 2
    report = inspect_inputs(args.start)
    print(json.dumps(report, indent=2))
    if report['status'] != 'READY':
        return 2
    if args.check:
        return 0
    execute(args)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
