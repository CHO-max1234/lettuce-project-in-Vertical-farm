"""Run frozen cycle-4 environment, CNN, A and C evaluation in one new run."""
from pathlib import Path
import sys
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT_ROOT),str(PROJECT_ROOT/'pipeline')]
import settings as CYCLE
from runtime import load_module,digest
import argparse
import ast
import json
import os
import shutil
import subprocess
from datetime import datetime

def check_models():
    """Verify the shared artifacts and distinct cycle-4 front model."""
    inspector=load_module('cycle4_model_inspector',CYCLE.MAIN_PIPELINE/'model_contract.py')
    report=inspector.inspect_artifacts(CYCLE.ARTIFACT_ROOT,CYCLE.C_MODEL_TAG)
    policy=CYCLE.LOCAL_ARTIFACT_ROOT/'front_inference_policy.json'
    front_ready=policy.is_file() and CYCLE.BED_FRONT_MODEL.is_file()
    if front_ready:
        front_ready=digest(CYCLE.BED_FRONT_MODEL)==json.loads(policy.read_text(encoding='utf-8'))['model_sha256']
    from runtime import load_legacy_components
    models,hashes=load_legacy_components() if report['status']=='FILES_VERIFIED' else ({},{})
    return dict(status='READY' if report['status']=='FILES_VERIFIED' and front_ready else 'MODELS_REQUIRED',shared=report,cycle4_front_ready=front_ready,feature_counts={k:len(v[0]) for k,v in models.items()},model_identity=CYCLE.C_MODEL_TAG,training=False,model_inference=False)

def check_inputs(start="models"):
    """Check required source paths, labels and disjoint output placement."""
    paths=dict(targets89=CYCLE.TARGET_FILE,labels115=CYCLE.SOURCE115,crops=CYCLE.CROP_ROOT,crop_database=CYCLE.CROP_DB,survey=CYCLE.GROWTH_FILE,environment=CYCLE.ENVIRONMENT_FILE)
    missing={k:str(v) for k,v in paths.items() if not v.exists()}
    problems=[]
    for p in PROJECT_ROOT.rglob('*.py'):
        compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
    out=CYCLE.OUTPUT_ROOT.resolve()
    for source in [CYCLE.SOURCE_ROOT,CYCLE.ARTIFACT_ROOT,CYCLE.TARGET_FILE.parent,CYCLE.SOURCE115.parent,CYCLE.ENVIRONMENT_FILE.parent]:
        source=source.resolve()
        if out==source or source in out.parents or out in source.parents:problems.append('Use a separate output directory: '+str(out))
    if start=='preprocessing':
        extra=dict(reviewed_ocr=CYCLE.OCR_POINTER,base_layout=CYCLE.LAYOUT_BASE_DB,
                   scale_recovery=CYCLE.CALIBRATION_ROOT/'scale_recovery_full.jsonl',
                   body_reviews=CYCLE.SOURCE_ROOT/'6. 시각화와 QC/_실행기록/261011_QC진단/reviewed_case_recommendations.csv')
        missing.update({k:str(v) for k,v in extra.items() if not v.exists()})
    models=check_models()
    counts=None
    if not missing:
        inputs=load_module('cycle4_input_preflight',PROJECT_ROOT/'model/2. Predict_stage.py')
        targets=inputs.read_sheet(CYCLE.TARGET_FILE,'생육매칭')
        history=inputs.read_sheet(CYCLE.TARGET_FILE,'과거Feature')
        all_targets=inputs.read_sheet(CYCLE.SOURCE115,'생육매칭')
        assert len(targets)==89 and len(history)==1313 and len(all_targets)==115
        counts=dict(current=89,history_only=1224,total=1313,front_separate=26)
    return dict(status='READY' if not missing and not problems and models['status']=='READY' else 'INPUTS_REQUIRED',missing=missing,problems=problems,models=models,counts=counts,writes=0,model_inference=False)

def execute(args):
    """Pass every newly completed run explicitly to its dependent stage."""
    run=args.output.resolve()/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['LETTUCE_S4_OUTPUT_ROOT']=str(run);env['LETTUCE_DEVICE']=args.device
    env['PYTHONDONTWRITEBYTECODE']='1'
    state=dict(status='RUNNING',start=args.start,model_identity=CYCLE.C_MODEL_TAG,training=False,stages={})
    def save():
        temporary=run/'execution.writing.json'
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8');temporary.replace(run/'execution.json')
    def stage(name,relative,arguments=()):
        print('[Run]',name,flush=True)
        subprocess.run([str(CYCLE.PYTHON),'-X','utf8','-B',str(PROJECT_ROOT/relative),*map(str,arguments)],cwd=str(PROJECT_ROOT),env=env,check=True)
        state['stages'][name]=dict(status='COMPLETE');save()
    def pointer(relative):
        data=json.loads((run/relative/'latest_completed.json').read_text(encoding='utf-8-sig'))
        path=Path(data['run']).resolve()
        assert path.is_relative_to(run) and (path/'completion.json').is_file(),str(path)
        return path
    save()
    try:
        if args.start=='preprocessing':
            env['LETTUCE_S4_PREPROCESS_ROOT']=str(run/'preprocessing')
            steps=[('reviewed_ocr','2. Load_reviewed_bed_numbers.py'),
                   ('brightness','3. Normalize_brightness.py'),('scale','4. Estimate_scale.py'),
                   ('geometry','5. Refine_slot_geometry.py'),('crops','6. Extract_plant_crops.py'),
                   ('features','7. Extract_features.py'),('calibration','8. Apply_feature_calibration.py')]
            for name,file in steps:stage(name,'preprocessing/'+file)
            state.update(status='PREPROCESSING_COMPLETE_REVIEW_REQUIRED',
                         source_boundary='Reviewed OCR, preserved base layout and review/calibration ledgers',
                         next_step='Review survey matching and supply reviewed labels before model evaluation',
                         models_executed=False)
            save();print('Preprocessing complete; reviewed matching/labels required:',run,flush=True)
            return run
        stage('environment' ,'model/1. Prepare_environment.py')
        environment=pointer('models/00_environment');state['environment_run']=str(environment)
        stage('input_and_cnn','model/2. Predict_stage.py')
        prepared=pointer('models/01_inputs');cnn=pointer('models/02_CNN')
        state.update(prepared_run=str(prepared),cnn_run=str(cnn))
        stage('a_and_evaluation','model/3. Predict_weight.py',['--prepared',prepared,'--cnn-run',cnn,'--environment-run',environment])
        a_run=pointer('models/03_A-step');a_eval=pointer('models/04_A_evaluation')
        state.update(a_run=str(a_run),a_evaluation_run=str(a_eval))
        stage('c_and_evaluation','model/4. Fuse_growth_history.py',['--a-run',a_run,'--prepared',prepared,'--environment-run',environment,'--a-evaluation',a_eval])
        c_run=pointer('models/05_C-step');state['c_run']=str(c_run)
        c_pointer=json.loads((run/'models/05_C-step/latest_completed.json').read_text(encoding='utf-8'))
        evaluation=Path(c_pointer['evaluation']);assert evaluation.resolve().is_relative_to(run)
        shutil.copy2(evaluation/'01_current89_A_C_comparison.csv',run/'predictions_top89.csv')
        if not args.skip_front:
            stage('front_cnn_separate','model/5. Evaluate_front_stage.py')
            front=pointer('models/06_front_CNN');state['front_run']=str(front)
            shutil.copy2(front/'02_front26_predictions_and_truth.csv',run/'predictions_front26.csv')
        state.update(status='COMPLETE',final_output=str(run/'predictions_top89.csv'),d_step='NOT_RUN_NO_LATE_STAGE_TRUTH')
        save();print('Cycle complete:',run,flush=True)
    except BaseException as exc:
        state.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');save();raise
    return run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',choices=['models','preprocessing'],default='models')
    p.add_argument('--check',action='store_true')
    p.add_argument('--check-models',action='store_true')
    p.add_argument('--output',type=Path,default=CYCLE.OUTPUT_ROOT)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default=CYCLE.DEVICE)
    p.add_argument('--skip-front',action='store_true',help='Skip the separate front26 CNN diagnostic')
    args=p.parse_args();CYCLE.OUTPUT_ROOT=args.output.resolve()
    report=check_models() if args.check_models else check_inputs(args.start)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='READY':return 2
    if args.check or args.check_models:return 0
    execute(args);return 0
if __name__=='__main__':raise SystemExit(main())
