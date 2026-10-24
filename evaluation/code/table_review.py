"""Comparison outputs for saved predictions, operational counts and training logs."""
from pathlib import Path
import hashlib
import json
import pandas as pd

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare(rows,group,metric,observed,reported,decimals=None,source='',note=''):
    def display(value):
        if value is None:return 'N/A'
        return f'{value:.{decimals}f}' if decimals is not None else str(value)
    a,b=display(observed),display(reported)
    status='UNVERIFIED' if observed is None and reported is not None else ('MATCH' if a==b else 'MISMATCH')
    rows.append(dict(group=group,metric=metric,observed=observed,reported=reported,
        observed_display=a,reported_display=b,status=status,source=source,note=note))

def write_review(table,rows,inputs,output_dir,scope,details=None):
    output_dir=Path(output_dir)
    targets=[output_dir/f'{table}_comparison.csv',output_dir/f'{table}_verification.json']
    if any(p.exists() for p in targets):raise FileExistsError('Choose a new --output-dir')
    for path,digest in inputs.items():
        if sha256(path)!=digest:raise RuntimeError(f'Input changed: {path}')
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(targets[0],index=False,float_format='%.17g',encoding='utf-8-sig')
    counts={s:int(frame.status.eq(s).sum()) for s in ['MATCH','MISMATCH','UNVERIFIED']}
    report=dict(table=table,status='MATCH_AT_REPORTED_PRECISION' if counts['MATCH']==len(rows) else 'PARTIAL_WITH_LIMITATIONS',
        compared_items=len(rows),counts=counts,scope=scope,model_training=False,model_inference=False,
        source_files_unchanged=True,inputs={str(p):v for p,v in inputs.items()},details=details or {},
        comparison_sha256=sha256(targets[0]))
    targets[1].write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{table}: {report["status"]}; {counts}')
    return report
