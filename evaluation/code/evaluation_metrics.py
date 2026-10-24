"""Metrics and output checks for saved-prediction table reproduction."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_predictions(path, expected_rows, numeric_columns):
    before=sha256(path)
    frame=pd.read_csv(path,dtype={'sample_id':str})
    if len(frame)!=expected_rows or not frame.sample_id.is_unique:
        raise ValueError('Unexpected sample count or duplicate sample_id')
    if frame.sample_id.isna().any():raise ValueError('Missing sample_id')
    values=frame[numeric_columns].to_numpy(float)
    if not np.isfinite(values).all():raise ValueError('Non-finite evaluation input')
    return frame,before

def regression_metrics(observed,predicted):
    observed=np.asarray(observed,dtype=float);predicted=np.asarray(predicted,dtype=float)
    error=predicted-observed
    denominator=float(np.square(observed-observed.mean()).sum())
    return dict(n=len(observed),mae_g=float(np.abs(error).mean()),
        rmse_g=float(np.sqrt(np.square(error).mean())),
        r2=float(1-np.square(error).sum()/denominator) if denominator>0 else None,
        bias_g=float(error.mean()))

def binary_metrics(truth,predicted):
    truth=np.asarray(truth,dtype=bool);predicted=np.asarray(predicted,dtype=bool)
    tp=int((truth&predicted).sum());tn=int((~truth&~predicted).sum())
    fp=int((~truth&predicted).sum());fn=int((truth&~predicted).sum())
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    return dict(n=len(truth),tp=tp,tn=tn,fp=fp,fn=fn,actual_positives=tp+fn,
        predicted_positives=tp+fp,precision=precision,recall=recall,f1=f1,
        accuracy=(tp+tn)/len(truth),balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None)

def compare_reported(rows,reported,precisions):
    comparisons=[]
    for row in rows:
        for metric,reference in reported[row['model']].items():
            actual=row[metric];digits=precisions[metric]
            displayed='N/A' if actual is None else f'{actual:.{digits}f}'
            expected='N/A' if reference is None else f'{reference:.{digits}f}'
            comparisons.append(dict(model=row['model'],metric=metric,recomputed=actual,
                reported=reference,decimals=digits,recomputed_display=displayed,
                reported_display=expected,matches=displayed==expected))
    return pd.DataFrame(comparisons)

def write_results(table,rows,comparison,input_path,input_hash,output_dir,details):
    output_dir=Path(output_dir)
    files=[output_dir/f'{table}_metrics.csv',output_dir/f'{table}_comparison.csv',output_dir/f'{table}_verification.json']
    if any(p.exists() for p in files):raise FileExistsError('Output exists: choose another --output-dir')
    output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(files[0],index=False,float_format='%.17g',encoding='utf-8-sig')
    comparison.to_csv(files[1],index=False,float_format='%.17g',encoding='utf-8-sig')
    if sha256(input_path)!=input_hash:raise RuntimeError('Input changed during calculation')
    matches=bool(comparison.matches.all())
    report=dict(table=table,status='MATCH_AT_REPORTED_PRECISION' if matches else 'REPORTED_VALUE_MISMATCH',
        compared_cells=len(comparison),matched_cells=int(comparison.matches.sum()),
        input_file=str(Path(input_path).resolve()),input_sha256=input_hash,input_unchanged=True,
        model_training=False,model_inference=False,predictions_modified=False,excluded_rows=0,
        verification_scope='Metrics recalculated from saved held-out predictions; not training reproduction',
        output_hashes={p.name:sha256(p) for p in files[:2]},**details)
    files[2].write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'{table}: {report["status"]}; {report["matched_cells"]}/{report["compared_cells"]} reported cells match')
    if not matches:raise SystemExit(2)
    return report
