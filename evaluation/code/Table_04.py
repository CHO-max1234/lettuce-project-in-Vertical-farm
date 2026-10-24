"""Recalculate Table 4 from paired Hard/Soft saved test probabilities."""
from pathlib import Path
import argparse
import numpy as np
from evaluation_metrics import read_predictions,compare_reported,write_results

BASE=Path(__file__).resolve().parent.parent
INPUT=BASE/'data'/'Table_04_predictions.csv'
OUTPUT=BASE/'reproduced'
# Reference values transcribed from the supplied Table 4; used only for comparison.
REPORTED={
 'Hard':dict(accuracy_percent=80.5,macro_f1=.777,expected_stage_mae=.233,soft_cross_entropy=.710,large_stage_errors=11,large_stage_error_percent=1.6,stage3_recall_percent=36.8,stage4_recall_percent=95.7),
 'Soft':dict(accuracy_percent=79.3,macro_f1=.782,expected_stage_mae=.218,soft_cross_entropy=.594,large_stage_errors=3,large_stage_error_percent=.4,stage3_recall_percent=49.4,stage4_recall_percent=85.1),
}
PRECISION=dict(accuracy_percent=1,macro_f1=3,expected_stage_mae=3,soft_cross_entropy=3,large_stage_errors=0,large_stage_error_percent=1,stage3_recall_percent=1,stage4_recall_percent=1)

def calculate(frame):
    y=frame.human_stage.to_numpy(int)
    if not np.array_equal(y,frame.human_stage) or not np.isin(y,np.arange(5)).all():raise ValueError('Human primary stages must be integers 0–4')
    target=frame[[f'human_p{i}' for i in range(5)]].to_numpy(float)
    if not np.allclose(target.sum(1),1,atol=1e-6) or not ((target>=0)&(target<=1)).all():raise ValueError('Invalid human target probabilities')
    rows=[];matrices={}
    for name in ['Hard','Soft']:
        p=frame[[f'{name.lower()}_p{i}' for i in range(5)]].to_numpy(float)
        if not np.allclose(p.sum(1),1,atol=1e-6) or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid model probabilities')
        pred=p.argmax(1);cm=np.zeros((5,5),dtype=int);np.add.at(cm,(y,pred),1)
        denominator=cm.sum(0)+cm.sum(1)
        f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(5),where=denominator!=0)
        large=int((np.abs(y-pred)>=2).sum())
        rows.append(dict(model=name,n=len(y),beds=frame.bed.nunique(),accuracy_percent=float((y==pred).mean()*100),
            macro_f1=float(f1.mean()),expected_stage_mae=float(np.abs((p-target)@np.arange(5)).mean()),
            soft_cross_entropy=float(-(target*np.log(p.clip(1e-8,1))).sum(1).mean()),
            large_stage_errors=large,large_stage_error_percent=large/len(y)*100,
            stage3_recall_percent=float(cm[3,3]/cm[3].sum()*100),stage4_recall_percent=float(cm[4,4]/cm[4].sum()*100)))
        matrices[name]=cm.tolist()
    return rows,matrices

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    cols=['human_stage']+[f'{prefix}_p{i}' for prefix in ['human','hard','soft'] for i in range(5)]
    frame,digest=read_predictions(args.input,668,cols)
    if frame.bed.nunique()!=19:raise ValueError('Expected the original 19 held-out beds')
    rows,cms=calculate(frame)
    write_results('Table_04',rows,compare_reported(rows,REPORTED,PRECISION),args.input,digest,args.output_dir,
        dict(test_samples=668,held_out_beds=19,class_order=list(range(5)),confusion_matrices=cms,
             expected_stage_definition='Absolute difference between model and human probability-weighted stages',soft_ce_log_floor=1e-8))

if __name__=='__main__':main()
