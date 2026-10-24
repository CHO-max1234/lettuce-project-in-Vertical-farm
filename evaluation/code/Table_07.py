"""Recalculate four fixed harvest rules on the saved Table 7 test set."""
from pathlib import Path
import argparse
import numpy as np
from evaluation_metrics import read_predictions,binary_metrics,compare_reported,write_results

BASE=Path(__file__).resolve().parent.parent
INPUT=BASE/'data'/'Table_07_predictions.csv'
OUTPUT=BASE/'reproduced'
P4_THRESHOLD=.30
WEIGHT_THRESHOLD_G=120
REPORTED={
 'Stage probability':dict(precision=.462,recall=.824,f1=.592),
 'Predicted weight':dict(precision=.358,recall=.863,f1=.506),
 'Argmax stage AND weight':dict(precision=.744,recall=.569,f1=.644),
 'P4 AND weight':dict(precision=.725,recall=.725,f1=.725),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    frame,digest=read_predictions(args.input,268,['actual_stage','observed_weight_g','cnn_pred_stage','cnn_p4','predicted_weight_g'])
    for c in ['actual_stage','cnn_pred_stage']:
        v=frame[c].to_numpy(float)
        if not np.equal(v,np.floor(v)).all() or not np.isin(v,np.arange(5)).all():raise ValueError('Stages must be integers 0–4')
    if not frame.cnn_p4.between(0,1).all():raise ValueError('Invalid p4')
    truth=frame.actual_stage.eq(4)&frame.observed_weight_g.ge(WEIGHT_THRESHOLD_G)
    if int(truth.sum())!=51:raise ValueError('Original test positive count changed')
    rules={
        'Stage probability':frame.cnn_p4.ge(P4_THRESHOLD),
        'Predicted weight':frame.predicted_weight_g.ge(WEIGHT_THRESHOLD_G),
        'Argmax stage AND weight':frame.cnn_pred_stage.eq(4)&frame.predicted_weight_g.ge(WEIGHT_THRESHOLD_G),
        'P4 AND weight':frame.cnn_p4.ge(P4_THRESHOLD)&frame.predicted_weight_g.ge(WEIGHT_THRESHOLD_G),
    }
    rows=[dict(model=name,**binary_metrics(truth,pred)) for name,pred in rules.items()]
    final=rows[-1]
    if [final[k] for k in ['tp','tn','fp','fn']]!=[37,203,14,14]:raise ValueError('Original combined confusion counts differ')
    write_results('Table_07',rows,compare_reported(rows,REPORTED,dict(precision=3,recall=3,f1=3)),args.input,digest,args.output_dir,
        dict(test_samples=268,actual_positives=51,p4_threshold=P4_THRESHOLD,weight_threshold_g=WEIGHT_THRESHOLD_G,
             truth_definition='human primary stage == 4 AND observed weight >= 120 g',threshold_selection='Previously selected on validation; no retuning',growth_qc_used_in_rule=False))

if __name__=='__main__':main()
