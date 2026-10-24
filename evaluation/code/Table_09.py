"""Recalculate Table 9 from saved predictions, keeping each published cohort."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from evaluation_metrics import regression_metrics,binary_metrics
from table_review import compare,write_review,sha256

DATA=Path(__file__).resolve().parent.parent/'data'
P4_THRESHOLD=0.30
WEIGHT_THRESHOLD_G=120.0

def macro_f1(truth,predicted,labels):
    scores=[]
    for label in labels:
        tp=int(((truth==label)&(predicted==label)).sum())
        fp=int(((truth!=label)&(predicted==label)).sum())
        fn=int(((truth==label)&(predicted!=label)).sum())
        denominator=2*tp+fp+fn
        scores.append(2*tp/denominator if denominator else 0.0)
    return float(np.mean(scores))

def external_metrics(frame,labels,expected_targets):
    if len(frame)!=expected_targets or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('Unexpected target count or sample identifiers')
    for column in ['cnn_pred_stage','a_step_weight_g','c_step_weight_g','harvest_p4']:
        if np.isinf(frame[column].to_numpy(float)).any():raise ValueError('Infinite prediction')
    valid=frame[['cnn_pred_stage','a_step_weight_g','c_step_weight_g','harvest_p4']].notna().all(axis=1)
    for column in ['cnn_status','a_step_status','c_step_status']:
        if not np.array_equal(valid,frame[column].eq('OK')):raise ValueError('Availability/status mismatch')
    if frame[['human_stage','observed_weight_g']].isna().any().any():raise ValueError('Missing evaluation target')
    d=frame.loc[valid]
    if not d.harvest_p4.between(0,1).all():raise ValueError('Invalid probability')
    a=regression_metrics(d.observed_weight_g,d.a_step_weight_g)
    c=regression_metrics(d.observed_weight_g,d.c_step_weight_g)
    truth=(d.human_stage==4)&(d.observed_weight_g>=WEIGHT_THRESHOLD_G)
    predicted=(d.harvest_p4>=P4_THRESHOLD)&(d.c_step_weight_g>=WEIGHT_THRESHOLD_G)
    if 'archived_harvest_prediction' in d:
        if not np.array_equal(predicted.astype(int),d.archived_harvest_prediction):
            raise ValueError('Fixed rule differs from archived D decisions')
    harvest=binary_metrics(truth,predicted)
    if harvest['actual_positives']==0:harvest.update(f1=None,recall=None)
    return dict(valid=len(d),target=len(frame),cnn_n=len(d),macro_f1=macro_f1(d.human_stage.to_numpy(),d.cnn_pred_stage.to_numpy(),labels),
        a_mae=a['mae_g'],a_r2=a['r2'],c_mae=c['mae_g'],c_r2=c['r2'],**harvest)

def calculate(data_dir):
    names=['Table_04_predictions.csv','Table_05_predictions.csv','Table_06_predictions.csv','Table_07_predictions.csv',
        'Table_09_cycle2_predictions.csv','Table_09_cycle3_predictions.csv']
    inputs={data_dir/n:sha256(data_dir/n) for n in names}
    cnn,a,c,d,second,third=[pd.read_csv(data_dir/n,dtype={'sample_id':str}) for n in names]
    if len(cnn)!=668 or not cnn.sample_id.is_unique:raise ValueError('Reference CNN cohort mismatch')
    if not (a.sample_id.equals(c.sample_id) and a.sample_id.equals(d.sample_id) and len(a)==268):
        raise ValueError('Reference weight/harvest cohort mismatch')
    np.testing.assert_allclose(a.observed_weight_g,c.observed_weight_g)
    np.testing.assert_allclose(a.observed_weight_g,d.observed_weight_g)
    am=regression_metrics(a.observed_weight_g,a.pred_cnn_only)
    cm=regression_metrics(c.observed_weight_g,c.blend50)
    hm=binary_metrics((d.actual_stage==4)&(d.observed_weight_g>=WEIGHT_THRESHOLD_G),
        (d.cnn_p4>=P4_THRESHOLD)&(d.predicted_weight_g>=WEIGHT_THRESHOLD_G))
    first=dict(valid=268,target=268,cnn_n=668,
        macro_f1=macro_f1(cnn.human_stage.to_numpy(),cnn[[f'soft_p{i}' for i in range(5)]].to_numpy().argmax(axis=1),list(range(5))),
        a_mae=am['mae_g'],a_r2=am['r2'],c_mae=cm['mae_g'],c_r2=cm['r2'],**hm)
    labels2=sorted(set(second.human_stage.dropna().astype(int))|set(second.cnn_pred_stage.dropna().astype(int)))
    metrics=[first,external_metrics(second,labels2,32),external_metrics(third,[0,1,2],89)]
    refs=[dict(valid=268,target=268,macro_f1=.7822,a_mae=16.64,a_r2=.747,c_mae=13.57,c_r2=.836,actual_positives=51,f1=.725,recall=.725),
        dict(valid=31,target=32,macro_f1=.5217,a_mae=19.44,a_r2=.817,c_mae=22.45,c_r2=.740,actual_positives=3,f1=0.,recall=0.),
        dict(valid=85,target=89,macro_f1=.6620,a_mae=23.27,a_r2=-1.411,c_mae=14.28,c_r2=.094,actual_positives=0,f1=None,recall=None,tn=85,fp=0)]
    digits=dict(valid=0,target=0,macro_f1=4,a_mae=2,a_r2=3,c_mae=2,c_r2=3,actual_positives=0,f1=3,recall=1,tn=0,fp=0)
    rows=[]
    for i,(observed,reported) in enumerate(zip(metrics,refs),1):
        for key,reference in reported.items():
            value=observed[key]
            # The manuscript displays recall as a percentage.
            if key=='recall':
                value=None if value is None else 100*value
                reference=None if reference is None else 100*reference
            compare(rows,f'paper_cycle_{i}',key,value,reference,digits[key],source='saved predictions; fixed rule')
    return rows,inputs,dict(metrics=metrics,cnn_class_sets=[list(range(5)),labels2,[0,1,2]],
        crop_to_paper_cycle_mapping='research seasons 2/3/4 correspond to paper cycles 1/2/3',
        reference_cnn_n=668,reference_weight_harvest_n=268,
        cycle3_harvest_scope='Rule recalculation only; no archived D-step execution. Zero actual positives.',
        unavailable_rows_preserved=[0,1,4],p4_threshold=P4_THRESHOLD,weight_threshold_g=WEIGHT_THRESHOLD_G)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows,inputs,details=calculate(args.data_dir)
    write_review('Table_09',rows,inputs,args.output_dir,
        'Saved prediction metrics, not training/inference; paper cycle 3 harvest is a derived fixed-rule check.',details)

if __name__=='__main__':main()
