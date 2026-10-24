"""Reconstruct Table 3 counts and compare saved mask-validation training logs."""
from pathlib import Path
import argparse
import pandas as pd
from table_review import compare,write_review,sha256

DATA=Path(__file__).resolve().parent.parent/'data'

def calculate(data_dir):
    names=['Table_03_crop_inventory.csv','Table_03_paired_inventory.csv']+[
        f'Table_S04_{model}_training_log.csv' for model in ['bed_top','bed_front','lettuce']]
    inputs={data_dir/n:sha256(data_dir/n) for n in names}
    crops=pd.read_csv(data_dir/names[0],dtype=str)
    pairs=pd.read_csv(data_dir/names[1],dtype=str)
    if crops.duplicated(['pair_id','camera']).any() or pairs.pair_id.duplicated().any():
        raise ValueError('Duplicate crop or pair key')
    counts=crops.groupby('pair_id').size()
    if (counts>2).any():raise ValueError('Unexpected camera multiplicity')
    front=set(crops.loc[crops.camera.eq('cam2'),'pair_id'])
    top=set(crops.loc[crops.camera.isin(['cam0','cam1']),'pair_id'])
    if set(pairs.pair_id)!=front&top:raise ValueError('Pair inventory differs from camera intersection')
    start=pd.to_datetime(pairs.capture_date.min(),format='%Y%m%d')
    end=pd.to_datetime(pairs.capture_date.max(),format='%Y%m%d')
    rows=[];details={}
    for model,reported in [('bed_top',(.994,.866)),('bed_front',(.981,.802)),('lettuce',(.902,.728))]:
        logs=pd.read_csv(data_dir/f'Table_S04_{model}_training_log.csv')
        # One row per model, selected independently of manuscript numbers.
        peak=logs.loc[logs.map50_95.idxmax()]
        details[model]=dict(log_peak_epoch=int(peak.epoch),selection='maximum mask mAP50-95',
            any_single_epoch_matches_both=bool(((logs.map50.map(lambda x:f'{x:.3f}')==f'{reported[0]:.3f}')&
                (logs.map50_95.map(lambda x:f'{x:.3f}')==f'{reported[1]:.3f}')).any()))
        for metric,value in zip(['map50','map50_95'],reported):
            compare(rows,model,metric,float(peak[metric]),value,3,'training-log peak; not AP recomputation')
    assigned=int(crops.assignment_status.eq('ASSIGNED').sum())
    operational=dict(assigned=assigned,eligible=len(crops),assigned_percent=100*assigned/len(crops),
        paired=len(pairs),available_front=len(front),paired_percent=100*len(pairs)/len(front),
        plants=len(pairs[['bed_id','physical_slot']].drop_duplicates()),observations=len(pairs),
        calendar_span_days=(end-start).days+1)
    reported=dict(assigned=58018,eligible=64596,assigned_percent=89.8,paired=28953,available_front=32847,
        paired_percent=88.1,plants=858,observations=28953,calendar_span_days=58)
    for key,value in reported.items():compare(rows,'operational',key,operational[key],value,1 if 'percent' in key else 0,'saved crop/pair inventory')
    compare(rows,'OCR','bed_number_accuracy_percent',None,92.6,1,'not located',
        'The 1,000-image OCR ground-truth/prediction evaluation was not located in the inspected handoff.')
    details.update(date_min=start.strftime('%Y-%m-%d'),date_max=end.strftime('%Y-%m-%d'),
        observed_dates=int(pairs.capture_date.nunique()),calendar_span_days=(end-start).days+1,
        longitudinal_identity='unique bed_id + physical_slot in the paired dataset',
        limitations='Training-log comparison does not recompute AP from images/masks. No OCR accuracy reconstruction.')
    return rows,inputs,details

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows,inputs,details=calculate(args.data_dir)
    write_review('Table_03',rows,inputs,args.output_dir,
        'Operational-count reconstruction and training-log comparison; missing evidence remains UNVERIFIED.',details)

if __name__=='__main__':main()
