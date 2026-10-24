"""Check the four fixed Table 8 cases and reconstruct their projected-area QC.

Uses saved predictions and the original eligible daily area history. No model
inference, image replacement, case reselection or biological growth diagnosis.
"""
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
from table_review import compare,write_review,sha256

DATA=Path(__file__).resolve().parent.parent/'data'
WINDOW_DAYS=7
MIN_VALID_DATES=3
MIN_SPAN_DAYS=2
GOOD_VALID_DATES=5
MAX_GAP_DAYS=3
AREA_THRESHOLD_PCT_DAY=1.0
SLOWING_RATIO=0.5
P4_THRESHOLD=0.30
WEIGHT_THRESHOLD_G=120.0

def window_rate(records):
    if len(records)<MIN_VALID_DATES:return None
    span=(records[-1]['date']-records[0]['date']).days
    if span<MIN_SPAN_DAYS:return None
    first,last=records[0]['area_cm2_top'],records[-1]['area_cm2_top']
    return dict(rate=100*math.expm1(math.log(last/first)/span),span_days=span,valid_dates=len(records))

def classify_history(frame):
    """Keep the stored eligibility and camera/scale/segment boundaries."""
    if frame.duplicated(['lettuce_key','date']).any():raise ValueError('Duplicate track/date')
    results={}
    for track,group in frame.groupby('lettuce_key',sort=False):
        history=[];last=None;signature=None
        for row in group.sort_values('date').to_dict('records'):
            current=dict(growth_qc='Insufficient observations',rate=None,previous_rate=None,valid_dates=0,span_days=None)
            if row['source_qc']!='USABLE':
                current['growth_qc']='Not evaluable'
                results[(track,row['date'])]=current
                continue
            if not math.isfinite(row['area_cm2_top']) or row['area_cm2_top']<=0:
                raise ValueError('Non-positive or missing usable area')
            identity=(row['top_camera'],row['scale_signature'])
            if last is not None and ((row['date']-last).days>MAX_GAP_DAYS or identity!=signature):history=[]
            last=row['date'];signature=identity
            history.append(row)
            history=[r for r in history if (row['date']-r['date']).days<2*WINDOW_DAYS]
            recent=[r for r in history if (row['date']-r['date']).days<WINDOW_DAYS]
            previous=[r for r in history if WINDOW_DAYS<=(row['date']-r['date']).days<2*WINDOW_DAYS]
            current['valid_dates']=len(recent)
            measured=window_rate(recent)
            if measured:
                old=window_rate(previous)
                rate=measured['rate'];prior=old['rate'] if old else None
                current.update(measured,previous_rate=prior)
                if rate < -AREA_THRESHOLD_PCT_DAY:current['growth_qc']='Area decrease'
                elif abs(rate)<=AREA_THRESHOLD_PCT_DAY:
                    if len(recent)>=GOOD_VALID_DATES:current['growth_qc']='Plateau candidate'
                elif prior is not None and prior>AREA_THRESHOLD_PCT_DAY and rate<prior*SLOWING_RATIO:
                    current['growth_qc']='Growth slowing'
                else:current['growth_qc']='Area increase'
            results[(track,row['date'])]=current
    return results

def calculate(data_dir):
    paths=[data_dir/'Table_08_cases.csv',data_dir/'Table_08_daily_history.csv']
    hashes={p:sha256(p) for p in paths}
    cases=pd.read_csv(paths[0]);daily=pd.read_csv(paths[1],parse_dates=['date'],keep_default_na=False)
    daily['area_cm2_top']=pd.to_numeric(daily.area_cm2_top,errors='coerce')
    if len(cases)!=4 or not cases.sample_id.is_unique or set(cases['case'])!={1,2,3,4}:
        raise ValueError('The four archived cases must remain fixed')
    predictions=cases[[f'p{i}' for i in range(5)]].to_numpy(float)
    if not np.isfinite(predictions).all() or (predictions<0).any() or (predictions>1).any():raise ValueError('Invalid probability')
    np.testing.assert_allclose(predictions.sum(axis=1),1,atol=1e-6,rtol=0)
    if not np.array_equal(predictions.argmax(axis=1),cases.saved_pred_stage):raise ValueError('Saved stage differs from probability argmax')
    qc=classify_history(daily)
    refs=[dict(stage=4,p4=.435,predicted_weight_g=135.9,observed_weight_g=138.0,growth_qc='Plateau candidate'),
        dict(stage=4,p4=.385,predicted_weight_g=117.4,observed_weight_g=118.0,growth_qc='Growth slowing'),
        dict(stage=2,p4=.031,predicted_weight_g=157.1,observed_weight_g=170.0,growth_qc='Area increase'),
        dict(stage=2,p4=.041,predicted_weight_g=97.9,observed_weight_g=99.0,growth_qc='Area decrease')]
    comparisons=[];details=[]
    for row in cases.sort_values('case').to_dict('records'):
        cutoff=pd.Timestamp(row['date'])
        if daily.loc[daily.lettuce_key.eq(row['track_id']),'date'].gt(cutoff).any():raise ValueError('History after the fixed case date')
        current=qc[(row['track_id'],cutoff)]
        for key,archived in [('rate','archived_area_rate'),('previous_rate','archived_previous_area_rate'),
            ('valid_dates','archived_valid_dates'),('span_days','archived_span_days')]:
            np.testing.assert_allclose(current[key],row[archived],atol=1e-9,rtol=0)
        if current['growth_qc']!=row['archived_growth_qc']:raise ValueError('QC differs from archive')
        eligible=row['p4']>=P4_THRESHOLD and row['predicted_weight_g']>=WEIGHT_THRESHOLD_G
        if int(eligible)!=row['archived_harvest_decision']:raise ValueError('Harvest rule differs from archive')
        observed=dict(stage=int(row['saved_pred_stage']),p4=row['p4'],predicted_weight_g=row['predicted_weight_g'],
            observed_weight_g=row['observed_weight_g'],growth_qc=current['growth_qc'])
        for key,reference in refs[int(row['case'])-1].items():
            compare(comparisons,f'case_{row["case"]}',key,observed[key],reference,
                dict(stage=0,p4=3,predicted_weight_g=1,observed_weight_g=1).get(key),
                source='saved final predictions; reconstructed daily-area QC')
        details.append(dict(case=int(row['case']),sample_id=row['sample_id'],capture_key=row['capture_key'],
            harvest_eligible=bool(eligible),**observed,**{k:v for k,v in current.items() if k!='growth_qc'}))
    return comparisons,hashes,dict(cases=details,daily_rows=len(daily),
        scope='Four archived operational examples, not an independent held-out performance estimate.',
        case2_stage_discrepancy='Manuscript 4; archived probabilities and final workbook give argmax 3.',
        qc_scope='Supplementary projected-area trend rules; not biological stagnation or measured-weight change.',
        photo_scope='The supplied table image is retained; photos were not searched for, replaced or regenerated.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows,inputs,details=calculate(args.data_dir)
    write_review('Table_08',rows,inputs,args.output_dir,
        'Saved case-value comparison and reconstructed temporal area QC; no training, inference or photo reproduction.',details)

if __name__=='__main__':main()
