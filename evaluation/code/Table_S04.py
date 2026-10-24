"""Compare Table S4 with archived configurations, split inventory and mask logs."""
from pathlib import Path
import argparse
import json
import pandas as pd
from table_review import compare,write_review,sha256

DATA=Path(__file__).resolve().parent.parent/'data'
MODELS=['bed_top','bed_front','lettuce']
COMMON=dict(pretrained=True,optimizer='auto',lrf=.01,momentum=.937,weight_decay=.0005,
    cos_lr=True,warmup_epochs=3.,amp=True,deterministic=True,overlap_mask=True,mask_ratio=4)
SPECIFIC=[dict(imgsz=1024,epochs=120,patience=25,batch=4,seed=260823,lr0=.01,degrees=1.5,translate=.02),
    dict(imgsz=1024,epochs=120,patience=25,batch=4,seed=260825,lr0=.01,degrees=1.,translate=.02),
    dict(imgsz=960,epochs=80,patience=20,batch=8,seed=42,lr0=.001,degrees=5.,translate=.05)]

def calculate(data_dir):
    names=['Table_S04_training_metadata.json','Table_S04_lettuce_split_inventory.csv']+[
        f'Table_S04_{model}_training_log.csv' for model in MODELS]
    inputs={data_dir/n:sha256(data_dir/n) for n in names}
    metadata=json.loads((data_dir/names[0]).read_text(encoding='utf-8'))
    split=pd.read_csv(data_dir/names[1])
    if not split.image_name.is_unique or not split.label_present.all():raise ValueError('Split inventory inconsistency')
    rows=[];details={}
    for i,model in enumerate(MODELS):
        saved=metadata[model]
        compare(rows,model,'architecture',saved['architecture'],['yolo11s-seg.yaml','yolo11s-seg.yaml','yolov8m-seg.yaml'][i],source='retained checkpoint metadata')
        compare(rows,model,'class',saved['class_names']['0'],['bed_roi','bed_roi','lettuce'][i],source='retained checkpoint metadata')
        for key,value in {**COMMON,**SPECIFIC[i]}.items():
            compare(rows,model,key,saved['config'][key],value,source='archived args.yaml')
        for key,value in zip(['unique_source_images','train','val','test'],[[1000,790,210,None],[1100,879,221,None],[712,581,68,63]][i]):
            actual=None if model!='lettuce' else (int(split.source_id.nunique()) if key=='unique_source_images' else int(split.split.eq(key).sum()))
            compare(rows,model,key,actual,value,source='file inventory' if model=='lettuce' else 'split manifest not located',
                note='Unique original source paths from copy manifest; 712 output files include 90 repeated copies. Pixel-level identity was not tested.' if model=='lettuce' and key=='unique_source_images' else '')
            if model!='lettuce' and key=='test':rows[-1]['status']='UNVERIFIED';rows[-1]['note']='A dash in the manuscript is not evidence of an empty test split.'
        compare(rows,model,'split_strategy',None,['CAM0/1 pooled','Grouped by bed + date','Existing split + new data split'][i],
            source='historical split procedure not re-executed',note='Saved counts do not independently establish the split construction rule.')
        logs=pd.read_csv(data_dir/f'Table_S04_{model}_training_log.csv')
        peak=logs.loc[logs.map50_95.idxmax()]
        refs=[[.995,.990,.994,.866],[.981,.981,.986,.802],[.828,.931,.902,.728]][i]
        for metric,reference in zip(['precision','recall','map50','map50_95'],refs):
            compare(rows,model,metric,float(peak[metric]),reference,3,'training-log peak; maximum mask mAP50-95')
        details[model]=dict(log_peak_epoch=int(peak.epoch),logged_epochs=len(logs),
            retained_checkpoint_metrics=saved['retained_checkpoint']['train_metrics'],
            any_single_epoch_matches_all_four=bool(pd.concat([logs[col].map(lambda x:f'{x:.3f}').eq(f'{ref:.3f}')
                for col,ref in zip(['precision','recall','map50','map50_95'],refs)],axis=1).all(axis=1).any()))
    details['limitations']=['Configured maximum epochs differ from completed/logged epochs.',
        'Pretrained=True in args does not verify the identity of the initialization checkpoint.',
        'Lettuce split has 712 output files but 622 unique original source paths; 90 repeated copies. Pixel-level identity was not tested.',
        'Retained checkpoint metadata and training-log peaks disagree for front bed and lettuce; no source was altered.',
        'No model training or image-based AP evaluation was performed.']
    return rows,inputs,details

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows,inputs,details=calculate(args.data_dir)
    write_review('Table_S04',rows,inputs,args.output_dir,
        'Configuration/log comparison and split filename counts; no training or fresh AP evaluation.',details)

if __name__=='__main__':main()
