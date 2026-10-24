"""Recalculate the three single-time-point regression models in Table 5."""
from pathlib import Path
import argparse
from evaluation_metrics import read_predictions,regression_metrics,compare_reported,write_results

BASE=Path(__file__).resolve().parent.parent
INPUT=BASE/'data'/'Table_05_predictions.csv'
OUTPUT=BASE/'reproduced'
MODELS={'Fusion CNN':'pred_cnn_only','Direct CatBoost':'pred_direct_catboost','Residual CatBoost':'pred_residual_catboost'}
REPORTED={'Fusion CNN':dict(mae_g=16.64,r2=.747),'Direct CatBoost':dict(mae_g=18.91,r2=.702),'Residual CatBoost':dict(mae_g=17.16,r2=.741)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    frame,digest=read_predictions(args.input,268,['observed_weight_g']+list(MODELS.values()))
    rows=[dict(model=name,**regression_metrics(frame.observed_weight_g,frame[column])) for name,column in MODELS.items()]
    write_results('Table_05',rows,compare_reported(rows,REPORTED,dict(mae_g=2,r2=3)),args.input,digest,args.output_dir,
        dict(test_samples=268,models=list(MODELS),scope_note='A-step model comparison; not the C-step Direct/Residual models'))

if __name__=='__main__':main()
