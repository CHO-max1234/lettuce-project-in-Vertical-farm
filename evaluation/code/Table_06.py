"""Recalculate Table 6 from aligned saved baseline and C-step predictions."""
from pathlib import Path
import argparse
import numpy as np
from evaluation_metrics import read_predictions,regression_metrics,compare_reported,write_results

BASE=Path(__file__).resolve().parent.parent
INPUT=BASE/'data'/'Table_06_predictions.csv'
OUTPUT=BASE/'reproduced'
MODELS={'Baseline CNN':'baseline_cnn','Direct':'direct','Direct stacking':'direct_stacking','Residual':'residual','50:50 blend':'blend50'}
REPORTED={
 'Baseline CNN':dict(mae_g=16.64,rmse_g=24.00,r2=.747),
 'Direct':dict(mae_g=13.99,rmse_g=20.06,r2=.823),
 'Direct stacking':dict(mae_g=14.02,rmse_g=19.98,r2=.825),
 'Residual':dict(mae_g=14.00,rmse_g=19.51,r2=.833),
 '50:50 blend':dict(mae_g=13.57,rmse_g=19.32,r2=.836),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=INPUT);parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args()
    frame,digest=read_predictions(args.input,268,['observed_weight_g']+list(MODELS.values()))
    np.testing.assert_allclose(frame.blend50,.5*frame.direct_stacking+.5*frame.residual,rtol=0,atol=1e-10)
    rows=[dict(model=name,**regression_metrics(frame.observed_weight_g,frame[column])) for name,column in MODELS.items()]
    write_results('Table_06',rows,compare_reported(rows,REPORTED,dict(mae_g=2,rmse_g=2,r2=3)),args.input,digest,args.output_dir,
        dict(test_samples=268,blend_formula_verified=True,blend_formula='0.5 * Direct stacking + 0.5 * Residual',test_used_for_model_selection=False))

if __name__=='__main__':main()
