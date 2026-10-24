"""Plot the saved Direct stacking importance_mean, without refitting."""
from pathlib import Path
import hashlib
import json
import textwrap
import openpyxl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(r'D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\3. C-step direct&residual regression\4. 학습,평가,비교,시각화')
SOURCE = OUTPUT / 'C_step_results_current_and_historical.xlsx'
STEM = 'Figure 11'

def main():
    global OUTPUT
    OUTPUT = output_directory()
    book = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
    rows = list(book['Feature_importance'].iter_rows(values_only=True))
    header_index = next(i for i,r in enumerate(rows) if r[:4] == ('feature','importance_mean','importance_sd','component'))
    selected = [dict(zip(rows[header_index], r)) for r in rows[header_index+1:] if len(r)>3 and r[3]=='direct_stacking']
    book.close()
    assert len(selected)==311 and len({r['feature'] for r in selected})==311
    assert np.isclose(sum(r['importance_mean'] for r in selected),100)
    top = sorted(selected,key=lambda r:r['importance_mean'],reverse=True)[:13]
    values = [r['importance_mean'] for r in top]
    labels = ['\n'.join(textwrap.wrap(r['feature'].replace('_',' '),width=18,break_long_words=False)) for r in top]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':12,'axes.linewidth':.8})
    fig,ax=plt.subplots(figsize=(13,5.7))
    fig.subplots_adjust(left=.065,right=.985,bottom=.27,top=.95)
    x=np.arange(13)
    bars=ax.bar(x,values,width=.7,color='#43858F',edgecolor='white',linewidth=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#68757D')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True,color='#E6EBEE',linewidth=.65)
    ax.set_xticks(x,labels,rotation=48,ha='right',rotation_mode='anchor',fontsize=9)
    ax.tick_params(axis='x',length=0,pad=8)
    ax.set_ylabel('Mean feature importance (%)')
    ax.set_ylim(0,max(values)*1.14)
    ax.set_xlim(-.65,12.65)
    for bar,val in zip(bars,values):
        ax.annotate(f'{val:.2f}',(bar.get_x()+bar.get_width()/2,val),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=9,color='#304750')
    fig.savefig(OUTPUT/f'{STEM}.png',dpi=600)
    plt.close(fig)
    audit=OUTPUT/'_검증자료'/STEM
    audit.mkdir(parents=True,exist_ok=True)
    report={'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'sheet':'Feature_importance','component':'direct_stacking','source_feature_count':len(selected),'top13':top,'status':'PASS'}
    (audit/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUTPUT/f'{STEM}_caption.md').write_text('''**Figure X. The 13 highest-ranked features in the Direct stacking model.** Bars show mean CatBoost feature importance across three random seeds, ranked in descending order. Values are taken from the saved importance_mean column and retain the full-model normalization; the displayed subset is not renormalized. Tick labels replace underscores with spaces for readability. These importances describe the Direct stacking component of the selected 50:50 blend, not the blended predictor as a whole, and do not establish causal effects.\n''',encoding='utf-8')
    print(json.dumps({'status':'PASS','figure':str(OUTPUT/f'{STEM}.png'),'top13_sum':sum(values)},ensure_ascii=False))

OUTPUT_NAME = 'Figure 11.png'

def output_directory():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'reproduced')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / OUTPUT_NAME).exists():
        raise FileExistsError('Choose another --output-dir to preserve the existing figure')
    return args.output_dir

if __name__=='__main__':
    main()
