"""Reproduce manuscript Figure 9 from saved CNN results; no training."""
from pathlib import Path

ROOT=Path(r'D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\1. 생육 단계 CNN-Soft label\5. CNN 재학습')

OUT=ROOT/'논문에 넣을그림'

import sys,json,hashlib,shutil

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib import font_manager

from matplotlib.patches import Patch

from PIL import Image

FINAL=ROOT/'03. 최종 비교와 결론'

BLUE='#0072B2'

ORANGE='#D55E00'

MM=1/25.4

FONT='Arial' if any(x.name=='Arial' for x in font_manager.fontManager.ttflist) else 'Liberation Sans'

plt.rcParams.update({'font.family':FONT,'font.size':9,'axes.labelsize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.linewidth':.8,'savefig.facecolor':'white'})

SOURCES={}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    SOURCES[str(p)]=digest(p)
    return pd.read_csv(p)

def csv(name,d):d.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def save(fig,name):
    fig.savefig(OUT/OUTPUT_NAME,dpi=600)
    plt.close(fig)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e5e5e5',linewidth=.5,zorder=0)
    ax.set_axisbelow(True);ax.tick_params(width=.7,length=3)

def tag(fig,ax,letter):
    b=ax.get_position();fig.text(b.x0-.035,b.y1+.018,f'({letter})',weight='bold',fontsize=11.5,ha='left',va='bottom')

def metrics(d):
    y=d.stage.to_numpy(int);p=d[[f'p{i}' for i in range(5)]].to_numpy();t=d[[f'soft_{i}' for i in range(5)]].to_numpy()
    pred=p.argmax(1);cm=np.zeros((5,5),dtype=int);np.add.at(cm,(y,pred),1)
    den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(5),where=den!=0)
    return dict(accuracy=float((y==pred).mean()),macro_f1=float(f1.mean()),expected_mae=float(np.abs((p-t)@np.arange(5)).mean()),
      soft_ce=float(-(t*np.log(p.clip(1e-8,1))).sum(1).mean()),large_errors=int((np.abs(y-pred)>=2).sum()),
      stage3_recall=float(cm[3,3]/cm[3].sum()),stage4_recall=float(cm[4,4]/cm[4].sum())),cm

def load_data():
    h=read(FINAL/'hard_final/test_predictions.csv');s=read(FINAL/'soft_final/test_predictions.csv')
    assert h.uid.is_unique and s.uid.is_unique and set(h.uid)==set(s.uid)
    h=h.set_index('uid').sort_index();s=s.set_index('uid').sort_index()
    for c in ['stage','camera','bed','image_name',*[f'soft_{i}' for i in range(5)]]:assert h[c].equals(s[c]),c
    assert len(h)==668 and h.bed.nunique()==19
    t=s[[f'soft_{i}' for i in range(5)]].to_numpy();one=np.eye(5)[t.argmax(1)]
    pure=np.isclose(t,one,atol=1e-3,rtol=0).all(1)
    assert np.array_equal(pure,t.max(1)>=.999) and int(pure.sum())==340
    h['label_type']=s['label_type']=np.where(pure,'pure','boundary')
    results={};cms={};stored=read(FINAL/'hard_soft_locked_test_comparison.csv').set_index('model')
    for name,d in [('hard',h),('soft',s)]:
        results[name],cms[name]=metrics(d)
        assert np.allclose(d[[f'p{i}' for i in range(5)]].sum(1),1,atol=1e-6)
        assert np.array_equal(d.pred_stage,d[[f'p{i}' for i in range(5)]].to_numpy().argmax(1))
        for key,value in results[name].items():
            if key!='large_errors':assert np.isclose(value,stored.loc[name,key],atol=1e-7),key
        source_cm=read(FINAL/f'{name}_final/test_confusion.csv')[[f'pred_{i}' for i in range(5)]].to_numpy()
        assert np.array_equal(cms[name],source_cm)
    return h,s,results,cms

def figure1(cms):
    fig=plt.figure(figsize=(180*MM,87*MM));gs=fig.add_gridspec(1,3,width_ratios=[1,1,.055],left=.083,right=.925,bottom=.18,top=.91,wspace=.23)
    for n,(name,cm) in enumerate(cms.items()):
        ax=fig.add_subplot(gs[0,n]);z=cm/cm.sum(1,keepdims=True)*100
        im=ax.imshow(z,cmap='Blues',vmin=0,vmax=100)
        for i in range(5):
            for j in range(5):ax.text(j,i,'0' if cm[i,j]==0 else f'{cm[i,j]}\n{z[i,j]:.1f}%',ha='center',va='center',fontsize=8,color='white' if z[i,j]>=55 else '#17212b',linespacing=1.25)
        ax.set(xticks=range(5),yticks=range(5),xlabel='Predicted stage',ylabel='Human-assigned stage' if n==0 else '')
        ax.tick_params(length=0,pad=5);ax.spines[:].set_visible(False)
        ax.text(0,1.055,f'({chr(97+n)})',transform=ax.transAxes,weight='bold',fontsize=11.5)
        ax.text(.5,1.055,name.title(),transform=ax.transAxes,ha='center',fontsize=10)
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[0,2]),ticks=[0,25,50,75,100]);cb.set_label('Within-stage proportion (%)',fontsize=9);cb.outline.set_visible(False)
    save(fig,'Fig1_confusion')

OUTPUT_NAME = 'Figure 9.png'

def output_directory():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'reproduced')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / OUTPUT_NAME).exists():
        raise FileExistsError('Choose another --output-dir to preserve the existing figure')
    return args.output_dir

def main():
    global OUT
    OUT = output_directory()
    h,s,results,cms=load_data(); figure1(cms)
    assert all(digest(Path(p)) == h for p,h in SOURCES.items()), 'Source changed'
    (OUT / (Path(OUTPUT_NAME).stem + '.json')).write_text(json.dumps(dict(status='REPRODUCED', source_hashes=SOURCES, model_inference=False, output_sha256=digest(OUT/OUTPUT_NAME)),indent=2),encoding='utf-8')
    print('REPRODUCED', OUT/OUTPUT_NAME)

if __name__ == '__main__':
    main()
