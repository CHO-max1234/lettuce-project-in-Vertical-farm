"""Reproduce manuscript Figure S1 from saved CNN results; no training."""
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

def figure_s1():
    histories={};colors=['#0072B2','#D55E00','#009E73','#CC79A7','#555555'];styles=['-','--','-.',':',(0,(5,2))]
    for name,folder in [('Hard',ROOT/'01. Hard'),('Soft',ROOT/'02. Soft/camera_balanced')]:histories[name]=[read(folder/f'fold_{f}/history.csv') for f in range(1,6)]
    fig,axs=plt.subplots(2,3,figsize=(180*MM,135*MM));fig.subplots_adjust(left=.11,right=.985,top=.94,bottom=.15,wspace=.47,hspace=.55)
    for r,(name,frames) in enumerate(histories.items()):
        for c,(key,label) in enumerate(zip(['train_loss','macro_f1','expected_mae'],['Training loss','Validation macro-F1','Validation expected-stage MAE'])):
            ax=axs[r,c]
            for f,h in enumerate(frames):ax.plot(h.epoch,h[key],color=colors[f],linestyle=styles[f],lw=1.1,label=f'Fold {f+1}')
            ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.set_xlim(0,30 if r==0 else 20);ax.set_xticks([0,10,20,30] if r==0 else [0,5,10,15,20]);clean(ax)
            if c==1:
                lo=min(x.macro_f1.min() for y in histories.values() for x in y)
                ax.set_ylim(min(.45,lo-.03),.85)
            elif c==2:
                hi=max(x.expected_mae.max() for y in histories.values() for x in y);ax.set_ylim(.15,np.ceil(hi*10)/10)
            else:ax.set_ylim(bottom=0)
            tag(fig,ax,chr(97+r*3+c))
        fig.text(.014,.745 if r==0 else .34,name,rotation=90,ha='center',va='center',fontsize=10,weight='bold')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),ncol=5,frameon=False)
    save(fig,'FigS1_learning_curves')

OUTPUT_NAME = 'FigS1_learning_curves.png'

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
    figure_s1()
    assert all(digest(Path(p)) == h for p,h in SOURCES.items()), 'Source changed'
    (OUT / (Path(OUTPUT_NAME).stem + '.json')).write_text(json.dumps(dict(status='REPRODUCED', source_hashes=SOURCES, model_inference=False, output_sha256=digest(OUT/OUTPUT_NAME)),indent=2),encoding='utf-8')
    print('REPRODUCED', OUT/OUTPUT_NAME)

if __name__ == '__main__':
    main()
