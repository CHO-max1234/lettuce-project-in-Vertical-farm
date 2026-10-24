"""Observed fresh-weight and CNN-stage distributions; no model fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA_ROOT = Path(r'D:\Googledrive\newfolder\양상추사진\양상추_테라웨이브\전처리,모델 성능개선\260915_0925_5. 분류와 회귀, 수확판단\3. C-step direct&residual regression')
SOURCE = DATA_ROOT / '1. 데이터 생성 및 병합/05. 최종병합 및 검증/C_step_2nd_wide_1288x454.csv'
OUTPUT = DATA_ROOT / '4. 학습,평가,비교,시각화'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', choices=['may04'], default='may04')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced')
    global OUTPUT
    args = parser.parse_args()
    OUTPUT=args.output_dir
    if (OUTPUT/'FigS6.png').exists(): raise FileExistsError('Choose another output directory')
    data = pd.read_csv(SOURCE)
    assert len(data) == 1288
    dates = pd.to_datetime(pd.to_numeric(data['survey_date']).astype('Int64').astype(str), format='%y%m%d')
    frame = data.loc[dates.eq(pd.Timestamp('2026-05-04'))].copy() if args.cohort == 'may04' else data.copy()
    assert frame[['fresh_weight_g', 'cnn_pred_stage']].notna().all().all()
    assert frame.cnn_pred_stage.isin(range(5)).all()
    assert frame.manifest_row_id.is_unique
    if args.cohort == 'may04':
        assert len(frame) == 1143 and frame.stable_plant_key.is_unique
        assert pd.to_numeric(frame.image_date_for_model).eq(260504).all()
    weights = frame.fresh_weight_g.to_numpy(float)
    counts = frame.cnn_pred_stage.value_counts().reindex(range(5), fill_value=0).sort_index()
    assert counts.sum() == len(frame)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.labelsize':11,
                         'axes.linewidth':0.8, 'xtick.labelsize':9, 'ytick.labelsize':9,
                         'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 3.9))
    fig.subplots_adjust(left=.075, right=.982, bottom=.18, top=.89, wspace=.29)
    for label, ax in zip(['(a)', '(b)'], axes):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#596570')
        ax.tick_params(length=3, color='#596570')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E7ECEF', linewidth=.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.text(-.13, 1.055, label, transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of plants' if args.cohort == 'may04' else 'Number of observations')
    ax = axes[0]
    edges = np.arange(np.floor(weights.min()/10)*10, np.ceil(weights.max()/10)*10+10, 10)
    hist, _, _ = ax.hist(weights, bins=edges, color='#468B93', edgecolor='white', linewidth=.65)
    assert int(hist.sum()) == len(frame)
    ax.axvline(120, color='#BD7447', linewidth=1.5, linestyle=(0, (4, 3)), label='120 g reference')
    ax.legend(frameon=False, loc='upper right', fontsize=9, handlelength=2.5)
    ax.set_xlabel('Observed fresh weight (g)')
    ax.set_xlim(edges[0]-5, edges[-1]+5)
    ax.set_ylim(0, hist.max()*1.22)
    ax.text(.98, .83, f'$n$ = {len(frame):,}', transform=ax.transAxes, ha='right', color='#47525C', fontsize=9)
    ax = axes[1]
    colors = ['#DCE5E9','#BDD4D9','#8CB9BC','#4F9299','#23646F']
    bars = ax.bar(np.arange(5), counts.to_numpy(), width=.63, color=colors, edgecolor='white', linewidth=.7)
    ax.set_xticks(range(5))
    ax.set_xlabel('CNN-predicted stage')
    ax.set_ylim(0, counts.max()*1.25)
    for bar, count in zip(bars, counts):
        ax.annotate(f'{count:,}\n({count/len(frame):.1%})', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=9, color='#283D47')
    stem = 'FigS6'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f'{stem}.png', dpi=600)
    plt.close(fig)
    audit = OUTPUT / '_검증자료' / stem
    audit.mkdir(parents=True, exist_ok=True)
    frame[['manifest_row_id','stable_plant_key','survey_date','image_date_for_model','split','fresh_weight_g','cnn_pred_stage']].to_csv(audit/'plotted_rows.csv', index=False, encoding='utf-8-sig')
    report = {'source':str(SOURCE), 'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'cohort':args.cohort,'n':len(frame),'unique_plants':int(frame.stable_plant_key.nunique()),
              'stage_counts':{str(k):int(v) for k,v in counts.items()},'histogram_bin_width_g':10,
              'weight_summary':frame.fresh_weight_g.describe().to_dict(),'splits':frame.split.value_counts().to_dict(),
              'n_at_least_120g':int((weights>=120).sum()), 'status':'PASS'}
    (audit/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cohort = '1,143 plants measured on 4 May 2026 (59 days after transplanting)' if args.cohort=='may04' else 'all 1,288 observations across survey dates'
    note = f'''# Figure caption\n\nDistributions of observed fresh weight and CNN-predicted stage for {cohort}. (a) Observed fresh-weight distribution in 10-g bins; the dashed line marks a 120-g reference. (b) CNN-predicted stage frequencies for the same observations. Labels show counts and percentages. All training, validation, and test subsets are pooled for this descriptive figure. Stage assignments are model predictions, not independently annotated developmental stages.\n\n# 생성 및 해석\n\n- 원본 1288행에서 선택한 범위만 사용하며 두 패널의 표본이 동일함.\n- 실측 생체중과 cnn_pred_stage를 사용. 결측 보간, 이상치 제거, 재학습 없음.\n- 120 g은 첨부 구성의 참고선이며 이 그림으로 수확 기준의 타당성을 검증한 것은 아님.\n- CNN 단계 분포는 실측 결구 단계 분포로 표현하지 않음.\n- 전체 제목은 그림 안에 넣지 않음. 영어 표기, PNG 600 dpi, PDF 미생성.\n- 개체별 원자료와 수치 검증은 _검증자료/{stem}에 저장.\n'''
    (OUTPUT/f'{stem}_caption.md').write_text(note,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__ == '__main__':
    main()
