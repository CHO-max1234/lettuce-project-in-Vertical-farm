"""완료된 4작기 QC를 FigS11 형식의 (a)–(e) PNG로 재배치한다.

QC/Feature 재계산 없음. (a) 기존 안정 예시5개체, (b) 전체 윗면89개체.
주어진 사진3장은 서로 다른 개체이며 동일 개체의 생장 시퀀스로 해석하지 않는다.
사진3장 모두 원본 QC의 정확한 record_id와 연결하여 계산된 판정을 표시한다.
(c)는 7월26일 S057, (d)는 8월5일 S064, (e)는 기존 S098을 유지한다. 캡션은 PNG에 넣지 않는다.
"""
from pathlib import Path
ROOT=Path(r'E:\newfolder_양상추\4작기 실증 적용\6. 시각화와 QC')
QC=ROOT/'생육 QC'
SOURCE_CSV=QC/'growth_qc.csv'
SELECTION=QC/'_근거'/'selected_examples.csv'
FEATURE_DB=ROOT/'Figures_v2.2_159행'/'_실행기록'/'feature_v2_2.sqlite3'
STEM='FigS11_mm2 ver'
OUTPUT_DIR=Path(__file__).resolve().parents[1]/'reproduced'
PHOTO_INPUTS=[
    (Path('E:\\newfolder_양상추\\4작기 실증 적용\\5. bbox to crop, feature\\crops\\cam0\\260726\\bed17_20260726_024711_cam0_rf__b1.png'), '0436c5729044a556aa8df4ed'),
    (Path('E:\\newfolder_양상추\\4작기 실증 적용\\5. bbox to crop, feature\\crops\\cam0\\260805\\bed19_20260805_021723_cam0_rf__t4.png'), 'ac2e8d6e2a2339a47909d40c'),
    (ROOT/'matching_crops_115'/'lettuce-crops'/'S098_260818_bed27_b3_top_cam0.png', '1fda70fa4727ac9346033ee4'),
]
WIDTH=7.2;HEIGHT=5.9;DPI=600;MAX_GAP=3
STATES=['GROWTH_OBSERVED','GROWTH_SLOWING','PLATEAU_CANDIDATE','DECREASE_OBSERVED']
STATE_LABELS=['Growth observed','Growth slowing','Plateau candidate','Decrease observed']
STATE_COLORS=['#258A58','#D28C10','#8E65AC','#C95145']
import sys,json,hashlib,sqlite3,math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
sys.dont_write_bytecode=True

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_inputs():
    """완료 QC와 기존 v2.2 Feature에서 실제 사진과 정확한 record를 연결한다."""
    paths=[SOURCE_CSV,QC/'growth_qc.xlsx',QC/'growth_qc_completion.json',SELECTION]+[QC/n for n in ['01_growth_transition_top.png','02_beddate_bedslot_consistency.png','03_growth_qc_calendar.png','04_size_and_appearance.png']]
    protected={str(p):sha(p) for p in paths}
    d=pd.read_csv(SOURCE_CSV);d['date']=pd.to_datetime(d.date)
    top=d.loc[d.view.eq('top')].sort_values(['item_id','date']).copy()
    require(len(d)==1825 and top.item_id.nunique()==89,'완료된 QC 범위 변경')
    selected=pd.read_csv(SELECTION);ids=list(selected.item_id)
    require(len(ids)==5 and selected.bed_id.nunique()==5,'기존5개체 예시 선정 보존')
    examples=top.loc[top.item_id.isin(ids)].copy();examples['norm']=np.nan
    for item,g in examples.groupby('item_id'):
        valid=g.loc[g.measurement_valid];require(len(valid)>0,'유효 예시 없음')
        examples.loc[g.index,'norm']=g.area_cm2/valid.area_cm2.iloc[0]
    examples.loc[~examples.measurement_valid,'norm']=np.nan
    dates=pd.DatetimeIndex(sorted(top.date.unique()))
    matrix=examples.pivot(index='date',columns='item_id',values='norm').reindex(index=dates,columns=ids)
    overall=matrix.median(axis=1)
    # 미관측 날짜에는 0개를 만들어 넣지 않는다. 실제 촬영 날짜만 집계한다.
    counts=top.groupby(['date','growth_qc']).size().unstack(fill_value=0).reindex(index=dates,columns=STATES,fill_value=0)
    for day in dates:
        for state in STATES:require(int(counts.loc[day,state])==int(((top.date==day)&(top.growth_qc==state)).sum()),'독립 판정 수 대조')
    wal=Path(str(FEATURE_DB)+'-wal');require(not wal.exists() or wal.stat().st_size==0,'원본 DB 미반영 WAL')
    stat=FEATURE_DB.stat();db=sqlite3.connect(FEATURE_DB.as_uri()+'?mode=ro&immutable=1',uri=True)
    photos=[]
    try:
        for path,record_id in PHOTO_INPUTS:
            require(path.is_file(),'지정 사진 없음 '+str(path))
            rr=top.loc[top.record_id.eq(record_id)]
            require(len(rr)==1,'사진의 정확한 QC record 단일성')
            q=rr.iloc[0]
            feature=db.execute('select payload from features where id=?',(record_id,)).fetchone()
            require(feature is not None,'원본 Feature record 없음')
            r=json.loads(feature[0]);state=str(q.growth_qc);item=str(q.item_id)
            require(bool(q.measurement_valid) and not bool(q.review_required),'사진 QC 유효성/검토 상태 변경')
            require(math.isclose(float(q.area_cm2),float(r['area_cm2_v2_2']),rel_tol=1e-12),'최종 QC 면적과 Feature 불일치')
            digest=sha(path);require(digest==r['crop_sha256']==sha(r['crop_path']),'사진 픽셀 계보 불일치')
            require(not r['qc_hold_physical_size_v2_2'] and r['area_cm2_v2_2']>0,'면적 미산출/보류 사진')
            protected[str(path)]=digest
            with Image.open(path) as im:pixels=np.asarray(im.convert('RGB'))
            photos.append(dict(date=pd.Timestamp(datetime.strptime(str(r['capture_date']),'%Y%m%d')),
                record_id=r['record_id'],bed_id=r['bed_id'],physical_slot=r['physical_slot'],item_id=item,
                path=str(path),source_crop=str(r['crop_path']),area_cm2=float(r['area_cm2_v2_2']),
                area_cm2_original=float(r['area_cm2']),state=state,sha256=digest,
                qc_role=str(q.role),valid_obs_n=int(q.valid_obs_n),area_growth_pct_day=float(q.area_growth_pct_day),
                dimensions=[pixels.shape[1],pixels.shape[0]],pixels=pixels))
    finally:db.close()
    require([p['date'] for p in photos]==sorted(p['date'] for p in photos),'사진 날짜 순서')
    require([p['state'] for p in photos]==['GROWTH_OBSERVED','GROWTH_OBSERVED','GROWTH_SLOWING'],'사진의 원본 QC 상태 변경')
    return top,examples,selected,dates,matrix,overall,counts,photos,protected,dict(path=str(FEATURE_DB),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)

def style_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#555555');ax.spines[['bottom','left']].set_linewidth(.65)
    ax.tick_params(axis='both',length=3,width=.65,pad=3)
    ax.grid(axis='y',color='#E5E5E5',linewidth=.5);ax.set_axisbelow(True)

def configure_dates(ax,dates,labels):
    start,end=dates[0],dates[-1]
    ticks=pd.DatetimeIndex([pd.Timestamp(x) for x in ['2026-07-24','2026-07-28','2026-08-01','2026-08-05','2026-08-09','2026-08-13','2026-08-18']])
    ax.set_xlim(start-pd.Timedelta(hours=10),end+pd.Timedelta(hours=10))
    ax.set_xticks(ticks);ax.set_xticklabels([x.strftime('%b %d') for x in ticks] if labels else [])
    if labels:ax.set_xlabel('Observation date (2026)',labelpad=7)

def line(ax,values,dates,**kwargs):
    """3일 초과 공백은 연결하지 않고, 빠진 수치도 보간하지 않는다."""
    start=0
    for i in range(1,len(dates)+1):
        if i==len(dates) or (dates[i]-dates[i-1]).days>MAX_GAP:
            ax.plot(dates[start:i],np.asarray(values)[start:i],**kwargs)
            kwargs.pop('label',None);start=i

def text_layout(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    entries=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.width<=0 or b.height<=0:continue
        require(b.x0>=bounds.x0-1 and b.y0>=bounds.y0-1 and b.x1<=bounds.x1+1 and b.y1<=bounds.y1+1,'글자 경계 밖 '+t.get_text())
        require(all(ord(c)<128 for c in t.get_text()),'논문 그림 영문 형식')
        entries.append(dict(text=t.get_text(),size=t.get_fontsize()))
    return entries

def draw(examples,selected,dates,matrix,overall,counts,photos):
    font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','axes.unicode_minus':False,'font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.2,'figure.facecolor':'white','axes.facecolor':'white','text.color':'#222222','axes.labelcolor':'#222222','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=150)
    a=fig.add_axes([.115,.712,.855,.248]);b=fig.add_axes([.115,.451,.855,.216])
    bedcolors=['#79A9C7','#69AE8C','#A39AC6','#CAABA0','#B3B790']
    for r,color in zip(selected.itertuples(),bedcolors):
        # 각 bed에서 선택한 한 개체. 카메라 경계가 있으면 해당 개체 선은 끊는다.
        g=examples.loc[examples.item_id.eq(r.item_id)]
        for j,(_,part) in enumerate(g.groupby('temporal_segment_id',sort=False)):
            line(a,part.norm,part.date.to_list(),color=color,lw=.85,alpha=.78,label=f'Bed {int(r.bed_id[3:])}' if j==0 else None)
    line(a,overall,dates,color='#176A91',lw=1.8,marker='o',ms=3.1,label='Across-bed median',zorder=4)
    a.set_ylabel('Relative projected area\n(first valid area = 1)',labelpad=8)
    maximum=float(matrix.max().max());a.set_ylim(0,math.ceil((maximum*1.18)/5)*5);a.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    a.legend(loc='upper left',ncol=3,frameon=False,columnspacing=1.1,handlelength=1.8,handletextpad=.5,borderaxespad=.25,labelspacing=.45)
    for state,label,color,marker in zip(STATES,STATE_LABELS,STATE_COLORS,['o','s','^','D']):line(b,counts[state],dates,color=color,lw=1.2,marker=marker,ms=2.8,label=label)
    b.set_ylabel('Number of plants',labelpad=8)
    b.set_ylim(-2,math.ceil(float(counts.max().max())*1.17/10)*10);b.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    b.legend(loc='upper left',ncol=4,frameon=False,columnspacing=.85,handlelength=1.5,handletextpad=.45,borderaxespad=.25,fontsize=8)
    for letter,ax in [('a',a),('b',b)]:
        style_axes(ax);configure_dates(ax,dates,letter=='b')
        fig.text(.10,ax.get_position().y1+.008,f'({letter})',fontweight='bold',fontsize=12,va='bottom')
    photo_labels=dict(zip(STATES,STATE_LABELS));photo_labels.update(INSUFFICIENT_DATA='Insufficient observations',SOURCE_REVIEW='Measurement review',NO_MEASUREMENT='No measurement',NOT_ASSESSED='Not assessed')
    for j,p in enumerate(photos):
        left=.115+j*.2925;ax=fig.add_axes([left,.080,.265,.240]);ax.set_facecolor('black')
        h,w=p['pixels'].shape[:2];extent=max(h,w)*1.02
        ax.imshow(p['pixels'],extent=(-w/2,w/2,-h/2,h/2),origin='upper',interpolation='none')
        ax.set_xlim(-extent/2,extent/2);ax.set_ylim(-extent/2,extent/2);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        fig.text(left,.345,f'({chr(ord("c")+j)})',fontweight='bold',fontsize=12,va='center')
        fig.text(left+.15,.345,p['date'].strftime('%b %d'),ha='center',va='center',fontsize=9.5)
        fig.text(left+.1325,.055,f'Area: {p["area_cm2"] * 100.0:.0f} mm$^2$',ha='center',va='center',fontsize=9)
        p['display_label']=photo_labels[p['state']]
        p['display_label_source']='CALCULATED_QC'
        fig.text(left+.1325,.024,p['display_label'],ha='center',va='center',fontsize=8.5)
    texts=text_layout(fig);renderer=fig.canvas.get_renderer();boxes=[t.get_window_extent(renderer) for t in b.get_xticklabels()]
    require(all(x.x1+2<y.x0 for x,y in zip(boxes,boxes[1:])),'날짜 눈금 겹침')
    dest=OUTPUT_DIR/(STEM+'.png');fig.savefig(dest,dpi=DPI)
    preview=OUTPUT_DIR/'_근거'/(STEM+'_preview.png');fig.savefig(preview,dpi=150)
    plt.close(fig)
    with Image.open(dest) as im:require(im.size==(round(WIDTH*DPI),round(HEIGHT*DPI)),'600dpi 출력 크기')
    return dict(path=str(dest),preview=str(preview),dimensions=[round(WIDTH*DPI),round(HEIGHT*DPI)],dpi=DPI,font='Arial',text_layout_checked=True,text=texts)

def main():
    global OUTPUT_DIR
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    args=parser.parse_args()
    OUTPUT_DIR=args.output_dir
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/'_근거').mkdir(exist_ok=True)
    dest=OUTPUT_DIR/(STEM+'.png');prior=None
    if dest.exists():
        require('--replace-labels' in sys.argv,'기존 논문 그림 보존: 새 버전 경로 지정')
        prior=json.loads((OUTPUT_DIR/(STEM+'.json')).read_text(encoding='utf-8'))
        require(sha(dest)==prior['output_sha256'],'기존 그림이 변경되어 교체 중단')
        with Image.open(dest) as im:before=np.asarray(im.convert('RGB')).copy()
    top,examples,selected,dates,matrix,overall,counts,photos,protected,dbstat=read_inputs()
    output=draw(examples,selected,dates,matrix,overall,counts,photos)
    if prior:
        with Image.open(dest) as im:after=np.asarray(im.convert('RGB'))
        require(np.array_equal(before[:int(HEIGHT*DPI*.956)],after[:int(HEIGHT*DPI*.956)]),'사진 아래 표기 밖 픽셀 변경')
        require(np.array_equal(before[:,int(WIDTH*DPI*.68):],after[:,int(WIDTH*DPI*.68):]),'마지막 사진/판정 변경')
    require(all(sha(p)==h for p,h in protected.items()),'원본 수정')
    stat=FEATURE_DB.stat();require(stat.st_size==dbstat['bytes'] and stat.st_mtime_ns==dbstat['mtime_ns'],'원본 DB 변경')
    manifest=dict(status='PAPER_FIGURE_EXPORTED_USER_REVIEW_PENDING',created_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        code_sha256=sha(__file__),source_hashes=protected,feature_database=dbstat,source_unchanged=True,qc_recalculated=False,mean_imputations=0,model_inference=False,
        panels={'a':'Five existing selected stable examples, one plant per bed; median across currently observed selected beds',
                'b':'Daily counts of the four existing growth-QC states among all 89 top-view survey plants; insufficient/review outside four states',
                'c_e':'User-selected original crops of THREE DIFFERENT PLANTS; visual examples, not a longitudinal sequence'},
        selected_examples=selected.to_dict('records'),a_daily_observed_beds=matrix.notna().sum(axis=1).to_list(),
        dates=[x.strftime('%Y-%m-%d') for x in dates],relative_area=[None if pd.isna(x) else float(x) for x in overall],qc_counts=counts.to_dict('list'),
        photos=[{k:(v.strftime('%Y-%m-%d') if isinstance(v,pd.Timestamp) else v) for k,v in p.items() if k!='pixels'} for p in photos],
        photos_not_equal_physical_scale=True,caption_included=False,caption_requirements='Specify the different cohorts in panels (a)/(b) and the three distinct bed/slots in panels (c-e). All three photo labels come from the saved calculated growth-QC state for the exact image record. Panels (c)/(d) are S057 Bed17 b1 on July26 and S064 Bed19 t4 on August5; panel (e) remains S098 Bed27 b3 on August18.',
        label_only_revision=prior is not None,prior_output_sha256=prior['output_sha256'] if prior else None,
        scale_verified=False,agent_visual_review_complete=False,output_sha256=sha(output['path']),**output)
    (OUTPUT_DIR/(STEM+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','path','dimensions','dpi','source_unchanged','photos']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
