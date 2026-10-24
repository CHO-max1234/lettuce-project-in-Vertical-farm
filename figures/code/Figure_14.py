"""Replot the completed growth-QC data and original crops; no QC recalculation."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from Figure_14_storage import STORAGE_ROOT, moved_path

# Source and export settings. Original plots, labels and QC outputs stay unchanged.
SOURCE_ROOT = STORAGE_ROOT / '3작기 실증 적용' / '4. 시각화와 QC' / '2. 생육 QC'
SOURCE_CSV = SOURCE_ROOT / 'growth_qc.csv'
VIS_COMPLETION = SOURCE_ROOT.parent / '1. Feature 시각화' / 'visualization_completion.json'
STEM = 'Figure 14'
OUTPUT_ROOT = SOURCE_ROOT
FIGURE_WIDTH_IN = 7.2
FIGURE_HEIGHT_IN = 5.9
DPI = 600
CHUNK_ROWS = 20000
MAX_GAP_DAYS = 3
PHOTO_DATES = ['2026-05-27', '2026-06-12', '2026-06-20']
STATES = ['GROWTH_OBSERVED', 'GROWTH_SLOWING', 'PLATEAU_CANDIDATE', 'DECREASE_OBSERVED']
STATE_LABELS = ['Area increase', 'Growth slowing', 'Plateau candidate', 'Area decrease']
STATE_COLORS = ['#258A58', '#D28C10', '#8E65AC', '#C95145']
PHOTO_LABELS = {'GROWTH_SLOWING': 'Growth slowing', 'INSUFFICIENT_DATA': 'Insufficient observations',
                'PLATEAU_CANDIDATE': 'Plateau candidate'}

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_inputs():
    source_paths = [SOURCE_CSV, SOURCE_ROOT / 'growth_qc_completion.json', VIS_COMPLETION,
                    SOURCE_ROOT / '01_growth_transition_top.png', SOURCE_ROOT / '04_size_and_appearance.png']
    protected = {str(p): sha(p) for p in source_paths}
    vis = json.loads(VIS_COMPLETION.read_text(encoding='utf-8-sig'))
    selected_beds = sorted({r['lettuce_key'].split('|')[1] for r in vis['selected_tracks']})
    require(selected_beds == ['bed44','bed47','bed62','bed75'], 'Original representative bed selection changed')
    representative = vis['representative_track']
    require(representative == '3작기|bed47|b10', 'Original representative plant changed')
    columns = ['lettuce_key','date','bed_name','slot_name','source_qc','area_cm2_top',
               'growth_qc','representative_crop']
    parts, all_dates, total = [], set(), 0
    for chunk in tqdm(pd.read_csv(SOURCE_CSV, encoding='utf-8-sig', usecols=columns,
                                  chunksize=CHUNK_ROWS), desc='Read completed QC', unit='chunk'):
        total += len(chunk)
        all_dates.update(chunk['date'].unique())
        parts.append(chunk.loc[chunk.bed_name.isin(selected_beds)].copy())
    data = pd.concat(parts, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    dates = sorted(pd.Timestamp(d) for d in all_dates)
    require(total == 60282 and len(dates) == 29, 'Completed QC cohort/date count changed')
    require(not data.duplicated(['lettuce_key','date']).any(), 'Duplicated plant/date')
    usable = data.loc[data.source_qc.eq('USABLE')].sort_values(['lettuce_key','date']).copy()
    require(usable.area_cm2_top.gt(0).all(), 'Invalid normalized area denominator')
    usable['normalized'] = usable.area_cm2_top / usable.groupby('lettuce_key').area_cm2_top.transform('first')
    by_bed = usable.groupby(['bed_name','date']).normalized.median().unstack('bed_name').reindex(dates)
    overall = by_bed.median(axis=1)
    counts = data.groupby(['date','growth_qc']).size().unstack(fill_value=0).reindex(index=dates,columns=STATES,fill_value=0)
    photos = []
    track = data.loc[data.lettuce_key.eq(representative)].set_index('date')
    for day in PHOTO_DATES:
        row = track.loc[pd.Timestamp(day)]
        path = Path(moved_path(row.representative_crop))
        require(path.is_file(), 'Original crop missing: ' + str(path))
        protected[str(path)] = sha(path)
        require(row.growth_qc in PHOTO_LABELS, 'Original photo QC state changed')
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('RGB'))
        photos.append(dict(date=day, area_cm2=float(row.area_cm2_top), state=row.growth_qc,
                           path=str(path), pixels=pixels, dimensions=[pixels.shape[1],pixels.shape[0]]))
    # Recompute summaries independently from the source cohort, not from PNG pixels.
    for date in dates:
        rr = usable.loc[usable.date.eq(date)]
        expected = rr.groupby('bed_name').normalized.median().median()
        require(math.isclose(overall.loc[date], expected, rel_tol=1e-12), 'Bed aggregation mismatch')
        for state in STATES:
            require(int(counts.loc[date,state]) == int(((data.date==date)&(data.growth_qc==state)).sum()), 'QC count mismatch')
    require([round(p['area_cm2']) for p in photos] == [737,874,879], 'Original photo areas changed')
    return data, dates, by_bed, overall, counts, photos, protected


def segments(dates):
    start = 0
    for i in range(1,len(dates)+1):
        if i==len(dates) or (dates[i]-dates[i-1]).days>MAX_GAP_DAYS:
            yield slice(start,i)
            start=i


def coordinates(dates):
    x = np.arange(len(dates),dtype=float)
    for i in range(1,len(dates)):
        if (dates[i]-dates[i-1]).days>MAX_GAP_DAYS:
            x[i:] += 2
    return x


def line(ax, values, dates, **kwargs):
    for block in segments(dates):
        ax.plot(coordinates(dates)[block], np.asarray(values)[block], **kwargs)
        kwargs.pop('label',None)


def style_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#555555')
    ax.spines[['bottom','left']].set_linewidth(.65)
    ax.tick_params(axis='both',length=3,width=.65,pad=3)
    ax.grid(axis='y',color='#E5E5E5',linewidth=.5)
    ax.set_axisbelow(True)


def configure_dates(ax, dates, show_labels):
    chosen = [0,4,8,12,16,19,20,24,28]
    x = coordinates(dates)
    ax.set_xlim(-.35,x[-1]+.35)
    ax.set_xticks(x[chosen])
    ax.set_xticklabels([dates[i].strftime('%b %d') for i in chosen] if show_labels else [])
    for i in range(1,len(dates)):
        gap = (dates[i]-dates[i-1]).days-1
        if gap>=MAX_GAP_DAYS:
            ax.axvspan(x[i-1]+.5,x[i]-.5,facecolor='#F3F3F3',edgecolor='none',zorder=0)
            ax.axvline((x[i-1]+x[i])/2,color='#7A7A7A',lw=.8,ls=(0,(3,3)),zorder=1)
    if show_labels:
        ax.set_xlabel('Observation date (2026; gap compressed)',labelpad=7)


def assert_text_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    report = []
    bounds = fig.bbox
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        box = t.get_window_extent(renderer)
        if box.width<=0 or box.height<=0:
            continue
        require(box.x0>=bounds.x0-1 and box.y0>=bounds.y0-1 and box.x1<=bounds.x1+1 and box.y1<=bounds.y1+1,
                'Text extends outside final figure: ' + t.get_text())
        require(all(ord(c)<128 for c in t.get_text()), 'Non-English figure text: ' + t.get_text())
        report.append(dict(text=t.get_text(),font_size=t.get_fontsize()))
    return report


def draw(dates, by_bed, overall, counts, photos, root):
    font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':8.5,
                         'ytick.labelsize':8.5,'legend.fontsize':8.2,'figure.facecolor':'white',
                         'axes.facecolor':'white','text.color':'#222222','axes.labelcolor':'#222222',
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN,FIGURE_HEIGHT_IN),dpi=150)
    ax_a = fig.add_axes([.115,.712,.855,.248])
    ax_b = fig.add_axes([.115,.451,.855,.216])
    bed_colors = ['#79A9C7','#69AE8C','#A39AC6','#CAABA0']
    for bed,color in zip(by_bed.columns,bed_colors):
        line(ax_a,by_bed[bed],dates,color=color,lw=.85,alpha=.78,label=bed.replace('bed','Bed '))
    line(ax_a,overall,dates,color='#176A91',lw=1.8,marker='o',ms=3.1,label='Across-bed median',zorder=4)
    ax_a.set_ylabel('Relative projected area\n(first valid area = 1)',labelpad=8)
    ax_a.set_ylim(0,29)
    ax_a.set_yticks([0,5,10,15,20,25])
    legend = ax_a.legend(loc='upper left',ncol=3,frameon=False,columnspacing=1.1,handlelength=1.8,
                        handletextpad=.5,borderaxespad=.25,labelspacing=.45)
    ax_a.text(20.5,.975,'15-day gap',transform=ax_a.get_xaxis_transform(),ha='center',va='top',
              fontsize=8,color='#666666',bbox=dict(facecolor='white',edgecolor='none',pad=1))
    for state,label,color,marker in zip(STATES,STATE_LABELS,STATE_COLORS,['o','s','^','D']):
        line(ax_b,counts[state],dates,color=color,lw=1.2,marker=marker,ms=2.8,label=label)
    ax_b.set_ylabel('Number of plants',labelpad=8)
    ax_b.set_ylim(-1.2,46)
    ax_b.set_yticks([0,10,20,30,40])
    ax_b.legend(loc='upper left',ncol=4,frameon=False,columnspacing=.85,handlelength=1.5,
                handletextpad=.45,borderaxespad=.25,fontsize=8)
    for letter,ax in [('a',ax_a),('b',ax_b)]:
        style_axes(ax)
        configure_dates(ax,dates,show_labels=letter=='b')
        fig.text(.10,ax.get_position().y1+.008,f'({letter})',fontweight='bold',fontsize=12,va='bottom')
    # Each original crop is shown fully, fit to a common display panel, not to a common physical scale.
    for j,p in enumerate(photos):
        left = .115 + j*.2925
        ax = fig.add_axes([left,.080,.265,.240])
        ax.set_facecolor('black')
        h,w = p['pixels'].shape[:2]
        extent = max(h,w)*1.02
        ax.imshow(p['pixels'],extent=(-w/2,w/2,-h/2,h/2),origin='upper',interpolation='none')
        ax.set_xlim(-extent/2,extent/2); ax.set_ylim(-extent/2,extent/2)
        ax.set_aspect('equal')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(left,.345,f'({chr(ord("c")+j)})',fontweight='bold',fontsize=12,va='center')
        fig.text(left+.15,.345,pd.Timestamp(p['date']).strftime('%b %d'),ha='center',va='center',fontsize=9.5)
        fig.text(left+.1325,.055,f'Area: {p["area_cm2"]:.0f} cm$^2$',ha='center',va='center',fontsize=9)
        fig.text(left+.1325,.024,PHOTO_LABELS[p['state']],ha='center',va='center',fontsize=8.5)
    text_report = assert_text_layout(fig)
    renderer = fig.canvas.get_renderer()
    tick_boxes = [t.get_window_extent(renderer) for t in ax_b.get_xticklabels()]
    require(all(a.x1+2<b.x0 for a,b in zip(tick_boxes,tick_boxes[1:])), 'Date labels overlap')
    png = root/(STEM+'.png')
    pdf = root/(STEM+'.pdf')
    fig.savefig(png,dpi=DPI)
    fig.savefig(pdf,dpi=DPI,metadata={'Title':'','Author':'','Subject':'Season 3 growth-QC panels; original data and crops'})
    preview = root/'_paper_preview.png'
    fig.savefig(preview,dpi=150)
    with Image.open(png) as im:
        require(im.size==(round(FIGURE_WIDTH_IN*DPI),round(FIGURE_HEIGHT_IN*DPI)), 'Export dimensions mismatch')
        require(im.getbbox() is not None, 'Blank PNG')
        size = list(im.size)
    plt.close(fig)
    return dict(files=[str(png),str(pdf)],png_dimensions=size,dpi=DPI,
                width_mm=FIGURE_WIDTH_IN*25.4,font='Arial',text_layout_checked=True,
                text=text_report,preview=str(preview))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir','--output',dest='output',type=Path,default=Path(__file__).resolve().parents[1]/'reproduced')
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    data,dates,by_bed,overall,counts,photos,protected = read_inputs()
    if args.check:
        print('READ_ONLY_PREFLIGHT_PASS',flush=True)
        print([(p['date'],p['area_cm2'],p['state'],p['dimensions'],p['path']) for p in photos],flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    require(not any((args.output/(STEM+ext)).exists() for ext in ['.png','.pdf','.json']), 'Output exists; choose another folder, do not overwrite')
    exported = draw(dates,by_bed,overall,counts,photos,args.output)
    require(all(sha(p)==digest for p,digest in protected.items()), 'Source changed during export')
    manifest = dict(status='PAPER_FIGURE_EXPORTED',created_utc=datetime.now(timezone.utc).isoformat(),
                    source_hashes=protected,script_sha256=sha(__file__),
                    output_hashes={p:sha(p) for p in exported['files']},
                    original_sources_unchanged=True,model_inference=False,qc_recalculated=False,
                    panels={'a':'Bed medians of individual-normalized projected area; median across beds',
                            'b':'Original counts of four growth-QC states for the same four beds',
                            'c_e':'Original representative crops of bed47/b10; independently fit display, not equal physical scale'},
                    beds=list(by_bed.columns),representative_bed_rows=len(data),
                    dates=[d.strftime('%Y-%m-%d') for d in dates],
                    relative_area=overall.to_list(),qc_counts=counts.to_dict(orient='list'),
                    photos=[{k:v for k,v in p.items() if k!='pixels'} for p in photos],
                    axis='Observed dates compressed; no line across 15-day gap; no gap interpolation',
                    scale_verified=False,**exported)
    (args.output/(STEM+'.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('PAPER_FIGURE_EXPORTED',json.dumps({k:exported[k] for k in ['files','png_dimensions','dpi','width_mm']}),flush=True)


if __name__=='__main__':
    main()
