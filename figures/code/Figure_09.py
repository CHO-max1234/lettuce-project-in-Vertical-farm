"""English paper figure from the completed season-2 QC workbook and original crops."""
from pathlib import Path
import argparse
import hashlib
import json

WORKBOOK = Path(r'C:\Users\user\Downloads\growth_qc_2nd_20260308_20260504(1).xlsx')
DATA_ROOT = Path(r'E:\newfolder_양상추\2작기 ver2_전처리')
OUTPUT_ROOT = DATA_ROOT / '9. 생육QC' / 'visualizations' / '2작기'
STEM = 'Figure 9'
PHOTO_SPECS = [
    ('20260320', '085556', 'GROWTH_OBSERVED', 'Growth observed'),
    ('20260410', '191823', 'STRUCTURAL_TRANSITION_SIGNAL', 'Structural transition'),
    ('20260503', '230124', 'PLATEAU_CANDIDATE', 'Plateau candidate'),
]
STATES = ['GROWTH_OBSERVED', 'GROWTH_SLOWING', 'STRUCTURAL_TRANSITION_SIGNAL',
          'PLATEAU_CANDIDATE', 'MIXED_CHANGE', 'DECREASE_OBSERVED']
LABELS = ['Growth observed', 'Growth slowing', 'Structural transition',
          'Plateau candidate', 'Mixed change', 'Decrease observed']
COLORS = ['#258A58', '#D28C10', '#2D7FA5', '#8E65AC', '#797979', '#C95145']
HIGHLIGHT_WINDOWS = [
    ('2026-03-15', '2026-03-20', 'GROWTH_OBSERVED', '#258A58'),
    ('2026-04-02', '2026-04-10', 'STRUCTURAL_TRANSITION_SIGNAL', '#2D7FA5'),
    ('2026-04-28', '2026-05-04', 'PLATEAU_CANDIDATE', '#8E65AC'),
]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd
import openpyxl
from PIL import Image
from tqdm import tqdm


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_inputs():
    workbook_hash = sha(WORKBOOK)
    wb = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    rows = wb['all_required_columns'].iter_rows(values_only=True)
    header = next(rows)
    data = pd.DataFrame(list(rows), columns=header)
    wb.close()
    require(len(data) == 3480, 'Source workbook row count changed')
    data['date'] = pd.to_datetime(data['date'])
    require(not data.duplicated(['track_id', 'date']).any(), 'Duplicate plant/date')
    eligible = data.analysis_eligible.astype(str).str.lower().eq('true')
    usable = data.loc[eligible & pd.to_numeric(data.area_cm2_top, errors='coerce').gt(0)].copy()
    usable = usable.sort_values(['track_id', 'date'])
    usable['relative_area'] = usable.area_cm2_top / usable.groupby('track_id').area_cm2_top.transform('first')
    dates = sorted(data.date.unique())
    dates = pd.DatetimeIndex(dates)
    by_bed = usable.groupby(['date', 'bed_name']).relative_area.median().unstack().reindex(dates)
    overall = by_bed.median(axis=1)
    classified = data.loc[eligible & data.growth_qc.isin(STATES)]
    counts = classified.groupby(['date', 'growth_qc']).size().unstack(fill_value=0).reindex(index=dates, columns=STATES, fill_value=0)
    require(int(counts.to_numpy().sum()) == 934, 'Original QC classified count changed')
    require(len(dates) == 49, 'Observed date count changed')
    # Source crop names changed after relocation; verify identity using stored content hashes.
    matches = []
    for chunk in tqdm(pd.read_csv(DATA_ROOT / '8. feature' / 'features_all.csv', chunksize=20000), desc='Verify photo lineage'):
        matched = chunk.loc[chunk.bed_id.eq('bed45') & chunk.physical_slot.eq('b9') & chunk.camera.eq('cam1')]
        matches.append(matched)
    features = pd.concat(matches, ignore_index=True)
    photos = []
    for day, time, state, label in PHOTO_SPECS:
        path = DATA_ROOT / '7. lettuce-crop' / 'crops' / 'bed45' / day / f'bed45_{day}_{time}_cam1_rf_b9.png'
        digest = sha(path)
        record = features.loc[features.crop_sha256.eq(digest)]
        require(len(record) == 1, 'Photo feature identity is not unique')
        record = record.iloc[0]
        row = data.loc[data.track_id.eq('2작기|bed45|b9') & data.date.eq(pd.Timestamp(day))]
        require(len(row) == 1, 'Photo daily QC identity is not unique')
        row = row.iloc[0]
        require(row.growth_qc == state, 'Original photo QC changed')
        require(str(record.record_id) in str(row.top_record_ids), 'Photo is not part of the original daily QC row')
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('RGB'))
        photos.append(dict(date=pd.Timestamp(day).strftime('%Y-%m-%d'), path=str(path), sha256=digest,
                           record_id=str(record.record_id), state=state, label=label,
                           photo_area_cm2=float(record.area_cm2), daily_median_area_cm2=float(row.area_cm2_top), pixels=pixels))
    return data, usable, dates, by_bed, overall, counts, photos, workbook_hash


def plot_observed(ax, dates, values, **kwargs):
    # Connect available observations visually without adding missing-date values.
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    ax.plot(dates[valid], values[valid], **kwargs)


def draw(dates, by_bed, overall, counts, photos, output):
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'axes.labelsize': 9.5,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(7.2, 6.15), dpi=150)
    a = fig.add_axes([.115, .718, .855, .242])
    b = fig.add_axes([.115, .450, .855, .218])
    # User-selected visual emphasis, not inferred state boundaries or new QC labels.
    for start, end, state, color in HIGHLIGHT_WINDOWS:
        b.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, alpha=.10,
                  linewidth=0, zorder=-1)
    for bed, color in zip(by_bed.columns, ['#83AEC5', '#A6A0C6', '#84B99A']):
        plot_observed(a, dates, by_bed[bed], color=color, lw=.85, alpha=.85, label=bed.replace('bed', 'Bed '))
    plot_observed(a, dates, overall, color='#176A91', lw=1.7, marker='o', ms=2.5, label='Across-bed median', zorder=4)
    a.set_ylabel('Relative projected area\n(first valid area = 1)', labelpad=8)
    a.set_ylim(0, float(by_bed.max().max()) * 1.24)
    a.legend(loc='upper left', ncol=4, frameon=False, columnspacing=1.0, handlelength=1.6)
    for state, label, color in zip(STATES, LABELS, COLORS):
        plot_observed(b, dates, counts[state], color=color, lw=1.05, marker='o', ms=2.2, label=label)
    b.set_ylabel('Number of plants', labelpad=8)
    b.set_ylim(-.8, float(counts.max().max()) * 1.38 + 1)
    b.legend(loc='upper left', ncol=3, frameon=False, columnspacing=1.15, handlelength=1.6, labelspacing=.5)
    ticks = pd.to_datetime(['2026-03-08', '2026-03-17', '2026-03-26', '2026-04-04', '2026-04-13', '2026-04-22', '2026-05-04'])
    for letter, ax in [('a', a), ('b', b)]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(.65)
        ax.tick_params(length=3, width=.65, pad=3)
        ax.grid(axis='y', color='#E5E5E5', lw=.5)
        ax.set_axisbelow(True)
        ax.set_xlim(dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=1))
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        if letter == 'a':
            ax.tick_params(labelbottom=False)
        fig.text(.100, ax.get_position().y1 + .008, f'({letter})', fontsize=12, fontweight='bold', va='bottom')
    b.set_xlabel('Observation date (2026)', labelpad=7)
    for i, p in enumerate(photos):
        left = .115 + i * .2925
        ax = fig.add_axes([left, .080, .265, .240], facecolor='black')
        h, w = p['pixels'].shape[:2]
        size = max(h, w) * 1.02
        ax.imshow(p['pixels'], extent=(-w/2, w/2, -h/2, h/2), origin='upper', interpolation='none')
        ax.set_xlim(-size/2, size/2)
        ax.set_ylim(-size/2, size/2)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(left, .345, f'({chr(99+i)})', fontsize=12, fontweight='bold', va='center')
        fig.text(left+.15, .345, pd.Timestamp(p['date']).strftime('%b %d'), fontsize=9.5, ha='center', va='center')
        fig.text(left+.1325, .055, f'Area: {p["photo_area_cm2"]:.0f} cm$^2$', fontsize=9, ha='center', va='center')
        fig.text(left+.1325, .024, p['label'], fontsize=8.5, ha='center', va='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1,
                'Text outside figure: ' + text.get_text())
        require(all(ord(c) < 128 for c in text.get_text()), 'Non-English figure text')
    boxes = [t.get_window_extent(renderer) for t in b.get_xticklabels()]
    require(all(x.x1+2 < y.x0 for x,y in zip(boxes, boxes[1:])), 'Overlapping date labels')
    paths = [output / (STEM + '.png')]
    fig.savefig(paths[0], dpi=600)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir','--output',dest='output', type=Path, default=Path(__file__).resolve().parents[1]/'reproduced')
    parser.add_argument('--overwrite', action='store_true', help='Replace only this figure PNG and its provenance JSON')
    args = parser.parse_args()
    data, usable, dates, by_bed, overall, counts, photos, digest = read_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    require(args.overwrite or not any((args.output / (STEM+ext)).exists() for ext in ['.png', '.json']), 'Output already exists')
    paths = draw(dates, by_bed, overall, counts, photos, args.output)
    require(sha(WORKBOOK) == digest and all(sha(p['path']) == p['sha256'] for p in photos), 'Source changed')
    manifest = dict(status='PAPER_FIGURE_EXPORTED', workbook=str(WORKBOOK), workbook_sha256=digest,
                    source_rows=len(data), area_rows=len(usable), qc_classified_rows=int(counts.to_numpy().sum()),
                    qc_unchanged=True, model_inference=False, scale_verified=False,
                    line_display='Connect available observations across missing dates; no missing-date values added or imputed',
                    highlight_windows=[dict(start=start, end=end, state=state, color=color)
                                       for start, end, state, color in HIGHLIGHT_WINDOWS],
                    highlight_notes='User-selected illustrative periods in panel b only; not data-derived state boundaries',
                    area_summary='Normalize each eligible track by its first valid projected area; daily bed medians, then median across beds',
                    qc_summary='Original six classified QC states among analysis-eligible rows; INSUFFICIENT_DATA omitted, never set to zero growth',
                    photo_notes='One physical slot bed45/b9. Original crops fit separately, not to equal physical scale. Photo area is per capture; QC state is daily.',
                    photos=[{k:v for k,v in p.items() if k != 'pixels'} for p in photos],
                    dates=[d.strftime('%Y-%m-%d') for d in dates],
                    relative_area=[None if pd.isna(x) else float(x) for x in overall],
                    qc_counts=counts.to_dict(orient='list'), output_hashes={str(p):sha(p) for p in paths},
                    png_dpi=600, width_mm=182.88, english_only=True, bounds_checked=True)
    (args.output / (STEM+'.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('PAPER_FIGURE_EXPORTED', [str(p) for p in paths], flush=True)


if __name__ == '__main__':
    main()
