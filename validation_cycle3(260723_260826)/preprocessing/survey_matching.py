"""Survey matching calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
DATA_ROOT = CYCLE.SOURCE_ROOT / '6. 시각화와 QC'
SOURCE_BOOK = DATA_ROOT / 'Figures_v2.2_159행/feature_생육조사매칭_ver2.xlsx'
EXPECTED_BOOK_SHA = '603fe7ce12318d7f677c8a817cf58f5c4e95bf67ecbd3d787fb14927147165c4'
CROP_DATABASE = CYCLE.CROP_DB
OUTPUT = CYCLE.PREPROCESS_OUTPUT / '6. 시각화와 QC/matching_crops_115'
REFERENCE_ROOT = CYCLE.MAIN_ROOT / 'model'
A_CODE = CYCLE.MAIN_ROOT / 'model/2. Predict_weight.py'
C_CODE = CYCLE.MAIN_ROOT / 'model/3. Fuse_growth_history.py'
S3_C_CODE = CYCLE.THIRD_ROOT / 'model/3. Fuse_growth_history.py'
WORKERS = 4
FONT_PATH = Path('C:\\Windows\\Fonts\\malgun.ttf')
import argparse, csv, hashlib, html, io, json, math, shutil, sqlite3, sys, zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import openpyxl
from PIL import Image, ImageDraw, ImageFont
try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(items, total, desc='', unit='', mininterval=3):
        import time
        start = last = time.monotonic()
        for (i, item) in enumerate(items, 1):
            now = time.monotonic()
            if now - last >= mininterval or i == total:
                print(f'{desc}: {i}/{total} {unit}, elapsed {now - start:.1f}s', flush=True)
                last = now
            yield item
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def save_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys((k for r in rows for k in r)))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for (k, v) in r.items()})

def timestamp(row):
    return datetime.strptime(str(row['capture_date']) + str(row['capture_time']).zfill(6), '%Y%m%d%H%M%S')

def read_inputs():
    assert sha(SOURCE_BOOK) == EXPECTED_BOOK_SHA, 'Input contract failed: sha(SOURCE_BOOK) == EXPECTED_BOOK_SHA'
    with closing(openpyxl.load_workbook(SOURCE_BOOK, read_only=True, data_only=True)) as book:

        def rows(name):
            it = book[name].iter_rows(values_only=True)
            headers = next(it)
            assert len(set(headers)) == len(headers)
            return [dict(zip(headers, r)) for r in it]
        (matching, crops, daily) = (rows('매칭'), rows('Crop_Feature'), rows('일별Feature'))
    surveys = [r for r in matching if r['생육조사'] == 'O']
    targets = [r for r in surveys if r['image_date']]
    assert len(surveys) == 128 and len(targets) == 115
    assert len({r['item_id'] for r in targets}) == len({r['track_id'] for r in targets}) == 115
    by_id = {r['record_id']: r for r in crops}
    assert len(by_id) == len(crops)
    groups = defaultdict(list)
    for r in crops:
        if r['feature_status'] != 'OK' or r['qc_exclude_body_v2_2'] is True:
            continue
        if r['view'] == 'top' and r['selected_for_top'] is not True:
            continue
        groups[r['track_id'], r['view']].append(r)
    for g in groups.values():
        g.sort(key=lambda r: (timestamp(r), r['record_id']))
    return (targets, by_id, groups, daily)

def inventory(targets, by_id, groups, daily):
    (per_target, selected, changed, views) = ([], [], [], [])
    totals = Counter()
    ids = defaultdict(set)
    for target in targets:
        t = dict(item_id=target['item_id'], survey_date=target['survey_date'], image_date=target['image_date'], bed_id=target['bed_id'], physical_slot=target['physical_slot'], pair_status=target['pair_status'])
        primary = 'top' if target['record_id_top'] else 'front'
        t['primary_view'] = primary
        t['existing_A_C_current_view_eligible'] = primary == 'top'
        for view in ['top', 'front']:
            rid = target['record_id_' + view]
            if not rid:
                for n in ['current', 'past3', 'past7', 'full_including_current', 'past3_only', 'past7_only', 'full_past_only']:
                    t[view + '_' + n] = 0
                continue
            cur = by_id[rid]
            assert cur['track_id'] == target['track_id'] and cur['view'] == view
            assert cur['capture_date'] == target['image_date']
            assert cur['crop_path'] == target['crop_path_' + view]
            cutoff = timestamp(cur)
            survey_day = datetime.strptime(target['survey_date'], '%Y%m%d').date()
            assert 0 <= (survey_day - cutoff.date()).days <= 3
            assert int(target['sample_number']) == int(target['physical_slot'][1:]) + (12 if target['physical_slot'][0] == 'b' else 0)
            candidates = [r for r in groups[cur['track_id'], view] if timestamp(r) <= cutoff]
            by_date = {}
            for r in candidates:
                by_date[r['capture_date']] = r
            assert by_date[cur['capture_date']]['record_id'] == rid, "Input contract failed: by_date[cur['capture_date']]['record_id'] == rid"
            rows = list(by_date.values())
            counts = {span: sum((timestamp(r).date() >= cutoff.date() - timedelta(days=span) for r in rows)) for span in [3, 7]}
            totals[view + '_current'] += 1
            totals[view + '_past3_including_current'] += counts[3]
            totals[view + '_past7_including_current'] += counts[7]
            totals[view + '_full_including_current'] += len(rows)
            totals[view + '_raw_candidates_until_cutoff'] += len(candidates)
            t.update({view + '_cutoff': cutoff.isoformat(sep=' '), view + '_current': 1, view + '_past3': counts[3], view + '_past7': counts[7], view + '_full_including_current': len(rows), view + '_past3_only': counts[3] - 1, view + '_past7_only': counts[7] - 1, view + '_full_past_only': len(rows) - 1, view + '_camera_change_days': sum((r['camera'] != cur['camera'] for r in rows)), view + '_first_observed_date': rows[0]['capture_date'], view + '_window3_at_least_two_days': counts[3] >= 2, view + '_window7_at_least_two_days': counts[7] >= 2})
            existing = {r['capture_date']: r for r in daily if r['item_id'] == target['item_id'] and r['view'] == view and (timestamp(r) <= cutoff)}
            for r in rows:
                lead = (cutoff.date() - timestamp(r).date()).days
                is_current = r['record_id'] == rid
                item = dict(item_id=target['item_id'], survey_date=target['survey_date'], target_image_date=target['image_date'], target_record_id=rid, target_cutoff=cutoff.isoformat(sep=' '), bed_id=target['bed_id'], physical_slot=target['physical_slot'], view=view, primary_view=primary, role='CURRENT' if is_current else 'HISTORY_ONLY', in_past3=lead <= 3, in_past7=lead <= 7, lead_days_from_image=lead, record_id=r['record_id'], capture_date=r['capture_date'], capture_time=r['capture_time'], camera=r['camera'], source_camera=r['source_camera'], native_width=r['native_width'], native_height=r['native_height'], camera_differs_from_current=r['camera'] != cur['camera'], crop_path=r['crop_path'], crop_sha256=r['crop_sha256'], area_cm2_original=r['area_cm2'], area_cm2_v2_2=r['area_cm2_v2_2'], scale_qc_v2_2=r['scale_qc_v2_2'], body_qc_v2_2=r['body_qc_v2_2'], qc_hold_physical_size_v2_2=r['qc_hold_physical_size_v2_2'], human_qc_status='미검토', final_model_input_confirmed=False)
                selected.append(item)
                ids[view + '_full'].add(r['record_id'])
                if lead <= 3:
                    ids[view + '_3'].add(r['record_id'])
                if lead <= 7:
                    ids[view + '_7'].add(r['record_id'])
                old = existing.get(r['capture_date'])
                if old and old['record_id'] != r['record_id']:
                    changed.append(dict(item_id=target['item_id'], view=view, capture_date=r['capture_date'], v2_2_daily_record_id=old['record_id'], legacy_last_record_id=r['record_id'], policy='C_STEP_LAST_PER_DAY_BEFORE_CURRENT_CAPTURE; NOT_ADOPTED_FOR_QC', earlier_v2_2_values_preserved=True))
            stem = f"{target['item_id']}_{target['lettuce_key']}_{view}_{cur['camera']}"
            views.append(dict(target=target, crop=cur, view=view, primary=view == primary, stem=stem))
        if primary == 'front':
            cut = timestamp(by_id[target['record_id_front']])
            prior = [r for r in groups[target['track_id'], 'top'] if timestamp(r) <= cut]
            allowable = [r for r in prior if 0 <= (datetime.strptime(target['survey_date'], '%Y%m%d').date() - timestamp(r).date()).days <= 3]
            t['front_only_top_candidate_records_within_survey_0_3d'] = len(allowable)
            t['front_only_top_candidate_record_ids'] = [r['record_id'] for r in allowable]
            t['front_only_prior_top_days'] = len({r['capture_date'] for r in prior})
        per_target.append(t)
    totals['all_views_current'] = len(views)
    totals['all_views_full'] = len(selected)
    totals['all_views_past3_including_current'] = sum((r['in_past3'] for r in selected))
    totals['all_views_past7_including_current'] = sum((r['in_past7'] for r in selected))
    totals['primary_current'] = len(targets)
    totals['primary_full'] = sum((r['view'] == r['primary_view'] for r in selected))
    totals['primary_past3_including_current'] = sum((r['in_past3'] and r['view'] == r['primary_view'] for r in selected))
    totals['primary_past7_including_current'] = sum((r['in_past7'] and r['view'] == r['primary_view'] for r in selected))
    totals['v22_daily_vs_legacy_last_different_days'] = len(changed)
    totals['top_at_least_two_days_in_3_window'] = sum((r.get('top_window3_at_least_two_days', False) for r in per_target))
    totals['top_at_least_two_days_in_7_window'] = sum((r.get('top_window7_at_least_two_days', False) for r in per_target))
    totals['front_only_targets_with_top_candidate_within_survey_0_3d'] = sum((bool(r.get('front_only_top_candidate_records_within_survey_0_3d')) for r in per_target))
    totals['selected_history_physical_size_hold'] = sum((r['qc_hold_physical_size_v2_2'] is True for r in selected))
    assert len({r['record_id'] for r in selected}) == len(selected), "Input contract failed: len({r['record_id'] for r in selected}) == len(selected)"
    assert len({v['crop']['record_id'] for v in views}) == len(views) == 180
    return (per_target, selected, changed, views, dict(totals))

def readonly_database():
    wal = Path(str(CROP_DATABASE) + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Input contract failed: not wal.exists() or wal.stat().st_size == 0'
    return sqlite3.connect(CROP_DATABASE.as_uri() + '?mode=ro&immutable=1', uri=True)

def unpack_record(blob):
    return json.loads(zlib.decompress(blob))

def unpack_mask(record, blob):
    shape = record['target_mask_shape']
    mask = np.unpackbits(np.frombuffer(zlib.decompress(blob), dtype=np.uint8), bitorder='little', count=math.prod(shape)).reshape(shape).astype(bool)
    assert int(mask.sum()) == record['mask_pixels']
    assert hashlib.sha256(mask.tobytes()).hexdigest() == record['target_mask_pixels_sha256']
    return mask

def mask_edge(mask):
    inside = mask.copy()
    inside[1:, :] &= mask[:-1, :]
    inside[:-1, :] &= mask[1:, :]
    inside[:, 1:] &= mask[:, :-1]
    inside[:, :-1] &= mask[:, 1:]
    inside[0, :] = inside[-1, :] = False
    inside[:, 0] = inside[:, -1] = False
    return mask & ~inside
