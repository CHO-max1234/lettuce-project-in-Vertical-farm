"""Add final scale/body review fields while preserving original feature values."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = CYCLE.PREPROCESS_OUTPUT / '6. 시각화와 QC'
OUT = ROOT / 'Figures_v2.2_159행'
REC = OUT / '_실행기록'
OLD_DB = ROOT / '_실행기록/261011_Feature/feature_state.sqlite3'
NEW_DB = REC / 'feature_v2_2.sqlite3'
OLD_SAMPLE = ROOT / 'Figures_v2.1_159행/_실행기록'
SCALE_RUN = CYCLE.PREPROCESS_OUTPUT / '3. 밝기보정과 Scale/_실행기록/261009_SCALE'
PHYSICAL = ['mm_per_px', 'area_cm2', 'front_height_cm', 'front_height_cm2', 'volume_proxy']
FLAGS = ['scale_qc_v2_2', 'qc_hold_physical_size_v2_2', 'body_qc_v2_2', 'qc_exclude_body_v2_2', 'scale_physical_verified_v2_2']
ADDED = [f + '_v2_2' for f in PHYSICAL] + FLAGS
BODY_REPLACEMENTS = {'037e3cb74c5518421798a5ac': 'f29594dd056f7ba36142fe2a', '83e0e3f7007acf9a72d0b387': '77668637699b50214a4ceb1e', '4a308e684128ebf4417c6ac7': 'f645c0d7a87085876a1333b7'}
import sys, json, sqlite3, hashlib, math, time, importlib.util
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_calculation import FEATURES
from workbook_stream import write_workbook, verify_workbook

def load_module(name, file):
    s = resolved_spec(name, HERE / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

def ro(path):
    wal = Path(str(path) + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0
    return sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)

def js(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)

def save(name, obj):
    (REC / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda : f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def scale_updates():
    results = [json.loads(l) for l in (CYCLE.CALIBRATION_ROOT / 'scale_recovery_full.jsonl').read_text(encoding='utf-8').splitlines()]
    manual = CYCLE.CALIBRATION_ROOT / 'visually_verified_reference_row_cases.json'
    if manual.exists():
        results += json.loads(manual.read_text(encoding='utf-8'))
    updates = {r['source_path']: dict(status=r['status'], new_scale=None if r['status'].startswith('HOLD') else r['new_scale'], image_key=r['image_key']) for r in results}
    bykey = {r['image_key']: r for r in results}
    scale = pd.read_csv(SCALE_RUN / 'scale_map.csv').replace({np.nan: None})
    allkeys = scale.set_index('image_key').to_dict('index')
    references = []
    for (_, row) in scale.loc[scale.scale_source.eq('MEDIAN_SAME_GROUP')].iterrows():
        keys = [row.get('reference_' + str(i)) for i in range(1, 4)]
        keys = [k for k in keys if k]
        if not any((k in bykey for k in keys)):
            continue
        if any((k in bykey and bykey[k]['status'].startswith('HOLD') for k in keys)):
            status = 'HOLD_REFERENCE_CONTAINS_UNRESOLVED_INNER_RIM'
            new = None
        else:
            diams = [50 / bykey[k]['new_scale'] if k in bykey else float(allkeys[k]['diameter_px']) for k in keys]
            new = 50 / float(np.median(diams))
            status = 'SAME_REFERENCES_REESTIMATED_AFTER_OUTER_RIM_RECOVERY'
        updates[row.source_path] = dict(status=status, new_scale=new, image_key=row.image_key)
        references.append(dict(image_key=row.image_key, source_path=row.source_path, old_scale=row.mm_per_px, new_scale=new, status=status, reference_keys=keys))
    save('reference_scale_update_audit.json', references)
    return updates

def augment(r, updates, invalid):
    out = dict(r)
    update = updates.get(r['source_path'])
    status = update['status'] if update else 'UNCHANGED_UPSTREAM_ESTIMATE'
    hold = status.startswith('HOLD')
    s = None if hold else update['new_scale'] if update else r['mm_per_px']
    out.update(scale_qc_v2_2=status, qc_hold_physical_size_v2_2=hold, body_qc_v2_2='EXCLUDE_VISUALLY_CONFIRMED_PARTIAL' if r['record_id'] in invalid else 'UPSTREAM_REVIEW_PRESERVED', qc_exclude_body_v2_2=r['record_id'] in invalid, scale_physical_verified_v2_2=False)
    out['mm_per_px_v2_2'] = s
    ratio = s / r['mm_per_px'] if s is not None else None
    for (field, power) in [('area_cm2', 2), ('front_height_cm', 1), ('front_height_cm2', 2), ('volume_proxy', 3)]:
        value = r.get(field)
        out[field + '_v2_2'] = value * ratio ** power if value is not None and ratio is not None else None
    assert all((out[k] == v for (k, v) in r.items())), 'Input contract failed: all((out[k] == v for (k, v) in r.items()))'
    return out

def prepare_database():
    assert not (REC / 'database_application_complete.json').exists(), "Input contract failed: not (REC / 'database_application_complete.json').exists()"
    updates = scale_updates()
    review = pd.read_csv(CYCLE.SOURCE_ROOT / '6. 시각화와 QC/_실행기록/261011_QC진단/reviewed_case_recommendations.csv')
    invalid = set(review.loc[review.recommendation.eq('EXCLUDE_WHOLE_PLANT_MEASUREMENT_OR_RECOVER_ACTUAL_CROP'), 'record_id'])
    old = ro(OLD_DB)
    new = sqlite3.connect(NEW_DB)
    new.execute('create table if not exists features(id text primary key,lettuce_key text,camera text,sequence integer,payload text)')
    new.execute('create index if not exists feature_key_camera on features(lettuce_key,camera)')
    new.execute('create table if not exists matching(lettuce_key text primary key,payload text)')
    counts = Counter()
    n = 0
    for (id, key, cam, seq, p) in tqdm(old.execute('select id,lettuce_key,camera,sequence,payload from features order by sequence'), total=243025, desc='Processing', mininterval=5):
        r = augment(json.loads(p), updates, invalid)
        counts[r['scale_qc_v2_2']] += 1
        new.execute('insert or replace into features values(?,?,?,?,?)', (id, key, cam, seq, js(r)))
        n += 1
        if n % 512 == 0:
            new.commit()
    new.commit()
    m = 0
    for (key, p) in tqdm(old.execute('select lettuce_key,payload from matching order by lettuce_key'), total=198045, desc='Processing', mininterval=5):
        r = json.loads(p)
        out = dict(r)
        for view in ('top', 'front'):
            rid = r.get('record_id_' + view)
            row = new.execute('select payload from features where id=?', (rid,)).fetchone() if rid else None
            z = json.loads(row[0]) if row else None
            for c in ADDED:
                out[c + '_' + view] = z.get(c) if z else None
        a = out.get('area_cm2_v2_2_top')
        h = out.get('front_height_cm_v2_2_front')
        out['volume_proxy_top_front_v2_2'] = a * h if a is not None and h is not None else None
        assert all((out[k] == v for (k, v) in r.items()))
        new.execute('insert or replace into matching values(?,?)', (key, js(out)))
        m += 1
        if m % 512 == 0:
            new.commit()
    new.commit()
    assert n == 243025 and m == 198045
    new.close()
    old.close()
    save('database_application_complete.json', dict(status='FULL_FEATURE_ADDITIVE_UPDATE_COMPLETE', crop_rows=n, matching_rows=m, crop_states=dict(counts), confirmed_partial_records=len(invalid), original_fields_preserved=True, mask_reinference=False, raw_excel_unchanged=True, completed_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
    print(js(dict(crop_rows=n, matching_rows=m, crop_states=dict(counts))), flush=True)

def read_features(db, ids=None):
    if ids is None:
        for (p,) in db.execute('select payload from features order by lettuce_key,camera'):
            yield json.loads(p)
    else:
        for id in ids:
            row = db.execute('select payload from features where id=?', (id,)).fetchone()
            assert row is not None
            yield json.loads(row[0])

def schema_and_export(do_export):
    legacy = load_module('legacy_full_schema', '7. Extract_features.py')
    db = ro(NEW_DB)
    specs = [('Feature_all_ver2', legacy.ALL_COLUMNS + ADDED, 243025, 'features'), ('Feature_matching_ver2', legacy.MATCH_COLUMNS + [c + '_' + v for v in ('top', 'front') for c in ADDED] + ['volume_proxy_top_front_v2_2'], 198045, 'matching')]

    def records(table):
        if table == 'features':
            yield from read_features(db)
        else:
            for (p,) in db.execute('select payload from matching order by lettuce_key'):
                yield json.loads(p)
    verified = []
    for (name, columns, n, table) in specs:
        sample = []
        for r in records(table):
            sample.append(r)
            if len(sample) == 6:
                break
        if not do_export:
            condition = "json_extract(payload,'$.scale_qc_v2_2')='OUTER_RIM_RECOVERED_ESTIMATED'" if table == 'features' else "json_extract(payload,'$.scale_qc_v2_2_front')='OUTER_RIM_RECOVERED_ESTIMATED'"
            corrected = [json.loads(p) for (p,) in db.execute(f'select payload from {table} where ' + condition + ' limit 3')]
            if corrected:
                sample = sample[:3] + corrected
        save(name + '_schema.json', dict(columns=columns, rows=n, sample=sample))
        if do_export:
            target = OUT / (name + '.xlsx')
            write_workbook(REC / (name + '_template.xlsx'), target, columns, records(table), n)
            result = verify_workbook(target, columns, records(table), n)
            result.update(path=str(target), sha256=digest(target), bytes=target.stat().st_size)
            verified.append(result)
    db.close()
    if do_export:
        save('full_workbook_verification.json', verified)
        print(json.dumps(verified, ensure_ascii=False, indent=2))

def main():
    REC.mkdir(parents=True, exist_ok=True)
    if (REC / 'database_application_complete.json').exists():
        raise ValueError('Use a fresh output directory')
    prepare_database()
    schema_and_export(True)
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
