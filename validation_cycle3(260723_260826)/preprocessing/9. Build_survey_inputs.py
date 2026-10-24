"""Build label-free target/history workbooks using reviewed survey selections."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE = Path(__file__).resolve().parent
ROOT = CYCLE.PREPROCESS_OUTPUT / '6. 시각화와 QC/matching_crops_115'
RECORDS = ROOT / '_실행기록/261014_최종89_115'
MODEL = CYCLE.A_MODEL
CSTEP = CYCLE.MAIN_ROOT / 'model/3. Fuse_growth_history.py'
FILES = {89: 'Feature 최종_생육매칭__89rows.xlsx', 115: 'Feature 최종_생육매칭_115rows.xlsx'}
import sys, importlib.util, json, csv, math, hashlib
from datetime import datetime
from collections import Counter, defaultdict
from contextlib import closing
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')

def load(name, p):
    spec = resolved_spec(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def date_serial(value):
    return (datetime.strptime(str(value), '%Y%m%d') - datetime(1899, 12, 30)).days

def clean(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def table(name, rows, date_cols=()):
    headers = list(dict.fromkeys((k for r in rows for k in r)))
    assert len(headers) == len(set(headers)) and rows
    return dict(name=name, columns=headers, rows=[[clean(r.get(k)) for k in headers] for r in rows], date_columns=list(date_cols))

def numeric(r, core):
    out = {k: r.get(k) for k in core}
    out.update(area_cm2=r['area_cm2_v2_2'])
    for k in ['inner_ratio_core', 'ridge_circularity_core', 'ridge_std_r_core', 'ridge_mean_r_core', 'angular_cv_core', 'front_height_cm', 'front_height_cm2', 'volume_proxy']:
        out[k] = r.get(k + '_v2_2') if k in ['front_height_cm', 'front_height_cm2', 'volume_proxy'] else r.get(k)
    return out

def main():
    RECORDS.mkdir(parents=True, exist_ok=True)
    assert not any(((ROOT / name).exists() for name in FILES.values())), 'Input contract failed: not any(((ROOT / name).exists() for name in FILES.values()))'
    m = load('final_collect_inputs', CODE / 'survey_matching.py')
    cm = load('final_causal_numeric', CSTEP)
    (targets, by_id, groups, daily) = m.read_inputs()
    (per_target, history, _, jobs, counts) = m.inventory(targets, by_id, groups, daily)
    core = json.loads((MODEL / 'selected_model_bundle.json').read_text(encoding='utf-8'))['core_numeric_features']
    assert len(core) == 25 and 'area_cm2' in core
    amounts = {r['item_id']: r for r in per_target}
    recommendations = {}
    selected = defaultdict(list)
    for r in history:
        if r['view'] == r['primary_view']:
            selected[r['item_id']].append(r)
    (rows, history_rows, past_rows) = ([], [], [])
    with closing(m.readonly_database()) as db:
        for t in targets:
            view = 'top' if t['record_id_top'] else 'front'
            cur = by_id[t['record_id_' + view]]
            record = m.unpack_record(db.execute('SELECT payload FROM slots WHERE record_id=?', (cur['record_id'],)).fetchone()[0])
            vals = numeric(cur, core)
            source_candidate = recommendations.get(t['item_id'], {})
            row = dict(item_id=t['item_id'], Stage_human=None, survey_date=date_serial(t['survey_date']), image_date=date_serial(t['image_date']), bed_id=t['bed_id'], physical_slot=t['physical_slot'], sample_number=int(t['sample_number']), view=view, camera=cur['camera'], model_route='TOP_PRIMARY' if view == 'top' else 'FRONT_EXPLORATORY', fresh_weight_g=t['조사_무게(g)'], record_id=cur['record_id'], crop_path=cur['crop_path'], capture_time=cur['capture_time'], capture_date=cur['capture_date'], capture_timestamp=m.timestamp(cur).isoformat(sep=' '), season_id='4작기', lettuce_key=cur['lettuce_key'], survey_lettuce_key=t['lettuce_key'], matching_bools=t['matching_bools'], lead_days=int(t['lead_days']), source_camera=cur['source_camera'], native_width=int(cur['native_width']), native_height=int(cur['native_height']), crop_sha256=cur['crop_sha256'], mask_pixels=record['mask_pixels'], crop_source_box=json.dumps(record['crop_source_box']), selected_camera=cur['camera'], selected_record_id=cur['record_id'], slot_namespace_version=record['slot_namespace_version'], model_slot_namespace_version='physical_12col_v1', slot_mapping_basis='USER_t1_1_t12_12_b1_13_b12_24; stored slot name unchanged', feature_status=cur['feature_status'], A_C_main_view_eligible=view == 'top', crop_use_decision='USER_APPROVED_TOP_89' if view == 'top' else 'FRONT_SEPARATE_DIAGNOSTIC', pair_status=t['pair_status'], survey_time_known=t['survey_time_known'], identity_confirmed=t['identity_confirmed'], scale_verified=False, mm_per_px=cur['mm_per_px_v2_2'], scale_qc_v2_2=cur['scale_qc_v2_2'], qc_hold_physical_size_v2_2=cur['qc_hold_physical_size_v2_2'], body_qc_v2_2=cur['body_qc_v2_2'], qc_exclude_body_v2_2=cur['qc_exclude_body_v2_2'], feature_qc=cur['feature_qc'])
            row.update(vals)
            row.update(mm_per_px_original=cur['mm_per_px'], area_cm2_original=cur['area_cm2'], front_height_cm_original=cur['front_height_cm'], front_height_cm2_original=cur['front_height_cm2'], volume_proxy_original=cur['volume_proxy'], numeric_missing_count=sum((not (isinstance(vals[k], (int, float)) and math.isfinite(vals[k])) for k in core)), history_n_days=amounts[t['item_id']][view + '_full_including_current'], window3_n_days=amounts[t['item_id']][view + '_past3'], window7_n_days=amounts[t['item_id']][view + '_past7'], collected_crop_path=str(ROOT / ('lettuce-crops/' + next((j['stem'] for j in jobs if j['crop']['record_id'] == cur['record_id'])) + '.png')), collected_bed_overlay_path=str(ROOT / ('bed-overlay/' + next((j['stem'] for j in jobs if j['crop']['record_id'] == cur['record_id'])) + '_bed-overlay.png')), top_candidate_record_id=source_candidate.get('proposed_top_record_id') or None, top_candidate_image_date=source_candidate.get('image_date') or None, top_candidate_matching_bools=source_candidate.get('proposed_matching_bools') or None, candidate_applied=False)
            assert m.sha(row['crop_path']) == row['crop_sha256']
            rows.append(row)
            hr = []
            for link in selected[t['item_id']]:
                r = by_id[link['record_id']]
                h = dict(item_id=t['item_id'], role=link['role'], target_record_id=cur['record_id'], target_cutoff=row['capture_timestamp'], record_id=r['record_id'], view=view, camera=r['camera'], source_camera=r['source_camera'], capture_date=r['capture_date'], capture_time=r['capture_time'], capture_timestamp=m.timestamp(r).isoformat(sep=' '), history_lead_days=link['lead_days_from_image'], in_window3=link['in_past3'], in_window7=link['in_past7'], crop_path=r['crop_path'], crop_sha256=r['crop_sha256'], native_width=r['native_width'], native_height=r['native_height'])
                h.update(numeric(r, core))
                h.update(mm_per_px=r['mm_per_px_v2_2'], mm_per_px_original=r['mm_per_px'], area_cm2_original=r['area_cm2'], qc_hold_physical_size_v2_2=r['qc_hold_physical_size_v2_2'], scale_qc_v2_2=r['scale_qc_v2_2'], body_qc_v2_2=r['body_qc_v2_2'], camera_differs_from_current=link['camera_differs_from_current'], Cstep_volume_proxy=max(float(h['area_cm2']), 0) ** 1.5 if h['area_cm2'] is not None else None)
                assert m.timestamp(r) <= m.timestamp(cur)
                hr.append(h)
            history_rows.extend(hr)
            frame = pd.DataFrame(hr)
            frame['capture_ts'] = pd.to_datetime(frame.capture_timestamp)
            frame['crop_filename'] = frame.crop_path.map(lambda p: Path(p).name)
            frame['volume_proxy'] = frame['Cstep_volume_proxy']
            frame['source_tier'] = 'raw_direct'
            (history_values, h) = cm.causal_history(frame, pd.Timestamp(row['capture_timestamp']))
            numeric_history = {k: clean(v) for (k, v) in history_values.items() if not any((s in k for s in ['cnn_', 'embedding_', 'stage', 'source', 'confidence']))}
            p = dict(item_id=t['item_id'], record_id=cur['record_id'], view=view, target_cutoff=row['capture_timestamp'], window3_n_days=row['window3_n_days'], window7_n_days=row['window7_n_days'])
            p.update(numeric_history)
            past_rows.append(p)
            assert len(h) == row['history_n_days']
    top_ids = {r['item_id'] for r in rows if r['view'] == 'top'}
    assert len(rows) == 115 and len(top_ids) == 89 and (len(history_rows) == 1825)
    specs = []
    for size in [89, 115]:
        include = top_ids if size == 89 else {r['item_id'] for r in rows}
        main = [r for r in rows if r['item_id'] in include]
        hh = [r for r in history_rows if r['item_id'] in include]
        pp = [r for r in past_rows if r['item_id'] in include]
        raw = [r for r in targets if r['item_id'] in include]
        assert len(main) == len(pp) == len(raw) == size
        assert len(hh) == (1313 if size == 89 else 1825)
        specs.append(dict(filename=FILES[size], rows=size, tables=[table('생육매칭', main, ['survey_date', 'image_date']), table('Cstep_수치과거', pp), table('과거Feature', hh), table('원본매칭', raw)]))
    spec = dict(status='DATA_PREPARED_NOT_EXPORTED', source_book=str(m.SOURCE_BOOK), source_book_sha256=m.sha(m.SOURCE_BOOK), worksheets=specs, core_numeric_features=core, schema_notes=dict(Stage_human='빈칸, 실제 선택 crop의 인간 단계 라벨0..4', main_area='현재 v2.2 면적; original 별도 보존', window='[D-3,D] / [D-7,D], selected capture timestamp cutoff', Cstep='수치 이력만 계산. CNN확률·예측중량·PCAembedding·환경17변수는 모델 실행 단계에서 생성/연결', front='26개는별도진단경로. 기존윗면CAM01입력으로위장하지않음', candidates='후보3개체는정보만. 이번89/115의선택은변경하지않음'), hashes={'Cstep_code': m.sha(CSTEP), 'A_model_bundle': m.sha(MODEL / 'selected_model_bundle.json')})
    m.save_json(RECORDS / 'workbook_spec.json', spec)
    print(json.dumps(dict(prepared=[dict(filename=s['filename'], sheets={t['name']: [len(t['rows']), len(t['columns'])] for t in s['tables']}) for s in specs], Stage_human_blank=True, numeric_missing=dict(Counter((r['numeric_missing_count'] for r in rows)))), ensure_ascii=False))
    from runtime import write_workbooks
    write_workbooks([dict(path=str(ROOT / s['filename']), sheets=[dict(name=t['name'], columns=t['columns'], rows=t['rows'], date_columns=t.get('date_columns',[])) for t in s['tables']]) for s in specs])
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
