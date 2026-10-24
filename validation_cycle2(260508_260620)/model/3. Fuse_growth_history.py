"""Apply the frozen main C models to current targets and causal same-plant history. Environmental observations remain excluded and all legacy environment columns remain NaN."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))
import settings as CYCLE
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PROJECT = CYCLE.PROJECT_ROOT
CNN_CODE = CYCLE.PROJECT_ROOT / 'model'
sys.path.insert(0, str(CNN_CODE))
import importlib.util
spec = importlib.util.spec_from_file_location('season3_inputs', CNN_CODE / '2. Predict_stage_and_weight.py')
s3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3)
OUTPUT_ROOT = CYCLE.MODEL_OUTPUT / '04_C-step'
PREP_ROOT = CYCLE.MODEL_OUTPUT / '03_C-input'
BATCH_SIZE = CYCLE.BATCH_SIZE
(SEASON_START, SEASON_END) = ('2026-05-08', '2026-06-20')
EXPECTED_TARGETS = CYCLE.EXPECTED_TARGETS
A_RUN = None
import argparse
from collections import Counter
from contextlib import closing
import csv
import json
import math
import sqlite3
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def key(row):
    return tuple((str(row[k]) for k in ['season_id', 'bed_id', 'physical_slot', 'camera']))

def timestamp(row):
    return str(row['capture_date']) + str(row['capture_time']).zfill(6)

def call_main(mod, args):
    old = sys.argv
    try:
        sys.argv = [mod.__name__] + args
        mod.main()
    finally:
        sys.argv = old

def targets():
    if A_RUN is None:
        raise ValueError('Pass the newly completed target A run explicitly')
    run = Path(A_RUN)
    done = s3.read_json(run / 'completion.json')
    rows = read_csv(run / '01_A_step_predictions.csv')
    s3.require(len(rows) == EXPECTED_TARGETS and done['counts']['top_rows'] == len(rows), 'Target A rows changed')
    s3.require(len(set((key(r) for r in rows))) == len(rows), 'Nonunique target physical/camera identity')
    for (i, r) in enumerate(rows):
        s3.require(int(r['embedding_row_index']) == i, 'Target A embedding index changed')
    return (run, rows)

def prepare():
    (target_run, current) = targets()
    cm = s3.load_reference('s3_c_prepare', 'model/3. Fuse_growth_history.py')
    (models, protected) = cm.load_models()
    for p in [s3.CNN_MODEL, s3.REFERENCE / 'model' / '1. Predict_stage.py', s3.REFERENCE / 'model' / '2. Predict_weight.py', s3.REFERENCE / 'pipeline/crop_integrity.py']:
        protected[str(p)] = s3.sha(p)
    for fold in s3.read_json(s3.A_MODEL / 'selected_model_bundle.json')['folds']:
        for rec in fold['files']:
            p = s3.A_MODEL / rec['path']
            s3.require(s3.sha(p) == rec['sha256'], 'Frozen A model changed')
            protected[str(p)] = rec['sha256']
    import torch
    from torchvision.models import EfficientNet_B0_Weights
    ckpt = CYCLE.BACKBONE_FILE
    s3.require(ckpt.is_file() and s3.sha(ckpt) == '7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934', 'Fixed ImageNet checkpoint missing/changed')
    protected.update({str(ckpt): s3.sha(ckpt), str(s3.LABEL_FILE): s3.sha(s3.LABEL_FILE), str(Path(__file__)): s3.sha(__file__), str(s3.REFERENCE / 'model' / '3. Fuse_growth_history.py'): s3.sha(s3.REFERENCE / 'model' / '3. Fuse_growth_history.py')})
    for p in [target_run / 'completion.json', target_run / '01_A_step_predictions.csv', target_run / 'fold_predictions.npy']:
        protected[str(p)] = s3.sha(p)
    for fold in range(1, 6):
        for name in ['predictions.csv', 'embedding.npy']:
            p = target_run / f'fold_{fold}' / name
            protected[str(p)] = s3.sha(p)
        a = np.load(target_run / f'fold_{fold}' / 'embedding.npy', mmap_mode='r')
        s3.require(a.shape == (32, 128), 'Target supervised embedding shape mismatch')
        for (i, r) in enumerate(current):
            s3.require(bool(np.isfinite(a[i]).all()) == (r['a_step_status'] == 'OK'), 'Target supervised embedding/status mismatch')
    survey = s3.rows_xlsx(s3.LABEL_FILE, 'feature') + s3.rows_xlsx(s3.LABEL_FILE, '미매칭_조사')
    for q in survey:
        sample = int(q['샘플'])
        q['_survey_bed'] = f"bed{int(q['베드(No.)']):02d}"
        q['_survey_slot'] = f't{sample}' if sample <= 12 else f'b{sample - 12}'
    for r in current:
        earlier = [q for q in survey if q['_survey_bed'] == r['bed_id'] and q['_survey_slot'] == r['physical_slot'] and q.get('survey_date') and (q['survey_date'].strftime('%Y%m%d') < r['capture_date'])]
        s3.require(not earlier, 'Earlier destructive sample at target physical slot: ' + r['record_id'])
    cutoffs = {key(r): timestamp(r) for r in current}
    beds = sorted(set((r['bed_id'] for r in current)))
    snapshots = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in [s3.FEATURE_DB, s3.CROP_DB, s3.MASK_DB]}
    (daily, rejected) = ({}, Counter())
    query = "SELECT payload FROM features WHERE json_extract(payload,'$.bed_id') IN (" + ','.join(('?' for _ in beds)) + ") AND json_extract(payload,'$.feature_status')='OK'"
    with closing(s3.readonly_db(s3.FEATURE_DB)) as db:
        for item in tqdm(db.execute(query, beds), desc='Label-free history inventory', mininterval=5):
            f = json.loads(item[0])
            k = key(f)
            if k not in cutoffs or timestamp(f) > cutoffs[k]:
                continue
            if f['ocr_status'] != 'OCR_CANDIDATE_OK':
                rejected[f['ocr_status']] += 1
                continue
            dk = k + (f['capture_date'],)
            if dk not in daily or timestamp(f) > timestamp(daily[dk]):
                daily[dk] = f
    target_ids = {r['record_id'] for r in current}
    extras = sorted([r for r in daily.values() if r['record_id'] not in target_ids], key=lambda r: (key(r), timestamp(r), r['record_id']))
    s3.require(target_ids <= {r['record_id'] for r in daily.values()}, 'A target differs from daily last valid <= cutoff policy')
    core = s3.read_json(s3.A_MODEL / 'selected_model_bundle.json')['core_numeric_features']
    gate_mod = s3.load_reference('s3_history_gate', 'pipeline/crop_integrity.py')
    (rows, gates, manifest) = ([], [], [])
    with closing(s3.readonly_db(s3.CROP_DB)) as crops, closing(s3.readonly_db(s3.MASK_DB)) as slots:
        for f in tqdm(extras, desc='History PNG / mask / identity validation', mininterval=5):
            c0 = crops.execute('SELECT payload FROM crops WHERE record_id=?', (f['record_id'],)).fetchone()
            b0 = slots.execute('SELECT bbox_json,length(mask) FROM slots WHERE record_id=?', (f['record_id'],)).fetchone()
            s3.require(c0 and b0 and (b0[1] > 0), 'Historical crop/mask missing')
            (c, b) = (json.loads(c0[0]), json.loads(b0[0]))
            for name in ['record_id', 'bed_id', 'physical_slot', 'camera', 'capture_date', 'capture_time']:
                s3.require(str(b[name]) == str(f[name]), 'Historical slot identity mismatch: ' + name)
            s3.require(b['target_available'] and b['slot_namespace_version'] in ['physical_12col_v1', 'image_left_to_right_v1'], 'Unverified historical slot')
            path = Path(s3.moved_path(f['crop_path'])).resolve()
            s3.require(path.is_relative_to(s3.CROP_ROOT.resolve()) and path.is_file(), 'Missing/outside historical crop')
            s3.require(s3.sha(path) == f['crop_sha256'] == c['crop_sha256'], 'Historical crop hash mismatch')
            match = s3.IMAGE_NAME.fullmatch(path.name)
            s3.require(match and match.groups() == tuple((str(f[n]) for n in ['bed_id', 'capture_date', 'capture_time', 'camera', 'physical_slot'])), 'Historical filename/identity mismatch')
            mask = gate_mod.unpack(c)
            with Image.open(path) as im:
                rgb = np.asarray(im.convert('RGB'))
            s3.require(mask.shape == rgb.shape[:2] and np.count_nonzero(mask) == int(f['mask_pixels']) and (not np.any(rgb[mask == 0])), 'Historical mask/pixel mismatch')
            c.update(crop_path=str(path), has_crop='True')
            (allow, status, reason) = gate_mod.judge(c, straight_ratio=0.8, minimum=20, main_share=0.9)
            gates.append(dict(record_id=f['record_id'], model_eligible=allow, integrity_status=status, integrity_reason=reason))
            row = {n: f[n] for n in ['record_id', 'season_id', 'bed_id', 'physical_slot', 'camera', 'capture_date', 'capture_time', 'crop_sha256', 'feature_status', 'mask_pixels', 'scale_verified', 'shape_qc', 'feature_qc'] + core}
            row.update(crop_path=str(path), source_crop_path=f['crop_path'], selected_record_id=f['record_id'], selected_camera=f['camera'], slot_namespace_version='physical_12col_v1', source_slot_namespace_version=b['slot_namespace_version'], slot_mapping_basis='USER_CONFIRMED_t1_1_t12_12_b1_13_b12_24; original slot unchanged', front_height_cm=None, volume_proxy=None, crop_source_box=json.dumps(c['crop_source_box']))
            row['분석가능'] = True
            rows.append(row)
            manifest.append(dict(record_id=f['record_id'], season_id=f['season_id'], bed_id=f['bed_id'], physical_slot=f['physical_slot'], camera=f['camera'], capture_timestamp=timestamp(f), target_cutoff=cutoffs[key(f)], source_feature_status=f['feature_status'], source_shape_qc=f['shape_qc'], source_ocr_status=f['ocr_status'], model_eligible=allow, integrity_status=status))
    for (p, before) in snapshots.items():
        s3.require([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == before, 'Source SQLite changed')
    s3.require(not any((set(r) & {'human_label', 'fresh_weight_g', 'survey_date', 'lead_days'} for r in rows)), 'Labels in history model input')
    counts = dict(targets=len(current), history_extra=len(rows), combined=len(current) + len(rows), history_gate_ok=sum((g['model_eligible'] for g in gates)), history_gate_review=sum((not g['model_eligible'] for g in gates)))
    report = dict(status='INPUTS_PREPARED', target_a_run=str(target_run), counts=counts, source_hashes=protected, source_db_snapshots=snapshots, history_date_counts=dict(sorted(Counter((r['capture_date'] for r in rows)).items())), ocr_rejected_before_daily_selection=dict(rejected), history_selection='same season/bed/physical slot/camera; feature OK; confirmed OCR; latest per observed day <= individual target capture cutoff', environment_observed_days=0, environment_features=cm.ENV_COLS, environment_policy='May unavailable and June rejected; all 17 variables NaN, no zero-fill, no interpolation', ec_schedule_used=False, model_feature_counts={k: len(v[0]) for (k, v) in models.items()}, embedding_contract='A supervised: five separate fold Nx128; C common: fixed B0 + frozen train-only PCA Nx128; no coordinate mixing', scale_verified=False, model_training=False, future_qc_used=False)
    signature = s3.sha(s3.LABEL_FILE) + s3.sha(__file__) + s3.sha(target_run / '01_A_step_predictions.csv')
    prep = PREP_ROOT / ('run_' + s3.stamp())
    prep.mkdir(parents=True, exist_ok=False)
    s3.save_workbooks([dict(path=str(prep / 'history_model_input.xlsx'), sheets=[s3.sheet_spec('전체', rows)])], 'c_history_input')
    save_csv(prep / 'history_crop_integrity.csv', gates)
    save_csv(prep / 'history_manifest.csv', manifest)
    save_csv(prep / 'environment_excluded.csv', [], cm.RAW)
    report.update(input_signature=signature, output_hashes={str(p): s3.sha(p) for p in prep.iterdir() if p.is_file()})
    s3.write_json(prep / 'completion.json', report)
    s3.write_json(PREP_ROOT / 'latest_prepared.json', dict(run=str(prep)))
    print(json.dumps(counts, indent=2), flush=True)
    return (prep, report)

def history_inference(prep, report):
    root = prep / 'history_A'
    if (root / 'selected_run.json').is_file():
        run = Path(s3.read_json(root / 'selected_run.json')['run'])
        s3.require(s3.read_json(run / 'completion.json')['status'].startswith('COMPLETE'), 'Historical A incomplete')
        return run
    cnn_root = prep / 'history_CNN'
    cnn_root.mkdir(parents=True, exist_ok=False)
    mod = s3.load_reference('s3_c_history_cnn', 'model/1. Predict_stage.py')
    mod.IMAGE_NAME = s3.IMAGE_NAME
    mod.FEATURE_XLSX = prep / 'history_model_input.xlsx'
    (mod.CROP_ROOT, mod.MODEL_PT) = (s3.CROP_ROOT, s3.CNN_MODEL)
    mod.OUTPUT_XLSX = cnn_root / 'crop_features_CNN.xlsx'
    mod.CACHE_SQLITE = cnn_root / 'cnn_predictions_cache.sqlite3'
    (mod.GATE_CSV, mod.LABEL_CSV, mod.DEVICE) = (prep / 'history_crop_integrity.csv', None, 'auto')
    mod.main()
    mod = s3.load_reference('s3_c_history_a', 'model/2. Predict_weight.py')
    mod.IMAGE_NAME = s3.IMAGE_NAME
    call_main(mod, ['--input', str(cnn_root / 'crop_features_CNN.xlsx'), '--sheet', '전체', '--output-dir', str(root), '--run', str(s3.A_MODEL), '--crop-root', str(s3.CROP_ROOT), '--device', 'auto', '--batch-size', str(BATCH_SIZE)])
    runs = [p for p in root.glob('run_*') if (p / 'completion.json').is_file()]
    s3.require(len(runs) == 1, 'Ambiguous historical A run')
    s3.require(s3.read_json(runs[0] / 'completion.json')['counts']['top_rows'] == report['counts']['history_extra'], 'Historical A rows mismatch')
    s3.write_json(root / 'selected_run.json', dict(run=str(runs[0])))
    return runs[0]

def combine(prep, report, history_run):
    out = prep / 'combined_A'
    out.mkdir(parents=True, exist_ok=False)
    source_runs = [Path(report['target_a_run']), history_run]
    originals = [read_csv(p / '01_A_step_predictions.csv') for p in source_runs]
    s3.require(list(originals[0][0]) == list(originals[1][0]), 'Target/history A schema mismatch')
    (mapping, combined) = ([], [])
    for (run, rows, role) in zip(source_runs, originals, ['EVALUATION_TARGET', 'HISTORY_ONLY']):
        for (old_i, row) in enumerate(rows):
            s3.require(int(row['embedding_row_index']) == old_i, 'Source A index mismatch')
            row = dict(row)
            idx = len(combined)
            row['embedding_row_index'] = str(idx)
            combined.append(row)
            mapping.append(dict(embedding_row_index=idx, record_id=row['record_id'], row_role=role, source_a_run=str(run), source_embedding_row_index=old_i))
    n = len(combined)
    s3.require(n == report['counts']['combined'] and len({r['record_id'] for r in combined}) == n, 'Combined A rows/IDs mismatch')
    save_csv(out / '01_A_step_predictions.csv', combined)
    save_csv(out / 'row_alignment.csv', mapping)
    for name in ['fold_predictions.npy']:
        np.save(out / name, np.concatenate([np.load(p / name, allow_pickle=False) for p in source_runs], axis=0), allow_pickle=False)
    for fold in range(1, 6):
        dest = out / f'fold_{fold}'
        dest.mkdir()
        embeddings = np.concatenate([np.load(p / f'fold_{fold}' / 'embedding.npy', allow_pickle=False) for p in source_runs])
        s3.require(embeddings.shape == (n, 128), 'Combined supervised embedding shape mismatch')
        np.save(dest / 'embedding.npy', embeddings, allow_pickle=False)
        joined = []
        for (offset, run) in zip([0, len(originals[0])], source_runs):
            for row in read_csv(run / f'fold_{fold}' / 'predictions.csv'):
                row['embedding_row_index'] = str(int(row['embedding_row_index']) + offset)
                s3.require(combined[int(row['embedding_row_index'])]['record_id'] == row['record_id'], 'Combined fold record/index mismatch')
                joined.append(row)
        save_csv(dest / 'predictions.csv', joined)
        seen = {int(r['embedding_row_index']) for r in joined}
        s3.require(len(seen) == len(joined) == int(np.isfinite(embeddings).all(axis=1).sum()), 'Combined fold successes/embedding mismatch')
        original = np.load(source_runs[0] / f'fold_{fold}' / 'embedding.npy')
        s3.require(np.array_equal(embeddings[:32], original, equal_nan=True), 'Target embedding values changed')
    s3.require(read_csv(out / '01_A_step_predictions.csv')[:32] == originals[0], 'Existing target A prediction values changed')
    done = dict(status='COMPLETE_WITH_ROW_ERRORS' if any((r['a_step_status'] != 'OK' for r in combined)) else 'COMPLETE', selected_model='cnn_only', prediction_role='operational_ensemble_not_oof', model_bundle_sha256=s3.sha(s3.A_MODEL / 'selected_model_bundle.json'), counts=dict(top_rows=n, input_rows=n, prediction_OK=sum((r['a_step_status'] == 'OK' for r in combined))), input=str(prep / 'history_model_input.xlsx'), source_runs=[str(p) for p in source_runs], targets_preserved_exactly=True, targets=32, history_extra=n - 32, assembly='concatenation only; indices rebased by record_id; no refitting or averaging fold embedding coordinates', output_hashes={str(p): s3.sha(p) for p in out.rglob('*') if p.is_file()})
    s3.write_json(out / 'completion.json', done)
    return out

def run_c(prep, report, combined):
    cm = s3.load_reference('s3_c_external', 'model/3. Fuse_growth_history.py')
    cm.IMAGE_NAME = s3.IMAGE_NAME
    (cm.SUPPORTED_SEASON, cm.SEASON_START, cm.SEASON_END) = ('3작기', SEASON_START, SEASON_END)
    (models, model_hashes) = cm.load_models()
    (raw, daily) = cm.prepare_environment(prep / 'environment_excluded.csv')
    s3.require(len(raw) == len(daily) == 0, 'Excluded environment gained observations')
    (run, upstream) = cm.choose_a_run(combined)
    out = OUTPUT_ROOT / ('run_' + s3.stamp())
    out.mkdir(parents=True, exist_ok=False)
    started = dict(status='RUNNING', a_run=str(run), target_a_run=report['target_a_run'], input_preparation=str(prep), model_scope='frozen_season2_external_season3_environment_missing', source_hashes=model_hashes, model_training=False, environment_observed_days=0, ec_schedule_used=False, prediction_role='operational_ensemble_not_oof', selected='454columns_blend50')
    s3.write_json(out / 'started.json', started)
    t = time.time()
    with closing(sqlite3.connect(out / 'working.sqlite3')) as db:
        (headers, n) = cm.ingest(db, run)
        s3.require(n == upstream['counts']['top_rows'] == report['counts']['combined'], 'C/A/embedding row count mismatch')
        probs = cm.stage_probabilities(db, run, out, n)
        recovered = cm.recover_history_only(db, probs, out, SimpleNamespace(device='auto', batch_size=BATCH_SIZE))
        device = cm.infer_common_embedding(db, probs, out, n, SimpleNamespace(device='auto', batch_size=BATCH_SIZE))
        stats = cm.export_predictions(db, headers, models, daily, out, n)
    for (p, digest) in report['source_hashes'].items():
        s3.require(s3.sha(p) == digest, 'Protected source changed: ' + p)
    started.update(status='COMPLETE_WITH_ROW_ERRORS' if stats['unavailable'] else 'COMPLETE', counts=stats, device=device, elapsed_seconds=round(time.time() - t, 2), history_only_recovered=recovered, target_rows=32, history_only_rows=n - 32, history_scope=report['history_selection'], environment_scope=report['environment_policy'], embedding='fixed ImageNet B0 256 + frozen train-only PCA128; no A-coordinate mixing', output_hashes={str(p): s3.sha(p) for p in out.iterdir() if p.is_file()})
    s3.write_json(out / 'completion.json', started)
    return out

def verify_and_evaluate(out, prep, report):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    current = read_csv(Path(report['target_a_run']) / '01_A_step_predictions.csv')
    all_rows = read_csv(out / '01_C_step_predictions.csv')
    n = report['counts']['combined']
    s3.require(len(all_rows) == n and len({r['record_id'] for r in all_rows}) == n, 'C output lost/duplicated observations')
    lookup = {r['record_id']: r for r in all_rows}
    cemb = np.load(out / 'common_embedding_128.npy', mmap_mode='r')
    probs = np.load(out / 'stage_probabilities.npy', mmap_mode='r')
    s3.require(cemb.shape == (n, 128) and probs.shape == (n, 5), 'C embedding/probability shape mismatch')
    feature_rows = read_csv(out / '02_C_step_model_features.csv')
    cm = s3.load_reference('s3_c_verify', 'model/3. Fuse_growth_history.py')
    (models, _) = cm.load_models()
    environment = read_csv(out / '03_environment_by_capture_day.csv')
    s3.require(all((row[c] in ['', 'nan', 'NaN'] for row in environment for c in cm.ENV_COLS)), 'Excluded environment was filled')
    for row in feature_rows:
        idx = int(row['embedding_row_index'])
        s3.require(all_rows[idx]['record_id'] == row['record_id'], 'C feature/embedding ID mismatch')
        s3.require(np.isfinite(cemb[idx]).all(), 'Missing C embedding for usable feature row')
        s3.require(all((row[c] in ['', 'nan', 'NaN'] for c in cm.ENV_COLS if c in row)), 'Selected model environment was filled')
        s3.require(np.allclose(cemb[idx], [float(row[c]) for c in cm.EMB_COLS], rtol=0, atol=1e-05), 'C feature uses wrong embedding row')
    feature_map = {r['record_id']: r for r in feature_rows}
    combined = Path(s3.read_json(out / 'completion.json')['a_run'])
    fold_rows = [{r['record_id']: r for r in read_csv(combined / f'fold_{fold}' / 'predictions.csv')} for fold in range(1, 6)]
    for row in all_rows:
        idx = int(row['embedding_row_index'])
        if row['a_step_status'] == 'OK':
            expected = np.mean([[float(fold[row['record_id']][f'stage_p{k}']) for k in range(5)] for fold in fold_rows], axis=0)
            s3.require(np.allclose(probs[idx], expected, rtol=0, atol=1e-12), 'C stage probabilities not matched five-fold mean')
    with closing(sqlite3.connect((out / 'working.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        for target in current:
            identity = '|'.join((target[k] for k in ['season_id', 'bed_id', 'physical_slot']))
            records = []
            for (ts, payload, feat) in db.execute("SELECT ts,payload,feat FROM rows WHERE identity=? AND error='' AND feat IS NOT NULL ORDER BY ts", (identity,)):
                (r, v) = (json.loads(payload), json.loads(feat))
                records.append(dict(v, capture_ts=pd.Timestamp(ts), crop_filename=Path(r['crop_path']).name, source_tier='raw_direct', volume_proxy=max(v['area_cm2'], 0) ** 1.5))
            if not records:
                continue
            history = pd.DataFrame(records)
            cutoff = pd.to_datetime(timestamp(target), format='%Y%m%d%H%M%S')
            (prefix_features, _) = cm.causal_history(history, cutoff)
            future = dict(records[-1], capture_ts=cutoff + pd.Timedelta(days=1), area_cm2=1000000000000.0, pred_weight_cnn=1000000000000.0)
            (injected_features, _) = cm.causal_history(pd.concat([history, pd.DataFrame([future])], ignore_index=True), cutoff)
            pd.testing.assert_series_equal(pd.Series(prefix_features), pd.Series(injected_features), check_names=False)
    candidates = s3.rows_xlsx(s3.LABEL_FILE, 'feature')
    labels = {r['record_id']: r for r in candidates if r['match_eligible'] is True}
    evaluated = []
    for a in current:
        c = lookup[a['record_id']]
        label = labels[a['record_id']]
        idx = int(c['embedding_row_index'])
        for (name, value) in a.items():
            s3.require(c[name] == value, 'Target A metadata/prediction changed in C: ' + name)
        if c['c_step_status'] == 'OK':
            s3.require(c['c_step_history_last_timestamp'].replace('-', '').replace(':', '').replace(' ', '') <= timestamp(c), 'Future history in target')
            s3.require(c['c_step_env_source_date'] == '' and int(c['c_step_env_7d_observed_days']) == 0, 'Environment observations inserted')
            saved = feature_map[a['record_id']]
            f = pd.DataFrame([{name: float(saved[name]) if saved[name] else np.nan for name in models['direct_stacking'][0]}])
            pred = cm.predict(f, models)
            s3.require(math.isclose(float(c['c_step_pred_weight_g']), float(pred['blend50'][0]), rel_tol=1e-12), 'C saved prediction mismatch')
            s3.require(np.isfinite(cemb[idx]).all(), 'Target common embedding missing')
        evaluated.append(dict(sample_key=label['sample_key'], record_id=a['record_id'], embedding_row_index=idx, survey_date=label['survey_date'].strftime('%Y-%m-%d'), capture_date=a['capture_date'], lead_days=label['lead_days'], bed_id=a['bed_id'], physical_slot=a['physical_slot'], human_label=label['human_label'], fresh_weight_g=label['fresh_weight_g'], a_step_status=a['a_step_status'], a_step_pred_weight_g=float(a['pred_weight_g']) if a['pred_weight_g'] else None, **{k: float(v) if k.endswith('_g') and v else int(v) if k == 'c_step_history_n_days' and v else v for (k, v) in c.items() if k.startswith('c_step_')}))
    metrics = []
    groups = [('ALL_MATCHED', evaluated)] + [(d, [r for r in evaluated if r['survey_date'] == d]) for d in sorted(set((r['survey_date'] for r in evaluated)))]
    groups += [('SAME_CALENDAR_DAY', [r for r in evaluated if r['lead_days'] == 0]), ('PAST_1_TO_3_DAYS', [r for r in evaluated if r['lead_days'] > 0])]
    for (group, rr) in groups:
        good = [r for r in rr if r['c_step_status'] == 'OK']
        for (name, col) in [('A_baseline', 'a_step_pred_weight_g'), ('C_stacking', 'c_step_stacking_g'), ('C_residual', 'c_step_residual_g'), ('C_blend50', 'c_step_pred_weight_g')]:
            y = np.array([r['fresh_weight_g'] for r in good], float)
            p = np.array([r[col] for r in good], float)
            metrics.append(dict(group=group, model=name, targets=len(rr), predicted=len(good), unavailable=len(rr) - len(good), MAE_g=mean_absolute_error(y, p), RMSE_g=math.sqrt(mean_squared_error(y, p)), bias_g=float((p - y).mean()), R2=r2_score(y, p) if len(y) > 1 else None, environment_observed_days=0))
    ev = out / 'evaluation'
    ev.mkdir(exist_ok=False)
    save_csv(ev / 'metrics.csv', metrics)
    save_csv(ev / 'target_predictions.csv', evaluated)
    s3.save_workbooks([dict(path=str(ev / 'Cstep_evaluation.xlsx'), sheets=[s3.sheet_spec('Metrics', metrics), s3.sheet_spec('Target_predictions', evaluated)])], 'c_evaluation')
    report_out = dict(status='PASS', targets=len(evaluated), predicted=sum((r['c_step_status'] == 'OK' for r in evaluated)), unavailable=sum((r['c_step_status'] != 'OK' for r in evaluated)), aligned_observations=n, supervised_embedding_shapes=[n, 128], common_embedding_shape=list(cemb.shape), stage_probabilities_shape=list(probs.shape), target_A_values_unchanged=True, environment_17_all_nan=True, future_history_used=False, future_injection_invariance_pass=True, five_fold_stage_mean_verified=True, generated_environment_feature_count=17, selected_environment_feature_count=sum((c in feature_rows[0] for c in cm.ENV_COLS)), verification_script_sha256=s3.sha(__file__), inference_script_sha256=report['source_hashes'][str(Path(__file__))], scope='32 external matched targets only; history-only rows excluded from metrics; not OOF or complete-environment validation', metrics=metrics, protected_sources_unchanged=True, model_training=False, output_hashes={str(p): s3.sha(p) for p in ev.iterdir() if p.is_file()})
    s3.write_json(ev / 'completion.json', report_out)
    s3.write_json(OUTPUT_ROOT / 'latest_completed.json', dict(run=str(out), evaluation=str(ev)))
    print(json.dumps(report_out, ensure_ascii=False, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'c'], default='prepare')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--a-run', type=Path)
    parser.add_argument('--evaluate-run', type=Path, help='Validate completed inference without running models again')
    args = parser.parse_args()
    global A_RUN
    A_RUN = args.a_run
    if args.evaluate_run:
        done = s3.read_json(args.evaluate_run / 'completion.json')
        prep = Path(done['input_preparation'])
        report = s3.read_json(prep / 'completion.json')
        for (p, digest) in report['source_hashes'].items():
            if Path(p) != Path(__file__):
                s3.require(s3.sha(p) == digest, 'Protected model/source changed: ' + p)
        for (p, digest) in done['output_hashes'].items():
            s3.require(s3.sha(p) == digest, 'Completed inference output changed: ' + p)
        verify_and_evaluate(args.evaluate_run, prep, report)
        return
    if args.prepared:
        prep = args.prepared
        report = s3.read_json(prep / 'completion.json')
        for (p, digest) in report['source_hashes'].items():
            s3.require(s3.sha(p) == digest, 'Prepared source changed: ' + p)
        for (p, digest) in report['output_hashes'].items():
            s3.require(s3.sha(p) == digest, 'Prepared output changed: ' + p)
    else:
        (prep, report) = prepare()
    if args.stage == 'c':
        history = history_inference(prep, report)
        combined = combine(prep, report, history)
        out = run_c(prep, report, combined)
        verify_and_evaluate(out, prep, report)
if __name__ == '__main__':
    main()
