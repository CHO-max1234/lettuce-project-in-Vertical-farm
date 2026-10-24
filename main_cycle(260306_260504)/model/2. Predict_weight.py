"""Infer fresh weight with the saved five-fold image/numeric fusion model. Fold embeddings remain separate; prediction averages are operational outputs, not OOF training data. Same-day front measurements retain their recorded time offsets."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from pathlib import Path
INPUT_FILE = CYCLE.OUTPUT_ROOT / '01_CNN' / 'crop_features_CNN.xlsx'
INPUT_SHEET = '전체'
CROP_ROOT = CYCLE.CROP_ROOT
RESULT_ROOT = CYCLE.A_MODEL
MODEL_RUN = CYCLE.A_MODEL
OUTPUT_DIR = CYCLE.OUTPUT_ROOT / '02_A-step'
BATCH_SIZE = CYCLE.BATCH_SIZE
IMAGE_WORKERS = CYCLE.WORKERS
ROW_CHUNK = 256
DEVICE = CYCLE.DEVICE
USE_AMP = True
import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import sqlite3
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MT = '{' + NS['m'] + '}'
RT = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
IMAGE_NAME = re.compile('^(bed\\d+)_(20\\d{6})_(\\d{6})_(cam[012])_r(?:f)?_([bt]\\d+)(?:_p\\d+)?\\.(?:png|jpg|jpeg)$', re.I)
STAGE_NAMES = ['cnn_expected_stage', 'cnn_max_prob', 'cnn_entropy', *[f'cnn_p{i}' for i in range(5)]]
FRONT_COLUMNS = ['record_id', 'season_id', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'front_height_cm', 'volume_proxy', '식별검토사유', 'scale_qc']
ADDED = ['embedding_row_index', 'a_step_source_row', 'a_step_resolved_path', 'front_height_cm', 'volume_proxy', 'front_pair_status', 'front_record_id', 'front_capture_time', 'front_time_gap_seconds', 'front_after_top', 'front_candidate_count', 'front_identity_review', 'front_scale_qc', 'a_step_status', 'a_step_error', 'a_step_numeric_missing_count', 'pred_weight_g', 'a_step_model', 'prediction_role']

def text(x):
    return '' if x is None else str(x).strip()

def yes(x):
    return text(x).lower() in {'true', '1', '1.0', 'yes'}

def number(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except (TypeError, ValueError):
        return float('nan')

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def save_json(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def stream_rows(path, sheet, db):
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            for (i, row) in enumerate(csv.reader(f), 1):
                yield (i, row)
        return
    if path.suffix.lower() != '.xlsx':
        raise ValueError("Invalid input or state in stream_rows (rejected condition: path.suffix.lower() != '.xlsx')")
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rel = {e.attrib['Id']: e.attrib['Target'] for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        matching = [s for s in wb.find('m:sheets', NS) if s.attrib['name'] == sheet]
        if len(matching) != 1:
            raise ValueError(f'Invalid input or state in stream_rows (rejected condition: len(matching) != 1): {sheet!r} ')
        target = rel[matching[0].attrib[RT + 'id']]
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        db.execute('CREATE TEMP TABLE strings (id INTEGER PRIMARY KEY, value TEXT)')
        if 'xl/sharedStrings.xml' in z.namelist():
            with z.open('xl/sharedStrings.xml') as f:
                events = ET.iterparse(f, events=('start', 'end'))
                (_, root) = next(events)
                k = 0
                for (event, e) in events:
                    if event == 'end' and e.tag == MT + 'si':
                        db.execute('INSERT INTO strings VALUES (?,?)', (k, ''.join((t.text or '' for t in e.iter(MT + 't')))))
                        k += 1
                        e.clear()
                        root.remove(e)
        with z.open(member) as f:
            parent = None
            for (event, e) in ET.iterparse(f, events=('start', 'end')):
                if event == 'start' and e.tag == MT + 'sheetData':
                    parent = e
                if event != 'end' or e.tag != MT + 'row':
                    continue
                values = []
                for cell in e:
                    if cell.find('m:f', NS) is not None:
                        raise ValueError("Invalid input or state in stream_rows (rejected condition: cell.find('m:f', NS) is not None)")
                    col = 0
                    for ch in re.match('[A-Z]+', cell.attrib['r']).group():
                        col = col * 26 + ord(ch) - 64
                    values.extend([None] * max(0, col - len(values)))
                    kind = cell.attrib.get('t', 'n')
                    v = cell.find('m:v', NS)
                    raw = v.text if v is not None else None
                    if kind == 'inlineStr':
                        value = ''.join((t.text or '' for t in cell.iter(MT + 't')))
                    elif raw is None:
                        value = None
                    elif kind == 's':
                        value = db.execute('SELECT value FROM strings WHERE id=?', (int(raw),)).fetchone()[0]
                    elif kind == 'b':
                        value = raw == '1'
                    elif kind == 'n':
                        value = float(raw)
                    else:
                        value = raw
                    values[col - 1] = value
                index = int(e.attrib['r'])
                e.clear()
                if parent is not None:
                    parent.remove(e)
                yield (index, values)

def identity(row):
    """Parse bed, date, capture time and camera from a source filename."""
    season = text(row.get('season_id'))
    bed = text(row.get('bed_id')).lower()
    date = re.sub('\\.0$', '', text(row.get('capture_date')))
    hour = re.sub('\\.0$', '', text(row.get('capture_time'))).zfill(6)
    slot = text(row.get('physical_slot')).lower()
    if not re.fullmatch('bed\\d+', bed) or not re.fullmatch('[tb](?:[1-9]|1[0-2])', slot):
        raise ValueError('BED_OR_PHYSICAL_SLOT_INVALID')
    if not re.fullmatch('20\\d{6}', date) or not re.fullmatch('\\d{6}', hour):
        raise ValueError('DATE_OR_TIME_INVALID')
    dt = datetime.strptime(date + hour, '%Y%m%d%H%M%S')
    return ((season, bed, date, slot), dt.hour * 3600 + dt.minute * 60 + dt.second)

def create_tables(db):
    db.executescript('\n    PRAGMA temp_store=FILE;\n    CREATE TABLE top_rows (idx INTEGER PRIMARY KEY, record_id TEXT UNIQUE, source_row INTEGER, k TEXT, sec INTEGER, payload TEXT);\n    CREATE TABLE fronts (record_id TEXT PRIMARY KEY, k TEXT, sec INTEGER, payload TEXT);\n    CREATE INDEX front_key ON fronts(k,sec);\n    ')

def prepare_input(path, sheet, db, core, out):
    counts = Counter()
    create_tables(db)
    rows = stream_rows(path, sheet, db)
    (first, headers) = next(rows)
    headers = [text(v) for v in headers]
    required = {'record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'crop_path', 'crop_sha256', 'selected_record_id', 'selected_camera', 'slot_namespace_version', '분석가능', 'feature_status', 'cnn_status', 'cnn_inference_ok', *STAGE_NAMES, *core, 'front_height_cm', 'volume_proxy'}
    if first != 1 or not all(headers) or len(headers) != len(set(headers)):
        raise ValueError('Invalid input or state in prepare_input (rejected condition: first != 1 or not all(headers) or len(headers) != len(set(headers)))')
    if required - set(headers):
        raise ValueError(f'Invalid input or state in prepare_input (rejected condition: required - set(headers)): {sorted(required - set(headers))} ')
    if 'embedding_row_index' in headers:
        raise ValueError("Invalid input or state in prepare_input (rejected condition: 'embedding_row_index' in headers)")
    with (out / '03_input_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        audit = csv.writer(f)
        audit.writerow(['source_row', 'record_id', 'camera', 'action', 'reason'])
        for (excel_row, values) in tqdm(rows, desc='prepare input: Input', unit='row'):
            if not any((text(v) for v in values)):
                continue
            if len(values) > len(headers):
                raise ValueError(f'Invalid input or state in prepare_input (rejected condition: len(values) > len(headers)): {excel_row} ')
            row = dict(zip(headers, values + [None] * (len(headers) - len(values))))
            counts['input_rows'] += 1
            cam = text(row['camera']).lower()
            rid = text(row['record_id'])
            if not rid:
                raise ValueError(f'Invalid input or state in prepare_input (rejected condition: not rid): {excel_row} ')
            if cam not in {'cam0', 'cam1', 'cam2'}:
                raise ValueError(f"Invalid input or state in prepare_input (rejected condition: cam not in {{'cam0', 'cam1', 'cam2'}}): {excel_row} {cam} ")
            selected = text(row['selected_camera']).lower() == cam and text(row['selected_record_id']) == rid
            if not selected:
                counts['nonselected_rows'] += 1
                audit.writerow([excel_row, rid, cam, 'NOT_SELECTED', 'upstream camera selection'])
                continue
            err = ''
            try:
                (key, sec) = identity(row)
                if text(row['slot_namespace_version']) != 'physical_12col_v1':
                    raise ValueError('SLOT_NAMESPACE_MISMATCH')
                k = json.dumps(key, ensure_ascii=False)
            except ValueError as exc:
                (k, sec, err) = ('', -1, str(exc))
            if cam == 'cam2':
                (h, v) = (number(row['front_height_cm']), number(row['volume_proxy']))
                if err or not yes(row['분석가능']) or text(row['feature_status']) != 'OK' or (not (h > 0 and v > 0)):
                    counts['front_unavailable_rows'] += 1
                    audit.writerow([excel_row, rid, cam, 'FRONT_UNAVAILABLE', err or 'missing/invalid front feature'])
                    continue
                small = {c: row.get(c) for c in FRONT_COLUMNS}
                small.update(front_height_cm=h, volume_proxy=v)
                db.execute('INSERT INTO fronts VALUES (?,?,?,?)', (rid, k, sec, json.dumps(small, ensure_ascii=False)))
                counts['front_available_rows'] += 1
            else:
                idx = counts['top_rows']
                row.pop('front_height_cm2', None)
                row.update(embedding_row_index=idx, a_step_source_row=excel_row, a_step_error=err)
                db.execute('INSERT INTO top_rows VALUES (?,?,?,?,?,?)', (idx, rid, excel_row, k, sec, json.dumps(row, ensure_ascii=False)))
                counts['top_rows'] += 1
                if err:
                    audit.writerow([excel_row, rid, cam, 'RETAIN_WITH_ERROR', err])
            if counts['input_rows'] % ROW_CHUNK == 0:
                db.commit()
    db.commit()
    if not counts['top_rows']:
        raise ValueError("Invalid input or state in prepare_input (rejected condition: not counts['top_rows'])")
    return ([c for c in headers if c != 'front_height_cm2'], counts)

def attach_front(db, out, counts):
    with (out / '02_CAM2_height_volume.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FRONT_COLUMNS)
        writer.writeheader()
        for (payload,) in db.execute('SELECT payload FROM fronts ORDER BY record_id'):
            writer.writerow(json.loads(payload))
    cursor = db.execute('SELECT idx,k,sec,payload FROM top_rows ORDER BY idx')
    while True:
        chunk = cursor.fetchmany(ROW_CHUNK)
        if not chunk:
            break
        for (idx, k, sec, payload) in chunk:
            row = json.loads(payload)
            row.update(front_height_cm=None, volume_proxy=None, front_pair_status='NO_FRONT_MATCH', front_record_id='', front_capture_time='', front_time_gap_seconds=None, front_after_top=False, front_candidate_count=0, front_identity_review='', front_scale_qc='')
            candidates = db.execute('SELECT sec,payload FROM fronts WHERE k=? ORDER BY ABS(sec-?),sec,record_id', (k, sec)).fetchall() if k else []
            if candidates:
                (front_sec, raw) = candidates[0]
                front = json.loads(raw)
                gap = front_sec - sec
                row.update(front_height_cm=front['front_height_cm'], volume_proxy=front['volume_proxy'], front_pair_status='EXACT_TIME' if gap == 0 else 'SAME_DATE_NEAREST_TIME', front_record_id=front['record_id'], front_capture_time=front['capture_time'], front_time_gap_seconds=gap, front_after_top=gap > 0, front_candidate_count=len(candidates), front_identity_review=front.get('식별검토사유'), front_scale_qc=front.get('scale_qc'))
            counts['pair_' + row['front_pair_status']] += 1
            db.execute('UPDATE top_rows SET payload=? WHERE idx=?', (json.dumps(row, ensure_ascii=False), idx))
        db.commit()

class WeightNet(nn.Module):

    def __init__(self, num):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.image_embedding = nn.Sequential(nn.Linear(1280, 128), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.image_head = nn.Linear(128, 1)
        self.numeric_net = nn.Sequential(nn.Linear(num, 32), nn.ReLU(inplace=True), nn.Dropout(0.2))
        self.fusion_head = nn.Sequential(nn.Linear(160, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x, n):
        emb = self.image_embedding(self.backbone(x))
        return (self.fusion_head(torch.cat([emb, self.numeric_net(n)], 1)).squeeze(1), emb)

def checked_bundle(run):
    bundle = read_json(run / 'selected_model_bundle.json')
    if bundle['selected_model'] != 'cnn_only' or len(bundle['folds']) != 5:
        raise ValueError("Invalid input or state in checked_bundle (rejected condition: bundle['selected_model'] != 'cnn_only' or len(bundle['folds']) != 5)")
    if read_json(run / 'MODEL_SELECTION_FROZEN.json')['selected'] != 'cnn_only':
        raise ValueError("Invalid input or state in checked_bundle (rejected condition: read_json(run / 'MODEL_SELECTION_FROZEN.json')['selected'] != 'cnn_only')")
    core = bundle['core_numeric_features']
    if len(core) != 25:
        raise ValueError('Invalid input or state in checked_bundle (rejected condition: len(core) != 25)')
    for fold in bundle['folds']:
        for artifact in fold['files']:
            p = (run / artifact['path']).resolve()
            if not p.is_relative_to(run.resolve()) or sha256(p) != artifact['sha256']:
                raise ValueError(f"Invalid input or state in checked_bundle (rejected condition: not p.is_relative_to(run.resolve()) or sha256(p) != artifact['sha256']): {p} ")
        d = run / f"fold_{fold['fold']}"
        if read_json(d / 'numeric_feature_names.json') != core + STAGE_NAMES:
            raise ValueError("Invalid input or state in checked_bundle (rejected condition: read_json(d / 'numeric_feature_names.json') != core + STAGE_NAMES)")
    return (bundle, core)

def prepare_image(row, tf):
    try:
        if text(row.get('a_step_error')):
            raise ValueError(row['a_step_error'])
        if not yes(row['분석가능']):
            raise ValueError('FEATURE_NOT_ANALYSABLE')
        if text(row['feature_status']) != 'OK':
            raise ValueError('FEATURE_STATUS_NOT_OK')
        if not text(row['crop_path']):
            raise ValueError('NO_CROP_PATH')
        path = Path(text(row['crop_path']))
        if not path.is_file():
            (key, _) = identity(row)
            path = CROP_ROOT / key[1] / key[2] / path.name
        match = IMAGE_NAME.fullmatch(path.name)
        (key, _) = identity(row)
        hour = re.sub('\\.0$', '', text(row['capture_time'])).zfill(6)
        expected = (key[1], key[2], hour, text(row['camera']).lower(), key[3])
        if match is None or tuple((s.lower() for s in match.groups())) != expected:
            raise ValueError('CROP_FILENAME_IDENTITY_MISMATCH')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if not text(row['crop_sha256']) or digest != text(row['crop_sha256']).lower():
            raise ValueError('CROP_HASH_MISMATCH_OR_MISSING')
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert('RGB')
            result = {size: tf(rgb.resize((size, size), Image.Resampling.BILINEAR)) for size in (384, 256)}
        return (row, result, str(path), '')
    except (OSError, ValueError) as exc:
        return (row, None, '', f'{type(exc).__name__}: {exc}')

def stage_columns(p):
    p = p.astype(np.float64)
    return np.column_stack([p @ np.arange(5), p.max(1), -(p * np.log(np.clip(p, 1e-12, 1))).sum(1), p])

@torch.inference_mode()
def infer(db, run, out, bundle, core, n, device, batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    all_pred = np.lib.format.open_memmap(out / 'fold_predictions.npy', mode='w+', dtype='float32', shape=(n, 5))
    all_pred[:] = np.nan
    for fold_info in bundle['folds']:
        fold = fold_info['fold']
        d = run / f'fold_{fold}'
        dest = out / f'fold_{fold}'
        dest.mkdir()
        stage = efficientnet_b0(weights=None)
        stage.classifier[1] = nn.Linear(1280, 5)
        stage.load_state_dict(torch.load(d / 'stage.pt', map_location='cpu', weights_only=True)['model_state'])
        cp = torch.load(d / 'weight.pt', map_location='cpu', weights_only=True)
        if cp['numeric_dim'] != 33:
            raise ValueError("Invalid input or state in infer (rejected condition: cp['numeric_dim'] != 33)")
        weight = WeightNet(33)
        weight.load_state_dict(cp['model_state'])
        del cp
        for model in (stage, weight):
            model.to(device, memory_format=torch.channels_last).eval()
        proc = read_json(d / 'numeric_preprocessor.json')
        (median, mean, std) = (np.asarray(proc[k], float) for k in ('median', 'mean', 'std'))
        if any((a.shape != (33,) for a in (median, mean, std))) or not all((np.isfinite(a).all() for a in (median, mean, std))) or (std <= 0).any():
            raise ValueError('Invalid input or state in infer (rejected condition: any((a.shape != (33,) for a in (median, mean, std))) or not all((np.isfinite(a).all() for a in (median, mean, std))) or (std <= 0).any())')
        emb = np.lib.format.open_memmap(dest / 'embedding.npy', mode='w+', dtype='float32', shape=(n, 128))
        emb[:] = np.nan
        with (dest / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f, ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            writer = csv.writer(f)
            writer.writerow(['embedding_row_index', 'record_id', 'pred_weight_raw_g', *[f'stage_p{i}' for i in range(5)]])
            cursor = db.execute('SELECT payload FROM top_rows ORDER BY idx')
            with tqdm(total=n, desc=f'infer: Embedding/ fold{fold}', unit='image') as bar:
                while True:
                    chunk = cursor.fetchmany(batch_size)
                    if not chunk:
                        break
                    decoded = list(pool.map(lambda r: prepare_image(r, tf), [json.loads(r[0]) for r in chunk]))
                    good = [item for item in decoded if item[1] is not None]
                    for (row, _, path, error) in decoded:
                        if fold == 1:
                            row.update(a_step_resolved_path=path, a_step_error=error, a_step_status='IMAGE_ERROR' if error else 'PENDING', a_step_numeric_missing_count=sum((not math.isfinite(number(row.get(c))) for c in core)))
                            db.execute('UPDATE top_rows SET payload=? WHERE idx=?', (json.dumps(row, ensure_ascii=False), row['embedding_row_index']))
                    if good:
                        a = torch.stack([r[1][384] for r in good]).to(device, memory_format=torch.channels_last)
                        b = torch.stack([r[1][256] for r in good]).to(device, memory_format=torch.channels_last)
                        context = lambda : torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' and USE_AMP else nullcontext()
                        with context():
                            prob = torch.softmax(stage(a).float(), 1).cpu().numpy()
                        numeric = np.column_stack([np.array([[number(r[0].get(c)) for c in core] for r in good]), stage_columns(prob)])
                        numeric = ((np.where(np.isfinite(numeric), numeric, median) - mean) / std).astype(np.float32)
                        with context():
                            (prediction, embedding) = weight(b, torch.from_numpy(numeric).to(device))
                        pred = prediction.float().cpu().numpy()
                        e = embedding.float().cpu().numpy()
                        if not all((np.isfinite(v).all() for v in (prob, pred, e))):
                            raise ValueError('Invalid input or state in infer (rejected condition: not all((np.isfinite(v).all() for v in (prob, pred, e))))')
                        for (j, (row, _, _, _)) in enumerate(good):
                            idx = row['embedding_row_index']
                            all_pred[idx, fold - 1] = pred[j]
                            emb[idx] = e[j]
                            writer.writerow([idx, row['record_id'], float(pred[j]), *prob[j].tolist()])
                    bar.update(len(chunk))
                    db.commit()
        all_pred.flush()
        emb.flush()
        del emb, stage, weight
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return all_pred

def export_result(db, out, headers, fold_predictions, counts):
    columns = list(dict.fromkeys(headers + ADDED))
    with (out / '01_A_step_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (idx, payload) in db.execute('SELECT idx,payload FROM top_rows ORDER BY idx'):
            row = json.loads(payload)
            pred = fold_predictions[idx]
            ok = bool(np.isfinite(pred).all())
            row.update(pred_weight_g=max(0.0, float(pred.astype(float).mean())) if ok else None, a_step_model='cnn_only_20260918_5fold', prediction_role='operational_ensemble_not_oof', a_step_status='OK' if ok else 'PREDICTION_UNAVAILABLE')
            if not ok and (not text(row.get('a_step_error'))):
                row['a_step_error'] = 'ONE_OR_MORE_FOLDS_UNAVAILABLE'
            counts['prediction_' + row['a_step_status']] += 1
            writer.writerow(row)

def main():
    global CROP_ROOT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', type=Path, default=INPUT_FILE)
    p.add_argument('--sheet', default=INPUT_SHEET)
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    p.add_argument('--run', type=Path, default=MODEL_RUN)
    p.add_argument('--crop-root', type=Path, default=CROP_ROOT)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=DEVICE)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = p.parse_args()
    CROP_ROOT = args.crop_root.resolve()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive.')
    if True and (not args.input.is_file()):
        raise FileNotFoundError(f'Invalid input or state in main (rejected condition: True and (not args.input.is_file())): {args.input} ')
    start = time.time()
    (bundle, core) = checked_bundle(args.run)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable.')
    torch.set_num_threads(6)
    out = args.output_dir / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    before = args.input.stat()
    summary = dict(status='RUNNING', input=str(args.input), model_run=str(args.run), model_bundle_sha256=sha256(args.run / 'selected_model_bundle.json'), script_sha256=sha256(__file__), device=str(device), torch=str(torch.__version__), batch_size=args.batch_size, selected_model='cnn_only', front_used_in_prediction=False, embedding_layout='fold_N/embedding.npy: row=embedding_row_index; columns=emb_000..127; separate coordinates per fold', prediction_role='operational_ensemble_not_oof', top_policy='all upstream-selected CAM01 rows retained', front_policy='same season/bed/image date/physical slot; nearest time; no cross-date; may be after top, see front_after_top')
    save_json(out / 'run_started.json', summary)
    try:
        with closing(sqlite3.connect(out / 'working.sqlite3')) as db:
            (headers, counts) = prepare_input(args.input, args.sheet, db, core, out)
            attach_front(db, out, counts)
            pred = infer(db, args.run, out, bundle, core, counts['top_rows'], device, args.batch_size)
            export_result(db, out, headers, pred, counts)
            del pred
        if before is not None:
            after = args.input.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Invalid input or state in main (rejected condition: (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns))')
            summary.update(input_size_bytes=before.st_size, input_mtime_ns=before.st_mtime_ns, input_sha256=sha256(args.input))
        summary.update(status='COMPLETE' if not counts['prediction_PREDICTION_UNAVAILABLE'] else 'COMPLETE_WITH_ROW_ERRORS', counts=dict(counts), elapsed_seconds=round(time.time() - start, 2), completed_at=datetime.now().isoformat())
        save_json(out / 'completion.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'main: :{out}', flush=True)
    except Exception as exc:
        save_json(out / 'failure.json', dict(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=round(time.time() - start, 2)))
        raise
if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
