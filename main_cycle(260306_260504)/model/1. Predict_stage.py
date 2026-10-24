"""Append frozen five-stage CNN predictions to feature rows. Input order, unavailable rows and original numeric fields are preserved. No training is performed."""

import sys as _sys

from pathlib import Path as _Path

_CYCLE_ROOT = _Path(__file__).resolve().parents[1]

if str(_CYCLE_ROOT) not in _sys.path:

    _sys.path.insert(0, str(_CYCLE_ROOT))

import settings as CYCLE
_sys.path.insert(0, str(_CYCLE_ROOT / 'pipeline'))

from pathlib import Path

import argparse

import csv

FEATURE_XLSX = CYCLE.FEATURE_FILE

FEATURE_SHEET = '전체'

MODEL_ROOT = CYCLE.ARTIFACT_ROOT / 'cnn'

MODEL_PT = CYCLE.STAGE_MODEL

CROP_ROOT = CYCLE.CROP_ROOT

OUTPUT_XLSX = CYCLE.OUTPUT_ROOT / '01_CNN' / 'crop_features_CNN.xlsx'

CACHE_SQLITE = OUTPUT_XLSX.with_name('cnn_predictions_cache.sqlite3')

GATE_CSV = None

LABEL_CSV = None

EXPECTED_MODEL_SHA256 = 'dc2b180b471ba47f67e617d4d6aa3f43c08016460dee6cf9cfb8ead226587eb2'

DEVICE = CYCLE.DEVICE

ROW_CHUNK = 256

BATCH_SIZE = CYCLE.BATCH_SIZE

IMAGE_WORKERS = CYCLE.WORKERS

USE_AMP = True

INFERENCE_VERSION = '20260917_rgb_bilinear_softmax_fp32_v1'

import hashlib

import io

import itertools

import json

import math

import os

import posixpath

import re

import sqlite3

import sys

import tempfile

import time

import zipfile

from collections import Counter

from concurrent.futures import ThreadPoolExecutor

from contextlib import closing, nullcontext, suppress

import xml.etree.ElementTree as ET

import numpy as np

from PIL import Image

from openpyxl import Workbook

from openpyxl.cell import WriteOnlyCell

from openpyxl.utils import get_column_letter

import torch

from torch import nn

from torchvision import transforms

from torchvision.models import efficientnet_b0

from tqdm import tqdm

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

MAIN_NS = '{' + NS['m'] + '}'

REL_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

REQUIRED_COLUMNS = {'record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'crop_path', '분석가능'}

CNN_COLUMNS = ['cnn_inference_ok', 'cnn_expected_stage', 'cnn_pred_stage', 'cnn_max_prob', 'cnn_entropy', *[f'cnn_p{i}' for i in range(5)], 'cnn_status', 'cnn_error', 'cnn_resolved_path', 'cnn_model_sha256', 'cnn_source_excel_row']

GATE_COLUMNS = ['crop_integrity_status', 'crop_integrity_reason', 'crop_integrity_version']

MATCH_COLUMNS = ['lettuce_key', 'growth_match_status']

IMAGE_NAME = re.compile('^(bed\\d+)_(\\d{8})_(\\d{6})_(cam[012])_r(?:f)?_([bt]\\d+)(?:_p\\d+)?\\.(?:png|jpg|jpeg)$', re.I)



def text(value):

    return '' if value is None else str(value).strip()



def sha256_file(path):

    h = hashlib.sha256()

    with open(path, 'rb') as handle:

        for block in iter(lambda : handle.read(1024 * 1024), b''):

            h.update(block)

    return h.hexdigest()



def cell_column(reference):

    result = 0

    for letter in reference:

        if not letter.isalpha():

            break

        result = result * 26 + ord(letter.upper()) - ord('A') + 1

    return result - 1



def worksheet_member(archive, sheet_name):

    workbook = ET.fromstring(archive.read('xl/workbook.xml'))

    relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))

    targets = {e.attrib['Id']: e.attrib['Target'] for e in relationships}

    sheets = workbook.find('m:sheets', NS)

    for (index, sheet) in enumerate(sheets):

        if sheet.attrib['name'] != sheet_name:

            continue

        target = targets[sheet.attrib[REL_NS + 'id']]

        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)

        total = None

        for name in workbook.findall('m:definedNames/m:definedName', NS):

            if name.attrib.get('name') == '_xlnm._FilterDatabase' and name.attrib.get('localSheetId') == str(index):

                match = re.search(':\\$?[A-Z]+\\$?(\\d+)$', name.text or '')

                if match:

                    total = max(0, int(match.group(1)) - 1)

        return (member, total)

    raise ValueError(f'Invalid input or state in worksheet_member: {sheet_name!r} ')



def load_shared_strings(archive, connection):

    connection.execute('CREATE TEMP TABLE xlsx_strings (id INTEGER PRIMARY KEY, value TEXT)')

    if 'xl/sharedStrings.xml' not in archive.namelist():

        return

    with archive.open('xl/sharedStrings.xml') as handle:

        context = ET.iterparse(handle, events=('start', 'end'))

        (_, root) = next(context)

        index = 0

        for (event, element) in context:

            if event == 'end' and element.tag == MAIN_NS + 'si':

                value = ''.join((t.text or '' for t in element.iter(MAIN_NS + 't')))

                connection.execute('INSERT INTO xlsx_strings VALUES (?, ?)', (index, value))

                index += 1

                element.clear()

                root.remove(element)

    connection.commit()



def decode_cell(cell, connection):

    if cell.find('m:f', NS) is not None:

        raise ValueError("Invalid input or state in decode_cell (rejected condition: cell.find('m:f', NS) is not None)")

    kind = cell.attrib.get('t', 'n')

    if kind == 'inlineStr':

        return ''.join((t.text or '' for t in cell.iter(MAIN_NS + 't')))

    value = cell.find('m:v', NS)

    if value is None or value.text is None:

        return None

    raw = value.text

    if kind == 's':

        found = connection.execute('SELECT value FROM xlsx_strings WHERE id=?', (int(raw),)).fetchone()

        if found is None:

            raise ValueError(f'Invalid input or state in decode_cell (rejected condition: found is None): {raw} ')

        return found[0]

    if kind == 'b':

        return raw == '1'

    if kind == 'n':

        value = float(raw)

        if not math.isfinite(value):

            raise ValueError(f"Invalid input or state in decode_cell (rejected condition: not math.isfinite(value)): {cell.attrib.get('r')} ")

        return int(raw) if re.fullmatch('[-+]?\\d+', raw) else value

    if kind in {'str', 'd', 'e'}:

        return raw

    raise ValueError(f'Invalid input or state in decode_cell: {kind} ')



def iter_feature_rows(archive, member, connection):

    with archive.open(member) as handle:

        data_element = None

        for (event, element) in ET.iterparse(handle, events=('start', 'end')):

            if event == 'start' and element.tag == MAIN_NS + 'sheetData':

                data_element = element

            elif event == 'end' and element.tag == MAIN_NS + 'row':

                pairs = [(cell_column(c.attrib['r']), decode_cell(c, connection)) for c in element]

                width = max((index for (index, _) in pairs), default=-1) + 1

                values = [None] * width

                for (index, value) in pairs:

                    values[index] = value

                excel_row = int(element.attrib['r'])

                element.clear()

                if data_element is not None:

                    data_element.remove(element)

                yield (excel_row, values)



def load_model():

    model_hash = sha256_file(MODEL_PT)

    if EXPECTED_MODEL_SHA256 and model_hash != EXPECTED_MODEL_SHA256:

        raise ValueError('Invalid input or state in load_model (rejected condition: EXPECTED_MODEL_SHA256 and model_hash != EXPECTED_MODEL_SHA256)')

    checkpoint = torch.load(MODEL_PT, map_location='cpu', weights_only=True)

    if checkpoint.get('backbone') != 'efficientnet_b0' or checkpoint.get('num_classes') != 5:

        raise ValueError("Invalid input or state in load_model (rejected condition: checkpoint.get('backbone') != 'efficientnet_b0' or checkpoint.get('num_classes') != 5)")

    config = checkpoint['config']

    size = int(checkpoint['img_size'])

    if size != 384 or config['image_size'] != size:

        raise ValueError("Invalid input or state in load_model (rejected condition: size != 384 or config['image_size'] != size)")

    device = torch.device('cuda' if DEVICE == 'auto' and torch.cuda.is_available() else 'cpu' if DEVICE == 'auto' else DEVICE)

    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and (not torch.cuda.is_available())):

        raise ValueError(f"Invalid input or state in load_model (rejected condition: device.type not in {{'cpu', 'cuda'}} or (device.type == 'cuda' and (not torch.cuda.is_available()))): {device} ")

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    model.load_state_dict(checkpoint['model_state'], strict=True)

    model.to(device, memory_format=torch.channels_last).eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(config['normalization_mean'], config['normalization_std'])])

    signature_data = dict(version=INFERENCE_VERSION, model=model_hash, size=size, mean=config['normalization_mean'], std=config['normalization_std'], torch=str(torch.__version__), device=str(device), amp=USE_AMP and device.type == 'cuda')

    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode('utf-8')).hexdigest()

    del checkpoint

    return (model, transform, device, size, model_hash, signature)



def blank_result(excel_row, model_hash):

    result = dict.fromkeys(CNN_COLUMNS)

    result.update(cnn_inference_ok=False, cnn_status='PENDING', cnn_error='', cnn_resolved_path='', cnn_model_sha256=model_hash, cnn_source_excel_row=excel_row)

    return result



def prepare_image(item, transform, size, model_hash):

    (index, excel_row, row) = item

    result = blank_result(excel_row, model_hash)

    raw_path = text(row.get('crop_path'))

    if not raw_path:

        result['cnn_status'] = 'NO_CROP'

        return (index, result, None, None)

    flag = text(row.get('분석가능')).lower()

    if flag not in {'true', '1', '1.0'}:

        result['cnn_status'] = 'QC_EXCLUDED' if flag in {'false', '0', '0.0'} else 'ANALYSIS_FLAG_UNKNOWN'

        return (index, result, None, None)

    try:

        resolver = getattr(CYCLE, 'resolve_crop_path', None)
        path = resolver(raw_path, CROP_ROOT) if resolver else Path(raw_path)

        if not path.is_absolute():

            path = FEATURE_XLSX.parent / path

        path = path.resolve()

    except (OSError, ValueError) as exc:

        result.update(cnn_status='IMAGE_PATH_ERROR', cnn_error=f'{type(exc).__name__}: {exc}')

        return (index, result, None, None)

    result['cnn_resolved_path'] = str(path)

    if not path.is_relative_to(CROP_ROOT.resolve()):

        result.update(cnn_status='OUTSIDE_CROP_ROOT', cnn_error='crop_path is outside the configured crop root')

        return (index, result, None, None)

    match = IMAGE_NAME.fullmatch(path.name)

    expected = (text(row['bed_id']).lower(), text(row['capture_date']), text(row['capture_time']).zfill(6), text(row['camera']).lower(), text(row['physical_slot']).lower())

    if match is None or tuple((v.lower() for v in match.groups())) != expected:

        result.update(cnn_status='IDENTITY_MISMATCH', cnn_error='crop 파일명과 feature의 bed/date/time/camera/slot이 다릅니다.')

        return (index, result, None, None)

    try:

        raw = path.read_bytes()

        image_hash = hashlib.sha256(raw).hexdigest()

        expected_hash = text(row.get('crop_sha256')).lower()

        if expected_hash and image_hash != expected_hash:

            result.update(cnn_status='CROP_HASH_MISMATCH', cnn_error='feature 계산 당시 crop과 파일 내용이 다릅니다.')

            return (index, result, None, None)

        with Image.open(io.BytesIO(raw)) as image:

            rgb = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)

            tensor = transform(rgb)

        return (index, result, tensor, image_hash)

    except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:

        result.update(cnn_status='IMAGE_READ_ERROR', cnn_error=f'{type(exc).__name__}: {exc}')

        return (index, result, None, None)



def fill_prediction(result, probabilities):

    p = np.asarray(probabilities, dtype=np.float64)

    if p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any() or (abs(p.sum() - 1) > 1e-05):

        raise ValueError('Invalid input or state in fill_prediction (rejected condition: p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any() or (abs(p.sum() - 1) > 1e-05))')

    result.update(cnn_inference_ok=True, cnn_status='OK', cnn_expected_stage=float(p @ np.arange(5)), cnn_pred_stage=int(p.argmax()), cnn_max_prob=float(p.max()), cnn_entropy=float(-(p * np.log(np.clip(p, 1e-12, 1))).sum()))

    result.update({f'cnn_p{i}': float(p[i]) for i in range(5)})



def infer_chunk(records, headers, model, transform, device, size, model_hash, signature, connection, pool, counters):

    results = [None] * len(records)

    for start in range(0, len(records), BATCH_SIZE):

        items = [(i, records[i][0], dict(zip(headers, records[i][1]))) for i in range(start, min(start + BATCH_SIZE, len(records)))]

        loaded = pool.map(lambda item: prepare_image(item, transform, size, model_hash), items)

        (pending, tensors) = ([], [])

        for (index, result, tensor, image_hash) in loaded:

            results[index] = result

            if tensor is None:

                continue

            cached = connection.execute('SELECT probabilities FROM predictions WHERE signature=? AND image_sha256=?', (signature, image_hash)).fetchone()

            if cached is not None:

                fill_prediction(result, json.loads(cached[0]))

                counters['cache_hits'] += 1

            else:

                pending.append((index, image_hash))

                tensors.append(tensor)

        if tensors:

            batch = torch.stack(tensors).to(device, memory_format=torch.channels_last)

            amp = torch.autocast('cuda', dtype=torch.float16) if USE_AMP and device.type == 'cuda' else nullcontext()

            with torch.inference_mode(), amp:

                logits = model(batch)

            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()

            for ((index, image_hash), p) in zip(pending, probabilities):

                fill_prediction(results[index], p)

                connection.execute('INSERT OR REPLACE INTO predictions VALUES (?, ?, ?)', (signature, image_hash, json.dumps(p.tolist())))

                counters['inferred_images'] += 1

            del batch, logits, probabilities

    connection.commit()

    assert all((result is not None for result in results))

    return results



def excel_values(sheet, values):

    for value in values:

        if isinstance(value, str):

            cell = WriteOnlyCell(sheet, value=value)

            cell.data_type = 's'

            yield cell

        else:

            yield value



def main():

    if min(ROW_CHUNK, BATCH_SIZE, IMAGE_WORKERS) < 1:

        raise ValueError('chunk, batch and workers must be positive.')

    if FEATURE_XLSX.resolve() == OUTPUT_XLSX.resolve():

        raise ValueError('Invalid input or state in main (rejected condition: FEATURE_XLSX.resolve() == OUTPUT_XLSX.resolve())')

    if not FEATURE_XLSX.is_file() or not MODEL_PT.is_file() or (not CROP_ROOT.is_dir()):

        raise FileNotFoundError('Invalid input or state in main (rejected condition: not FEATURE_XLSX.is_file() or not MODEL_PT.is_file() or (not CROP_ROOT.is_dir()))')

    if GATE_CSV is None or not Path(GATE_CSV).is_file():

        raise FileNotFoundError('Invalid input or state in main (rejected condition: GATE_CSV is None or not Path(GATE_CSV).is_file())')

    torch.set_num_threads(6)

    (model, transform, device, size, model_hash, signature) = load_model()

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    input_stat = FEATURE_XLSX.stat()

    counters = Counter()

    started = time.time()

    print(f'Input: {FEATURE_XLSX} [{FEATURE_SHEET}main: ] :{MODEL_PT}main: :{device}\nOutput: {OUTPUT_XLSX}', flush=True)

    with closing(sqlite3.connect(CACHE_SQLITE)) as connection, zipfile.ZipFile(FEATURE_XLSX) as archive:

        connection.execute('PRAGMA temp_store=FILE')

        connection.execute('CREATE TABLE IF NOT EXISTS predictions (signature TEXT, image_sha256 TEXT, probabilities TEXT, PRIMARY KEY(signature,image_sha256))')

        connection.execute('CREATE TABLE IF NOT EXISTS crop_gate (record_id TEXT PRIMARY KEY, model_eligible INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL)')

        connection.execute('DELETE FROM crop_gate')

        gate_count = 0

        with Path(GATE_CSV).open('r', encoding='utf-8-sig', newline='') as gate_file:

            gate_reader = csv.DictReader(gate_file)

            required_gate = {'record_id', 'model_eligible', 'integrity_status', 'integrity_reason'}

            if not required_gate.issubset(gate_reader.fieldnames or []):

                raise ValueError(f'crop gate Missing required columns: {sorted(required_gate - set(gate_reader.fieldnames or []))}')

            for gate_row in gate_reader:

                connection.execute('INSERT INTO crop_gate VALUES (?,?,?,?)', (gate_row['record_id'], int(gate_row['model_eligible'].lower() in {'true', '1'}), gate_row['integrity_status'], gate_row['integrity_reason']))

                gate_count += 1

        connection.commit()

        if LABEL_CSV is not None:

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

            from growth_labels import image_key

            connection.execute('CREATE TABLE IF NOT EXISTS growth_keys (lettuce_key TEXT PRIMARY KEY, status TEXT NOT NULL)')

            connection.execute('DELETE FROM growth_keys')

            with Path(LABEL_CSV).open('r', encoding='utf-8-sig', newline='') as label_file:

                for label in csv.DictReader(label_file):

                    connection.execute('INSERT INTO growth_keys VALUES (?,?)', (label['lettuce_key'], label['status']))

            connection.commit()

        load_shared_strings(archive, connection)

        (member, total) = worksheet_member(archive, FEATURE_SHEET)

        if total is not None and total > 1048575:

            raise ValueError('Invalid input or state in main (rejected condition: total is not None and total > 1048575)')

        rows = iter_feature_rows(archive, member, connection)

        (header_row, raw_headers) = next(rows)

        headers = [text(v) for v in raw_headers]

        if header_row != 1 or not all(headers) or len(set(headers)) != len(headers):

            raise ValueError('Invalid input or state in main (rejected condition: header_row != 1 or not all(headers) or len(set(headers)) != len(headers))')

        if not REQUIRED_COLUMNS.issubset(headers):

            raise ValueError(f'Invalid input or state in main (rejected condition: not REQUIRED_COLUMNS.issubset(headers)): {sorted(REQUIRED_COLUMNS - set(headers))} ')

        if set(headers) & set(CNN_COLUMNS):

            raise ValueError('Invalid input or state in main (rejected condition: set(headers) & set(CNN_COLUMNS))')

        if set(headers) & (set(GATE_COLUMNS) | set(MATCH_COLUMNS)):

            raise ValueError('Invalid input or state in main (rejected condition: set(headers) & (set(GATE_COLUMNS) | set(MATCH_COLUMNS)))')

        if 'record_id' not in headers:

            raise ValueError("Invalid input or state in main (rejected condition: 'record_id' not in headers)")

        original_headers = list(headers)

        id_index = headers.index('record_id')

        (date_index, bed_index, slot_index) = (headers.index(k) for k in ('capture_date', 'bed_id', 'physical_slot'))

        analysis_index = headers.index('분석가능') if '분석가능' in headers else None

        output_headers = headers + ([] if analysis_index is not None else ['분석가능']) + GATE_COLUMNS + MATCH_COLUMNS

        with tempfile.TemporaryDirectory(prefix='cnn_merge_', dir=OUTPUT_XLSX.parent) as temp_dir:

            temporary_root = Path(temp_dir).resolve()

            output_root = OUTPUT_XLSX.parent.resolve()

            if temporary_root == output_root or not temporary_root.is_relative_to(output_root):

                raise RuntimeError('Invalid input or state in main (rejected condition: temporary_root == output_root or not temporary_root.is_relative_to(output_root))')

            previous_temp = tempfile.tempdir

            tempfile.tempdir = temp_dir

            workbook = None

            try:

                workbook = Workbook(write_only=True)

                sheet = workbook.create_sheet(FEATURE_SHEET)

                sheet.freeze_panes = 'A2'

                sheet.append(list(excel_values(sheet, output_headers + CNN_COLUMNS)))

                with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool, tqdm(total=total, desc='FEATURES + CNN', unit='row') as progress:

                    while True:

                        records = list(itertools.islice(rows, ROW_CHUNK))

                        if not records:

                            break

                        for (offset, (excel_row, values)) in enumerate(records):

                            if excel_row != counters['rows'] + offset + 2 or len(values) > len(original_headers):

                                raise ValueError(f"Invalid input or state in main (rejected condition: excel_row != counters['rows'] + offset + 2 or len(values) > len(original_headers)): {excel_row} ")

                            if excel_row > 1048576:

                                raise ValueError('Invalid input or state in main (rejected condition: excel_row > 1048576)')

                            values.extend([None] * (len(original_headers) - len(values)))

                            gate = connection.execute('SELECT model_eligible,status,reason FROM crop_gate WHERE record_id=?', (text(values[id_index]),)).fetchone()

                            if gate is None:

                                raise ValueError(f'Invalid input or state in main (rejected condition: gate is None): {values[id_index]} ')

                            if LABEL_CSV is not None:

                                try:

                                    key = image_key(values[date_index], values[bed_index], values[slot_index])

                                    found = connection.execute('SELECT status FROM growth_keys WHERE lettuce_key=?', (key,)).fetchone()

                                    match_status = found[0] if found is not None else 'NO_LABEL'

                                except ValueError:

                                    (key, match_status) = ('', 'INVALID_IMAGE_KEY')

                            else:

                                (key, match_status) = ('', 'ALL_IMAGES')

                            prior = True if analysis_index is None else text(values[analysis_index]).lower() in {'true', '1', '1.0'}

                            allowed = bool(gate[0]) and prior

                            if analysis_index is None:

                                values.append(allowed)

                            else:

                                values[analysis_index] = allowed

                            values.extend([gate[1], gate[2], 'all_crop_mask_integrity_v1_20260923'])

                            values.extend([key, match_status])

                            counters['crop_gate_review'] += int(not bool(gate[0]))

                            counters['growth_unmatched'] += int(match_status not in {'OK', 'ALL_IMAGES'})

                        results = infer_chunk(records, output_headers, model, transform, device, size, model_hash, signature, connection, pool, counters)

                        for ((_, values), result) in zip(records, results):

                            sheet.append(list(excel_values(sheet, values + [result[c] for c in CNN_COLUMNS])))

                            counters['rows'] += 1

                            counters['status_' + result['cnn_status']] += 1

                        progress.update(len(records))

                        progress.set_postfix(ok=counters['status_OK'], cached=counters['cache_hits'], refresh=False)

                if total is not None and counters['rows'] != total:

                    raise ValueError(f"Invalid input or state in main (rejected condition: total is not None and counters['rows'] != total): {total} {counters['rows']} ")

                if counters['rows'] != gate_count:

                    raise ValueError(f"Invalid input or state in main (rejected condition: counters['rows'] != gate_count): {counters['rows']} {gate_count} ")

                after = FEATURE_XLSX.stat()

                if (input_stat.st_size, input_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):

                    raise RuntimeError('Invalid input or state in main (rejected condition: (input_stat.st_size, input_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns))')

                sheet.auto_filter.ref = f"A1:{get_column_letter(len(output_headers) + len(CNN_COLUMNS))}{counters['rows'] + 1}"

                summary = workbook.create_sheet('CNN_summary')

                summary.append(['Item', 'Value'])

                metadata = dict(input=str(FEATURE_XLSX), input_sheet=FEATURE_SHEET, model=str(MODEL_PT), model_sha256=model_hash, inference_signature=signature, device=str(device), image_size=size, completed_at=time.strftime('%Y-%m-%d %H:%M:%S'), elapsed_seconds=round(time.time() - started, 1), **dict(counters))

                for (key, value) in metadata.items():

                    summary.append(list(excel_values(summary, [key, value])))

                summary.append(['Interpretation', '추론 성공은 학습 적합성·수확 판정 승인이 아닙니다. 기존 QC/식별검토/선택 열을 함께 사용하세요.'])

                summary.append(['Failed rows', 'NO_CROP/QC_EXCLUDED/읽기·해시·식별 오류 행도 보존하며 예측은 빈칸입니다.'])

                temporary_output = Path(temp_dir) / OUTPUT_XLSX.name

                print('main: . XLSX …', flush=True)

                workbook.save(temporary_output)

                os.replace(temporary_output, OUTPUT_XLSX)

            finally:

                if workbook is not None:

                    for worksheet in workbook.worksheets:

                        if not worksheet.closed:

                            with suppress(Exception):

                                worksheet.close()

                tempfile.tempdir = previous_temp

    print(f'main: :{OUTPUT_XLSX}\n{dict(counters)}', flush=True)

if __name__ == '__main__':

    if hasattr(sys.stdout, 'reconfigure'):

        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--feature', type=Path, required=True)

    parser.add_argument('--crop-root', type=Path, required=True)

    parser.add_argument('--model', type=Path, required=True)

    parser.add_argument('--output', type=Path, required=True)

    parser.add_argument('--gate-csv', type=Path, required=True)

    parser.add_argument('--label-csv', type=Path)

    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')

    args = parser.parse_args()

    (FEATURE_XLSX, CROP_ROOT, MODEL_PT, OUTPUT_XLSX) = (args.feature, args.crop_root, args.model, args.output)

    (CACHE_SQLITE, GATE_CSV, LABEL_CSV, DEVICE) = (OUTPUT_XLSX.with_name('cnn_predictions_cache.sqlite3'), args.gate_csv, args.label_csv, args.device)

    main()

