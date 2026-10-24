"""Processing utilities for pipeline existing bed."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import os
import pipeline_paths as P

def _index(roots, camera=None):
    result = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in P.IMAGE_EXTS:
                continue
            info = P.identity(path)
            datetime.strptime(info['capture_date'] + info['capture_time'], '%Y%m%d%H%M%S')
            if camera and info['camera'] != camera:
                raise ValueError(f"Invalid input or state in _index (rejected condition: camera and info['camera'] != camera): {path} ")
            key = (info['key'], info['camera'])
            if key in result:
                raise ValueError(f'Invalid input or state in _index (rejected condition: key in result): {result[key][0]} {path} ')
            result[key] = (path.resolve(), root.resolve(), info)
    return result

def inspect_sources(hash_files=True):
    from pipeline_runner import digest
    from PIL import Image
    from tqdm import tqdm
    if not P.ALLOW_UNKNOWN_RAW_OFFSET:
        raise ValueError('Invalid input or state in inspect_sources (rejected condition: not P.ALLOW_UNKNOWN_RAW_OFFSET)')
    inputs = [*P.RAW_ROOTS, *P.EXTERNAL_BED_ROOTS.values()]
    for root in inputs:
        if P.OUTPUT_ROOT == root or root in P.OUTPUT_ROOT.parents or P.OUTPUT_ROOT in root.parents:
            raise ValueError(f'Invalid input or state in inspect_sources (rejected condition: P.OUTPUT_ROOT == root or root in P.OUTPUT_ROOT.parents or P.OUTPUT_ROOT in root.parents): {root} ')
    print('inspect sources: Source· bed .', flush=True)
    raw = _index(P.RAW_ROOTS)
    beds = {}
    for (camera, root) in P.EXTERNAL_BED_ROOTS.items():
        beds.update(_index([root], camera))
    if not raw or not beds:
        raise ValueError('Invalid input or state in inspect_sources (rejected condition: not raw or not beds)')
    extras = set(beds) - set(raw)
    if extras:
        raise ValueError(f'Invalid input or state in inspect_sources (rejected condition: extras): {len(extras)} {sorted(extras)[:5]} ')

    def one(item):
        (key, (path, root, info)) = item
        row = dict(**info, raw_source_path=str(path), original_source_path=str(path), original_key=path.stem, relative_path=path.relative_to(root).as_posix(), raw_sha256=digest(path) if hash_files else '', raw_input_root=str(root), raw_bytes=path.stat().st_size, existing_bed_path='', existing_bed_sha256='', raw_offset_status='UNAVAILABLE_EXISTING_BED', x1='', y1='', x2='', y2='')
        match = beds.get(key)
        if match:
            (bed, bed_root, _) = match
            try:
                with Image.open(bed) as im:
                    im.load()
                    (width, height) = im.size
            except Exception as exc:
                raise ValueError(f'Invalid input or state in one (rejected condition: match): {bed} ') from exc
            if width <= 0 or height <= 0:
                raise ValueError(f'Invalid input or state in one (rejected condition: width <= 0 or height <= 0): {bed} ')
            row.update(existing_bed_path=str(bed), external_bed_root=str(bed_root), existing_bed_sha256=digest(bed) if hash_files else '', bed_width=width, bed_height=height, bed_bytes=bed.stat().st_size)
        return row
    records = []
    items = sorted(raw.items())
    with ThreadPoolExecutor(max_workers=4) as pool, tqdm(total=len(items), desc='inspect sources: Source·bed', unit='frame') as bar:
        for start in range(0, len(items), 64):
            for row in pool.map(one, items[start:start + 64]):
                records.append(row)
                bar.update()
    return records

def save_reference_xlsx(path, rows, fields):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing.xlsx')
    wb = Workbook(write_only=True)
    ws = wb.create_sheet('참조원장')
    ws.freeze_panes = 'A2'
    for (i, name) in enumerate(fields, 1):
        ws.column_dimensions[get_column_letter(i)].width = 55 if 'path' in name else 23
    header = []
    for name in fields:
        cell = WriteOnlyCell(ws, name)
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='244062')
        header.append(cell)
    ws.append(header)
    for row in rows:
        values = []
        for key in fields:
            value = row.get(key, '')
            if isinstance(value, str):
                cell = WriteOnlyCell(ws, value)
                cell.data_type = 's'
                values.append(cell)
            else:
                values.append(value)
        ws.append(values)
    ws.auto_filter.ref = f'A1:{get_column_letter(len(fields))}{len(rows) + 1}'
    wb.save(temporary)
    os.replace(temporary, path)

def bed_stage(raw):
    beds = []
    for source in raw:
        path = source['existing_bed_path']
        beds.append(dict(source, target_path=path, current_path=path, bed_sha256=source['existing_bed_sha256'], status='OK' if path else 'REVIEW', bed_status='REUSED' if path else 'MISSING_EXISTING_BED', reason='' if path else 'Existing bed image missing; no redetection or generated replacement'))
    fields = ['camera', 'bed_id', 'capture_date', 'capture_time', 'raw_source_path', 'raw_sha256']
    save_reference_xlsx(P.OUTPUT_ROOT / '0. 원본 선발' / 'source_reference.xlsx', beds, fields + ['bed_status', 'existing_bed_path'])
    save_reference_xlsx(P.BED_ROOT / 'bed_reference.xlsx', beds, fields + ['target_path', 'bed_sha256', 'bed_status', 'reason', 'bed_width', 'bed_height', 'raw_offset_status', 'x1', 'y1', 'x2', 'y2'])
    return beds

def logical_relative(frame):
    filename = frame.get('new_filename') or frame['old_filename']
    info = P.identity(Path(filename))
    return Path(info['camera']) / info['capture_date'] / filename

def ocr_stage(beds, mod, signature):
    from pipeline_runner import digest, keypath, read_csv, save_csv, save_json, reconcile_ocr_contract
    successful = [r for r in beds if r['status'] == 'OK']
    if not successful:
        raise ValueError('Invalid input or state in ocr_stage (rejected condition: not successful)')
    roots = [P.EXTERNAL_BED_ROOTS[k] for k in sorted(P.EXTERNAL_BED_ROOTS)]
    contract = dict(mode='external_bed_manifest_only', source_images={r['target_path']: r['bed_sha256'] for r in successful}, ocr_code=digest(Path(mod.__file__)))
    contract_path = P.MANAGEMENT / 'ocr_input_contract.json'
    if contract_path.is_file():
        contract = reconcile_ocr_contract(json.loads(contract_path.read_text(encoding='utf-8')), contract, mod)
    save_json(contract_path, contract)
    mapping = {keypath(r['target_path']): r for r in successful}
    initial = []
    for bed in successful:
        path = Path(bed['target_path'])
        info = P.identity(path)
        row = dict.fromkeys(mod.FIELDS, '')
        row.update(old_path=str(path), current_path=str(path), old_filename=path.name, input_roots_json=json.dumps([str(p) for p in roots]), version=mod.VERSION, status='DISCOVERED', old_bed_id=info['bed_id'], old_key=info['key'], camera=info['camera'], sha256=bed['bed_sha256'])
        initial.append(row)
    mod.GPU = P.DEVICE
    mod.run(roots, P.OCR_LOG, previews=False, rename_files=False, initial_rows=initial)
    frames = []
    logical = set()
    for row in read_csv(P.OCR_LOG / 'manifest.csv'):
        bed = mapping[keypath(row['old_path'])]
        if keypath(row['current_path']) != keypath(bed['target_path']) or row['sha256'] != bed['bed_sha256']:
            raise ValueError("Invalid input or state in ocr_stage (rejected condition: keypath(row['current_path']) != keypath(bed['target_path']) or row['sha256'] != bed['bed_sha256'])")
        if row['status'] not in {'MAPPED', 'UNCHANGED', 'REVIEW'}:
            raise RuntimeError(f"OCR incomplete or conflicting: {row['old_path']} / {row['status']}")
        row.update({k: bed[k] for k in ('raw_source_path', 'raw_sha256', 'original_source_path', 'original_key', 'external_bed_root', 'raw_offset_status')})
        row['bed_box'] = ['', '', '', '']
        relative = logical_relative(row)
        key = relative.with_suffix('.png').as_posix().casefold()
        if key in logical:
            raise ValueError('Output collision after OCR rename: ' + key)
        logical.add(key)
        frames.append(row)
    if len(frames) != len(successful) or {keypath(r['old_path']) for r in frames} != set(mapping):
        raise ValueError("Invalid input or state in ocr_stage (rejected condition: len(frames) != len(successful) or {keypath(r['old_path']) for r in frames} != set(mapping))")
    save_csv(P.OCR_ROOT / 'frame_manifest.csv', frames)
    return frames
