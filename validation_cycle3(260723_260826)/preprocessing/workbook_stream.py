"""Workbook stream calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
import hashlib, json, math, os, re, zipfile
from itertools import zip_longest
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET
from tqdm import tqdm
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'

def letters(i):
    result = ''
    while i:
        (i, r) = divmod(i - 1, 26)
        result = chr(65 + r) + result
    return result

def value(v):
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, separators=(',', ':'))
    if v is None:
        return None
    if isinstance(v, float) and (not math.isfinite(v)):
        raise ValueError('Invalid input or execution contract')
    if isinstance(v, str):
        if len(v) > 32767:
            raise ValueError('Invalid input or execution contract')
        if re.search('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]', v):
            raise ValueError('Invalid input or execution contract')
    return v

def row_xml(number, columns, record, styles):
    parts = [f'<row r="{number}">']
    for (i, k) in enumerate(columns, 1):
        v = value(record.get(k))
        if v is None:
            continue
        style = f' s="{styles[i - 1]}"' if styles[i - 1] else ''
        ref = f'{letters(i)}{number}'
        if isinstance(v, bool):
            parts.append(f'<c r="{ref}"{style} t="b"><v>{int(v)}</v></c>')
        elif isinstance(v, (float, int)):
            parts.append(f'<c r="{ref}"{style}><v>{repr(v)}</v></c>')
        else:
            parts.append(f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>')
    parts.append('</row>')
    return ''.join(parts).encode('utf-8')

def write_workbook(template, target, columns, rows, count):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet('Data')
    sheet.freeze_panes = 'C2'
    sheet.append(columns)
    n = 0
    for (n, record) in enumerate(tqdm(rows, total=count, desc=target.stem, mininterval=5), 1):
        cells = []
        for column in columns:
            item = value(record.get(column))
            cell = WriteOnlyCell(sheet, item)
            if isinstance(item, str):
                cell.data_type = 's'
            cells.append(cell)
        sheet.append(cells)
    assert n == count, (n, count)
    sheet.auto_filter.ref = f'A1:{letters(len(columns))}{count + 1}'
    temporary = target.with_suffix('.writing.xlsx')
    workbook.save(temporary)
    os.replace(temporary, target)

def verify_workbook(path, columns, expected_rows, count):
    expected = iter(expected_rows)
    compared = 0
    cells = 0
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            tree = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared = [''.join(n.itertext()) for n in tree]
        with archive.open('xl/worksheets/sheet1.xml') as source:
            root = None
            for (event, element) in ET.iterparse(source, events=('start', 'end')):
                if root is None:
                    root = element
                if event != 'end' or element.tag != f'{{{NS}}}row':
                    continue
                values = {}
                for cell in element:
                    address = cell.attrib.get('r', '')
                    col = re.sub('\\d', '', address)
                    i = 0
                    for c in col:
                        i = i * 26 + ord(c) - 64
                    t = cell.attrib.get('t')
                    node = cell.find(f'{{{NS}}}v')
                    if t == 'inlineStr':
                        v = ''.join(cell.find(f'{{{NS}}}is').itertext())
                    elif t == 's':
                        v = shared[int(node.text)]
                    elif t == 'b':
                        v = bool(int(node.text))
                    elif t == 'str':
                        v = node.text or ''
                    elif node is not None:
                        v = float(node.text)
                    else:
                        v = None
                    values[i - 1] = v
                if element.attrib['r'] == '1':
                    assert [values.get(i) for i in range(len(columns))] == columns, 'Input contract failed: [values.get(i) for i in range(len(columns))] == columns'
                else:
                    record = next(expected, None)
                    if record is None:
                        raise AssertionError('Excel 추가행')
                    for (i, k) in enumerate(columns):
                        a = values.get(i)
                        b = value(record.get(k))
                        if isinstance(b, (float, int)) and (not isinstance(b, bool)):
                            assert a is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (path.name, compared, k, a, b)
                        else:
                            assert a == b, (path.name, compared, k, a, b)
                    compared += 1
                    cells += len(columns)
                    if compared % 50000 == 0:
                        print(f'{path.name}: verified {compared:,}/{count:,}', flush=True)
                element.clear()
                if compared % 1000 == 0:
                    data = root.find(f'{{{NS}}}sheetData')
                    if data is not None:
                        data.clear()
    assert compared == count and next(expected, None) is None
    return dict(status='PASS', rows=compared, columns=len(columns), cells_compared=cells, full_cell_value_comparison=True, zip_crc_and_xml_read=True)
