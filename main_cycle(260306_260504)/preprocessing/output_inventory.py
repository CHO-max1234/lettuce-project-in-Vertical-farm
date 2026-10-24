"""Processing utilities for output inventory."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from collections import Counter
from pathlib import Path
import csv
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}
REPORT_NAMES = {'output_inventory.csv', 'output_counts.json'}
DEFAULT_EXCLUDED_DIRS = ('testimages', '0. 원본 선발')

def _within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def _csv_rows(path):
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(64 * 1024 * 1024)
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, strict=True)
            count = 0
            for row in reader:
                if None in row:
                    raise ValueError(f'Invalid input or state in _csv_rows (rejected condition: None in row): {path} {reader.line_num} ')
                if any((value is None for value in row.values())):
                    raise ValueError(f'Invalid input or state in _csv_rows (rejected condition: any((value is None for value in row.values()))): {path} {reader.line_num} ')
                count += 1
            return count
    except csv.Error as error:
        raise ValueError(f'Invalid input or state in _csv_rows: {path} {error} ') from error
    finally:
        csv.field_size_limit(previous_limit)

def _xlsx_rows(path):
    import posixpath
    from zipfile import ZipFile
    from xml.etree import ElementTree
    from xml.parsers import expat
    main_ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relation_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with path.open('rb') as source, ZipFile(source) as workbook:
        root = ElementTree.fromstring(workbook.read('xl/workbook.xml'))
        relations = ElementTree.fromstring(workbook.read('xl/_rels/workbook.xml.rels'))
        targets = {item.attrib['Id']: item for item in relations}
        counts = {}
        for sheet in root.find('{' + main_ns + '}sheets'):
            relation = targets[sheet.attrib['{' + relation_ns + '}id']]
            if relation.attrib.get('TargetMode') == 'External':
                raise ValueError(f"Invalid input or state in _xlsx_rows (rejected condition: relation.attrib.get('TargetMode') == 'External'): {path} ")
            target = relation.attrib['Target']
            member = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join('xl', target))
            if member.startswith('../') or member.startswith('/'):
                raise ValueError(f"Invalid input or state in _xlsx_rows (rejected condition: member.startswith('../') or member.startswith('/')): {path} {target} ")
            state = {'last_row': 0, 'dimension_rows': None}

            def start(tag, attrs):
                if tag == main_ns + ' row':
                    number = int(attrs.get('r', state['last_row'] + 1))
                    state['last_row'] = max(state['last_row'], number)
                elif tag == main_ns + ' dimension' and attrs.get('ref'):
                    import re
                    end = attrs['ref'].split(':')[-1]
                    match = re.search('(\\d+)$', end)
                    if match:
                        state['dimension_rows'] = int(match[1])
            parser = expat.ParserCreate(namespace_separator=' ')
            parser.StartElementHandler = start
            with workbook.open(member) as handle:
                for block in iter(lambda : handle.read(1024 * 1024), b''):
                    parser.Parse(block, False)
                parser.Parse(b'', True)
            rows = state['last_row']
            if state['dimension_rows'] is not None:
                rows = min(rows, state['dimension_rows'])
            counts[sheet.attrib['name']] = max(0, rows - 1)
        return counts

def collect(root, excluded_roots=()):
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f'Validation failed in collect: :{root}')
    excluded = [root / name for name in DEFAULT_EXCLUDED_DIRS]
    for item in excluded_roots:
        item = Path(item)
        excluded.append(item if item.is_absolute() else root / item)
    excluded = [item.resolve() for item in excluded]
    pending = [root]
    visited = set()
    while pending:
        directory = pending.pop()
        resolved = directory.resolve()
        if resolved in visited or not _within(resolved, root):
            continue
        if any((_within(resolved, item) for item in excluded)):
            continue
        visited.add(resolved)
        for path in directory.iterdir():
            resolved_path = path.resolve()
            if any((_within(resolved_path, item) for item in excluded)):
                continue
            if not _within(resolved_path, root):
                continue
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file() or path.name.casefold() in REPORT_NAMES:
                continue
            relative = path.relative_to(root)
            extension = path.suffix.lower()
            kind = 'other'
            data_rows = None
            sheet_rows = {}
            if extension in IMAGE_EXTENSIONS:
                kind = 'image'
            elif extension == '.csv':
                kind = 'csv'
                data_rows = _csv_rows(path)
            elif extension == '.xlsx':
                kind = 'xlsx'
                sheet_rows = _xlsx_rows(path)
                data_rows = sum(sheet_rows.values())
            elif extension == '.jsonl':
                kind = 'jsonl'
                with path.open('r', encoding='utf-8-sig') as handle:
                    data_rows = sum((1 for line in handle if line.strip()))
            elif extension == '.json':
                kind = 'json'
            elif extension in {'.sqlite', '.sqlite3', '.db'}:
                kind = 'sqlite'
            elif path.name.lower().endswith(('-wal', '-shm', '-journal')):
                kind = 'sqlite_aux'
            yield {'relative_path': relative.as_posix(), 'stage': relative.parts[0] if len(relative.parts) > 1 else '(root)', 'kind': kind, 'bytes': path.stat().st_size, 'data_rows': data_rows, 'sheet_rows': sheet_rows}

def summarize(rows):
    extension_counts = Counter()
    kind_counts = Counter()
    image_counts = Counter()
    csv_counts = {}
    xlsx_counts = {}
    total_bytes = 0
    total_files = 0
    for row in rows:
        name = row['relative_path']
        extension_counts[Path(name).suffix.lower() or '(none)'] += 1
        kind_counts[row['kind']] += 1
        total_files += 1
        total_bytes += row['bytes']
        if row['kind'] == 'image':
            image_counts[row['stage']] += 1
        elif row['kind'] == 'csv':
            csv_counts[name] = row['data_rows']
        elif row['kind'] == 'xlsx':
            xlsx_counts[name] = dict(row['sheet_rows'])
    return {'total_files': total_files, 'total_bytes': total_bytes, 'total_images': kind_counts['image'], 'counts_by_extension': dict(sorted(extension_counts.items())), 'counts_by_kind': dict(sorted(kind_counts.items())), 'image_counts_by_stage': dict(sorted(image_counts.items())), 'csv_rows': dict(sorted(csv_counts.items())), 'xlsx_rows': dict(sorted(xlsx_counts.items()))}
