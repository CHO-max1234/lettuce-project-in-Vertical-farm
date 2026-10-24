"""Layout manifest calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parent
WORKERS = 8
CHUNK_FRAMES = 64
import argparse
import csv
import io
import json
import os
import sqlite3
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
KINDS = ('frame', 'slot', 'segment', 'error')
NAMES = {k: k + '_manifest.csv' for k in KINDS}
NAMES['qc'] = 'qc_manifest.csv'
QC_FIELDS = ['frame_id', 'camera', 'image_name', 'qc_status', 'geometry_status', 'slot_count', 'geometry_supported_slots', 'geometry_review_slots', 'geometry_hold_slots', 'qc_reasons', 'overlay_path']

def connect(database):
    return sqlite3.connect('file:' + Path(database).resolve().as_posix() + '?mode=ro', uri=True)

def records(data, kind):
    if kind == 'frame':
        return [data['frame']]
    if kind == 'error':
        f = data['frame']
        return [dict(frame_id=f['frame_id'], camera=f['camera'], image_name=f['image_name'], **data['error'])] if data['error'] else []
    return data[kind]

def iter_chunk(database, start, stop):
    with connect(database) as db:
        for (blob,) in db.execute('SELECT payload FROM frames WHERE sequence>=? AND sequence<? ORDER BY sequence', (start, stop)):
            yield json.loads(zlib.decompress(blob).decode('utf-8'))

def schema_chunk(database, start, stop):
    fields = {k: [] for k in KINDS}
    seen = {k: set() for k in KINDS}
    count = 0
    for data in iter_chunk(database, start, stop):
        count += 1
        for kind in KINDS:
            for row in records(data, kind):
                for key in row:
                    if key not in seen[kind]:
                        seen[kind].add(key)
                        fields[kind].append(key)
    return (fields, count)

def csv_value(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) if isinstance(value, (list, dict)) else '' if value is None else value

def render_chunk(database, start, stop, fields):
    buffers = {k: io.StringIO(newline='') for k in NAMES}
    writers = {k: csv.DictWriter(buffers[k], fields[k] or ['frame_id', 'error']) for k in KINDS}
    writers['qc'] = csv.DictWriter(buffers['qc'], QC_FIELDS)
    counts = {k: 0 for k in NAMES}
    count = 0
    for data in iter_chunk(database, start, stop):
        count += 1
        f = data['frame']
        for kind in KINDS:
            for row in records(data, kind):
                writers[kind].writerow(row if kind == 'error' else {k: csv_value(v) for (k, v) in row.items()})
                counts[kind] += 1
        meta = f.get('geometry_metadata', {})
        reasons = meta.get('qc_flags') or meta.get('failure_reasons') or []
        qrow = {k: f.get(k, '') for k in QC_FIELDS}
        qrow['qc_reasons'] = '|'.join(reasons)
        writers['qc'].writerow(qrow)
        counts['qc'] += 1
    return ({k: b.getvalue() for (k, b) in buffers.items()}, counts, count)

def ordered_chunks(pool, function, database, ranges, fields=None, limit=WORKERS):
    pending = deque()
    iterator = iter(ranges)

    def fill():
        while len(pending) < limit:
            try:
                (start, stop) = next(iterator)
            except StopIteration:
                break
            args = (database, start, stop) if fields is None else (database, start, stop, fields)
            pending.append(pool.submit(function, *args))
    fill()
    while pending:
        result = pending.popleft().result()
        fill()
        yield result

def export_database(database, root, workers=WORKERS, chunk=CHUNK_FRAMES):
    database = Path(database)
    root = Path(root)
    with connect(database) as db:
        (total, minimum, maximum) = db.execute('SELECT COUNT(*),MIN(sequence),MAX(sequence) FROM frames').fetchone()
    if not total:
        raise ValueError('EMPTY_CHECKPOINT')
    ranges = [(i, min(i + chunk, maximum + 1)) for i in range(minimum, maximum + 1, chunk)]
    fields = {k: [] for k in KINDS}
    seen = {k: set() for k in KINDS}
    schema_count = 0
    root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with tqdm(total=total, desc='Parallel CSV schema', unit='frame') as bar:
            for (part, count) in ordered_chunks(pool, schema_chunk, str(database), ranges, limit=workers):
                schema_count += count
                bar.update(count)
                for kind in KINDS:
                    for key in part[kind]:
                        if key not in seen[kind]:
                            seen[kind].add(key)
                            fields[kind].append(key)
        if schema_count != total:
            raise ValueError('SCHEMA_FRAME_COUNT_MISMATCH')
        handles = {k: (root / (name + '.partial')).open('w', encoding='utf-8-sig', newline='') for (k, name) in NAMES.items()}
        for kind in KINDS:
            csv.DictWriter(handles[kind], fields[kind] or ['frame_id', 'error']).writeheader()
        csv.DictWriter(handles['qc'], QC_FIELDS).writeheader()
        exported = 0
        counts = {k: 0 for k in NAMES}
        try:
            with tqdm(total=total, desc='Parallel CSV export', unit='frame') as bar:
                for (text, part, count) in ordered_chunks(pool, render_chunk, str(database), ranges, fields, workers):
                    for kind in NAMES:
                        handles[kind].write(text[kind])
                        counts[kind] += part[kind]
                    exported += count
                    bar.update(count)
        finally:
            for handle in handles.values():
                handle.close()
        if exported != total or counts['frame'] != total or counts['qc'] != total:
            raise ValueError('EXPORT_FRAME_COUNT_MISMATCH')
    for name in NAMES.values():
        os.replace(root / (name + '.partial'), root / name)
    return dict(frames=total, row_counts={NAMES[k]: v for (k, v) in counts.items()}, workers=workers, chunk_frames=chunk)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--chunk', type=int, default=CHUNK_FRAMES)
    args = parser.parse_args()
    print(json.dumps(export_database(args.database, args.output, args.workers, args.chunk), ensure_ascii=False))
if __name__ == '__main__':
    main()
