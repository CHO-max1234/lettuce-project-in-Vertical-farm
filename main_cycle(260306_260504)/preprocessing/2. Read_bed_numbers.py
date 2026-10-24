"""Read CAM1 bed numbers and apply the same-date, same-time, same-old-bed KEY mapping. Missing CAM1, review states and filename collisions are not silently resolved."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import settings as CYCLE
from pathlib import Path
from pipeline_paths import OCR_IMAGES, OCR_LOG, DEVICE
INPUT_DIR = str(OCR_IMAGES)
OUTPUT_DIR = str(OCR_LOG)
SAVE_PREVIEWS = False
GPU = DEVICE
CHUNK_SIZE = 32
OCR_WORKERS = 8
READ_WORKERS = 4
DETECTION_MAG_RATIO = 2.0
MIN_CONFIDENCE = 0.8
(MIN_BED, MAX_BED) = (1, 95)
RIGHT_ROI = (0.88, 0.0, 1.0, 0.6)
VERSION = 'bed_ocr_parallel_v4'
RESUMABLE_VERSIONS = {'bed_ocr_rename_v2', 'bed_ocr_single_pass_v3', VERSION}
COMPATIBLE_CODE_HASHES = {'d8290fdeb53b8403ad4fa44f55e6bf7389150514e1a42e0aea37f144ffc46c82', '8b60232158315e4a798b563cd5349f7c3e740b9c53b439d4cb0a0b6d7c9a8daf', 'f9cd7b07e77bb92d09701434cba29f632547f77deaf8862170264bc6714bdb51'}
import argparse, csv, json, re, sys, os, hashlib
from threading import Lock, Semaphore
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Any, Sequence
import cv2
import numpy as np
from tqdm import tqdm

@dataclass(frozen=True)
class OCRCandidate:
    text: str
    number: int | None
    confidence: float
    variant: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def read_image_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None

def _parse_number(text: str, minimum: int, maximum: int) -> int | None:
    digits = re.sub('\\D', '', str(text))
    if not digits:
        return None
    value = int(digits)
    return value if minimum <= value <= maximum else None

def collect_candidates(reader: Any, image: np.ndarray) -> list[OCRCandidate]:
    raw_results = reader.readtext(image, detail=1, paragraph=False, allowlist='0123456789', mag_ratio=DETECTION_MAG_RATIO)
    candidates = []
    for item in raw_results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            confidence = 0.0
        candidates.append(OCRCandidate(text=text, number=_parse_number(text, MIN_BED, MAX_BED), confidence=confidence, variant='ROI_SINGLE'))
    return sorted(candidates, key=lambda c: (c.number is not None, c.confidence), reverse=True)

def choose_candidate(candidates: Sequence[OCRCandidate], *, minimum_confidence: float=0.45) -> tuple[OCRCandidate | None, str]:
    valid = [item for item in candidates if item.number is not None]
    if not valid:
        return (None, 'NO_VALID_OCR')
    best = valid[0]
    near = {item.number for item in valid if item.number != best.number and item.confidence >= max(minimum_confidence, best.confidence - 0.05)}
    if near:
        return (best, 'AMBIGUOUS_OCR')
    if best.confidence < minimum_confidence:
        return (best, 'LOW_CONFIDENCE')
    if len(re.sub('\\D', '', best.text)) == 1:
        return (best, 'SINGLE_DIGIT_REVIEW')
    return (best, 'OCR_CANDIDATE_OK')

def plate_regions(image):
    (h, w) = image.shape[:2]
    return [[round(v * d) for (v, d) in zip(RIGHT_ROI, (w, h, w, h))]]

def recognize(image, reader):
    if image is None:
        return dict(ocr_status='IMAGE_READ_ERROR')
    boxes = plate_regions(image)
    (x1, y1, x2, y2) = boxes[0]
    candidates = collect_candidates(reader, image[y1:y2, x1:x2])
    (best, status) = choose_candidate(candidates, minimum_confidence=MIN_CONFIDENCE)
    if not boxes:
        status = 'NO_PLATE'
    if best and status in {'OCR_CANDIDATE_OK', 'SINGLE_DIGIT_REVIEW'}:
        if status == 'SINGLE_DIGIT_REVIEW':
            (x1, y1, x2, y2) = boxes[0]
            edge = image[y1:y2, -2:]
            white = (edge.min(axis=2) > 175) & (edge.max(axis=2).astype(int) - edge.min(axis=2) < 65)
            status = 'FRAME_EDGE_REVIEW' if white.any(axis=1).sum() >= 8 else 'OCR_CANDIDATE_OK'
    return dict(ocr_status=status, ocr_number=best.number if best else '', confidence=best.confidence if best else '', selected_variant=best.variant if best else '', plate_boxes_json=json.dumps(boxes), candidates_json=json.dumps([c.to_dict() for c in candidates]), suggested_bed_id=f'bed{best.number:02d}' if best and best.number is not None else '', identity_verified=False)
NAME_RE = re.compile('(?P<bed>bed[0-9]+)_(?P<date>[0-9]{8})_(?P<time>[0-9]{6})_(?P<cam>cam[012])(?:_|$)', re.I)
FIELDS = ['old_key', 'new_key', 'camera', 'cameras_present', 'old_bed_id', 'new_bed_id', 'old_filename', 'new_filename', 'old_path', 'new_path', 'current_path', 'sha256', 'status', 'ocr_status', 'ocr_number', 'confidence', 'cam1_old_path', 'candidates_json', 'plate_boxes_json', 'error', 'input_roots_json', 'version']

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def atomic_manifest(path, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def append_manifest(path, rows):
    with path.open('a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
        f.flush()
        os.fsync(f.fileno())

def inventory(roots):
    found = set()
    rows = []
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                continue
            path = path.resolve()
            if str(path).casefold() in found:
                continue
            found.add(str(path).casefold())
            m = NAME_RE.search(path.stem)
            r = dict.fromkeys(FIELDS, '')
            r.update(old_path=str(path), current_path=str(path), old_filename=path.name, input_roots_json=json.dumps([str(p) for p in roots]), version=VERSION, status='DISCOVERED')
            if m:
                (bed, date, time, cam) = m.group('bed', 'date', 'time', 'cam')
                r.update(old_bed_id=bed.lower(), old_key=f'{date}_{time}_{bed.lower()}', camera=cam.lower())
            else:
                r.update(status='PARSE_ERROR', error='Filename must contain bedNN_YYYYMMDD_HHMMSS_camN')
            try:
                r['sha256'] = file_hash(path)
            except OSError as e:
                r.update(status='READ_ERROR', error=str(e))
            rows.append(r)
    return rows

def groups_of(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['old_key'] or r['old_path']].append(r)
    return groups

def load_manifest(path, roots):
    latest = {}
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('Manifest schema mismatch')
        for r in reader:
            if None in r or any((v is None for v in r.values())):
                continue
            if not r['old_path']:
                continue
            if r['version'] not in RESUMABLE_VERSIONS or json.loads(r['input_roots_json']) != [str(p) for p in roots]:
                raise ValueError('Resume input roots/version mismatch; use original configuration')
            latest[r['old_path']] = r
    if not latest:
        raise ValueError('Empty manifest')
    return list(latest.values())

def plan_group(group, result):
    present = ','.join(sorted((r['camera'] for r in group)))
    for r in group:
        r.update(cameras_present=present, ocr_status=result.get('ocr_status', 'OCR_ERROR'), ocr_number=result.get('ocr_number', ''), confidence=result.get('confidence', ''), cam1_old_path=next((a['old_path'] for a in group if a['camera'] == 'cam1'), ''), candidates_json=result.get('candidates_json', ''), plate_boxes_json=result.get('plate_boxes_json', ''))
        if r['status'] != 'DISCOVERED':
            continue
        r['version'] = VERSION
        if result.get('ocr_status') != 'OCR_CANDIDATE_OK':
            r['status'] = 'REVIEW'
            continue
        new_bed = f"bed{int(result['ocr_number']):02d}"
        old = Path(r['old_path'])
        m = NAME_RE.search(old.stem)
        new_name = old.name[:m.start('bed')] + new_bed + old.name[m.end('bed'):]
        if int(m.group('bed')[3:]) != int(result['ocr_number']):
            renamed = Path(new_name)
            if not renamed.stem.lower().endswith('_rf'):
                new_name = renamed.stem + '_rf' + renamed.suffix
        r.update(new_bed_id=new_bed, new_key=f"{m.group('date')}_{m.group('time')}_{new_bed}", new_filename=new_name, new_path=str(old.with_name(new_name)), status='READY')

def collision_check(rows):
    targets = defaultdict(list)
    for r in rows:
        if r['status'] == 'READY':
            targets[r['new_path'].casefold()].append(r)
    blocked = set()
    for (target, items) in targets.items():
        if len(items) > 1:
            blocked.update((r['old_key'] for r in items))
        for r in items:
            (old, new) = (Path(r['old_path']), Path(r['new_path']))
            if new != old and old.exists() and new.exists():
                blocked.add(r['old_key'])
    for r in rows:
        if r['old_key'] in blocked and r['status'] == 'READY':
            r.update(status='COLLISION', error='Destination exists or multiple old files target the same path; group not renamed')

def apply_group(group, manifest, rename_files=True):
    active = [r for r in group if r['status'] == 'READY']
    if not active:
        return
    try:
        for r in active:
            (old, new) = (Path(r['old_path']), Path(r['new_path']))
            current = old if old.exists() else new
            if not current.is_file() or file_hash(current) != r['sha256']:
                raise ValueError(f'Missing/changed image: {current}')
            if old != new and old.exists() and new.exists():
                raise ValueError(f'Destination appeared: {new}')
    except Exception as e:
        for r in active:
            r.update(status='RENAME_ERROR', error=str(e))
        append_manifest(manifest, active)
        return
    for r in active:
        (old, new) = (Path(r['old_path']), Path(r['new_path']))
        try:
            if not rename_files:
                r.update(status='UNCHANGED' if old == new else 'MAPPED', current_path=str(old), error='')
                append_manifest(manifest, [r])
                continue
            if old == new:
                status = 'UNCHANGED'
            elif old.exists():
                if new.exists():
                    raise FileExistsError(str(new))
                old.rename(new)
                status = 'RENAMED'
            else:
                status = 'RENAMED'
            r.update(status=status, current_path=str(new), error='')
            append_manifest(manifest, [r])
        except Exception as e:
            r.update(status='RENAME_ERROR', error=str(e))
            append_manifest(manifest, [r])
            raise

def save_preview(row, out):
    image = read_image_unicode(Path(row['current_path']))
    if image is None:
        return
    boxes = json.loads(row['plate_boxes_json'] or '[]')
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 3)
    width = min(image.shape[1], 1600)
    image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))
    panel = cv2.copyMakeBorder(image, 95, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    for (i, text) in enumerate([f"OLD: {row['old_filename']}", f"NEW: {row['new_filename'] or '(unresolved)'}", f"{row['status']} | OCR {row['ocr_number']} | {row['ocr_status']}"]):
        cv2.putText(panel, text, (10, 25 + i * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    directory = out / 'previews'
    directory.mkdir(exist_ok=True)
    unique = hashlib.sha256(row['old_path'].encode()).hexdigest()[:8]
    (ok, buf) = cv2.imencode('.jpg', panel)
    if ok:
        buf.tofile(str(directory / f"{Path(row['old_filename']).stem}_{unique}.jpg"))

def create_reader(device=None):
    import easyocr
    import torch
    requested = GPU if device is None else device
    if requested is False or str(requested).lower() == 'cpu':
        gpu = False
    elif requested is True or str(requested).lower() == 'auto':
        gpu = True if torch.cuda.is_available() else False
    else:
        gpu = str(requested)
        if gpu.isdigit():
            gpu = 'cuda:' + gpu
        if not gpu.startswith('cuda'):
            raise ValueError('OCR device must be cpu/auto/cuda:N/N')
        if not torch.cuda.is_available():
            raise RuntimeError('Requested OCR CUDA is unavailable')
    reader = easyocr.Reader(['en'], gpu=gpu, model_storage_directory=str(CYCLE.OCR_MODEL_DIR), download_enabled=False)
    lock = Lock()
    original_forward = reader.recognizer.forward

    def guarded_forward(*args, **kwargs):
        with lock:
            return original_forward(*args, **kwargs)
    reader.recognizer.forward = guarded_forward
    print(f'OCR: device={reader.device}; calls_per_cam1=1; input=original ROI; detection_mag={DETECTION_MAG_RATIO}', flush=True)
    return reader

def recognize_path(path, reader, read_gate):
    try:
        with read_gate:
            image = read_image_unicode(path)
        return recognize(image, reader)
    except Exception as e:
        result = {'ocr_status': 'OCR_ERROR', 'candidates_json': json.dumps({'error': str(e)})}
        if type(e).__name__ == 'OutOfMemoryError' or 'CUDA out of memory' in str(e):
            result['_retry_oom'] = True
        return result

def recognize_paths(paths, reader, pool, read_gate, parallel):
    work = lambda path: recognize_path(path, reader, read_gate)
    results = list(pool.map(work, paths)) if parallel else [work(path) for path in paths]
    if any((r.get('_retry_oom') for r in results)):
        import torch
        torch.cuda.empty_cache()
        print('recognize paths: OCR: GPU .', flush=True)
        parallel = False
        for (i, result) in enumerate(results):
            if result.pop('_retry_oom', False):
                results[i] = work(paths[i])
                results[i].pop('_retry_oom', None)
    return (results, parallel)

def run(input_dir, output_dir, reader=None, previews=SAVE_PREVIEWS, rename_files=True, initial_rows=None):
    roots = [Path(p).resolve() for p in ([input_dir] if isinstance(input_dir, (str, Path)) else input_dir)]
    if not roots or not all((p.is_dir() for p in roots)):
        raise ValueError('Set existing input bed crop folders')
    out = Path(output_dir).resolve()
    if any((out == p or p in out.parents for p in roots)):
        raise ValueError('Output must be outside input folders')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.csv'
    if manifest.exists():
        rows = load_manifest(manifest, roots)
    else:
        if any(out.iterdir()):
            raise ValueError('Use a new output folder or existing manifest run')
        rows = inventory(roots) if initial_rows is None else initial_rows
        if not rows:
            raise ValueError('No images found')
        atomic_manifest(manifest, rows)
    grouped = groups_of(rows)
    pending = [g for g in grouped.values() if any((r['status'] == 'DISCOVERED' for r in g))]
    for group in pending:
        counts = Counter((r['camera'] for r in group))
        reason = 'DUPLICATE_CAMERA' if any((n > 1 for n in counts.values())) else 'NO_CAM1' if not counts['cam1'] else ''
        if not reason and any((r['status'] in {'READ_ERROR', 'PARSE_ERROR'} for r in group)):
            reason = 'GROUP_INPUT_ERROR'
        if reason:
            plan_group(group, {'ocr_status': reason})
            append_manifest(manifest, group)
    pending = [g for g in pending if any((r['status'] == 'DISCOVERED' for r in g))]
    if pending and reader is None:
        reader = create_reader()
    workers = min(max(1, int(OCR_WORKERS)), max(1, int(CHUNK_SIZE))) if str(getattr(reader, 'device', 'cpu')).startswith('cuda') else 1
    read_gate = Semaphore(max(1, int(READ_WORKERS)))
    parallel = workers > 1
    if pending:
        print(f'OCR: workers={workers}; read_workers={READ_WORKERS}; chunk={CHUNK_SIZE}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(total=len(pending), desc='CAM1 OCR') as bar:
        for start in range(0, len(pending), CHUNK_SIZE):
            chunk = pending[start:start + CHUNK_SIZE]
            paths = [Path(next((r['old_path'] for r in g if r['camera'] == 'cam1'))) for g in chunk]
            (results, parallel) = recognize_paths(paths, reader, pool, read_gate, parallel)
            for (group, result) in zip(chunk, results):
                plan_group(group, result)
                append_manifest(manifest, group)
                bar.update()
    collision_check(rows)
    atomic_manifest(manifest, rows)
    for group in tqdm(grouped.values(), desc='Rename KEY groups' if rename_files else 'Map KEY groups (read-only images)'):
        apply_group(group, manifest, rename_files=rename_files)
    atomic_manifest(manifest, rows)
    if previews:
        for r in tqdm(rows, desc='Review previews'):
            if r['camera'] == 'cam1':
                save_preview(r, out)
    return dict(images=len(rows), groups=len(grouped), statuses=dict(Counter((r['status'] for r in rows))), manifest=str(manifest))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', default=INPUT_DIR)
    parser.add_argument('--output', default=OUTPUT_DIR)
    parser.add_argument('--previews', action='store_true', default=SAVE_PREVIEWS)
    args = parser.parse_args([] if 'ipykernel' in sys.modules else None)
    if not args.input or not args.output:
        raise SystemExit('Set INPUT_DIR and OUTPUT_DIR at the top, or --input / --output')
    print(json.dumps(run(args.input, args.output, previews=args.previews), ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
