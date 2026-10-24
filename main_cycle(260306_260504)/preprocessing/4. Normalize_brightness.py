"""Normalize brightness with the fixed RGB-common gain transform. Pixel coordinates, image shape and source files remain unchanged. Inputs and outputs are separate."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from pathlib import Path
from pipeline_paths import OCR_IMAGES, BRIGHT_ROOT
ROOT = BRIGHT_ROOT
INPUT_DIR = OCR_IMAGES
VERSION = 'final'
SELECTED_CANDIDATE = 'v2'
SHADOW_STRENGTH = 0.65
HIGHLIGHT_REDUCTION = 0.02
WORKERS = 4
CHUNK_SIZE = 8
import argparse, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_image(path):
    im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('Invalid input or state in read_image (rejected condition: im is None)')
    return im

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def tone_curve(x, strength, highlight):
    return x + strength * x * (1 - x) ** 2 - highlight * smoothstep((x - 0.7) / 0.3)

def correct_brightness(image, strength=SHADOW_STRENGTH, highlight=HIGHLIGHT_REDUCTION):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Invalid input or state in correct_brightness (rejected condition: image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3)')
    if not 0 <= strength <= 1.2 or not 0 <= highlight <= 0.05:
        raise ValueError('Invalid input or state in correct_brightness (rejected condition: not 0 <= strength <= 1.2 or not 0 <= highlight <= 0.05)')
    x = np.arange(256, dtype=np.float32) / 255.0
    y = tone_curve(x, strength, highlight)
    if np.any(np.diff(y) < -1e-07):
        raise ValueError('Invalid input or state in correct_brightness (rejected condition: np.any(np.diff(y) < -1e-07))')
    gain = np.ones(256, np.float32)
    gain[1:] = y[1:] / x[1:]
    maximum = np.max(image, axis=2)
    result = np.rint(image.astype(np.float32) * gain[maximum, None])
    if np.any(result < 0) or np.any(result > 255):
        raise ValueError('Invalid input or state in correct_brightness (rejected condition: np.any(result < 0) or np.any(result > 255))')
    return result.astype(np.uint8)

def write_png(path, im):
    if path.exists():
        if not np.array_equal(read_image(path), im):
            raise FileExistsError('Different output already exists: ' + str(path))
        return
    (ok, buf) = cv2.imencode('.png', im)
    if not ok:
        raise ValueError('PNG encoding failed')
    buf.tofile(str(path))

def work(path, out, strength, highlight, input_root=None, output_relative=None):
    try:
        original_hash = digest(path)
        before = read_image(path)
        after = correct_brightness(before, strength, highlight)
        relative = Path(output_relative) if output_relative is not None else path.relative_to(input_root) if input_root is not None else Path(path.name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid relative brightness output path')
        target = out / 'corrected' / relative.with_suffix('.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        write_png(target, after)
        assert before.shape == after.shape and np.array_equal(read_image(target), after)
        assert digest(path) == original_hash
        return {'name': path.name, 'source_path': str(path), 'source_sha256': original_hash, 'output_path': str(target), 'output_sha256': digest(target), 'shape': list(before.shape), 'camera': re.search('cam[012]', path.name)[0] if re.search('cam[012]', path.name) else 'unknown', 'status': 'OK'}
    except Exception as e:
        import traceback
        return {'name': path.name, 'source_path': str(path), 'status': 'ERROR', 'error': repr(e), 'traceback': traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_DIR)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--strength', type=float, default=SHADOW_STRENGTH)
    parser.add_argument('--highlight', type=float, default=HIGHLIGHT_REDUCTION)
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.output.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Input and output must be separate')
    files = sorted((p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}))
    if not files:
        raise ValueError('No input images')
    names = [str(p.relative_to(source).with_suffix('')).casefold() for p in files]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate output image names')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, tqdm(total=len(files), desc='Brightness', unit='image') as bar:
        for start in range(0, len(files), CHUNK_SIZE):
            for row in pool.map(lambda p: work(p, out, args.strength, args.highlight, source), files[start:start + CHUNK_SIZE]):
                rows.append(row)
                bar.update(1)
    (out / 'manifest.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    failed = sum((r['status'] != 'OK' for r in rows))
    print(json.dumps({'images': len(rows), 'failed': failed, 'output': str(out)}))
    if failed:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
