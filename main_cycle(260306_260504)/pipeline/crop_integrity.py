"""Processing utilities for crop integrity."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
_sys.path.insert(0, str(_CYCLE_ROOT / 'preprocessing'))
from mask_record import unpack
VERSION = 'all_crop_mask_integrity_v1_20260923'
FIELDS = ('record_id', 'model_eligible', 'integrity_status', 'integrity_reason')
csv.field_size_limit(max(csv.field_size_limit(), 100000000))

def yes(value: object) -> bool:
    return str(value).strip().lower() in {'true', '1', '1.0', 'yes'}

def _straight_sides(mask: np.ndarray, ratio: float, minimum: int) -> list[str]:
    (ys, xs) = np.nonzero(mask)
    if not len(xs):
        return ['EMPTY_MASK']
    (left, right, top, bottom) = (xs.min(), xs.max(), ys.min(), ys.max())
    crop = mask[top:bottom + 1, left:right + 1] > 0
    if crop.shape[0] < minimum or crop.shape[1] < minimum:
        return []
    sides = {'TOP_STRAIGHT_EDGE': crop[:2, :].any(axis=0), 'BOTTOM_STRAIGHT_EDGE': crop[-2:, :].any(axis=0), 'LEFT_STRAIGHT_EDGE': crop[:, :2].any(axis=1), 'RIGHT_STRAIGHT_EDGE': crop[:, -2:].any(axis=1)}
    return [name for (name, line) in sides.items() if len(line) >= minimum and np.count_nonzero(line) / len(line) >= ratio]

def judge(row: dict[str, str], *, straight_ratio: float, minimum: int, main_share: float) -> tuple[bool, str, str]:
    if not row.get('record_id'):
        raise ValueError("Invalid input or state in judge (rejected condition: not row.get('record_id'))")
    if not yes(row.get('has_crop')) or not row.get('crop_path'):
        return (False, 'NO_CROP', 'Missing crop')
    reasons: list[str] = []
    if 'False' in Path(row['crop_path']).parts:
        reasons.append('EXISTING_FALSE_FOLDER')
    flag = row.get('분석가능', '').strip().lower()
    if flag in {'false', '0', '0.0'}:
        reasons.append('EXISTING_ANALYSIS_FALSE')
    try:
        mask = unpack(row)
        binary = np.asarray(mask > 0, np.uint8)
        (labels, _, stats, _) = cv2.connectedComponentsWithStats(binary, 8)
        areas = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, labels)), reverse=True)
        if not areas:
            reasons.append('EMPTY_MASK')
        elif areas[0] / sum(areas) < main_share:
            reasons.append('DISCONNECTED_MASK')
        reasons.extend(_straight_sides(binary, straight_ratio, minimum))
    except Exception as exc:
        reasons.append(f'MASK_READ_ERROR:{type(exc).__name__}')
    if reasons:
        return (False, 'REVIEW', ';'.join(dict.fromkeys(reasons)))
    return (True, 'PASS', '')

def run(source: Path, output: Path, *, straight_ratio: float=0.8, minimum: int=20, main_share: float=0.9) -> dict:
    if not 0 < straight_ratio <= 1 or minimum < 1 or (not 0 < main_share <= 1):
        raise ValueError('Invalid input or state in run (rejected condition: not 0 < straight_ratio <= 1 or minimum < 1 or (not 0 < main_share <= 1))')
    source = source.resolve()
    output = output.resolve()
    if not source.is_file() or source == output:
        raise ValueError('Invalid input or state in run (rejected condition: not source.is_file() or source == output)')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.writing')
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        with source.open('r', encoding='utf-8-sig', newline='') as src, temp.open('w', encoding='utf-8-sig', newline='') as dst:
            reader = csv.DictReader(src)
            required = {'record_id', 'has_crop', 'crop_path', 'mask_encoding', 'mask_height', 'mask_width', 'mask_data', 'mask_sha256'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'crop manifest Missing required columns: {sorted(required - set(reader.fieldnames or []))}')
            writer = csv.DictWriter(dst, FIELDS)
            writer.writeheader()
            for row in tqdm(reader, desc='All crop mask integrity', unit='row', mininterval=5):
                rid = row['record_id']
                if rid in seen:
                    raise ValueError(f'Invalid input or state in run (rejected condition: rid in seen): {rid} ')
                seen.add(rid)
                (allow, status, reason) = judge(row, straight_ratio=straight_ratio, minimum=minimum, main_share=main_share)
                writer.writerow({'record_id': rid, 'model_eligible': allow, 'integrity_status': status, 'integrity_reason': reason})
                counts[status] += 1
                counts['rows'] += 1
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()
    report = {'version': VERSION, 'input': str(source), 'output': str(output), 'counts': dict(counts), 'policy': 'suspect mask -> REVIEW, no model inference', 'thresholds': {'straight_edge_ratio': straight_ratio, 'straight_edge_min_pixels': minimum, 'main_component_min_share': main_share}}
    output.with_suffix('.summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--straight-ratio', type=float, default=0.8)
    p.add_argument('--min-pixels', type=int, default=20)
    p.add_argument('--main-share', type=float, default=0.9)
    a = p.parse_args()
    print(json.dumps(run(a.input, a.output, straight_ratio=a.straight_ratio, minimum=a.min_pixels, main_share=a.main_share), ensure_ascii=False))
