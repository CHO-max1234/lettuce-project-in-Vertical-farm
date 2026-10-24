"""Cam01 selection policy calculations used by the final cycle pipeline."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
import math, re
from collections import defaultdict
from statistics import median
from typing import Any, Sequence
HARD_CENTER_LIMIT = 0.55
CAMERA_OVERRIDE_MARGIN = 0.15
SEG_CONF = 0.2
SLOT_RE = re.compile('^(?P<row>[tb])(?P<col>\\d+)$', re.I)

def robust_location_scale(values: Sequence[float], floor: float) -> tuple[float, float]:
    center = float(median(values))
    mad = float(median([abs(value - center) for value in values]))
    return (center, max(floor, 1.4826 * mad))

def add_quality_metrics(rows: list[dict[str, Any]], reference=None) -> None:
    if reference is None:
        clean = [row for row in rows if row['inference_status'] == 'OK' and row['mask_edge_touch'] == 0 and (row['center_distance_norm'] <= HARD_CENTER_LIMIT)]
        aspects: dict[tuple[str, ...], list[float]] = defaultdict(list)
        sizes: dict[tuple[str, ...], list[float]] = defaultdict(list)
        for row in clean:
            log_aspect = math.log(max(1e-08, row['mask_aspect_ratio']))
            log_size = math.log(max(1e-08, row['mask_area_ratio_input']))
            keys = [('date', row['capture_date'], row['camera'], row['row']), ('camera_row', row['camera'], row['row']), ('camera', row['camera']), ('all',)]
            for key in keys:
                aspects[key].append(log_aspect)
                sizes[key].append(log_size)
        reference: dict[tuple[str, ...], tuple[float, float, float, float, int]] = {}
        for key in aspects:
            (a_center, a_scale) = robust_location_scale(aspects[key], 0.05)
            (s_center, s_scale) = robust_location_scale(sizes[key], 0.1)
            reference[key] = (a_center, a_scale, s_center, s_scale, len(aspects[key]))
    for row in rows:
        row.update({'reference_level': '', 'reference_n': 0, 'aspect_robust_z': float('nan'), 'size_robust_z': float('nan'), 'hard_valid': 0, 'quality_score': float('nan'), 'quality_flags': row['inference_status']})
        if row['inference_status'] != 'OK':
            continue
        choices = [(('date', row['capture_date'], row['camera'], row['row']), 5, 'date_camera_row'), (('camera_row', row['camera'], row['row']), 20, 'camera_row'), (('camera', row['camera']), 20, 'camera'), (('all',), 1, 'all')]
        picked = None
        for (key, minimum, label) in choices:
            if key in reference and reference[key][4] >= minimum:
                picked = (reference[key], label)
                break
        if picked is None:
            raise RuntimeError('no robust reference available')
        ((a_center, a_scale, s_center, s_scale, n), level) = picked
        aspect_z = abs(math.log(max(1e-08, row['mask_aspect_ratio'])) - a_center) / a_scale
        size_z = abs(math.log(max(1e-08, row['mask_area_ratio_input'])) - s_center) / s_scale
        hard_valid = int(row['mask_edge_touch'] == 0 and row['center_distance_norm'] <= HARD_CENTER_LIMIT)
        border_score = min(1.0, max(0.0, row['edge_margin_norm'] / 0.1))
        center_score = max(0.0, 1.0 - row['center_distance_norm'] / HARD_CENTER_LIMIT)
        shape_score = math.exp(-0.5 * (min(aspect_z, 4.0) / 2.0) ** 2)
        size_score = math.exp(-0.5 * (min(size_z, 4.0) / 2.0) ** 2)
        conf_score = min(1.0, max(0.0, (row['target_conf'] - SEG_CONF) / (1.0 - SEG_CONF)))
        quality = 0.3 * border_score + 0.25 * center_score + 0.2 * shape_score + 0.15 * size_score + 0.1 * conf_score
        flags: list[str] = []
        if row['mask_edge_touch']:
            flags.append('MASK_EDGE_TOUCH')
        if row['center_distance_norm'] > HARD_CENTER_LIMIT:
            flags.append('CENTER_FAR')
        if aspect_z > 4:
            flags.append('SHAPE_OUTLIER')
        if size_z > 4:
            flags.append('SIZE_OUTLIER')
        row.update({'reference_level': level, 'reference_n': n, 'aspect_robust_z': aspect_z, 'size_robust_z': size_z, 'hard_valid': hard_valid, 'quality_score': quality, 'quality_flags': '|'.join(flags) if flags else 'OK'})

def track_primary_cameras(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[row['physical_key']].append(row)
    dual_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidates in by_key.values():
        if {row['camera'] for row in candidates} >= {'cam0', 'cam1'}:
            dual_by_track[candidates[0]['track_id']].extend(candidates)
    primary: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    tracks = {row['track_id'] for row in rows}
    for track in tracks:
        physical_col = int(SLOT_RE.match(track.rsplit('_', 1)[1]).group('col'))
        default = 'cam0' if physical_col <= 4 else 'cam1'
        alternate = 'cam1' if default == 'cam0' else 'cam0'
        dual = dual_by_track.get(track, [])
        grouped = {camera: [row for row in dual if row['camera'] == camera] for camera in ('cam0', 'cam1')}
        stats: dict[str, dict[str, float]] = {}
        for (camera, items) in grouped.items():
            ok = [row for row in items if row['inference_status'] == 'OK']
            stats[camera] = {'n': len(items), 'hard_valid_rate': sum((row['hard_valid'] for row in items)) / max(1, len(items)), 'median_quality': float(median([row['quality_score'] for row in ok])) if ok else 0.0}
        chosen = default
        reason = 'DEFAULT_SLOT_POLICY'
        if min(stats[default]['n'], stats[alternate]['n']) >= 3:
            hard_gain = stats[alternate]['hard_valid_rate'] - stats[default]['hard_valid_rate']
            quality_gain = stats[alternate]['median_quality'] - stats[default]['median_quality']
            if hard_gain >= 0.3 or (hard_gain >= 0.15 and quality_gain >= 0.08):
                chosen = alternate
                reason = 'TRACK_QUALITY_OVERRIDE'
        primary[track] = chosen
        evidence[track] = {'track_primary_camera': chosen, 'track_primary_reason': reason, 'track_dual_n': min(stats['cam0']['n'], stats['cam1']['n']), 'track_cam0_hard_valid_rate': stats['cam0']['hard_valid_rate'], 'track_cam1_hard_valid_rate': stats['cam1']['hard_valid_rate'], 'track_cam0_median_quality': stats['cam0']['median_quality'], 'track_cam1_median_quality': stats['cam1']['median_quality']}
    return (primary, evidence)

def choose_candidate(items: Sequence[dict[str, Any]], primary_camera: str) -> tuple[dict[str, Any] | None, str]:
    camera_best: dict[str, dict[str, Any]] = {}
    for row in items:
        score = row['quality_score'] if row['inference_status'] == 'OK' else -1.0
        previous = camera_best.get(row['camera'])
        previous_score = previous['quality_score'] if previous and previous['inference_status'] == 'OK' else -1.0
        if previous is None or score > previous_score:
            camera_best[row['camera']] = row
    if len(camera_best) == 1:
        only = next(iter(camera_best.values()))
        return (only, 'ONLY_AVAILABLE_CAMERA') if only['inference_status'] == 'OK' else (None, 'ONLY_CAMERA_NO_DETECTION')
    primary = camera_best.get(primary_camera)
    alternate = next((row for (camera, row) in camera_best.items() if camera != primary_camera), None)
    if primary is None:
        return (alternate, 'PRIMARY_UNAVAILABLE') if alternate and alternate['inference_status'] == 'OK' else (None, 'NO_USABLE_CAMERA')
    if alternate is None:
        return (primary, 'ALTERNATE_UNAVAILABLE') if primary['inference_status'] == 'OK' else (None, 'NO_USABLE_CAMERA')
    p_ok = primary['inference_status'] == 'OK'
    a_ok = alternate['inference_status'] == 'OK'
    if not p_ok and a_ok:
        return (alternate, 'PRIMARY_NO_DETECTION')
    if p_ok and (not a_ok):
        return (primary, 'ALTERNATE_NO_DETECTION')
    if not p_ok and (not a_ok):
        return (None, 'BOTH_NO_DETECTION')
    if not primary['hard_valid'] and alternate['hard_valid']:
        return (alternate, 'ALTERNATE_ONLY_HARD_VALID')
    if primary['hard_valid'] and (not alternate['hard_valid']):
        return (primary, 'PRIMARY_ONLY_HARD_VALID')
    if alternate['quality_score'] >= primary['quality_score'] + CAMERA_OVERRIDE_MARGIN:
        return (alternate, 'OBSERVATION_QUALITY_OVERRIDE')
    return (primary, 'TRACK_PRIMARY_KEPT')
