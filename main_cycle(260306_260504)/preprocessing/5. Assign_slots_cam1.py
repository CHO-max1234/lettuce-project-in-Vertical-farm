"""Assign physical planting slots to cam1 observations. Paths are configured in settings.py."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import importlib
import os
from pathlib import Path
from pipeline_paths import BRIGHT_IMAGES, NAME_ROOT, bootstrap, load_stage
bootstrap()
_core = importlib.import_module('slot_geometry')
_adapter = importlib.import_module('cam1_geometry')
_lettuce_seg = importlib.import_module('plant_segmentation')
_overlay_style = load_stage('name0')
INPUT_DIR = BRIGHT_IMAGES / 'cam1'
OUT_ROOT = NAME_ROOT / 'cam1'
VERSION_LABEL = 'ver3.4'
OUTPUT_IMAGE_DIRNAME = 'overlays'
SUMMARY_FILENAME = 'slot_summary.csv'
SLOTS_FILENAME = 'slots.csv'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect_images(image_dir=None) -> list[Path]:
    source=Path(image_dir or INPUT_DIR)
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS and 'cam1' in p.name.lower())
    if not files: raise FileNotFoundError('No cam1 bed images: '+str(source))
    return files

def _draw_evidence(canvas, evidence_by_row, color, cross: bool) -> None:
    import cv2
    for row in ('t', 'b'):
        for item in evidence_by_row.get(row, []):
            point = (int(round(item['x'])), int(round(item['y'])))
            if cross:
                cv2.drawMarker(canvas, point, color, cv2.MARKER_TILTED_CROSS, 22, _overlay_style.OVERLAY_LINE_THICKNESS)
            else:
                cv2.circle(canvas, point, 8, color, _overlay_style.OVERLAY_LINE_THICKNESS)

def draw_overlay(image, slots, meta, segments, output_path: Path) -> None:
    import cv2
    import numpy as np
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    segments_by_id = _lettuce_seg.segment_by_id(segments)
    bed_bbox = meta.get('bed_bbox')
    if bed_bbox:
        (x1, y1, x2, y2) = bed_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for slot in slots:
        if slot.local_col <= 0 or not np.isfinite(slot.x):
            continue
        color = _overlay_style._slot_color(slot)
        center = (int(round(slot.x)), int(round(slot.y)))
        prior = (int(round(slot.x_prior)), int(round(slot.y_prior)))
        segment = segments_by_id.get(slot.seg_id)
        if segment is not None:
            cv2.circle(canvas, center, _overlay_style.OVERLAY_CENTER_RADIUS, color, -1)
            polygon = np.asarray(segment['polygon'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(canvas, [polygon], -1, color, _overlay_style.OVERLAY_LINE_THICKNESS, cv2.LINE_AA)
        else:
            cv2.circle(canvas, prior, _overlay_style.OVERLAY_CENTER_RADIUS, color, _overlay_style.OVERLAY_LINE_THICKNESS)
        if slot.seg_id is not None and slot.bbox:
            (bx1, by1, bx2, by2) = slot.bbox
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, _overlay_style.OVERLAY_BBOX_THICKNESS)
        suffix = ' SEG' if slot.seg_id is not None else ' PRIOR'
        if slot.visibility == _core.Visibility.OCCLUDED:
            suffix = ' OCCLUDED'
        _overlay_style._put_label(canvas, (f'{slot.canonical_id[0]}{int(slot.canonical_id[1:]) + 2}' if slot.canonical_id else f'{slot.row}L{slot.local_col}?') + suffix, (center[0] + 10, center[1] - 12), color)
    flags = '|'.join(meta.get('qc_flags', [])) or 'none'
    header = f"CAM1 {VERSION_LABEL} | {meta.get('status')} | physical_visible={meta.get('n_visible_physical', 0)} | matched_evidence={meta.get('n_visible_evidence', 0)} | physical={(int(meta['start_col']) + 2 if meta.get('start_col') is not None else '?')}..12 | towel11/12={meta.get('towel_present')} | seg={meta.get('segment_count', 0)} | assigned={meta.get('assigned_segment_count', 0)} | rejected={meta.get('rejected_segment_count', 0)} | curve_review={meta.get('curve_rejected_segment_count', 0)} | inferred={meta.get('inferred_column_count', 0)} | QC={flags}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 60), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 42), font, _overlay_style.OVERLAY_FONT_SCALE, (255, 255, 255), _overlay_style.OVERLAY_FONT_THICKNESS, cv2.LINE_AA)
    _core.imwrite_unicode(output_path, canvas)

def process_one(path: Path, model) -> dict:
    image = _core.imread_unicode(path)
    if image is None:
        return {'image_name': path.name, 'status': 'READ_FAIL', 'rows': []}
    raw_segments = _lettuce_seg.predict_lettuce_segments(model, image)
    (segments, rejected_segments) = _adapter.filter_cam1_segments(image, raw_segments)
    (slots, meta) = _adapter.map_cam1_v3(image, segments)
    if slots:
        seg_meta = _lettuce_seg.refine_slots_with_segments(slots, segments, image.shape[1], image.shape[0], bed_bbox=meta.get('bed_bbox'), enforce_cell_corridor=True, conservative_prior_gate=False, prune_single_curve_outlier=not bool(meta.get('towel_present')))
    else:
        seg_meta = {'segment_count': len(segments), 'assigned_segment_count': 0, 'unassigned_segment_count': len(segments)}
    meta.update({key: value for (key, value) in seg_meta.items() if key != 'assigned_by_slot_object_id'})
    if meta.get('curve_rejected_segment_count', 0):
        meta.setdefault('qc_flags', []).append('AMBIGUOUS_SEG_PRIOR_KEPT')
    meta['raw_segment_count'] = len(raw_segments)
    meta['rejected_segment_count'] = len(rejected_segments)
    meta['rejected_segments'] = rejected_segments
    output_path = OUT_ROOT / OUTPUT_IMAGE_DIRNAME / path.name
    draw_overlay(image, slots, meta, segments, output_path)
    rows = []
    for slot in slots:
        row = slot.to_dict()
        row.update({'image_name': path.name, 'map_status': meta.get('status'), 'evidence_column_count': meta.get('n_visible_evidence'), 'physical_visible_column_count': meta.get('n_visible_physical'), 'qc_flags': '|'.join(meta.get('qc_flags', []))})
        from pipeline_contracts import remap_cam1
        row.update(camera='cam1', mapping_version='VER4_final CAM1 VER3.4')
        rows.append(remap_cam1(row))
    return {'image_name': path.name, 'status': meta.get('status'), 'evidence_column_count': meta.get('n_visible_evidence', 0), 'raw_evidence_column_count': meta.get('raw_evidence_column_count', 0), 'physical_visible_column_count': meta.get('n_visible_physical', 0), 'start_col': meta.get('start_col'), 'towel_present': meta.get('towel_present'), 'lettuce_segment_count': meta.get('segment_count', 0), 'raw_lettuce_segment_count': meta.get('raw_segment_count', 0), 'rejected_segment_count': meta.get('rejected_segment_count', 0), 'assigned_segment_count': meta.get('assigned_segment_count', 0), 'curve_rejected_segment_count': meta.get('curve_rejected_segment_count', 0), 'inferred_column_count': meta.get('inferred_column_count', 0), 'unmatched_column_count': meta.get('unmatched_column_count', 0), 'right_anchor_x': meta.get('right_anchor_x'), 'right_anchor_source': meta.get('right_anchor_source'), 't8_rescue_gap_pitch': meta.get('t8_rescue_gap_pitch'), 't8_rescue_endpoint_clearance_pitch': meta.get('t8_rescue_endpoint_clearance_pitch'), 't8_rescue_row_support': meta.get('t8_rescue_row_support', 0), 'pitch': meta.get('pitch'), 'adaptive_top_row_y': meta.get('adaptive_top_row_y'), 'adaptive_bottom_row_y': meta.get('adaptive_bottom_row_y'), 'qc_flags': '|'.join(meta.get('qc_flags', [])), 'rows': rows}

def main(argv=None):
    """Write slot assignments for all configured cam1 bed images."""
    global INPUT_DIR, OUT_ROOT
    import argparse
    import pandas as pd
    from tqdm import tqdm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=INPUT_DIR)
    parser.add_argument('--output-root',type=Path,default=OUT_ROOT)
    args=parser.parse_args(argv)
    INPUT_DIR=args.input_root.resolve();OUT_ROOT=args.output_root.resolve()
    if INPUT_DIR==OUT_ROOT or INPUT_DIR in OUT_ROOT.parents or OUT_ROOT in INPUT_DIR.parents:
        raise ValueError('Input and output directories must be separate')
    images=collect_images()
    model=_lettuce_seg.load_lettuce_seg_model()
    (OUT_ROOT/OUTPUT_IMAGE_DIRNAME).mkdir(parents=True,exist_ok=True)
    summary=[];rows=[]
    for path in tqdm(images,desc='cam1 slot assignment',unit='image'):
        result=process_one(path, model)
        if result is None:raise RuntimeError('Cannot process image: '+str(path))
        summary.append({k:v for k,v in result.items() if k!='rows'})
        rows.extend(result['rows'])
    pd.DataFrame(summary).to_csv(OUT_ROOT/SUMMARY_FILENAME,index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUT_ROOT/SLOTS_FILENAME,index=False,encoding='utf-8-sig')
    print('Slot assignments:',OUT_ROOT)
    return 0
if __name__ == '__main__':
    main()
