"""Assign physical planting slots to cam0 observations. Paths are configured in settings.py."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import importlib
import os
import sys
from pathlib import Path
from pipeline_paths import BRIGHT_IMAGES, NAME_ROOT, bootstrap
bootstrap()
_slot_core = importlib.import_module('slot_geometry')
_adapter = importlib.import_module('cam0_geometry')
_lettuce_seg = importlib.import_module('plant_segmentation')
INPUT_DIR = BRIGHT_IMAGES / 'cam0'
OUT_ROOT = NAME_ROOT / 'cam0'
VERSION_LABEL = 'ver3.3'
OUTPUT_IMAGE_DIRNAME = 'overlays'
SUMMARY_FILENAME = 'slot_summary.csv'
SLOTS_FILENAME = 'slots.csv'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
OVERLAY_LINE_THICKNESS = 4
OVERLAY_BBOX_THICKNESS = 5
OVERLAY_FONT_SCALE = 1.0
OVERLAY_FONT_THICKNESS = 3
OVERLAY_CENTER_RADIUS = 9


def collect_images(image_dir=None) -> list[Path]:
    source=Path(image_dir or INPUT_DIR)
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS and 'cam0' in p.name.lower())
    if not files: raise FileNotFoundError('No cam0 bed images: '+str(source))
    return files

def _draw_evidence(canvas, evidence_by_row, color, marker: str) -> None:
    import cv2
    for row in ('t', 'b'):
        for evidence in evidence_by_row.get(row, []):
            center = (int(round(evidence['x'])), int(round(evidence['y'])))
            if marker == 'cross':
                cv2.drawMarker(canvas, center, color, cv2.MARKER_TILTED_CROSS, markerSize=22, thickness=OVERLAY_LINE_THICKNESS)
            else:
                cv2.circle(canvas, center, 8, color, OVERLAY_LINE_THICKNESS)

def _slot_label(slot) -> str:
    if slot.canonical_id is not None:
        return slot.canonical_id
    return f'{slot.row}L{slot.local_col}?'

def _slot_color(slot) -> tuple[int, int, int]:
    import cv2
    import numpy as np
    col = slot.physical_col if slot.physical_col is not None else slot.local_col
    index = (int(col) - 1) * 2 + (0 if slot.row == 't' else 1)
    hue = int(index * 47 % 180)
    hsv = np.asarray([[[hue, 235, 255]]], dtype=np.uint8)
    return tuple((int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))

def _put_label(canvas, text: str, origin: tuple[int, int], color) -> None:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((width, height), baseline) = cv2.getTextSize(text, font, OVERLAY_FONT_SCALE, OVERLAY_FONT_THICKNESS)
    x = max(0, min(origin[0], canvas.shape[1] - width - 6))
    y = max(height + 5, min(origin[1], canvas.shape[0] - baseline - 3))
    cv2.rectangle(canvas, (x - 3, y - height - 4), (x + width + 4, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, OVERLAY_FONT_SCALE, color, OVERLAY_FONT_THICKNESS, cv2.LINE_AA)

def draw_overlay(img, slots, meta, segments, out_path: Path) -> None:
    import cv2
    import numpy as np
    canvas = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    segments_by_id = _lettuce_seg.segment_by_id(segments)
    visible_slots = [slot for slot in slots if slot.local_col > 0 and np.isfinite(slot.x) and np.isfinite(slot.y)]
    bed_bbox = meta.get('bed_bbox')
    if bed_bbox is not None:
        (x1, y1, x2, y2) = bed_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2 - 1), (0, 255, 255), 2)
    for (row, line_color) in (('t', (255, 255, 0)), ('b', (255, 0, 255))):
        row_slots = sorted((slot for slot in visible_slots if slot.row == row), key=lambda slot: slot.local_col)
        for slot in row_slots:
            slot_color = _slot_color(slot)
            prior = (int(round(slot.x_prior)), int(round(slot.y_prior)))
            center = (int(round(slot.x)), int(round(slot.y)))
            segment = segments_by_id.get(slot.seg_id)
            if segment is not None:
                cv2.circle(canvas, center, OVERLAY_CENTER_RADIUS, slot_color, -1)
                polygon = np.asarray(segment['polygon'], dtype=np.int32).reshape(-1, 1, 2)
                cv2.drawContours(canvas, [polygon], -1, slot_color, OVERLAY_LINE_THICKNESS, cv2.LINE_AA)
            else:
                cv2.circle(canvas, prior, OVERLAY_CENTER_RADIUS, slot_color, OVERLAY_LINE_THICKNESS)
            if slot.seg_id is not None and slot.bbox is not None:
                (bx1, by1, bx2, by2) = slot.bbox
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), slot_color, OVERLAY_BBOX_THICKNESS)
            suffix = ' SEG' if slot.seg_id is not None else ' PRIOR'
            _put_label(canvas, _slot_label(slot) + suffix, (center[0] + 10, center[1] - 12), slot_color)
    flags = ','.join(meta.get('qc_flags', [])) or 'none'
    header = f"CAM0 {VERSION_LABEL} | {meta.get('status', '?')} | observed_cols={meta.get('n_visible', 0)} | holes={meta.get('hole_count', 0)} | seg={meta.get('segment_count', 0)} | assigned={meta.get('assigned_segment_count', 0)} | pitch={meta.get('pitch', 0):.1f} | QC={flags}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 60), (0, 0, 0), -1)
    cv2.putText(canvas, header, (10, 42), font, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_FONT_THICKNESS, cv2.LINE_AA)
    _slot_core.imwrite_unicode(out_path, canvas)

def process_one(image_path: Path, out_root: Path, model) -> dict | None:
    img = _slot_core.imread_unicode(image_path)
    if img is None:
        print(f'process one: [Skip] :{image_path.name}')
        return None
    segments = _lettuce_seg.predict_lettuce_segments(model, img)
    (slots, meta) = _adapter.map_cam0_v3(img, segments)
    if slots:
        seg_meta = _lettuce_seg.refine_slots_with_segments(slots, segments, img.shape[1], img.shape[0], bed_bbox=meta.get('bed_bbox'), enforce_cell_corridor=True)
    else:
        seg_meta = {'segment_count': len(segments), 'assigned_segment_count': 0, 'unassigned_segment_count': len(segments)}
    meta.update({key: value for (key, value) in seg_meta.items() if key != 'assigned_by_slot_object_id'})
    overlay_path = out_root / OUTPUT_IMAGE_DIRNAME / image_path.name
    draw_overlay(img, slots, meta, segments, overlay_path)
    common = {'image_name': image_path.name, 'status': meta.get('status', 'UNKNOWN'), 'observed_column_count': int(meta.get('n_visible', 0)), 'raw_observed_column_count': int(meta.get('raw_observed_column_count', meta.get('n_visible', 0))), 'duplicate_merge_count': int(meta.get('duplicate_merge_count', 0)), 'mapped_slot_count': sum((slot.local_col > 0 for slot in slots)), 'hole_count': int(meta.get('hole_count', 0)), 'lettuce_peak_count': int(meta.get('lettuce_peak_count', 0)), 'lettuce_segment_count': int(meta.get('segment_count', 0)), 'assigned_segment_count': int(meta.get('assigned_segment_count', 0)), 'pitch': float(meta.get('pitch', float('nan'))), 'phase_origin_col': meta.get('phase_origin_col'), 'left_hole_support': meta.get('left_hole_support'), 'qc_flags': '|'.join(meta.get('qc_flags', [])), 'overlay': str(overlay_path)}
    rows = []
    for slot in slots:
        row = slot.to_dict()
        row.update({'image_name': image_path.name, 'map_status': common['status'], 'observed_column_count': common['observed_column_count'], 'qc_flags': common['qc_flags']})
        rows.append(row)
    return {**common, 'rows': rows}

def main(argv=None):
    """Write slot assignments for all configured cam0 bed images."""
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
    for path in tqdm(images,desc='cam0 slot assignment',unit='image'):
        result=process_one(path, OUT_ROOT, model)
        if result is None:raise RuntimeError('Cannot process image: '+str(path))
        summary.append({k:v for k,v in result.items() if k!='rows'})
        rows.extend(result['rows'])
    pd.DataFrame(summary).to_csv(OUT_ROOT/SUMMARY_FILENAME,index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUT_ROOT/SLOTS_FILENAME,index=False,encoding='utf-8-sig')
    print('Slot assignments:',OUT_ROOT)
    return 0
if __name__ == '__main__':
    main()
