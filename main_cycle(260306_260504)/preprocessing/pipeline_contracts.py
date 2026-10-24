"""Processing utilities for pipeline contracts."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import hashlib
import math
from pathlib import Path
from pipeline_paths import SEASON_ID, identity
from camera_selection import SELECTION_FIELDS
PROVENANCE_FIELDS = ['raw_source_path', 'original_source_path', 'original_key', 'old_key', 'new_key', 'ocr_status', 'ocr_action', 'raw_sha256', 'bed_x1_raw', 'bed_y1_raw', 'bed_x2_raw', 'bed_y2_raw', 'scale_image_key', 'scale_reference_count', 'scale_reference_1', 'scale_reference_2', 'scale_reference_3', 'photometric_policy', 'slot_namespace_version', 'legacy_slot_id', 'bbox_crop_path', 'bbox_overlay_path', 'bbox_crop_status', 'raw_offset_status']
PROVENANCE_FIELDS += SELECTION_FIELDS

def number(row, *keys):
    for key in keys:
        try:
            value = float(row.get(key, ''))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return None

def remap_cam1(row):
    """Map legacy CAM1 columns 1..10 to physical columns 3..12 exactly once."""
    row = dict(row)
    slot = row.get('canonical_id') or row.get('slot_name')
    if row.get('slot_namespace_version') == 'physical_12col_v1':
        if row['camera'] == 'cam1':
            old = row.get('legacy_slot_id', '')
            if not old or slot != f'{old[0]}{int(old[1:]) + 2}':
                raise ValueError('CAM1 physical-slot crosswalk mismatch')
        return row
    if not slot:
        raise ValueError('Missing physical slot')
    row['legacy_slot_id'] = slot
    if row['camera'] == 'cam1':
        col = int(slot[1:])
        if not 1 <= col <= 10:
            raise ValueError(f'Invalid CAM1 legacy slot: {slot}')
        row['canonical_id'] = f'{slot[0]}{col + 2}'
        row['col'] = col + 2
        if number(row, 'physical_col') is not None:
            row['physical_col'] = int(number(row, 'physical_col')) + 2
        row['mapping_version'] = str(row.get('mapping_version', '')) + ' | 12col_crosswalk_v1'
    row['slot_namespace_version'] = 'physical_12col_v1'
    if row.get('source_image_name'):
        canonical = row.get('canonical_id') or slot
        row['record_id'] = f"{row['camera']}|{row['source_image_name']}|{canonical}"
        row['crop_name'] = f"{Path(row['source_image_name']).stem}__{canonical}.jpg"
    return row

def connect_bbox(row, frame, scale, bright):
    """Join OCR, scale and normalized-image provenance without changing bed-pixel coordinates."""
    row = remap_cam1(row) if row['camera'] in ('cam0', 'cam1') else dict(row)
    source = Path(bright['output_path']).resolve()
    ident = identity(source)
    if str(Path(scale['source_path']).resolve()).casefold() != str(Path(frame['current_path']).resolve()).casefold():
        raise ValueError('Scale/OCR source paths differ')
    if scale.get('source_sha256') and scale['source_sha256'] != frame['sha256']:
        raise ValueError('Scale/OCR source checksums differ')
    if bright['source_sha256'] != frame['sha256']:
        raise ValueError('Brightness/OCR source checksums differ')
    if ident['camera'] != row['camera']:
        raise ValueError('bbox camera mismatch')
    if row.get('bed_id') and row['bed_id'].lower() != ident['bed_id']:
        raise ValueError('bbox/OCR identity mismatch')
    (height, width) = bright['shape'][:2]
    if number(row, 'source_width') != width or number(row, 'source_height') != height:
        raise ValueError('bbox and normalized image dimensions differ')
    slot = row.get('canonical_id') or row.get('slot_name')
    if not slot:
        raise ValueError('Missing slot name')
    ax = ay = None
    for (x, y) in [('anchor_x', 'anchor_y'), ('seg_center_x', 'seg_center_y'), ('slot_center_x', 'slot_center_y'), ('slot_prior_x', 'slot_prior_y'), ('x_prior', 'y_prior')]:
        if number(row, x) is not None and number(row, y) is not None:
            (ax, ay) = (number(row, x), number(row, y))
            break
    if ax is None or ay is None:
        raise ValueError('Missing anchor')
    (cw, ch) = (number(row, 'cell_width'), number(row, 'cell_height'))
    if cw is None or ch is None:
        raise ValueError('Missing cell dimensions')
    unavailable = not (0 <= ax < width and 0 <= ay < height) or cw <= 0 or ch <= 0
    if unavailable:
        row['visibility'] = 'OUT_OF_FOV'
    search = [max(0, round(ax - 0.725 * cw)), max(0, round(ay - 0.725 * ch)), min(width, round(ax + 0.725 * cw)), min(height, round(ay + 0.725 * ch))]
    qc = '|'.join((str(row[k]) for k in ('qc_flags', 'image_qc', 'map_status') if row.get(k)))
    if frame['status'] not in ('RENAMED', 'UNCHANGED', 'MAPPED'):
        qc += '|OCR_IDENTITY_REVIEW'
    row.update(source_path=str(source), source_sha256=bright['output_sha256'], source_coordinate_space='bed_pixels', physical_slot=slot, anchor_x=ax, anchor_y=ay, upstream_qc=qc, bed_id=ident['bed_id'], capture_date=ident['capture_date'], capture_time=ident['capture_time'], mapping_version=row.get('mapping_version') or 'CAM2_VER1__rail_is_t1_b1', season_id=SEASON_ID, old_key=frame['old_key'], new_key=frame.get('new_key') or frame['old_key'], original_key=frame['original_key'], ocr_status=frame['ocr_status'], ocr_action=frame['status'], raw_source_path=frame['raw_source_path'], original_source_path=frame.get('original_source_path', ''), raw_sha256=frame['raw_sha256'], photometric_policy='brightness_final_v2', scale_image_key=scale['image_key'], mm_per_px=scale.get('mm_per_px'), scale_ready=scale.get('scale_ready', False), scale_verified=scale.get('scale_verified', False), scale_source=scale.get('scale_source', 'UNRESOLVED'), scale_qc=scale.get('scale_qc') or scale.get('qc', ''), scale_version=scale.get('scale_version', ''), scale_reference_count=scale.get('reference_count', 0), slot_namespace_version='physical_12col_v1')
    for (i, axis) in enumerate(('x1', 'y1', 'x2', 'y2')):
        row['bed_' + axis + '_raw'] = frame['bed_box'][i]
        row['search_' + axis] = search[i]
    row['raw_offset_status'] = frame.get('raw_offset_status', 'KNOWN')
    for i in range(1, 4):
        row[f'scale_reference_{i}'] = scale.get(f'reference_{i}', '')
    row['track_id'] = f"{SEASON_ID}|{ident['bed_id']}|{slot}"
    row['pair_id'] = f"{SEASON_ID}|{ident['key']}|{slot}"
    row['record_id'] = hashlib.sha256(f'{source}|{slot}'.encode()).hexdigest()[:24]
    return row
