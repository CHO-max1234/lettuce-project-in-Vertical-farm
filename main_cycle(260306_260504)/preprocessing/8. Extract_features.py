"""Extract contour, masked grayscale and structural features from the stored target mask. Physical area uses the declared pixel scale. Missing scale or REVIEW states are preserved. Paths are defined in settings.py."""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
from pathlib import Path
from pipeline_paths import CROP_ROOT, FEATURE_ROOT
INPUT_PATH = str(CROP_ROOT / 'crop_manifest.csv')
OUTPUT_DIR = str(FEATURE_ROOT)
SCALE_JSON = ''
WORKERS = 4
CHUNK_SIZE = 256
import argparse, csv, hashlib, json, math, os, re, sqlite3, datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm
cv2.setNumThreads(1)
VERSION = 'crop_to_feature_inline_region_v2'
FEATURES = ['roi_area_px', 'area_cm2', 'perimeter_px', 'circularity', 'solidity', 'curvature', 'aspect_ratio', 'bottom_flatness', 'radial_min', 'radial_max_min_ratio', 'brightness_mean', 'dark_ratio', 'grad_std', 'inner_ratio_30', 'inner_ratio_50', 'ridge_circularity', 'ridge_std_r', 'ridge_mean_r', 'angular_cv', 'inner_ratio_core', 'ridge_circularity_core', 'ridge_std_r_core', 'ridge_mean_r_core', 'angular_cv_core', 'upper_area_px', 'lower_area_px', 'upper_lower_area_ratio', 'upper_circularity', 'upper_edge_density', 'blur_score', 'front_height_cm', 'front_height_cm2', 'volume_proxy']
META = ['record_id', 'camera', 'bed_id', 'capture_date', 'capture_time', 'physical_slot', 'source_path', 'crop_path', 'mask_path', 'crop_sha256', 'mask_sha256', 'status', 'auto_qc_pass', 'training_eligible', 'feature_status', 'feature_qc', 'feature_flags', 'feature_candidate', 'scale_status', 'mm_per_px', 'scale_version', 'scale_ready', 'scale_verified', 'scale_source', 'scale_qc', 'physical_units_estimated', 'mask_policy', 'mask_pixels', 'components', 'error', 'source_sha256', 'source_coordinate_space', 'crop_source_box', 'mapping_version', 'upstream_qc', 'metadata_warnings', 'reasons', 'run_signature', 'track_id', 'pair_id', 'season_id']
from pipeline_contracts import PROVENANCE_FIELDS
from mask_record import unpack as unpack_region
from pipeline_events import report_error
META = list(dict.fromkeys(META + PROVENANCE_FIELDS))
FIELDS = META + FEATURES

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda : f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def j(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def yes(v):
    return str(v).lower() in ('1', 'true', 'yes')

def image(path, flag=cv2.IMREAD_COLOR):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    if im is None:
        raise ValueError('Invalid input or state in image (rejected condition: im is None)')
    return im

def encoded_input(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return (data, hashlib.sha256(data).hexdigest())

def checked_image(path, flag=cv2.IMREAD_COLOR, prepared=None):
    (data, file_hash) = encoded_input(path) if prepared is None else prepared
    im = cv2.imdecode(data, flag)
    if im is None:
        raise ValueError('Invalid input or state in checked_image (rejected condition: im is None)')
    return (im, file_hash)

def iter_input(path):
    path = Path(path)
    if path.is_dir() and (path / 'crop_manifest.csv').is_file():
        path = path / 'crop_manifest.csv'
    if path.is_file():
        if path.suffix.lower() != '.csv':
            raise ValueError("Invalid input or state in iter_input (rejected condition: path.suffix.lower() != '.csv')")
        with path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                for name in ('crop_path', 'mask_path', 'source_path'):
                    if row.get(name):
                        q = Path(row[name])
                        row[name] = str(q if q.is_absolute() else (path.parent / q).resolve())
                if not row.get('record_id'):
                    raise ValueError("Invalid input or state in iter_input (rejected condition: not row.get('record_id'))")
                yield row
    else:
        root = path / 'crops' if (path / 'crops').is_dir() else path
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                continue
            if any((x in ('masks', 'features_final') for x in p.relative_to(root).parts[:-1])):
                continue
            m = re.search('^(bed\\d+)_(\\d{8})_(\\d{6})_(cam[012])(?:_[a-z0-9]+)*?_{1,2}([tb]\\d+)(?:_|$)', p.stem, re.I)
            mask = path / 'masks' / p.relative_to(root) if root != path else path.parent / 'masks' / p.relative_to(root)
            yield {'record_id': hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:24], 'crop_path': str(p.resolve()), 'mask_path': str(mask.resolve()) if mask.exists() else '', 'has_crop': True, 'auto_qc_pass': False, 'status': 'REVIEW_FOLDER_ONLY', 'camera': m[4].lower() if m else '', 'bed_id': m[1] if m else '', 'capture_date': m[2] if m else '', 'capture_time': m[3] if m else '', 'physical_slot': m[5] if m else ''}

def scale_metadata(row, scales):
    override = scales.get(row.get('camera'), {})
    if override:
        return dict(mm_per_px=override.get('mm_per_px'), scale_ready=yes(override.get('scale_ready', True)), scale_verified=yes(override.get('verified', override.get('scale_verified'))), scale_version=str(override.get('version', override.get('scale_version', '')) or ''), scale_source=override.get('scale_source', override.get('source', 'CAMERA_CONFIG')), scale_qc=override.get('scale_qc', override.get('qc', '')))
    return dict(mm_per_px=row.get('mm_per_px'), scale_ready=yes(row.get('scale_ready')), scale_verified=yes(row.get('scale_verified')), scale_version=row.get('scale_version', '') or '', scale_source=row.get('scale_source', '') or '', scale_qc=row.get('scale_qc', '') or '')

def scale_for(row, scales):
    meta = scale_metadata(row, scales)
    version = meta['scale_version']
    try:
        value = float(meta['mm_per_px'])
    except (ValueError, TypeError):
        return (None, 'MISSING_SCALE', version)
    if not math.isfinite(value) or value <= 0:
        return (None, 'INVALID_SCALE', version)
    if not meta['scale_ready']:
        return (None, 'SCALE_NOT_READY', version)
    source = str(meta['scale_source']).strip().upper()
    if source in ('', 'NONE', 'NULL', 'NAN'):
        return (None, 'MISSING_SCALE_SOURCE', version)
    if source in ('ERROR', 'UNRESOLVED'):
        return (None, 'UNRESOLVED_SCALE_SOURCE', version)
    status = 'VERIFIED' if meta['scale_verified'] and version else 'ESTIMATED'
    return (value, status, version)

def compute(row, scales, allow_rgb_mask=False, prepared=None):
    result = {k: row.get(k, '') for k in META}
    result.update(training_eligible=False, feature_candidate=False)
    if not yes(row.get('has_crop', bool(row.get('crop_path')))):
        return {**result, 'feature_status': 'NO_CROP', 'scale_status': 'NOT_APPLICABLE'}
    try:
        prepared = prepared or {}
        (crop, crop_hash) = checked_image(row['crop_path'], prepared=prepared.get('crop_path'))
        if row.get('crop_sha256') and crop_hash != row['crop_sha256']:
            raise ValueError("Invalid input or state in compute (rejected condition: row.get('crop_sha256') and crop_hash != row['crop_sha256'])")
        result['crop_sha256'] = crop_hash
        if row.get('mask_data'):
            mask = unpack_region(row)
            if mask.shape != crop.shape[:2]:
                raise ValueError('Invalid input or state in compute (rejected condition: mask.shape != crop.shape[:2])')
            result.update(mask_policy='EXPLICIT_BINARY_MASK', mask_sha256=row['mask_sha256'])
        elif row.get('mask_path'):
            (mask, mh) = checked_image(row['mask_path'], cv2.IMREAD_UNCHANGED, prepared.get('mask_path'))
            if mask.ndim != 2 or mask.shape != crop.shape[:2]:
                raise ValueError('Invalid input or state in compute (rejected condition: mask.ndim != 2 or mask.shape != crop.shape[:2])')
            if not np.all(np.isin(mask, [0, 1, 255])):
                raise ValueError('Invalid input or state in compute (rejected condition: not np.all(np.isin(mask, [0, 1, 255])))')
            if row.get('mask_sha256') and mh != row['mask_sha256']:
                raise ValueError("Invalid input or state in compute (rejected condition: row.get('mask_sha256') and mh != row['mask_sha256'])")
            result.update(mask_policy='EXPLICIT_BINARY_MASK', mask_sha256=mh)
        elif allow_rgb_mask:
            mask = np.max(crop, axis=2) > 8
            result['mask_policy'] = 'RGB_GT8_FALLBACK_REVIEW'
        else:
            raise ValueError('Invalid input or state in compute (rejected condition: allow_rgb_mask)')
        mask = (mask > 0).astype(np.uint8) * 255
        (scale, scale_status, scale_version) = scale_for(row, scales)
        result.update(scale_metadata(row, scales), scale_ready=scale is not None, physical_units_estimated=scale_status == 'ESTIMATED')
        if row.get('camera') not in ('cam0', 'cam1', 'cam2'):
            raise ValueError("Invalid input or state in compute (rejected condition: row.get('camera') not in ('cam0', 'cam1', 'cam2'))")
        (vals, qc) = FeatureExtractorAndQC().extract_30_features(crop, mask, scale or 1.0, row['camera'] == 'cam2')
        if qc['qc_status'] not in ('QC_PASS', 'QC_WARNING'):
            return {**result, 'feature_status': 'INVALID_MASK', 'feature_qc': qc['qc_status'], 'feature_flags': qc.get('qc_details', '')}
        if row['camera'] == 'cam2':
            vals['front_height_cm'] = math.sqrt(vals['front_height_cm2'])
        if any((not math.isfinite(float(v)) for v in vals.values())):
            raise ValueError('Invalid input or state in compute (rejected condition: any((not math.isfinite(float(v)) for v in vals.values())))')
        if scale is None:
            for k in ('area_cm2', 'front_height_cm', 'front_height_cm2', 'volume_proxy'):
                vals[k] = None
        (n, _, _, _) = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        result.update(vals)
        result.update(feature_status='OK', feature_qc=qc['qc_status'], feature_flags=qc.get('qc_flags', ''), scale_status=scale_status, mm_per_px=scale, scale_version=scale_version, mask_pixels=int((mask > 0).sum()), components=n - 1)
        result['feature_candidate'] = yes(row.get('auto_qc_pass')) and qc['qc_status'] == 'QC_PASS' and (result['mask_policy'] == 'EXPLICIT_BINARY_MASK') and (scale is not None)
        return result
    except Exception as e:
        return {**result, 'feature_status': 'ERROR', 'error': repr(e), 'error_traceback': traceback.format_exc()}

def export_workbook(path, rows, summary):
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill, Alignment
    wb = Workbook(write_only=True)
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='244062')
    header_alignment = Alignment(wrap_text=True)

    def safe_row(ws, values, header=False):
        out = []
        for v in values:
            if isinstance(v, (dict, list)):
                v = j(v)
            if isinstance(v, str) and len(v) > 32760:
                v = v[:32740] + ' [TRUNCATED; see CSV]'
            if header or isinstance(v, str):
                cell = WriteOnlyCell(ws, value=v)
                if isinstance(v, str):
                    cell.data_type = 's'
                if header:
                    cell.font = header_font
                    cell.fill = header_fill
                    cell.alignment = header_alignment
                out.append(cell)
            else:
                out.append(v)
        ws.append(out)
    intro = wb.create_sheet('Summary')
    intro.column_dimensions['A'].width = 30
    intro.column_dimensions['B'].width = 100
    safe_row(intro, ['Item', 'Value'], True)
    for (k, v) in summary.items():
        safe_row(intro, [k, v])
    safe_row(intro, ['Interpretation', 'Candidates pass automatic QC; they are not manual shape approvals or training splits.'])
    safe_row(intro, ['Physical units', 'scale_ready=True이고 유효한 mm_per_px와 출처가 있으면 물리 지표를 Calculation합니다. 검증 전 Value은 ESTIMATED로 표시하고 출처를 보존합니다. 수치 누락/오류만 빈칸입니다.'])
    safe_row(intro, ['Calculation', 'Calculates 30 image features and three front-view features from explicit masks. angular_cv describes contour-angle distribution.'])
    sheets = {}
    counts = {}
    pages = {}

    def append(group, row):
        if group not in sheets or counts[group] >= 1048575:
            if group in sheets:
                from openpyxl.utils import get_column_letter
                sheets[group].auto_filter.ref = f'A1:{get_column_letter(len(FIELDS))}{counts[group] + 1}'
            pages[group] = pages.get(group, 0) + 1
            ws = wb.create_sheet(group + ('_' + str(pages[group]) if pages[group] > 1 else ''))
            ws.freeze_panes = 'A2'
            for col in ('A', 'B', 'C', 'D', 'E', 'F'):
                ws.column_dimensions[col].width = 21
            safe_row(ws, FIELDS, True)
            sheets[group] = ws
            counts[group] = 0
        safe_row(sheets[group], [row.get(k) for k in FIELDS])
        counts[group] += 1
    for g in ('전체', '사용후보', '보류_오류'):
        ws = wb.create_sheet(g)
        ws.freeze_panes = 'A2'
        safe_row(ws, FIELDS, True)
        sheets[g] = ws
        counts[g] = 0
        pages[g] = 1
    for row in rows:
        append('전체', row)
        append('사용후보' if row.get('feature_candidate') else '보류_오류', row)
    from openpyxl.utils import get_column_letter
    for (g, ws) in sheets.items():
        ws.auto_filter.ref = f'A1:{get_column_letter(len(FIELDS))}{counts[g] + 1}'
    wb.save(path)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=INPUT_PATH)
    p.add_argument('--output', default=OUTPUT_DIR)
    p.add_argument('--scale-json', default=SCALE_JSON)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    p.add_argument('--allow-rgb-mask', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--force', action='store_true')
    return p

def prune_completed_history(db, token, outputs_complete=False):
    if not outputs_complete:
        return 0
    removed = db.execute('DELETE FROM current_rows WHERE token<>?', (token,)).rowcount
    db.commit()
    return removed

def main(argv=None):
    args = parser().parse_args(argv)
    if not args.input:
        raise SystemExit('Validation failed in main: INPUT_PATH --input')
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError('Invalid input or state in main (rejected condition: args.workers < 1 or args.chunk_size < 1)')
    inp = Path(args.input).resolve()
    base = inp if inp.is_dir() else inp.parent
    scales = json.loads(Path(args.scale_json).read_text(encoding='utf-8-sig')) if args.scale_json else {}
    if args.check:
        count = missing = 0
        for r in iter_input(inp):
            count += 1
            if yes(r.get('has_crop', bool(r.get('crop_path')))):
                if not Path(r.get('crop_path', '')).is_file():
                    missing += 1
                elif r.get('mask_data'):
                    try:
                        if unpack_region(r).shape != image(r['crop_path']).shape[:2]:
                            missing += 1
                    except Exception:
                        missing += 1
                elif not args.allow_rgb_mask and (not Path(r.get('mask_path', '')).is_file()):
                    missing += 1
        print(json.dumps({'rows': count, 'missing_crop_or_mask': missing, 'extraction': False, 'xlsx_written': False}))
        return 1 if missing or not count else 0
    out = Path(args.output).resolve() if args.output else base / 'features_final'
    if out == base or out in base.parents:
        raise ValueError('Invalid input or state in main (rejected condition: out == base or out in base.parents)')
    out.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (out / '.run.lock').open('a+b')
    lock.seek(0)
    if (out / '.run.lock').stat().st_size == 0:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    db = sqlite3.connect(out / 'feature_state.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS cache(id TEXT PRIMARY KEY,fingerprint TEXT,payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS current_rows(token TEXT,id TEXT,payload TEXT,PRIMARY KEY(token,id))')
    signature = hashlib.sha256((digest(Path(__file__)) + j(scales) + str(args.allow_rgb_mask)).encode()).hexdigest()
    token = datetime.datetime.now().isoformat()
    summary = {'version': VERSION, 'signature': signature, 'rows': 0, 'OK': 0, 'ERROR': 0, 'NO_CROP': 0, 'INVALID_MASK': 0, 'feature_candidates': 0}

    def work(item):
        (r, old) = item
        try:
            prepared = {k: encoded_input(r[k]) for k in ('crop_path', 'mask_path') if r.get(k) and Path(r[k]).is_file()}
            hashes = [item[1] for item in prepared.values()]
            fp = hashlib.sha256((signature + j(r) + j(hashes)).encode()).hexdigest()
        except Exception:
            fp = hashlib.sha256((signature + j(r)).encode()).hexdigest()
            prepared = {}
        if old and old[0] == fp and (not args.force):
            cached = json.loads(old[1])
            if cached.get('feature_status') != 'ERROR':
                return (fp, cached)
        return (fp, compute(r, scales, args.allow_rgb_mask, prepared))

    def chunks():
        batch = []
        for r in iter_input(inp):
            batch.append(r)
            if len(batch) >= args.chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in tqdm(chunks(), desc='crop → feature', unit='chunk'):
            tasks = [(r, db.execute('SELECT fingerprint,payload FROM cache WHERE id=?', (r['record_id'],)).fetchone()) for r in batch]
            for (fp, r) in pool.map(work, tasks):
                if r['feature_status'] == 'ERROR':
                    report_error(r.get('crop_path', ''), r.get('error', 'feature ERROR'), r.get('error_traceback'))
                rid = r['record_id']
                payload = j(r)
                db.execute('INSERT INTO current_rows VALUES(?,?,?)', (token, rid, payload))
                db.execute('INSERT OR REPLACE INTO cache VALUES(?,?,?)', (rid, fp, payload))
                summary['rows'] += 1
                summary[r['feature_status']] += 1
                summary['feature_candidates'] += int(r.get('feature_candidate', False))
            db.commit()
    if not summary['rows']:
        raise ValueError("Invalid input or state in main (rejected condition: not summary['rows'])")

    def rows(writer):
        for (payload,) in db.execute('SELECT payload FROM current_rows WHERE token=? ORDER BY id', (token,)):
            row = json.loads(payload)
            writer.writerow(row)
            yield row
    cp = out / 'features_all.csv'
    temp = out / 'features_all.writing'
    xp = out / 'crop_features.xlsx'
    xt = out / 'crop_features.writing.xlsx'
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, FIELDS, extrasaction='ignore')
        wr.writeheader()
        export_workbook(xt, rows(wr), summary)
    os.replace(temp, cp)
    os.replace(xt, xp)
    summary['xlsx_path'] = str(xp)
    summary['xlsx_sha256'] = digest(xp)
    (out / 'feature_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    prune_completed_history(db, token, outputs_complete=summary['ERROR'] == 0)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    db.close()
    lock.close()
    return 1 if summary['ERROR'] else 0

class FeatureExtractorAndQC:
    """Core FeatureExtractorAndQC calculations."""

    def __init__(self, blur_threshold: float=100.0):
        self.blur_threshold = blur_threshold

    @staticmethod
    def calc_blur_score(gray_img: np.ndarray) -> float:
        if gray_img is None or gray_img.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

    def extract_30_features(self, crop_bgr: np.ndarray, mask: np.ndarray, mm_per_px: float=0.735, is_front: bool=False) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Compute the declared contour, grayscale and structural features from an explicit mask."""
        if crop_bgr is None or mask is None or np.sum(mask > 0) < 30:
            return (self._empty_features(is_front), {'qc_status': 'EMPTY_OR_INVALID_MASK', 'qc_details': 'Mask area too small'})
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        bin_mask = (mask > 0).astype(np.uint8)
        (contours, _) = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (self._empty_features(is_front), {'qc_status': 'NO_CONTOUR', 'qc_details': 'No contour found'})
        cnt = max(contours, key=cv2.contourArea)
        roi_area_px = float(cv2.contourArea(cnt))
        if roi_area_px < 20:
            return (self._empty_features(is_front), {'qc_status': 'TOO_SMALL', 'qc_details': 'Area < 20px'})
        perimeter_px = float(cv2.arcLength(cnt, True))
        area_cm2 = roi_area_px * mm_per_px ** 2 / 100.0
        circularity = 4.0 * np.pi * roi_area_px / perimeter_px ** 2 if perimeter_px > 0 else 0.0
        circularity = float(np.clip(circularity, 0.0, 1.0))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = roi_area_px / hull_area if hull_area > 0 else 0.0
        solidity = float(np.clip(solidity, 0.0, 1.0))
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        hull_perimeter = cv2.arcLength(hull, True)
        curvature = perimeter_px / hull_perimeter if hull_perimeter > 0 else 1.0
        bottom_pts = cnt[cnt[:, :, 1] >= y + 0.85 * h]
        if len(bottom_pts) > 5:
            bottom_flatness = float(np.std(bottom_pts[:, 1]))
        else:
            bottom_flatness = 0.0
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            (cx, cy) = (x + w / 2.0, y + h / 2.0)
        pts = cnt[:, 0, :]
        distances = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        radial_min = float(np.min(distances)) if len(distances) > 0 else 0.0
        radial_max = float(np.max(distances)) if len(distances) > 0 else 1.0
        radial_max_min_ratio = radial_max / max(1.0, radial_min)
        masked_gray = gray[bin_mask > 0]
        brightness_mean = float(np.mean(masked_gray)) if len(masked_gray) > 0 else 0.0
        dark_ratio = float(np.mean(masked_gray < 50)) if len(masked_gray) > 0 else 0.0
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobelx, sobely)
        grad_std = float(np.std(grad_mag[bin_mask > 0])) if len(masked_gray) > 0 else 0.0
        r_avg = (w + h) / 4.0
        (y_grid, x_grid) = np.ogrid[:gray.shape[0], :gray.shape[1]]
        dist_from_c = np.hypot(x_grid - cx, y_grid - cy)
        mask_30 = (dist_from_c <= 0.3 * r_avg) & (bin_mask > 0)
        mask_50 = (dist_from_c <= 0.5 * r_avg) & (bin_mask > 0)
        inner_ratio_30 = float(np.sum(mask_30)) / roi_area_px if roi_area_px > 0 else 0.0
        inner_ratio_50 = float(np.sum(mask_50)) / roi_area_px if roi_area_px > 0 else 0.0
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        (angle_hist, _) = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
        angular_cv = float(np.std(angle_hist) / (np.mean(angle_hist) + 1e-06))
        ridge_mean_r = float(np.mean(distances)) if len(distances) > 0 else 0.0
        ridge_std_r = float(np.std(distances)) if len(distances) > 0 else 0.0
        ridge_circularity = float(ridge_mean_r / (ridge_std_r + 1.0))
        inner_ratio_core = inner_ratio_30
        ridge_circularity_core = ridge_circularity
        ridge_std_r_core = ridge_std_r
        ridge_mean_r_core = ridge_mean_r
        angular_cv_core = angular_cv
        upper_mask = bin_mask.copy()
        upper_mask[int(cy):, :] = 0
        lower_mask = bin_mask.copy()
        lower_mask[:int(cy), :] = 0
        upper_area_px = float(np.sum(upper_mask > 0))
        lower_area_px = float(np.sum(lower_mask > 0))
        upper_lower_area_ratio = upper_area_px / max(1.0, lower_area_px)
        (upper_cnts, _) = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if upper_cnts:
            u_cnt = max(upper_cnts, key=cv2.contourArea)
            u_peri = float(cv2.arcLength(u_cnt, True))
            upper_circularity = float(4.0 * np.pi * upper_area_px / u_peri ** 2) if u_peri > 0 else 0.0
            edges = cv2.Canny(gray, 50, 150)
            upper_edge_density = float(np.sum((edges > 0) & (upper_mask > 0)) / max(1.0, upper_area_px))
        else:
            upper_circularity = 0.0
            upper_edge_density = 0.0
        blur_score = self.calc_blur_score(gray)
        features = {'roi_area_px': roi_area_px, 'area_cm2': area_cm2, 'perimeter_px': perimeter_px, 'circularity': circularity, 'solidity': solidity, 'curvature': curvature, 'aspect_ratio': aspect_ratio, 'bottom_flatness': bottom_flatness, 'radial_min': radial_min, 'radial_max_min_ratio': radial_max_min_ratio, 'brightness_mean': brightness_mean, 'dark_ratio': dark_ratio, 'grad_std': grad_std, 'inner_ratio_30': inner_ratio_30, 'inner_ratio_50': inner_ratio_50, 'ridge_circularity': ridge_circularity, 'ridge_std_r': ridge_std_r, 'ridge_mean_r': ridge_mean_r, 'angular_cv': angular_cv, 'inner_ratio_core': inner_ratio_core, 'ridge_circularity_core': ridge_circularity_core, 'ridge_std_r_core': ridge_std_r_core, 'ridge_mean_r_core': ridge_mean_r_core, 'angular_cv_core': angular_cv_core, 'upper_area_px': upper_area_px, 'lower_area_px': lower_area_px, 'upper_lower_area_ratio': upper_lower_area_ratio, 'upper_circularity': upper_circularity, 'upper_edge_density': upper_edge_density, 'blur_score': blur_score}
        if is_front:
            front_height_cm = h * mm_per_px / 10.0
            features['front_height_cm2'] = float(front_height_cm ** 2)
            features['volume_proxy'] = float(area_cm2 * front_height_cm)
        qc_flags = []
        if blur_score < self.blur_threshold:
            qc_flags.append('BLUR_SUSPICIOUS')
        if circularity < 0.05:
            qc_flags.append('LOW_CIRCULARITY')
        if solidity < 0.25:
            qc_flags.append('LOW_SOLIDITY')
        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            qc_flags.append('ASPECT_OUTLIER')
        qc_info = {'qc_status': 'QC_PASS' if not qc_flags else 'QC_WARNING', 'qc_flags': ';'.join(qc_flags) if qc_flags else 'NONE'}
        return (features, qc_info)

    def _empty_features(self, is_front: bool=False) -> Dict[str, float]:
        keys = ['roi_area_px', 'area_cm2', 'perimeter_px', 'circularity', 'solidity', 'curvature', 'aspect_ratio', 'bottom_flatness', 'radial_min', 'radial_max_min_ratio', 'brightness_mean', 'dark_ratio', 'grad_std', 'inner_ratio_30', 'inner_ratio_50', 'ridge_circularity', 'ridge_std_r', 'ridge_mean_r', 'angular_cv', 'inner_ratio_core', 'ridge_circularity_core', 'ridge_std_r_core', 'ridge_mean_r_core', 'angular_cv_core', 'upper_area_px', 'lower_area_px', 'upper_lower_area_ratio', 'upper_circularity', 'upper_edge_density', 'blur_score']
        res = {k: 0.0 for k in keys}
        if is_front:
            res['front_height_cm2'] = 0.0
            res['volume_proxy'] = 0.0
        return res
if __name__ == '__main__':
    raise SystemExit(main())
