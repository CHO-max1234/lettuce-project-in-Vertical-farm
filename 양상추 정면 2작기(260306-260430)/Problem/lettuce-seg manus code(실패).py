# ============================================================
# Lettuce slot crop generation from best.pt (dynamic Y from p1~p4 CSV)
# - input : warped/front bed images
# - infer : YOLOv8-seg best.pt
# - assign: Hungarian matching to 12 slots (t1~t6, b1~b6)
# - save  : all detected pieces with slot names
# - output: crops + overlay + csv metadata
# ============================================================

import os
import gc
import cv2
import math
import time
import glob
import traceback
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

MODEL_PATH = r"/home/ubuntu/upload/best.pt"
IMG_ROOT = r"/home/ubuntu/work_brightness/test10_images"
OUT_ROOT = r"/home/ubuntu/work_brightness/lettuce_seg_test10_dynamic_y_output"
CSV_LOOKUP_PATH = r"/home/ubuntu/upload/per_image_results_v3.1.csv"

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
IMGSZ = 960
CONF_THRES = 0.25
IOU_THRES = 0.50
DEVICE = 'cpu'
MAX_DET = 64

PAD_PX = 14
MIN_BOX_W = 20
MIN_BOX_H = 20
MASK_BG = "black"
SAVE_EXT = ".png"

# X는 기존 유지, Y만 p1~p4 기반으로 동적 계산
X_POS = [0.10, 0.26, 0.42, 0.58, 0.74, 0.90]
SLOT_ORDER = [f"t{i}" for i in range(1, 7)] + [f"b{i}" for i in range(1, 7)]

# 사각형 내부 상대 row 비율
# top: 상단 edge에서 조금 아래, bot: 하단 edge 근처(혼입 억제 우선)
TOP_ROW_R = 0.28
BOT_ROW_R = 0.95

# ------------------------------------------------------------
# 빈 슬롯 허용용 안전장치
# - 현재 12 slot + Hungarian 구조는 유지
# - 단, t/b 위치에 맞지 않으면 강제 배정하지 않고 None 처리
# ------------------------------------------------------------
ROW_SPLIT_R = 0.50          # t와 b 중간선
ROW_MARGIN_R = 0.18         # 중간선 주변 허용 여유폭(세로 gap 비율)
PRIMARY_MAX_DIST_R = 0.65   # Hungarian 1차 배정 허용 거리 (세로 gap 비율)
EXTRA_MAX_DIST_R = 1.05     # 같은 slot 추가 조각(p02 이상) 허용 거리
UNASSIGNED_LABEL = "unassigned"
UNASSIGNED_STATUS = "row_guard_rejected"
UNASSIGNED_PART_NO = None
UNASSIGNED_SLOT_NAME = None

USE_BED_SUBFOLDER = True
CHUNK_SIZE = 10
SAVE_EVERY_CHUNK = True
N_SAVE_WORKERS = 8
SKIP_IF_DONE = False

SAVE_OVERLAY = True
OVERLAY_LINE_W = 2
FONT_SCALE = 0.7


def now_hms():
    return time.strftime("%H:%M:%S")


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def list_images_recursive(root, exts=IMG_EXTS):
    root = str(root)
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return sorted(list(set(files)))


def parse_base_key(img_path):
    return Path(img_path).stem


def parse_bed_name(base_key):
    parts = base_key.split("_")
    return parts[0] if len(parts) > 0 else "unknown_bed"


def load_csv_lookup(csv_path):
    df = pd.read_csv(csv_path)
    df['base_key'] = df['image_name'].astype(str).apply(lambda x: Path(x).stem)
    keep_cols = ['base_key', 'image_name', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y', 'matched_count', 'source', 'notes']
    for col in ['p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y', 'matched_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    out = {}
    for _, row in df[keep_cols].iterrows():
        out[row['base_key']] = row.to_dict()
    return out


def lerp(a, b, t):
    return a + (b - a) * t


def make_slot_anchors_dynamic(base_key, w, h, lookup):
    info = lookup.get(base_key)
    if info is None:
        anchors = {}
        for i, xr in enumerate(X_POS, start=1):
            anchors[f"t{i}"] = (int(round(w * xr)), int(round(h * 0.33)))
            anchors[f"b{i}"] = (int(round(w * xr)), int(round(h * 0.72)))
        return anchors, {'anchor_mode': 'fallback_fixed', 'top_row_r': None, 'bot_row_r': None}

    p1 = (float(info['p1_x']), float(info['p1_y']))
    p2 = (float(info['p2_x']), float(info['p2_y']))
    p3 = (float(info['p3_x']), float(info['p3_y']))
    p4 = (float(info['p4_x']), float(info['p4_y']))

    anchors = {}
    for i, xr in enumerate(X_POS, start=1):
        tx = lerp(p1[0], p2[0], xr)
        ty = lerp(p1[1], p2[1], xr)
        bx = lerp(p4[0], p3[0], xr)
        by = lerp(p4[1], p3[1], xr)

        top_ax = lerp(tx, bx, TOP_ROW_R)
        top_ay = lerp(ty, by, TOP_ROW_R)
        bot_ax = lerp(tx, bx, BOT_ROW_R)
        bot_ay = lerp(ty, by, BOT_ROW_R)

        anchors[f"t{i}"] = (int(round(top_ax)), int(round(top_ay)))
        anchors[f"b{i}"] = (int(round(bot_ax)), int(round(bot_ay)))

    meta = {
        'anchor_mode': 'dynamic_p1p4',
        'top_row_r': TOP_ROW_R,
        'bot_row_r': BOT_ROW_R,
        'p1_x': p1[0], 'p1_y': p1[1], 'p2_x': p2[0], 'p2_y': p2[1],
        'p3_x': p3[0], 'p3_y': p3[1], 'p4_x': p4[0], 'p4_y': p4[1],
        'matched_count_csv': int(info['matched_count']) if pd.notna(info['matched_count']) else None,
        'source_csv': info.get('source'),
        'notes_csv': info.get('notes'),
    }
    return anchors, meta


def mask_to_bbox(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_centroid(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def apply_mask_bg(crop_bgr, crop_mask_u8, mode="black"):
    if mode == "keep":
        return crop_bgr
    bg_val = 0 if mode == "black" else 255
    out = np.full_like(crop_bgr, bg_val)
    m = crop_mask_u8 > 0
    out[m] = crop_bgr[m]
    return out


def build_cost_matrix(det_centers, slot_points):
    cost = np.zeros((len(det_centers), len(slot_points)), dtype=np.float32)
    for i, (cx, cy) in enumerate(det_centers):
        for j, (sx, sy) in enumerate(slot_points):
            cost[i, j] = math.hypot(cx - sx, cy - sy)
    return cost


def get_slot_column_no(slot_name):
    return int(str(slot_name)[1:])


def get_row_pair_points(slot_name, anchors_dict):
    col_no = get_slot_column_no(slot_name)
    top_key = f"t{col_no}"
    bot_key = f"b{col_no}"
    top_pt = anchors_dict[top_key]
    bot_pt = anchors_dict[bot_key]
    return top_pt, bot_pt


def validate_slot_candidate(det_center, slot_name, anchors_dict, max_dist_r):
    cx, cy = det_center
    sx, sy = anchors_dict[slot_name]
    top_pt, bot_pt = get_row_pair_points(slot_name, anchors_dict)

    row_gap = math.hypot(bot_pt[0] - top_pt[0], bot_pt[1] - top_pt[1])
    row_gap = max(row_gap, 1.0)

    split_y = lerp(top_pt[1], bot_pt[1], ROW_SPLIT_R)
    row_margin_px = row_gap * ROW_MARGIN_R
    dist_px = math.hypot(cx - sx, cy - sy)
    max_dist_px = row_gap * max_dist_r

    if slot_name.startswith('t'):
        row_ok = (cy <= (split_y + row_margin_px))
    else:
        row_ok = (cy >= (split_y - row_margin_px))

    dist_ok = (dist_px <= max_dist_px)
    valid = bool(row_ok and dist_ok)

    return {
        'valid': valid,
        'row_ok': bool(row_ok),
        'dist_ok': bool(dist_ok),
        'dist_px': float(dist_px),
        'max_dist_px': float(max_dist_px),
        'split_y': float(split_y),
        'row_margin_px': float(row_margin_px),
        'anchor_x': float(sx),
        'anchor_y': float(sy),
    }


def assign_primary_slots(det_centers, anchors_dict):
    primary = {}
    if len(det_centers) == 0:
        return primary, []
    slot_names = list(anchors_dict.keys())
    slot_points = [anchors_dict[s] for s in slot_names]
    cost = build_cost_matrix(det_centers, slot_points)
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_det = set()
    for r, c in zip(row_ind, col_ind):
        if r < len(det_centers) and c < len(slot_names):
            primary[r] = slot_names[c]
            assigned_det.add(r)
    unassigned = [i for i in range(len(det_centers)) if i not in assigned_det]
    return primary, unassigned


def assign_all_pieces(det_centers, anchors_dict):
    slot_of_det = {}
    part_no_of_det = {}
    reject_reason_of_det = {}
    if len(det_centers) == 0:
        return slot_of_det, part_no_of_det, reject_reason_of_det

    primary_map, initial_unassigned = assign_primary_slots(det_centers, anchors_dict)
    slot_names = list(anchors_dict.keys())
    slot_groups = {s: [] for s in slot_names}
    recheck_det_idxs = list(initial_unassigned)

    for det_idx, slot_name in primary_map.items():
        check = validate_slot_candidate(det_centers[det_idx], slot_name, anchors_dict, PRIMARY_MAX_DIST_R)
        if check['valid']:
            slot_groups[slot_name].append(det_idx)
            slot_of_det[det_idx] = slot_name
        else:
            recheck_det_idxs.append(det_idx)
            reject_reason_of_det[det_idx] = 'primary_row_or_distance_fail'

    for det_idx in recheck_det_idxs:
        cx, cy = det_centers[det_idx]
        candidates = []
        for slot_name in slot_names:
            check = validate_slot_candidate((cx, cy), slot_name, anchors_dict, EXTRA_MAX_DIST_R)
            if check['valid']:
                candidates.append((slot_name, check['dist_px']))

        if len(candidates) == 0:
            if det_idx not in reject_reason_of_det:
                reject_reason_of_det[det_idx] = 'no_valid_slot_after_guard'
            continue

        candidates.sort(key=lambda x: x[1])
        best_slot = candidates[0][0]
        slot_groups[best_slot].append(det_idx)
        slot_of_det[det_idx] = best_slot
        if det_idx in reject_reason_of_det:
            reject_reason_of_det.pop(det_idx, None)

    for slot_name in slot_names:
        dets = slot_groups[slot_name]
        if len(dets) == 0:
            continue
        dets_sorted = sorted(
            dets,
            key=lambda i: math.hypot(
                det_centers[i][0] - anchors_dict[slot_name][0],
                det_centers[i][1] - anchors_dict[slot_name][1]
            )
        )
        for k, det_idx in enumerate(dets_sorted, start=1):
            part_no_of_det[det_idx] = k

    return slot_of_det, part_no_of_det, reject_reason_of_det


def draw_overlay(img, det_infos, anchors_dict):
    canvas = img.copy()
    for slot_name, (sx, sy) in anchors_dict.items():
        cv2.circle(canvas, (int(sx), int(sy)), 8, (0, 255, 255), -1)
        cv2.putText(canvas, slot_name, (int(sx) - 16, int(sy) - 12), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 2, cv2.LINE_AA)
    for d in det_infos:
        x1, y1, x2, y2 = d['bbox']
        if d['slot_name'] is None or d['part_no'] is None:
            label = UNASSIGNED_LABEL
            color = (0, 0, 255)
        else:
            label = f"{d['slot_name']}_p{d['part_no']:02d}"
            color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, OVERLAY_LINE_W)
        cv2.putText(canvas, label, (x1, max(22, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, 2, cv2.LINE_AA)
    return canvas


def build_output_paths(base_key, bed_name, slot_name, part_no):
    filename = f"{base_key}_{slot_name}_p{part_no:02d}{SAVE_EXT}"
    out_dir = os.path.join(OUT_ROOT, 'crops', bed_name) if USE_BED_SUBFOLDER else os.path.join(OUT_ROOT, 'crops_flat')
    safe_mkdir(out_dir)
    return os.path.join(out_dir, filename)


def save_crop_task(task):
    try:
        img = task['img']
        mask = task['mask']
        out_path = task['out_path']
        x1, y1, x2, y2 = task['bbox']
        x1p = max(0, x1 - PAD_PX)
        y1p = max(0, y1 - PAD_PX)
        x2p = min(img.shape[1] - 1, x2 + PAD_PX)
        y2p = min(img.shape[0] - 1, y2 + PAD_PX)
        if (x2p - x1p + 1) < MIN_BOX_W or (y2p - y1p + 1) < MIN_BOX_H:
            return {'ok': False, 'out_path': out_path, 'reason': 'too_small'}
        crop = img[y1p:y2p+1, x1p:x2p+1].copy()
        crop_mask = mask[y1p:y2p+1, x1p:x2p+1].copy()
        crop2 = apply_mask_bg(crop, crop_mask, MASK_BG)
        cv2.imwrite(out_path, crop2)
        return {'ok': True, 'out_path': out_path, 'skipped': False}
    except Exception as e:
        return {'ok': False, 'out_path': task.get('out_path', ''), 'reason': str(e)}


def process_one_image(img_path, model, lookup):
    base_key = parse_base_key(img_path)
    bed_name = parse_bed_name(base_key)
    img = cv2.imread(img_path)
    if img is None:
        return {'meta_rows': [], 'n_det': 0, 'n_saved': 0, 'n_fail': 1, 'overlay_path': None, 'error': f'cv2.imread failed: {img_path}'}

    h, w = img.shape[:2]
    anchors, anchor_meta = make_slot_anchors_dynamic(base_key, w, h, lookup)

    results = model.predict(source=img, imgsz=IMGSZ, conf=CONF_THRES, iou=IOU_THRES, device=DEVICE, max_det=MAX_DET, retina_masks=True, verbose=False)

    if len(results) == 0 or results[0].masks is None or results[0].boxes is None or len(results[0].boxes) == 0:
        meta_row = {
            'image_path': img_path, 'base_key': base_key, 'bed_name': bed_name,
            'slot_name': None, 'part_no': None, 'crop_filename': None, 'crop_path': None,
            'conf': None, 'cls': None, 'cx': None, 'cy': None,
            'bbox_x1': None, 'bbox_y1': None, 'bbox_x2': None, 'bbox_y2': None,
            'bbox_w': None, 'bbox_h': None, 'area_px': None, 'mask_pixels': None,
            'is_primary': None, 'n_det_in_image': 0, 'status': 'no_detection',
            **anchor_meta
        }
        return {'meta_rows': [meta_row], 'n_det': 0, 'n_saved': 0, 'n_fail': 0, 'overlay_path': None, 'error': None}

    r = results[0]
    xyxy = r.boxes.xyxy.detach().cpu().numpy()
    confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.zeros(len(xyxy), dtype=np.float32)
    clss = r.boxes.cls.detach().cpu().numpy() if r.boxes.cls is not None else np.zeros(len(xyxy), dtype=np.float32)
    masks_small = r.masks.data.detach().cpu().numpy()

    masks = []
    det_centers = []
    bbox_list = []

    for i in range(len(masks_small)):
        m = masks_small[i]
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) * 255
        else:
            m = (m > 0).astype(np.uint8) * 255
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        bb = mask_to_bbox(m)
        if bb is None:
            x1, y1, x2, y2 = map(int, xyxy[i])
            bb = (max(0, min(w - 1, x1)), max(0, min(h - 1, y1)), max(0, min(w - 1, x2)), max(0, min(h - 1, y2)))
            cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
        else:
            cen = mask_centroid(m)
            if cen is None:
                x1, y1, x2, y2 = bb
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            else:
                cx, cy = cen
        masks.append(m)
        det_centers.append((cx, cy))
        bbox_list.append(bb)

    slot_of_det, part_no_of_det, reject_reason_of_det = assign_all_pieces(det_centers, anchors)

    save_tasks = []
    meta_rows = []
    det_infos = []

    for i in range(len(masks)):
        x1, y1, x2, y2 = map(int, bbox_list[i])
        cx, cy = det_centers[i]
        slot_name = slot_of_det.get(i, UNASSIGNED_SLOT_NAME)
        part_no = part_no_of_det.get(i, UNASSIGNED_PART_NO)
        reject_reason = reject_reason_of_det.get(i, None)
        is_primary = (part_no == 1) if part_no is not None else False
        if slot_name is None or part_no is None:
            status = reject_reason if reject_reason is not None else UNASSIGNED_STATUS
            out_path = None
            crop_filename = None
        else:
            out_path = build_output_paths(base_key, bed_name, slot_name, part_no)
            crop_filename = os.path.basename(out_path)
            status = 'ok'
            save_tasks.append({'img': img, 'mask': masks[i], 'out_path': out_path, 'bbox': (x1, y1, x2, y2)})

        meta_rows.append({
            'image_path': img_path, 'base_key': base_key, 'bed_name': bed_name,
            'slot_name': slot_name, 'part_no': part_no, 'crop_filename': crop_filename, 'crop_path': out_path,
            'conf': float(confs[i]) if i < len(confs) else None,
            'cls': int(clss[i]) if i < len(clss) else None,
            'cx': float(cx), 'cy': float(cy),
            'bbox_x1': int(x1), 'bbox_y1': int(y1), 'bbox_x2': int(x2), 'bbox_y2': int(y2),
            'bbox_w': int(x2 - x1 + 1), 'bbox_h': int(y2 - y1 + 1),
            'area_px': int((x2 - x1 + 1) * (y2 - y1 + 1)),
            'mask_pixels': int((masks[i] > 0).sum()),
            'is_primary': int(is_primary), 'n_det_in_image': int(len(masks)), 'status': status,
            'slot_guard_reason': reject_reason,
            **anchor_meta
        })
        det_infos.append({'bbox': (x1, y1, x2, y2), 'slot_name': slot_name, 'part_no': part_no})

    n_saved = 0
    n_fail = 0
    if len(save_tasks) > 0:
        with ThreadPoolExecutor(max_workers=N_SAVE_WORKERS) as ex:
            futures = [ex.submit(save_crop_task, t) for t in save_tasks]
            for fut in as_completed(futures):
                rr = fut.result()
                if rr['ok']:
                    n_saved += 1
                else:
                    n_fail += 1

    overlay_path = None
    if SAVE_OVERLAY:
        overlay_dir = os.path.join(OUT_ROOT, 'overlay', bed_name)
        safe_mkdir(overlay_dir)
        overlay_path = os.path.join(overlay_dir, f'{base_key}_overlay.jpg')
        overlay = draw_overlay(img, det_infos, anchors)
        cv2.imwrite(overlay_path, overlay)

    del img, results, r, masks_small, masks
    gc.collect()
    return {'meta_rows': meta_rows, 'n_det': int(len(bbox_list)), 'n_saved': int(n_saved), 'n_fail': int(n_fail), 'overlay_path': overlay_path, 'error': None}


def save_chunk_csv(rows, chunk_idx):
    if len(rows) == 0:
        return None
    chunk_dir = os.path.join(OUT_ROOT, 'csv_chunks')
    safe_mkdir(chunk_dir)
    out_csv = os.path.join(chunk_dir, f'slot_crop_meta_chunk_{chunk_idx:04d}.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    return out_csv


def merge_chunk_csvs(final_csv_path):
    chunk_dir = os.path.join(OUT_ROOT, 'csv_chunks')
    csvs = sorted(glob.glob(os.path.join(chunk_dir, 'slot_crop_meta_chunk_*.csv')))
    if len(csvs) == 0:
        return None
    dfs = [pd.read_csv(p) for p in csvs]
    pd.concat(dfs, axis=0, ignore_index=True).to_csv(final_csv_path, index=False, encoding='utf-8-sig')
    return final_csv_path


def main():
    print(f'[{now_hms()}] [START dynamic_y]')
    safe_mkdir(OUT_ROOT)
    safe_mkdir(os.path.join(OUT_ROOT, 'crops'))
    safe_mkdir(os.path.join(OUT_ROOT, 'overlay'))
    safe_mkdir(os.path.join(OUT_ROOT, 'csv_chunks'))

    lookup = load_csv_lookup(CSV_LOOKUP_PATH)
    print(f'[INFO] csv lookup rows: {len(lookup)}')

    img_paths = list_images_recursive(IMG_ROOT)
    print(f'[INFO] found images: {len(img_paths)}')
    if len(img_paths) == 0:
        raise RuntimeError('No input images found.')
    img_paths = img_paths[:10]
    print(f'[INFO] test images selected: {len(img_paths)}')

    model = YOLO(MODEL_PATH)
    total_imgs = len(img_paths)
    n_chunks = (total_imgs - 1) // CHUNK_SIZE + 1
    t0 = time.time()
    total_det = total_saved = total_fail = 0
    error_logs = []
    processed_img_count = 0

    for chunk_idx, start in enumerate(range(0, total_imgs, CHUNK_SIZE), start=1):
        chunk_paths = img_paths[start:start+CHUNK_SIZE]
        chunk_rows = []
        chunk_det = chunk_saved = chunk_fail = 0
        print(f'\n[{now_hms()}] [CHUNK {chunk_idx}/{n_chunks}] start | images={len(chunk_paths)}')
        pbar = tqdm(chunk_paths, total=len(chunk_paths), desc=f'chunk {chunk_idx}/{n_chunks}')
        for img_path in pbar:
            try:
                ret = process_one_image(img_path, model, lookup)
                chunk_rows.extend(ret['meta_rows'])
                chunk_det += ret['n_det']
                chunk_saved += ret['n_saved']
                chunk_fail += ret['n_fail']
                if ret['error'] is not None:
                    error_logs.append({'image_path': img_path, 'error': ret['error']})
            except Exception as e:
                err = traceback.format_exc()
                error_logs.append({'image_path': img_path, 'error': str(e), 'traceback': err})
            processed_img_count += 1
            elapsed = time.time() - t0
            frac = processed_img_count / max(1, total_imgs)
            eta = elapsed * (1 - frac) / max(frac, 1e-9)
            pbar.set_postfix({'done': f'{processed_img_count}/{total_imgs}', 'det': chunk_det, 'save': chunk_saved, 'eta_min': f'{eta/60:.1f}'})
        total_det += chunk_det
        total_saved += chunk_saved
        total_fail += chunk_fail
        out_chunk_csv = save_chunk_csv(chunk_rows, chunk_idx)
        print(f'[{now_hms()}] chunk csv saved: {out_chunk_csv}')
        elapsed = time.time() - t0
        frac = processed_img_count / max(1, total_imgs)
        eta = elapsed * (1 - frac) / max(frac, 1e-9)
        print(f'[{now_hms()}] [CHUNK {chunk_idx}/{n_chunks}] done | det={chunk_det} save={chunk_saved} fail={chunk_fail} | elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m')
        gc.collect()

    final_csv = os.path.join(OUT_ROOT, 'slot_crop_metadata_all.csv')
    merged = merge_chunk_csvs(final_csv)
    print(f'[{now_hms()}] merged csv: {merged}')
    if len(error_logs) > 0:
        err_path = os.path.join(OUT_ROOT, 'error_log.csv')
        pd.DataFrame(error_logs).to_csv(err_path, index=False, encoding='utf-8-sig')
        print(f'[{now_hms()}] error log saved: {err_path}')
    elapsed = time.time() - t0
    print('\n' + '='*70)
    print(f'[{now_hms()}] [DONE dynamic_y]')
    print(f'images      : {total_imgs}')
    print(f'detections  : {total_det}')
    print(f'saved crops : {total_saved}')
    print(f'save fails  : {total_fail}')
    print(f'elapsed     : {elapsed/60:.1f} min')
    print(f'OUT_ROOT    : {OUT_ROOT}')
    print('='*70)


if __name__ == '__main__':
    main()
