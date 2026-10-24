"""Feature calculation calculations used by the final cycle pipeline."""
from pathlib import Path as _Path
import sys as _sys
_ROOT = _Path(__file__).resolve().parents[1]
for _folder in (_ROOT, _ROOT / 'pipeline', _ROOT / 'preprocessing'):
    _sys.path.insert(0, str(_folder))
import settings as CYCLE
from runtime import resolved_spec, run_source, moved_tree, moved_path, yolo_device
import math
from typing import Dict, Tuple
import cv2
import numpy as np
cv2.setNumThreads(1)
FEATURES = ['roi_area_px', 'area_cm2', 'perimeter_px', 'circularity', 'solidity', 'curvature', 'aspect_ratio', 'bottom_flatness', 'radial_min', 'radial_max_min_ratio', 'brightness_mean', 'dark_ratio', 'grad_std', 'inner_ratio_30', 'inner_ratio_50', 'ridge_circularity', 'ridge_std_r', 'ridge_mean_r', 'angular_cv', 'inner_ratio_core', 'ridge_circularity_core', 'ridge_std_r_core', 'ridge_mean_r_core', 'angular_cv_core', 'upper_area_px', 'lower_area_px', 'upper_lower_area_ratio', 'upper_circularity', 'upper_edge_density', 'blur_score', 'front_height_cm', 'front_height_cm2', 'volume_proxy']

class FeatureExtractorAndQC:

    def __init__(self, blur_threshold: float=100.0):
        self.blur_threshold = blur_threshold

    @staticmethod
    def calc_blur_score(gray_img: np.ndarray) -> float:
        if gray_img is None or gray_img.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())

    def extract_30_features(self, crop_bgr: np.ndarray, mask: np.ndarray, mm_per_px: float=0.735, is_front: bool=False) -> Tuple[Dict[str, float], Dict[str, str]]:
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
