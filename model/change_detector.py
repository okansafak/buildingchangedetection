import os
import base64
import numpy as np
import cv2
from PIL import Image

from model import geo

# Map location (Istanbul, Başakşehir) used for bounds when detect() is given none
DEFAULT_CENTER = (41.1070, 28.7900)

# Per-type labels, colors, change-mask value and overlay RGBA
BUILDING_TYPES = {
    "new": {
        "type_tr": "Yeni Eklenen Bina",
        "icon": "🟢",
        "color": "#10b981",
        "mask_value": 1,
        "overlay_rgba": [16, 185, 129, 210],
    },
    "demolished": {
        "type_tr": "Yıkılan / Kaldırılan Bina",
        "icon": "🔴",
        "color": "#ef4444",
        "mask_value": 2,
        "overlay_rgba": [239, 68, 68, 210],
    },
    "existing": {
        "type_tr": "Mevcut / Korunan Bina",
        "icon": "⚪",
        "color": "#64748b",
        "mask_value": 3,
        "overlay_rgba": [100, 116, 139, 140],
    },
}


class BuildingChangeDetector:
    """
    Building Detection and Temporal Change Analyzer.
    Identifies individual building footprints in T1 (past) and T2 (present),
    and classifies each structure into:
      - Yeni Eklenen Bina (New Building)
      - Yıkılan / Kaldırılan Bina (Demolished Building)
      - Mevcut / Korunan Bina (Existing / Unchanged Building)
    """
    def __init__(self):
        pass

    def _extract_building_candidates(self, img_rgb, min_area_px=20):
        """
        Extracts candidate building rooftop/structure contours.
        Uses edge detection and morphological operations.
        Returns rectangular approximations (minAreaRect) for cleaner building footprints.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        max_area_px = h * w * 0.15 # Max 15% of image area

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate and close to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        mag_norm = edges # for compatibility with existing code returning mag_norm

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px or area > max_area_px:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Ignore if contour touches the image boundaries
            if x <= 2 or y <= 2 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
                continue

            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
            if aspect_ratio > 4.0:
                continue

            # Get minimum area rectangle to make it look like a building footprint
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            approx_box = box.reshape((4, 1, 2))

            candidates.append({
                'cnt': approx_box,
                'area_px': area,
                'bbox': (x, y, bw, bh)
            })

        return candidates, mag_norm, gray

    def detect(self, img1, img2, ground_truth=None, threshold=0.45, min_area_m2=35.0, gsd=0.5, bounds=None):
        """
        Runs building detection and classifies individual buildings into
        New, Demolished, and Existing.
        """
        img1_rgb, img2_rgb = self._load_pair(img1, img2)
        h, w = img1_rgb.shape[:2]
        pixel_area_m2 = gsd * gsd
        min_pixels = max(4, int(min_area_m2 / pixel_area_m2))

        # 1. Structural difference (optionally guided by ground truth)
        diff_map, g1, g2 = self._difference_map(img1_rgb, img2_rgb, ground_truth)

        # 2. Candidate buildings in T1 and T2
        _, mag1, _ = self._extract_building_candidates(img1_rgb, min_area_px=min_pixels)
        b_t2, mag2, _ = self._extract_building_candidates(img2_rgb, min_area_px=min_pixels)

        # 3. Regions of significant change
        change_contours = self._changed_contours(diff_map, threshold)

        if bounds is None:
            bounds = geo.bounds_from_center(DEFAULT_CENTER[0], DEFAULT_CENTER[1], w, h, gsd)

        # 4. Building-level classification; the mask holds BUILDING_TYPES mask values
        change_type_mask = np.zeros((h, w), dtype=np.uint8)
        detections = self._changed_buildings(
            change_contours, min_pixels, gsd, diff_map, mag1, mag2, g1, g2, change_type_mask
        )
        detections += self._existing_buildings(b_t2, gsd, change_type_mask)

        buildings_list = []
        features = []
        counts = {t: 0 for t in BUILDING_TYPES}
        areas = {t: 0.0 for t in BUILDING_TYPES}
        for b_idx, d in enumerate(detections, start=1):
            b_info, feature = self._build_record(b_idx, d, w, h, bounds)
            buildings_list.append(b_info)
            features.append(feature)
            counts[d["type"]] += 1
            areas[d["type"]] += d["area_m2"]

        changed_m2 = areas["new"] + areas["demolished"]
        return {
            "success": True,
            "image_size": [w, h],
            "bounds": bounds,
            "stats": {
                "total_buildings": len(buildings_list),
                "new_buildings_count": counts["new"],
                "demolished_count": counts["demolished"],
                "existing_count": counts["existing"],
                "total_changed_m2": round(changed_m2, 1),
                "total_changed_hectares": round(changed_m2 / 10000.0, 3),
                "new_buildings_m2": round(areas["new"], 1),
                "demolished_m2": round(areas["demolished"], 1),
                "existing_m2": round(areas["existing"], 1),
                "patch_dimensions": f"{w}x{h} px ({int(w*gsd)}m x {int(h*gsd)}m)",
                "gsd": gsd
            },
            "buildings": buildings_list,
            "geojson": {
                "type": "FeatureCollection",
                "features": features
            },
            # 5. Overlays (Base64)
            "overlays": self._render_overlays(change_type_mask, diff_map, img1_rgb, img2_rgb)
        }

    def _load_pair(self, img1, img2):
        """Opens paths/PIL images as RGB arrays and resizes both to the larger size."""
        if isinstance(img1, str):
            img1 = Image.open(img1)
        if isinstance(img2, str):
            img2 = Image.open(img2)

        img1_rgb = np.array(img1.convert('RGB'))
        img2_rgb = np.array(img2.convert('RGB'))

        # Ensure identical sizes
        h1, w1 = img1_rgb.shape[:2]
        h2, w2 = img2_rgb.shape[:2]
        target_size = (max(w1, w2), max(h1, h2))
        if (w1, h1) != target_size:
            img1_rgb = cv2.resize(img1_rgb, target_size)
        if (w2, h2) != target_size:
            img2_rgb = cv2.resize(img2_rgb, target_size)
        return img1_rgb, img2_rgb

    def _difference_map(self, img1_rgb, img2_rgb, ground_truth):
        """Returns (diff_map in [0, 1], gray T1, gray T2); a ground-truth mask dominates the map when given."""
        h, w = img1_rgb.shape[:2]
        g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
        b1 = cv2.GaussianBlur(g1, (5, 5), 1.0)
        b2 = cv2.GaussianBlur(g2, (5, 5), 1.0)
        diff_intensity = cv2.absdiff(b1, b2).astype(np.float32) / 255.0

        lab1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        color_diff = np.linalg.norm(lab1 - lab2, axis=2) / 255.0

        diff_map = np.clip(0.6 * diff_intensity + 0.4 * color_diff, 0.0, 1.0)

        # If ground truth mask provided, incorporate it
        gt_mask = None
        if ground_truth is not None:
            if isinstance(ground_truth, str) and os.path.exists(ground_truth):
                gt_img = Image.open(ground_truth).convert('L')
                gt_mask = np.array(gt_img.resize((w, h), Image.NEAREST))
            elif isinstance(ground_truth, np.ndarray):
                gt_mask = cv2.resize(ground_truth, (w, h), interpolation=cv2.INTER_NEAREST)
            if gt_mask is not None:
                gt_bin = (gt_mask > 127).astype(np.uint8) * 255
                diff_map = 0.8 * (gt_bin.astype(np.float32) / 255.0) + 0.2 * diff_map
                diff_map = np.clip(diff_map, 0.0, 1.0)

        return diff_map, g1, g2

    def _changed_contours(self, diff_map, threshold):
        change_binary = (diff_map >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        change_cleaned = cv2.morphologyEx(change_binary, cv2.MORPH_CLOSE, kernel)
        change_cleaned = cv2.morphologyEx(change_cleaned, cv2.MORPH_OPEN, kernel)

        change_contours, _ = cv2.findContours(change_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return change_contours

    def _classify_change(self, c_mask, mag1, mag2, g1, g2):
        """'new' if edges and brightness held up or grew from T1 to T2, else 'demolished'."""
        edge_t1 = cv2.mean(mag1, mask=c_mask)[0]
        edge_t2 = cv2.mean(mag2, mask=c_mask)[0]
        val_t1 = cv2.mean(g1, mask=c_mask)[0]
        val_t2 = cv2.mean(g2, mask=c_mask)[0]
        if edge_t2 >= edge_t1 * 0.9 and (val_t2 >= val_t1 * 0.85):
            return "new"
        return "demolished"

    def _changed_buildings(self, change_contours, min_pixels, gsd, diff_map, mag1, mag2, g1, g2, change_type_mask):
        """Detections for changed regions; marks each one in change_type_mask."""
        pixel_area_m2 = gsd * gsd
        h, w = change_type_mask.shape
        detections = []
        for cnt in change_contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_pixels:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)

            b_type = self._classify_change(c_mask, mag1, mag2, g1, g2)
            confidence = float(cv2.mean(diff_map, mask=c_mask)[0])
            change_type_mask[c_mask > 0] = BUILDING_TYPES[b_type]["mask_value"]

            detections.append({
                "type": b_type,
                "polygon": approx,
                "moment_contour": cnt,
                "area_m2": float(area_px * pixel_area_m2),
                "perimeter_m": float(cv2.arcLength(approx, True) * gsd),
                "confidence_pct": round(confidence * 100, 1),
            })
        return detections

    def _existing_buildings(self, b_t2, gsd, change_type_mask):
        """Detections for T2 candidates that don't overlap a change; marks each one in change_type_mask."""
        pixel_area_m2 = gsd * gsd
        h, w = change_type_mask.shape
        detections = []
        for candidate in b_t2[:25]: # limit for clean display
            cnt = candidate['cnt']
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)

            # Check overlap with changed regions
            overlap = cv2.bitwise_and(change_type_mask, change_type_mask, mask=c_mask)
            if np.count_nonzero(overlap) > (candidate['area_px'] * 0.3):
                continue # already part of change

            change_type_mask[c_mask > 0] = BUILDING_TYPES["existing"]["mask_value"]
            detections.append({
                "type": "existing",
                "polygon": cnt,
                "moment_contour": cnt,
                "area_m2": float(candidate['area_px'] * pixel_area_m2),
                "perimeter_m": float(cv2.arcLength(cnt, True) * gsd),
                "confidence_pct": 95.0,
            })
        return detections

    def _build_record(self, b_idx, detection, w, h, bounds):
        """Building info dict and matching GeoJSON feature for one detection."""
        building_type = BUILDING_TYPES[detection["type"]]
        polygon = detection["polygon"]

        M = cv2.moments(detection["moment_contour"])
        cx_px = M["m10"] / (M["m00"] + 1e-5)
        cy_px = M["m01"] / (M["m00"] + 1e-5)
        c_lon, c_lat = geo.pixel_to_lonlat(cx_px, cy_px, w, h, bounds)

        px_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in polygon]
        if px_coords and px_coords[0] != px_coords[-1]:
            px_coords.append(px_coords[0])

        b_info = {
            "id": b_idx,
            "type": detection["type"],
            "type_tr": building_type["type_tr"],
            "icon": building_type["icon"],
            "color": building_type["color"],
            "area_m2": round(detection["area_m2"], 1),
            "perimeter_m": round(detection["perimeter_m"], 1),
            "confidence_pct": detection["confidence_pct"],
            "centroid": [c_lat, c_lon],
            "centroid_px": [round(float(cx_px), 1), round(float(cy_px), 1)],
            "px_coords": px_coords
        }
        feature = {
            "type": "Feature",
            "id": b_idx,
            "properties": b_info,
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo.ring_to_lonlat(polygon, w, h, bounds)]
            }
        }
        return b_info, feature

    def _render_overlays(self, change_type_mask, diff_map, img1_rgb, img2_rgb):
        h, w = change_type_mask.shape
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for building_type in BUILDING_TYPES.values():
            overlay_rgba[change_type_mask == building_type["mask_value"]] = building_type["overlay_rgba"]

        heatmap_uint8 = (diff_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[:, :, 3] = (diff_map * 190).astype(np.uint8)

        return {
            # OpenCV encodes BGR(A): convert, or "demolished" red is written as blue
            "mask_png_base64": self._mat_to_base64(cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA)),
            "heatmap_png_base64": self._mat_to_base64(heatmap_rgba),
            "t1_png_base64": self._mat_to_base64(cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)),
            "t2_png_base64": self._mat_to_base64(cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
        }

    def _mat_to_base64(self, mat):
        success, encoded_img = cv2.imencode('.png', mat)
        if not success:
            return ""
        return f"data:image/png;base64,{base64.b64encode(encoded_img).decode('utf-8')}"
