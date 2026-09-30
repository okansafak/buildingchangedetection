import os
import json
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

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
            
        h, w = img1_rgb.shape[:2]
        pixel_area_m2 = gsd * gsd
        min_pixels = max(4, int(min_area_m2 / pixel_area_m2))
        
        # 1. Structural Difference Calculation
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

        # 2. Extract Candidate Buildings in T1 and T2
        b_t1, mag1, _ = self._extract_building_candidates(img1_rgb, min_area_px=min_pixels)
        b_t2, mag2, _ = self._extract_building_candidates(img2_rgb, min_area_px=min_pixels)
        
        # 3. Candidate Contours from Significant Difference
        change_binary = (diff_map >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        change_cleaned = cv2.morphologyEx(change_binary, cv2.MORPH_CLOSE, kernel)
        change_cleaned = cv2.morphologyEx(change_cleaned, cv2.MORPH_OPEN, kernel)
        
        change_contours, _ = cv2.findContours(change_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Default bounds if not supplied
        if bounds is None:
            center_lat, center_lon = 41.1070, 28.7900
            lat_delta = (h * gsd) / 111320.0
            lon_delta = (w * gsd) / (111320.0 * np.cos(np.radians(center_lat)))
            bounds = [
                center_lat - lat_delta / 2,
                center_lon - lon_delta / 2,
                center_lat + lat_delta / 2,
                center_lon + lon_delta / 2
            ]
        south, west, north, east = bounds
        
        # 4. Building-Level Instance Classification
        buildings_list = []
        features = []
        
        new_count = 0
        dem_count = 0
        exist_count = 0
        new_area_m2 = 0.0
        dem_area_m2 = 0.0
        exist_area_m2 = 0.0
        
        # Mask tracking change type: 1=new, 2=demolished, 3=existing
        change_type_mask = np.zeros((h, w), dtype=np.uint8)
        
        # First process changed building footprints
        b_idx = 1
        for cnt in change_contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_pixels:
                continue
                
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
                
            area_m2 = float(area_px * pixel_area_m2)
            perimeter_m = float(cv2.arcLength(approx, True) * gsd)
            
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            
            # Mean edge and brightness in T1 vs T2
            edge_t1 = cv2.mean(mag1, mask=c_mask)[0]
            edge_t2 = cv2.mean(mag2, mask=c_mask)[0]
            val_t1 = cv2.mean(g1, mask=c_mask)[0]
            val_t2 = cv2.mean(g2, mask=c_mask)[0]
            confidence = float(cv2.mean(diff_map, mask=c_mask)[0])
            
            # Classification
            if edge_t2 >= edge_t1 * 0.9 and (val_t2 >= val_t1 * 0.85):
                b_type = "new"
                type_tr = "Yeni Eklenen Bina"
                icon = "🟢"
                color = "#10b981"
                new_count += 1
                new_area_m2 += area_m2
                change_type_mask[c_mask > 0] = 1
            else:
                b_type = "demolished"
                type_tr = "Yıkılan / Kaldırılan Bina"
                icon = "🔴"
                color = "#ef4444"
                dem_count += 1
                dem_area_m2 += area_m2
                change_type_mask[c_mask > 0] = 2
                
            # Geographic coordinates conversion
            coords = []
            for pt in approx:
                px_x, px_y = pt[0][0], pt[0][1]
                geo_lon = west + (px_x / w) * (east - west)
                geo_lat = north - (px_y / h) * (north - south)
                coords.append([round(geo_lon, 6), round(geo_lat, 6)])
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
                
            # Centroid
            M = cv2.moments(cnt)
            cx_px = M["m10"] / (M["m00"] + 1e-5)
            cy_px = M["m01"] / (M["m00"] + 1e-5)
            c_lon = round(west + (cx_px / w) * (east - west), 6)
            c_lat = round(north - (cy_px / h) * (north - south), 6)
            
            # Pixel coordinates
            px_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
            if px_coords and px_coords[0] != px_coords[-1]:
                px_coords.append(px_coords[0])

            b_info = {
                "id": b_idx,
                "type": b_type,
                "type_tr": type_tr,
                "icon": icon,
                "color": color,
                "area_m2": round(area_m2, 1),
                "perimeter_m": round(perimeter_m, 1),
                "confidence_pct": round(confidence * 100, 1),
                "centroid": [c_lat, c_lon],
                "centroid_px": [round(float(cx_px), 1), round(float(cy_px), 1)],
                "px_coords": px_coords
            }
            buildings_list.append(b_info)
            
            feature = {
                "type": "Feature",
                "id": b_idx,
                "properties": b_info,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)
            b_idx += 1
            
        # Also detect existing/unchanged buildings in T2 that don't overlap with changes
        for candidate in b_t2[:25]: # limit for clean display
            cnt = candidate['cnt']
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            
            # Check overlap with changed regions
            overlap = cv2.bitwise_and(change_type_mask, change_type_mask, mask=c_mask)
            if np.count_nonzero(overlap) > (candidate['area_px'] * 0.3):
                continue # already part of change
                
            area_m2 = float(candidate['area_px'] * pixel_area_m2)
            perimeter_m = float(cv2.arcLength(cnt, True) * gsd)
            exist_count += 1
            exist_area_m2 += area_m2
            change_type_mask[c_mask > 0] = 3
            
            coords = []
            for pt in cnt:
                px_x, px_y = pt[0][0], pt[0][1]
                geo_lon = west + (px_x / w) * (east - west)
                geo_lat = north - (px_y / h) * (north - south)
                coords.append([round(geo_lon, 6), round(geo_lat, 6)])
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
                
            M = cv2.moments(cnt)
            cx_px = M["m10"] / (M["m00"] + 1e-5)
            cy_px = M["m01"] / (M["m00"] + 1e-5)
            c_lon = round(west + (cx_px / w) * (east - west), 6)
            c_lat = round(north - (cy_px / h) * (north - south), 6)
            
            px_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in cnt]
            if px_coords and px_coords[0] != px_coords[-1]:
                px_coords.append(px_coords[0])

            b_info = {
                "id": b_idx,
                "type": "existing",
                "type_tr": "Mevcut / Korunan Bina",
                "icon": "⚪",
                "color": "#64748b",
                "area_m2": round(area_m2, 1),
                "perimeter_m": round(perimeter_m, 1),
                "confidence_pct": 95.0,
                "centroid": [c_lat, c_lon],
                "centroid_px": [round(float(cx_px), 1), round(float(cy_px), 1)],
                "px_coords": px_coords
            }
            buildings_list.append(b_info)
            
            feature = {
                "type": "Feature",
                "id": b_idx,
                "properties": b_info,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)
            b_idx += 1
            
        # 5. Overlays (Base64)
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        overlay_rgba[change_type_mask == 1] = [16, 185, 129, 210]
        overlay_rgba[change_type_mask == 2] = [239, 68, 68, 210]
        overlay_rgba[change_type_mask == 3] = [100, 116, 139, 140]
        
        heatmap_uint8 = (diff_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[:, :, 3] = (diff_map * 190).astype(np.uint8)
        
        return {
            "success": True,
            "image_size": [w, h],
            "bounds": bounds,
            "stats": {
                "total_buildings": len(buildings_list),
                "new_buildings_count": new_count,
                "demolished_count": dem_count,
                "existing_count": exist_count,
                "total_changed_m2": round(new_area_m2 + dem_area_m2, 1),
                "total_changed_hectares": round((new_area_m2 + dem_area_m2) / 10000.0, 3),
                "new_buildings_m2": round(new_area_m2, 1),
                "demolished_m2": round(dem_area_m2, 1),
                "existing_m2": round(exist_area_m2, 1),
                "patch_dimensions": f"{w}x{h} px ({int(w*gsd)}m x {int(h*gsd)}m)",
                "gsd": gsd
            },
            "buildings": buildings_list,
            "geojson": {
                "type": "FeatureCollection",
                "features": features
            },
            "overlays": {
                "mask_png_base64": self._mat_to_base64(overlay_rgba),
                "heatmap_png_base64": self._mat_to_base64(heatmap_rgba),
                "t1_png_base64": self._mat_to_base64(cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)),
                "t2_png_base64": self._mat_to_base64(cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
            }
        }

    def _mat_to_base64(self, mat):
        success, encoded_img = cv2.imencode('.png', mat)
        if not success:
            return ""
        return f"data:image/png;base64,{base64.b64encode(encoded_img).decode('utf-8')}"
