"""
Deep-learning building change detection: ChangeStar building masks for T1 and
T2 (model/segmenter.py), matched as objects (model/object_change.py).

Returns the BuildingChangeDetector.detect() result contract (stats, buildings,
geojson, overlays, image_size, bounds) plus engine / analysis_mode / georef /
metrics. Without a georeference, coordinates stay in pixels: bounds, georef and
every centroid are None and the GeoJSON is marked "georeferenced": false.
"""
import base64
import os

import cv2
import numpy as np
from PIL import Image

from model.change_detector import BUILDING_TYPES
from model.object_change import classify_objects
from model.segmenter import BuildingSegmenter

Image.MAX_IMAGE_PIXELS = 250_000_000  # 8192 x 4468 aerial mosaics are legitimate input

ASSUMED_GSD = 0.5            # m/px, only to size pixel thresholds when the GSD is unknown; never reported
MODEL_GSD = 0.5              # the segmenter works best near this resolution
UPSAMPLE_MIN_GSD = 0.75      # only clearly coarse imagery (e.g. Wayback zoom 17, ~0.9 m) is upsampled toward MODEL_GSD
PARALLAX_TOLERANCE_M = 8.0   # a roof may move this far between dates and still be the same building
FAST_MIN_SIDE = 2048         # "Hızlı" halves only images larger than this ...
FAST_MAX_GSD = 1.2           # ... and only while the halved GSD stays at or below this (2 m/px was unusable)
ORDER = {"new": 0, "demolished": 1, "existing": 2}


def analysis_scale(w, h, gsd, mode):
    """Resize factor for the pair. "fast" halves large images unless that would make them too
    coarse, and never enlarges a large image; otherwise clearly coarse imagery (e.g. Wayback
    zoom 17) is upsampled up to 2x toward MODEL_GSD, and everything else stays native."""
    coarse = gsd is not None and gsd >= UPSAMPLE_MIN_GSD
    if mode == "fast" and max(w, h) > FAST_MIN_SIDE:
        too_coarse_to_halve = coarse or (gsd is not None and gsd * 2 > FAST_MAX_GSD)
        return 1.0 if too_coarse_to_halve else 0.5
    if coarse:
        return min(2.0, gsd / MODEL_GSD)
    return 1.0


def load_rgb(image):
    """H x W x 3 uint8 RGB from a path or PIL image; TIFFs PIL cannot read go through tifffile."""
    if not isinstance(image, str):
        return np.array(image.convert("RGB"))
    try:
        with Image.open(image) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        if not image.lower().endswith((".tif", ".tiff")):
            raise
    import tifffile
    arr = tifffile.imread(image)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    arr = arr[:, :, :3].astype(np.float32)
    if arr.max() > 255:
        lo, hi = np.percentile(arr, (2, 98))
        arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def change_metrics(label_mask, ground_truth):
    """Pixel precision / recall / F1 / IoU of new+demolished against a change label (path or array)."""
    h, w = label_mask.shape
    if isinstance(ground_truth, str):
        with Image.open(ground_truth) as im:
            ground_truth = np.array(im.convert("L"))
    gt = cv2.resize(ground_truth.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 127
    pred = (label_mask == BUILDING_TYPES["new"]["mask_value"]) | (label_mask == BUILDING_TYPES["demolished"]["mask_value"])
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "iou": round(tp / (tp + fp + fn), 3)}


def _record(b_idx, obj, georef, gsd):
    """Building info dict and GeoJSON feature for one classified object."""
    building_type = BUILDING_TYPES[obj["type"]]
    polygon = obj["polygon"]
    M = cv2.moments(polygon)
    cx = M["m10"] / (M["m00"] + 1e-5)
    cy = M["m01"] / (M["m00"] + 1e-5)
    px_coords = [[int(p[0][0]), int(p[0][1])] for p in polygon]
    px_coords.append(px_coords[0])

    if georef is not None:
        lon, lat = georef.to_lonlat([p[0] for p in px_coords] + [cx], [p[1] for p in px_coords] + [cy])
        ring = [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon[:-1], lat[:-1])]
        centroid = [round(float(lat[-1]), 7), round(float(lon[-1]), 7)]
    else:
        ring = px_coords
        centroid = None

    b_info = {
        "id": b_idx,
        "type": obj["type"],
        "type_tr": building_type["type_tr"],
        "icon": building_type["icon"],
        "color": building_type["color"],
        "area_m2": round(obj["area_px"] * gsd * gsd, 1) if gsd else None,
        "area_px": obj["area_px"],
        "perimeter_m": round(obj["perimeter_px"] * gsd, 1) if gsd else None,
        "confidence_pct": obj["confidence_pct"],
        "centroid": centroid,
        "centroid_px": [round(float(cx), 1), round(float(cy), 1)],
        "px_coords": px_coords,
    }
    feature = {
        "type": "Feature",
        "id": b_idx,
        "properties": b_info,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    return b_info, feature


def _encode(ext, mat, out_dir, url_prefix, name):
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, mat, params)
    if not ok:
        return ""
    if out_dir:
        with open(os.path.join(out_dir, name + ext), "wb") as f:
            f.write(buf.tobytes())
        return f"{url_prefix}/{name}{ext}"
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('utf-8')}"


def render_overlays(label_mask, prob_t1, prob_t2, t1_rgb, t2_rgb, out_dir=None, url_prefix=""):
    """Overlay images keyed like the classic contract. With out_dir the images are written
    there and the values are URLs (url_prefix/<name>); otherwise they are data URIs."""
    h, w = label_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for building_type in BUILDING_TYPES.values():
        rgba[label_mask == building_type["mask_value"]] = building_type["overlay_rgba"]
    change = np.maximum(prob_t2 * (1.0 - prob_t1), prob_t1 * (1.0 - prob_t2))
    heat = cv2.cvtColor(cv2.applyColorMap((change * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2BGRA)
    heat[:, :, 3] = (change * 190).astype(np.uint8)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return {
        # OpenCV writes BGR(A): convert, or "demolished" red comes out blue
        "mask_png_base64": _encode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), out_dir, url_prefix, "mask"),
        "heatmap_png_base64": _encode(".png", heat, out_dir, url_prefix, "heatmap"),
        "t1_png_base64": _encode(".jpg", cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2BGR), out_dir, url_prefix, "t1"),
        "t2_png_base64": _encode(".jpg", cv2.cvtColor(t2_rgb, cv2.COLOR_RGB2BGR), out_dir, url_prefix, "t2"),
    }


class MLChangeDetector:
    def __init__(self, segmenter=None):
        self.segmenter = segmenter or BuildingSegmenter()

    def detect(self, img1, img2, georef=None, gsd=None, min_area_m2=30.0, threshold=0.5,
               mode="fast", ground_truth=None, progress=None, out_dir=None, url_prefix=""):
        """
        img1/img2: paths or PIL images (T1 is resized onto T2's pixel grid).
        georef: GeoRef of T2's grid or None. gsd: m/px override (manual entry).
        mode: "fast" | "deep". progress(fraction 0..1, Turkish stage text).
        """
        report = progress or (lambda fraction, stage: None)
        report(0.01, "Görüntüler okunuyor")
        t1 = load_rgb(img1)
        t2 = load_rgb(img2)
        h, w = t2.shape[:2]
        if t1.shape[:2] != (h, w):
            t1 = cv2.resize(t1, (w, h), interpolation=cv2.INTER_AREA)
        if gsd is None and georef is not None:
            gsd = georef.gsd_m(w, h)

        scale = analysis_scale(w, h, gsd, mode)
        if scale != 1.0:
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            t1 = cv2.resize(t1, (nw, nh), interpolation=interp)
            t2 = cv2.resize(t2, (nw, nh), interpolation=interp)
            if georef is not None:
                georef = georef.resampled(w, h, nw, nh)
            if gsd is not None:
                gsd = gsd / scale
            w, h = nw, nh
        px_gsd = gsd if gsd is not None else ASSUMED_GSD / scale

        report(0.03, "Bina modeli hazırlanıyor")
        p1 = self.segmenter.predict(t1, lambda d, n: report(0.05 + 0.45 * d / n, f"T1 binaları bulunuyor ({d}/{n} karo)"))
        p2 = self.segmenter.predict(t2, lambda d, n: report(0.50 + 0.45 * d / n, f"T2 binaları bulunuyor ({d}/{n} karo)"))

        report(0.96, "Değişimler sınıflandırılıyor")
        min_px = max(12, int(min_area_m2 / px_gsd ** 2))
        objects, label_mask = classify_objects(p1, p2, threshold=threshold, min_px=min_px,
                                               tolerance_px=PARALLAX_TOLERANCE_M / px_gsd)
        objects.sort(key=lambda o: (ORDER[o["type"]], -o["area_px"]))

        buildings, features = [], []
        counts = {t: 0 for t in BUILDING_TYPES}
        areas_px = {t: 0 for t in BUILDING_TYPES}
        for b_idx, obj in enumerate(objects, start=1):
            b_info, feature = _record(b_idx, obj, georef, gsd)
            buildings.append(b_info)
            features.append(feature)
            counts[obj["type"]] += 1
            areas_px[obj["type"]] += obj["area_px"]

        def m2(px):
            return round(px * gsd * gsd, 1) if gsd else None

        changed_px = areas_px["new"] + areas_px["demolished"]
        report(0.98, "Sonuçlar hazırlanıyor")
        return {
            "success": True,
            "engine": "ml",
            "analysis_mode": mode,
            "image_size": [w, h],
            "bounds": georef.bounds_wgs84(w, h) if georef is not None else None,
            "georef": georef.summary(w, h) if georef is not None else None,
            "stats": {
                "total_buildings": len(buildings),
                "new_buildings_count": counts["new"],
                "demolished_count": counts["demolished"],
                "existing_count": counts["existing"],
                "total_changed_m2": m2(changed_px),
                "total_changed_hectares": round(changed_px * gsd * gsd / 10000.0, 3) if gsd else None,
                "new_buildings_m2": m2(areas_px["new"]),
                "demolished_m2": m2(areas_px["demolished"]),
                "existing_m2": m2(areas_px["existing"]),
                "total_changed_px": changed_px,
                "patch_dimensions": f"{w}x{h} px ({int(w * gsd)}m x {int(h * gsd)}m)" if gsd else f"{w}x{h} px (ölçeksiz)",
                "gsd": round(gsd, 3) if gsd else None,
            },
            "buildings": buildings,
            "geojson": {"type": "FeatureCollection", "georeferenced": georef is not None, "features": features},
            "overlays": render_overlays(label_mask, p1, p2, t1, t2, out_dir, url_prefix),
            "metrics": change_metrics(label_mask, ground_truth) if ground_truth is not None else None,
        }
