"""
Object-level building change from two building-probability maps.

Buildings are matched as connected components, not pixels: parallax and
misregistration between dates shift roofs by a few pixels, and a pixel XOR
turns every building edge into a false "new"/"demolished" sliver.
"""
import cv2
import numpy as np

# Label values in the returned mask (match BUILDING_TYPES mask_value)
LABEL_VALUES = {"new": 1, "demolished": 2, "existing": 3}

SUPPORT_PROB = 0.25   # the other date "supports" a pixel when its probability exceeds this
NO_SUPPORT = 0.10     # an object with less supported area than this is wholly new/demolished
SLIVER_KERNEL = 7     # opening size (px) that removes misregistration slivers from partial parts

_OPEN_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_OPEN_SLIVER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIVER_KERNEL, SLIVER_KERNEL))


def _components(binary, min_px):
    """(labels, [(label, x, y, w, h)]) for components of a uint8 mask with at least min_px pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [
        (i, stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_px
    ]
    return labels, comps


def _parts(part_mask, min_px):
    """Sliver-free connected parts (bool masks, same crop) of a partial object."""
    opened = cv2.morphologyEx(part_mask.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SLIVER)
    labels, comps = _components(opened, min_px)
    return [labels == i for i, *_ in comps]


def _emit(objects, label_mask, change_type, mask, x, y, p_same, p_other):
    """Append one object (mask is a bool crop at offset x, y) and paint it into label_mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(polygon) < 3:
        polygon = contour
    same = float(p_same[mask].mean())
    other = float(p_other[mask].mean())
    confidence = same * (1.0 - other) if change_type != "existing" else same * other
    label_mask[y:y + mask.shape[0], x:x + mask.shape[1]][mask] = LABEL_VALUES[change_type]
    objects.append({
        "type": change_type,
        "polygon": (polygon + np.array([x, y], dtype=np.int32)).astype(np.int32),
        "area_px": int(mask.sum()),
        "perimeter_px": float(cv2.arcLength(polygon, True)),
        "confidence_pct": round(100.0 * confidence, 1),
    })


def _tolerant_support(prob, tolerance_px):
    """Pixels within tolerance_px of the other date's building support (absorbs roof parallax)."""
    support = (prob > SUPPORT_PROB).astype(np.uint8)
    if tolerance_px >= 1:
        k = 2 * int(round(tolerance_px)) + 1
        support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return support.astype(bool)


def classify_objects(prob_t1, prob_t2, threshold=0.5, min_px=40, tolerance_px=0):
    """
    Returns (objects, label_mask). Each object: type ("new" | "demolished" | "existing"),
    polygon (N x 1 x 2 int32, full-image px), area_px, perimeter_px, confidence_pct.
    label_mask holds LABEL_VALUES. tolerance_px: how far a roof may move between
    dates (off-nadir parallax, misregistration) and still count as the same building.
    """
    h, w = prob_t2.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    objects = []
    support_t1 = _tolerant_support(prob_t1, tolerance_px)
    support_t2 = _tolerant_support(prob_t2, tolerance_px)

    # T2 objects -> new / existing (partially supported objects are split)
    b2 = cv2.morphologyEx((prob_t2 >= threshold).astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SMALL)
    labels, comps = _components(b2, min_px)
    for i, x, y, cw, ch in comps:
        obj = labels[y:y + ch, x:x + cw] == i
        s1 = support_t1[y:y + ch, x:x + cw]
        p1 = prob_t1[y:y + ch, x:x + cw]
        p2 = prob_t2[y:y + ch, x:x + cw]
        support = s1[obj].mean()
        if support < NO_SUPPORT:
            _emit(objects, label_mask, "new", obj, x, y, p2, p1)
            continue
        new_parts = _parts(obj & ~s1, min_px)
        if not new_parts:
            _emit(objects, label_mask, "existing", obj, x, y, p2, p1)
            continue
        for part in _parts(obj & s1, min_px):
            _emit(objects, label_mask, "existing", part, x, y, p2, p1)
        for part in new_parts:
            _emit(objects, label_mask, "new", part, x, y, p2, p1)

    # T1 objects -> demolished (existing ones were already emitted from the T2 side)
    b1 = cv2.morphologyEx((prob_t1 >= threshold).astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SMALL)
    labels, comps = _components(b1, min_px)
    for i, x, y, cw, ch in comps:
        obj = labels[y:y + ch, x:x + cw] == i
        s2 = support_t2[y:y + ch, x:x + cw]
        p1 = prob_t1[y:y + ch, x:x + cw]
        p2 = prob_t2[y:y + ch, x:x + cw]
        if s2[obj].mean() < NO_SUPPORT:
            _emit(objects, label_mask, "demolished", obj, x, y, p1, p2)
            continue
        for part in _parts(obj & ~s2, min_px):
            _emit(objects, label_mask, "demolished", part, x, y, p1, p2)

    return objects, label_mask
