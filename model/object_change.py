"""
Object-level building change from two building-probability maps.

Buildings are matched as connected components, not pixels: parallax and
misregistration between dates shift roofs by a few pixels, and a pixel XOR
turns every building edge into a false "new"/"demolished" sliver.

A T2 building standing on ground that was already built in T1 is "existing"
only when its footprint still matches the T1 buildings it touches; otherwise
it was "rebuilt" (urban transformation: a block replacing several houses).
"""
import cv2
import numpy as np

# Label values in the returned mask (match BUILDING_TYPES mask_value)
LABEL_VALUES = {"new": 1, "demolished": 2, "existing": 3, "rebuilt": 4}

SUPPORT_PROB = 0.25   # the other date "supports" a pixel when its probability exceeds this
NO_SUPPORT = 0.10     # an object with less supported area than this is wholly new/demolished
SLIVER_KERNEL = 7     # opening size (px) that removes misregistration slivers from partial parts
# Footprint overlap (vs. the T1 buildings it touches) below which an "existing" T2 building counts as
# rebuilt. Measured on Wayback 2014 -> 2026: median overlap 0.89 for an unchanged suburb (1/211 flagged),
# 0.60-0.73 in transformation areas (Fikirtepe 58/201, Örnekköy 64/243 flagged).
REBUILT_IOU = 0.45

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
    if change_type == "existing":
        confidence = same * other
    elif change_type == "rebuilt":
        confidence = same  # both dates have a building here by definition
    else:
        confidence = same * (1.0 - other)
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


def _footprint_match(obj, x, y, t1_labels, t1_tolerant, pad):
    """(overlap, number of T1 buildings touched, share of those T1 buildings inside the object) for a T2 object
    (bool crop at x, y). The overlap's intersection uses the T1 mask dilated by the parallax tolerance, its union
    only the touching T1 buildings (not bbox neighbours)."""
    h, w = t1_labels.shape
    y0, x0 = max(0, y - pad), max(0, x - pad)
    y1, x1 = min(h, y + obj.shape[0] + pad), min(w, x + obj.shape[1] + pad)
    m = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    m[y - y0:y - y0 + obj.shape[0], x - x0:x - x0 + obj.shape[1]] = obj
    l1 = t1_labels[y0:y1, x0:x1]
    under = np.unique(l1[m])
    under = under[under > 0]
    if not len(under):
        return 0.0, 0, 0.0
    touching = np.isin(l1, under)
    overlap = float((m & t1_tolerant[y0:y1, x0:x1]).sum() / max(1, (m | touching).sum()))
    return overlap, len(under), float((touching & m).sum() / max(1, touching.sum()))


def classify_objects(prob_t1, prob_t2, threshold=0.5, min_px=40, tolerance_px=0):
    """
    Returns (objects, label_mask). Each object: type ("new" | "demolished" | "existing" | "rebuilt"),
    polygon (N x 1 x 2 int32, full-image px), area_px, perimeter_px, confidence_pct.
    label_mask holds LABEL_VALUES. tolerance_px: how far a roof may move between
    dates (off-nadir parallax, misregistration) and still count as the same building.
    """
    h, w = prob_t2.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    objects = []
    support_t1 = _tolerant_support(prob_t1, tolerance_px)
    support_t2 = _tolerant_support(prob_t2, tolerance_px)
    t1_raw = (prob_t1 >= threshold).astype(np.uint8)
    _, t1_labels = cv2.connectedComponents(t1_raw)
    k = 2 * int(round(tolerance_px)) + 1
    t1_tolerant = cv2.dilate(t1_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    pad = int(round(tolerance_px))

    def rebuilt(obj, x, y):
        """Low footprint overlap AND it replaced several T1 buildings or does not contain its T1 building
        (an extension keeps the old building inside: that stays existing + new part)."""
        overlap, n_t1, contained = _footprint_match(obj, x, y, t1_labels, t1_tolerant, pad)
        return overlap < REBUILT_IOU and (n_t1 >= 2 or contained < 0.7)

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
        if rebuilt(obj, x, y):
            _emit(objects, label_mask, "rebuilt", obj, x, y, p2, p1)
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


def estimate_parallax_px(prob_t1, prob_t2, max_shift_px, max_objects=300, min_matches=20, min_px=200, quality=0.75):
    """How far roofs move between the two dates, measured from the data (off-nadir parallax,
    misregistration). A fixed-seed random sample of T2 buildings (>= min_px) is template-matched
    against the T1 probability map within ±max_shift_px; for the reliable matches (buildings that
    exist on both dates) the 75th percentile of the shift length plus a 2 px margin is returned.
    The sample is random, not the largest buildings: parallax grows with height, and the largest
    footprints in a city are low halls (largest-first gave 4.5 m on the whole Başakşehir mosaic vs
    6.6 m on a crop of it; random sampling gives 11.7 vs 11.2 px). None when fewer than min_matches
    buildings match (small images): the caller then uses a small default."""
    radius = int(round(max_shift_px))
    b2 = (prob_t2 >= 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b2, connectivity=8)
    h, w = prob_t2.shape
    candidates = [
        i for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_px
        and stats[i, cv2.CC_STAT_LEFT] >= radius and stats[i, cv2.CC_STAT_TOP] >= radius
        and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] <= w - radius
        and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] <= h - radius
    ]
    if len(candidates) > max_objects:
        candidates = np.random.default_rng(0).choice(candidates, size=max_objects, replace=False)
    shifts = []
    for i in candidates:
        x, y, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        template = (labels[y:y + ch, x:x + cw] == i).astype(np.float32)
        search = prob_t1[y - radius:y + ch + radius, x - radius:x + cw + radius].astype(np.float32)
        response = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(response)
        if best > quality:
            shifts.append(np.hypot(bx - radius, by - radius))
    if len(shifts) < min_matches:
        return None
    return float(np.percentile(shifts, 75)) + 2.0
