"""When did each change happen? Captures taken between T1 and T2 are segmented too, and inside every
changed building's area each capture is compared with T1 and with T2: it shows the "after" state when its
buildings match T2's clearly better than T1's, the "before" state in the opposite case, and is undecided
otherwise (an empty construction lot under a rebuilt block, or a blurred capture where the model finds
nothing). The change happened at the split of the capture sequence into before / after that contradicts
the fewest decided captures (the latest such split on a tie); the period runs from the last decided capture
before the split to the first one after it.

One rule serves every change type: a new building matches T2 once it stands, a demolished one once it is
gone, and a rebuilt block once its new footprint replaces the old houses."""
import cv2
import numpy as np

CHANGE_TYPES = ("new", "demolished", "rebuilt")
DECISION_MARGIN = 0.1  # IoU difference below which a capture matches T1 and T2 equally well


def _iou(a, b):
    # plain IoU on purpose: a shift-tolerant overlap (dilating by the parallax tolerance) also absorbs the
    # footprint difference of rebuilt blocks and leaves most of them undecided (measured on Fikirtepe)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def change_periods(objects, t1_mask, t2_mask, captures, dates, tolerance_px):
    """objects: classified objects ("type", "polygon") on the T2 grid. t1_mask / t2_mask: bool building masks.
    captures: [(building mask, invalid mask or None)] of the intermediate captures, oldest first.
    dates: labels of [T1, *captures, T2]. Returns {"from", "to"} per changed object, None for the others."""
    assert len(dates) == len(captures) + 2
    h, w = t2_mask.shape
    pad = max(1, int(round(tolerance_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    periods = []
    for obj in objects:
        if obj["type"] not in CHANGE_TYPES:
            periods.append(None)
            continue
        x, y, bw, bh = cv2.boundingRect(obj["polygon"])
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        region = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(region, [obj["polygon"] - [x0, y0]], -1, 1, -1)
        region = cv2.dilate(region, kernel).astype(bool)  # the roof may have moved between captures

        def inside(mask):
            return mask[y0:y1, x0:x1] & region

        before, after = inside(t1_mask), inside(t2_mask)
        states = [False]
        for mask, invalid in captures:
            if invalid is not None and invalid[y0:y1, x0:x1][region].any():
                states.append(None)  # no imagery over this building in that capture
                continue
            cap = inside(mask)
            diff = _iou(cap, after) - _iou(cap, before)
            states.append(None if abs(diff) < DECISION_MARGIN else bool(diff > 0))
        states.append(True)
        decided = [i for i, state in enumerate(states) if state is not None]
        # split after decided[k]: decided[:k+1] are "before", the rest "after"; T1 and T2 are always decided
        cost = [sum(states[i] for i in decided[:k + 1]) + sum(not states[i] for i in decided[k + 1:])
                for k in range(len(decided) - 1)]
        k = max(k for k in range(len(cost)) if cost[k] == min(cost))
        periods.append({"from": dates[decided[k]], "to": dates[decided[k + 1]]})
    return periods


def period_summary(objects, periods, dates):
    """Number of new / demolished / rebuilt buildings per interval between consecutive captures (oldest
    first, intervals without changes left out), and of those whose period spans several intervals
    because the captures in between were undecided."""
    intervals = {(a, b): {t: 0 for t in CHANGE_TYPES} for a, b in zip(dates, dates[1:])}
    uncertain = {t: 0 for t in CHANGE_TYPES}
    for obj, period in zip(objects, periods):
        if period is not None:
            intervals.get((period["from"], period["to"]), uncertain)[obj["type"]] += 1
    rows = [{"from": a, "to": b, **c} for (a, b), c in intervals.items() if any(c.values())]
    return {"periods": rows, "uncertain": uncertain}
