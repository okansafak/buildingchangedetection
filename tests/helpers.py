import copy
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "static", "samples")


def sample_path(scenario_id, name):
    """Absolute path of static/samples/<scenario_id>/<name>.png (name: A, B or label)."""
    return os.path.join(SAMPLES, scenario_id, f"{name}.png")


def normalize_result(result):
    """Drop timing and hash the large base64 overlays so a detection result can be snapshotted."""
    out = copy.deepcopy(result)
    out.pop("inference_time_sec", None)
    if "overlays" in out:
        out["overlays"] = {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in out["overlays"].items()
        }
    return out
