import copy
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "tests", "fixtures", "samples")


def sample_path(scenario_id, name):
    """Absolute path of tests/fixtures/samples/<scenario_id>/<name>.png (name: A, B or label).
    LEVIR-CD / DSIFN pairs kept as test data only; the app no longer ships sample datasets."""
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


def upload_pair(client, sample_id):
    """Uploads a fixture pair through /api/upload and returns its JSON (path_A, path_B, ...)."""
    with open(sample_path(sample_id, "A"), "rb") as fa, open(sample_path(sample_id, "B"), "rb") as fb:
        res = client.post(
            "/api/upload",
            data={"image_t1": (fa, "A.png"), "image_t2": (fb, "B.png")},
            content_type="multipart/form-data",
        )
    assert res.status_code == 200
    return res.get_json()


def detect_classic(client, sample_id, **options):
    """Uploads a fixture pair and runs the classic engine on it; returns the response."""
    up = upload_pair(client, sample_id)
    return client.post("/api/detect", json={"scenario_id": "custom", "path_A": up["path_A"], "path_B": up["path_B"], **options})
