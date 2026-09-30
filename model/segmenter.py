"""
Building footprint segmentation with the ChangeStar ViT-B ONNX export
(geobase/changestar-building-segmentation-vitb). Large images are processed
as overlapping 1024 px tiles blended with a feathered weight.
"""
import os

import numpy as np

MODEL_REPO = "geobase/changestar-building-segmentation-vitb"
MODEL_FILE = "onnx/model_quantized.onnx"
MODEL_REVISION = "8a6d7676d2fc9ea787b8786f2f05a65989c2606c"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

TILE = 1024     # the exported ViT has fixed positional embeddings: tiles must be exactly 1024 px
OVERLAP = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelUnavailable(RuntimeError):
    """The ONNX model could not be downloaded or loaded."""


def tile_origins(length, tile=TILE, overlap=OVERLAP):
    """Tile start offsets covering [0, length); the last tile is aligned to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    starts.append(length - tile)
    return starts


def _feather(size, overlap):
    ramp = np.ones(size, dtype=np.float32)
    edge = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


class BuildingSegmenter:
    def __init__(self, session=None, model_dir=MODEL_DIR):
        self._session = session
        self.model_dir = model_dir

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, cache_dir=self.model_dir)
                self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise ModelUnavailable(str(e)) from e
        return self._session

    def count_tiles(self, h, w):
        return len(tile_origins(h)) * len(tile_origins(w))

    def predict(self, rgb, progress=None):
        """Building probability (H x W float32 in [0, 1]) for an H x W x 3 uint8 RGB image.
        progress(done_tiles, total_tiles) is called after every tile."""
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        h, w = rgb.shape[:2]
        accum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        feather = _feather(TILE, OVERLAP)
        ys, xs = tile_origins(h), tile_origins(w)
        total, done = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                tile = rgb[y:y + TILE, x:x + TILE].astype(np.float32) / 255.0
                th, tw = tile.shape[:2]
                if th < TILE or tw < TILE:
                    tile = np.pad(tile, ((0, TILE - th), (0, TILE - tw), (0, 0)), mode="reflect")
                batch = ((tile - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
                prob = np.asarray(session.run(None, {input_name: batch})[0]).reshape(TILE, TILE)[:th, :tw]
                accum[y:y + th, x:x + tw] += prob * feather[:th, :tw]
                weight[y:y + th, x:x + tw] += feather[:th, :tw]
                done += 1
                if progress:
                    progress(done, total)
        return accum / np.maximum(weight, 1e-6)
