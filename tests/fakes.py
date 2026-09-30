from types import SimpleNamespace

import numpy as np

from model.segmenter import ModelUnavailable, _MEAN, _STD


class FakeSession:
    """Stands in for an onnxruntime session: 'building' = bright pixels (mean RGB > 0.8)."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def run(self, _outputs, feeds):
        self.calls += 1
        x = feeds["image"][0].transpose(1, 2, 0) * _STD + _MEAN
        return [(x.mean(axis=2) > 0.8).astype(np.float32)[None, None]]


class FakeSegmenter:
    """BuildingSegmenter stand-in: bright pixels are buildings; records requested image sizes."""

    def __init__(self):
        self.sizes = []

    def count_tiles(self, h, w):
        return 1

    def predict(self, rgb, progress=None):
        self.sizes.append(rgb.shape[:2])
        if progress:
            progress(1, 1)
        return (rgb.mean(axis=2) > 200).astype(np.float32)


class OfflineSegmenter(FakeSegmenter):
    """The model download failed."""

    def predict(self, rgb, progress=None):
        raise ModelUnavailable("bağlantı yok")
