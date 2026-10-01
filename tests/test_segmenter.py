import numpy as np
import pytest

from fakes import FakeSession
from model.segmenter import TILE, BuildingSegmenter, tile_origins


def test_tile_origins_cover_and_align_to_end():
    assert tile_origins(500) == [0]
    assert tile_origins(1024) == [0]
    starts = tile_origins(2500)
    assert starts[0] == 0 and starts[-1] == 2500 - TILE
    assert all(b - a <= TILE for a, b in zip(starts, starts[1:]))


def test_predict_small_and_large_images():
    for h, w in [(256, 256), (1100, 2500)]:
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[50:120, 60:200] = 255
        rgb[h - 90:h - 20, w - 150:w - 10] = 255
        session = FakeSession()
        calls = []
        prob = BuildingSegmenter(session=session).predict(rgb, progress=lambda d, t: calls.append((d, t)))
        assert prob.shape == (h, w)
        assert ((prob > 0.5) == (rgb.mean(axis=2) > 200)).all()
        n = BuildingSegmenter(session=session).count_tiles(h, w)
        assert calls[-1] == (n, n) and session.calls == n


def _bright_square():
    rgb = np.zeros((300, 300, 3), np.uint8)
    rgb[50:150, 60:200] = 255
    return rgb


def test_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("ATLAS_DEVICE", raising=False)
    seg = BuildingSegmenter()
    seg._open_cuda_session = lambda: FakeSession()
    seg._open_cpu_session = lambda: pytest.fail("the CPU model must not be loaded when CUDA works")
    prob = seg.predict(_bright_square())
    assert seg.device == "cuda" and (prob > 0.5).sum() == 100 * 140


def test_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.delenv("ATLAS_DEVICE", raising=False)
    seg = BuildingSegmenter()
    seg._open_cuda_session = lambda: None
    seg._open_cpu_session = lambda: FakeSession()
    seg.predict(_bright_square())
    assert seg.device == "cpu"


def test_falls_back_to_cpu_when_cuda_fails_while_running(monkeypatch):
    """A 4 GB laptop GPU can run out of memory mid-image: the image is redone on the CPU."""
    monkeypatch.delenv("ATLAS_DEVICE", raising=False)

    class OutOfMemory(FakeSession):
        def run(self, _outputs, feeds):
            raise RuntimeError("BFCArena: available memory is smaller than requested bytes")

    seg = BuildingSegmenter()
    seg._open_cuda_session = lambda: OutOfMemory()
    seg._open_cpu_session = lambda: FakeSession()
    prob = seg.predict(_bright_square())
    assert seg.device == "cpu" and (prob > 0.5).sum() == 100 * 140
    seg.predict(_bright_square())  # stays on the CPU afterwards
    assert seg.device == "cpu"


def test_atlas_device_cpu_skips_cuda(monkeypatch):
    monkeypatch.setenv("ATLAS_DEVICE", "cpu")
    seg = BuildingSegmenter()
    seg._open_cuda_session = lambda: pytest.fail("ATLAS_DEVICE=cpu must not try CUDA")
    seg._open_cpu_session = lambda: FakeSession()
    seg.predict(_bright_square())
    assert seg.device == "cpu"
