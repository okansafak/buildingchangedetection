import numpy as np

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
