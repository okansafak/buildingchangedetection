import json
import os

import pytest

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Rewrite tests/snapshots/*.json from the current code's output.",
    )


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Tests must stay offline: any real urlopen call raises. Tests that need tiles override it (see fake_net)."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access blocked in tests")

    monkeypatch.setattr("urllib.request.urlopen", blocked)


@pytest.fixture
def snapshot(request):
    update = request.config.getoption("--update-snapshots")

    def check(name, data):
        path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
        if update:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, sort_keys=True, ensure_ascii=False)
            return
        assert os.path.exists(path), f"missing snapshot {path}"
        with open(path, encoding="utf-8") as f:
            expected = json.load(f)
        assert data == expected, f"output differs from snapshot '{name}'"

    return check


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    """The Flask app module with DB, uploads and tile cache redirected into tmp_path."""
    import app as app_mod
    from fakes import FakeSegmenter
    from model.jobs import JobRunner
    from model.live_satellite import LiveSatelliteFetcher
    from model.ml_detector import MLChangeDetector
    from model.project_manager import ProjectManager

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app_mod, "project_mgr", ProjectManager(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(app_mod, "live_fetcher", LiveSatelliteFetcher(cache_dir=str(tmp_path / "live_cache")))
    monkeypatch.setitem(app_mod.app.config, "UPLOAD_FOLDER", str(upload_dir))
    monkeypatch.setitem(app_mod.app.config, "RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setattr(app_mod, "ml_detector", MLChangeDetector(FakeSegmenter()))
    monkeypatch.setattr(app_mod, "jobs", JobRunner(sync=True))  # jobs finish before submit() returns
    app_mod.LAST_RESULTS.clear()
    yield app_mod
    app_mod.LAST_RESULTS.clear()


@pytest.fixture
def client(app_module):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
