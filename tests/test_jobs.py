import time

from model.jobs import JobRunner


def test_threaded_job_reports_progress_and_result():
    runner = JobRunner()

    def work(report):
        report(0.5, "yarı")
        time.sleep(0.05)
        return {"ok": 1}

    job = runner.wait(runner.submit(work), timeout=5)
    assert job["status"] == "done" and job["result"] == {"ok": 1}
    assert job["progress"] == 1.0 and job["stage"] == "Tamamlandı"


def test_failed_job_keeps_error_and_sync_mode_finishes_inline():
    runner = JobRunner(sync=True)

    def boom(report):
        raise RuntimeError("patladı")

    job = runner.get(runner.submit(boom))
    assert job["status"] == "error" and "patladı" in job["error"]
    assert runner.get("yok") is None
