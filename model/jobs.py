"""
In-memory background jobs for long detections (single user, like LAST_RESULTS).
fn(report) runs on a daemon thread; report(fraction, stage) updates progress.
"""
import threading
import time
import traceback
import uuid


class JobRunner:
    def __init__(self, sync=False):
        self.sync = sync  # tests: run inline so the job is finished when submit() returns
        self._jobs = {}
        self._threads = {}
        self._lock = threading.Lock()

    def submit(self, fn):
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "running", "progress": 0.0, "stage": "Sırada",
               "result": None, "error": None, "started": time.time()}
        with self._lock:
            self._jobs[job_id] = job

        def report(fraction, stage):
            job["progress"] = round(min(max(float(fraction), 0.0), 1.0), 3)
            job["stage"] = stage

        def run():
            try:
                job["result"] = fn(report)
                job["progress"] = 1.0
                job["stage"] = "Tamamlandı"
                job["status"] = "done"
            except Exception as e:
                traceback.print_exc()
                job["error"] = str(e)
                job["status"] = "error"

        if self.sync:
            run()
        else:
            thread = threading.Thread(target=run, daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return job_id

    def get(self, job_id):
        """Snapshot of the job dict, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def wait(self, job_id, timeout=None):
        thread = self._threads.get(job_id)
        if thread:
            thread.join(timeout)
        return self.get(job_id)
