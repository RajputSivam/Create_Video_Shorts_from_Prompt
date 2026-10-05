import logging
import multiprocessing
import threading
from datetime import datetime, timezone

from . import database
from .pipeline import process_job

logger = logging.getLogger("uvicorn.error")


class JobQueue:
    def __init__(self, heartbeat_timeout_seconds: int = 900, poll_interval_seconds: float = 0.5):
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: multiprocessing.Process | None = None
        self._active_job_id: str | None = None
        self._process_lock = threading.Lock()

    def start(self) -> None:
        recovered = database.recover_interrupted_jobs()
        logger.info("Render queue startup: recovered=%d queue=%s", len(recovered), database.queue_state())
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="render-queue", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._wake_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        with self._process_lock:
            process = self._process
            job_id = self._active_job_id
        if process and process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if job_id:
                database.requeue_job(job_id, "Queued after server shutdown")
        logger.info("Render queue stopped: queue=%s", database.queue_state())

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job_id = database.claim_next_queued_job()
            if not job_id:
                self._wake_event.clear()
                self._wake_event.wait(self.poll_interval_seconds)
                continue
            logger.info("Render queue claimed job=%s queue=%s", job_id, database.queue_state())
            process = multiprocessing.get_context("spawn").Process(
                target=process_job,
                args=(job_id,),
                name=f"video-job-{job_id[:8]}",
            )
            with self._process_lock:
                self._process = process
                self._active_job_id = job_id
            try:
                process.start()
                self._monitor(process, job_id)
            except Exception as error:
                logger.exception("Could not run queued job=%s: %s", job_id, error)
                database.update_job(job_id, status="failed", stage="Render worker failed", error=str(error))
            finally:
                if process.pid is not None:
                    process.join(timeout=1)
                with self._process_lock:
                    self._process = None
                    self._active_job_id = None
                logger.info("Render queue released job=%s exit=%s queue=%s", job_id, process.exitcode, database.queue_state())

    def _monitor(self, process: multiprocessing.Process, job_id: str) -> None:
        while process.is_alive():
            process.join(timeout=1)
            if self._stop_event.is_set():
                process.terminate()
                process.join(timeout=5)
                database.requeue_job(job_id, "Queued after server shutdown")
                return
            job = database.get_job(job_id)
            heartbeat = job.get("heartbeat_at") if job else None
            if heartbeat and self._heartbeat_age(heartbeat) > self.heartbeat_timeout_seconds:
                logger.error("Render heartbeat expired for job=%s; terminating worker", job_id)
                process.terminate()
                process.join(timeout=5)
                database.update_job(
                    job_id,
                    status="failed",
                    stage="Render timed out",
                    error=f"Render worker stopped reporting progress for {self.heartbeat_timeout_seconds // 60} minutes.",
                )
                return
        job = database.get_job(job_id)
        if process.exitcode and job and job["status"] == "running":
            database.update_job(
                job_id,
                status="failed",
                stage="Render worker crashed",
                error=f"Render worker exited unexpectedly (code {process.exitcode}). Retry this job to resume cached work.",
            )
        elif job and job["status"] == "running":
            database.update_job(job_id, status="failed", stage="Render stopped", error="Render worker exited before completing the job.")

    @staticmethod
    def _heartbeat_age(value: str) -> float:
        heartbeat = datetime.fromisoformat(value)
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - heartbeat).total_seconds()