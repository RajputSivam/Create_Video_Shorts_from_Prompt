from __future__ import annotations

from .common import job_directory, save_json


def save_qa_report(job: dict, checks: dict, metrics: dict | None = None) -> dict:
    statuses = {item.get("status") for item in checks.values()}
    final_failure = checks.get("final_video", {}).get("status") == "FAIL"
    status = "needs_review" if final_failure or "WARN" in statuses else "failed" if "FAIL" in statuses else "pass"
    report = {
        "status": status,
        "checks": checks,
        "metrics": metrics or {},
    }
    save_json(job_directory(job) / "qa_report.json", report)
    return report


def check(status: str, reason: str, **numbers) -> dict:
    return {"status": status, "reason": reason, "numbers": numbers}
