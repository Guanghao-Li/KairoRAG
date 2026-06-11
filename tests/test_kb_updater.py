from pathlib import Path
from shutil import copyfile

from kairorag.ingestion.jd_loader import load_jobs
from kairorag.maintenance.job_verifier import verify_job
from kairorag.maintenance.knowledge_base_updater import apply_verification_result


def test_kb_updater_dry_run_and_apply(tmp_path):
    jobs_file = tmp_path / "jobs.csv"
    copyfile(Path("data/raw/jobs.csv"), jobs_file)
    jobs = {job.job_id: job for job in load_jobs(jobs_file)}
    result = verify_job(jobs["job_003"])

    dry = apply_verification_result(result, jobs_file=jobs_file, apply=False)
    assert dry.action == "dry_run"
    assert dry.new_record["verification_status"] == "closed"

    applied = apply_verification_result(result, jobs_file=jobs_file, apply=True)
    assert applied.action == "update"
    refreshed = {job.job_id: job for job in load_jobs(jobs_file)}
    assert refreshed["job_003"].verification_status == "closed"
    assert (tmp_path.parent / "maintenance" / "audit_log.jsonl").exists() or applied.audit_log_path

