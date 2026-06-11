from kairorag.ingestion.jd_loader import load_jobs
from kairorag.maintenance.job_verifier import verify_job


def test_verify_active_closed_and_updated_jobs():
    jobs = {job.job_id: job for job in load_jobs("data/raw/jobs.csv")}
    assert verify_job(jobs["job_001"]).new_status == "active"
    assert verify_job(jobs["job_003"]).new_status == "closed"
    updated = verify_job(jobs["job_008"], search=True)
    assert updated.new_status == "updated"
    assert updated.needs_reindex

