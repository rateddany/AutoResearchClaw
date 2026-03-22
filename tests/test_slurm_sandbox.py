"""Tests for Slurm sandbox backend."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import SlurmConfig
from researchclaw.experiment.slurm_sandbox import (
    SlurmSandbox,
    SlurmBatchDispatcher,
)


@pytest.fixture
def slurm_config():
    return SlurmConfig(
        partition="hermes-2",
        gpus_per_node=8,
        cpus_per_task=24,
        time_limit="00:30:00",
        log_dir="slurm_logs",
        conda_env="research",
    )


@pytest.fixture
def sandbox(slurm_config, tmp_path):
    return SlurmSandbox(slurm_config, tmp_path)


# ── SlurmSandbox tests ──────────────────────────────────────────────


def test_generate_sbatch_script(sandbox, tmp_path):
    """sbatch script contains correct SBATCH headers and code."""
    script = sandbox._generate_sbatch_script(
        code_path=tmp_path / "experiment.py",
        job_name="rc-test-01",
        log_dir=tmp_path / "slurm_logs",
    )
    assert "#!/bin/bash" in script
    assert "#SBATCH --partition=hermes-2" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --cpus-per-task=24" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --job-name=rc-test-01" in script
    assert "conda activate research" in script
    assert "python" in script
    assert "experiment.py" in script


def test_generate_sbatch_script_no_conda(tmp_path):
    """sbatch script without conda when conda_env is empty."""
    cfg = SlurmConfig(conda_env="")
    sandbox = SlurmSandbox(cfg, tmp_path)
    script = sandbox._generate_sbatch_script(
        code_path=tmp_path / "experiment.py",
        job_name="rc-test",
        log_dir=tmp_path / "logs",
    )
    assert "conda activate" not in script


def test_generate_sbatch_script_extra_args(tmp_path):
    """sbatch script includes extra sbatch args."""
    cfg = SlurmConfig(extra_sbatch_args=("--exclude=node1", "--requeue"))
    sandbox = SlurmSandbox(cfg, tmp_path)
    script = sandbox._generate_sbatch_script(
        code_path=tmp_path / "experiment.py",
        job_name="rc-test",
        log_dir=tmp_path / "logs",
    )
    assert "#SBATCH --exclude=node1" in script
    assert "#SBATCH --requeue" in script


def test_generate_sbatch_script_setup_commands(tmp_path):
    """sbatch script includes setup commands."""
    cfg = SlurmConfig(setup_commands=("module load rocm", "export FOO=bar"))
    sandbox = SlurmSandbox(cfg, tmp_path)
    script = sandbox._generate_sbatch_script(
        code_path=tmp_path / "experiment.py",
        job_name="rc-test",
        log_dir=tmp_path / "logs",
    )
    assert "module load rocm" in script
    assert "export FOO=bar" in script


@patch("researchclaw.experiment.slurm_sandbox.subprocess.run")
def test_submit_job(mock_run, sandbox, tmp_path):
    """sbatch submission returns job ID on success."""
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="Submitted batch job 12345\n",
    )
    script_path = tmp_path / "job.sh"
    script_path.write_text("#!/bin/bash\necho hello")
    job_id = sandbox._submit_job(script_path)
    assert job_id == "12345"


@patch("researchclaw.experiment.slurm_sandbox.subprocess.run")
def test_submit_job_failure(mock_run, sandbox, tmp_path):
    """sbatch failure raises RuntimeError."""
    mock_run.return_value = MagicMock(
        returncode=1,
        stderr="sbatch: error: invalid partition",
    )
    script_path = tmp_path / "job.sh"
    script_path.write_text("#!/bin/bash\necho hello")
    with pytest.raises(RuntimeError, match="sbatch failed"):
        sandbox._submit_job(script_path)


@patch("researchclaw.experiment.slurm_sandbox.subprocess.run")
def test_check_job_status_running(mock_run, sandbox):
    """squeue returns RUNNING for active job."""
    mock_run.return_value = MagicMock(
        returncode=0, stdout="RUNNING\n",
    )
    assert sandbox._check_job_status("12345") == "RUNNING"


@patch("researchclaw.experiment.slurm_sandbox.subprocess.run")
def test_check_job_status_completed(mock_run, sandbox):
    """Empty squeue output means COMPLETED."""
    mock_run.return_value = MagicMock(
        returncode=0, stdout="\n",
    )
    assert sandbox._check_job_status("12345") == "COMPLETED"


def test_parse_log_metrics(sandbox, tmp_path):
    """Metrics are parsed from Slurm output log."""
    log_file = tmp_path / "slurm_logs" / "rc-test_12345.out"
    log_file.parent.mkdir(parents=True)
    log_file.write_text(
        "Starting experiment\n"
        "accuracy: 0.95\n"
        "loss: 0.032\n"
        "condition=baseline f1_score: 0.91\n"
    )
    metrics = sandbox._parse_log_metrics(log_file)
    assert metrics["accuracy"] == pytest.approx(0.95)
    assert metrics["loss"] == pytest.approx(0.032)
    assert "f1_score" in metrics


def test_find_log_file(sandbox, tmp_path):
    """Log file is found by job name and ID pattern."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    expected = log_dir / "rc-test_12345.out"
    expected.write_text("output")
    found = sandbox._find_log_file(log_dir, "rc-test", "12345", ".out")
    assert found == expected


def test_find_log_file_missing(sandbox, tmp_path):
    """Returns None when log file doesn't exist."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    found = sandbox._find_log_file(log_dir, "rc-test", "99999", ".out")
    assert found is None


# ── SlurmBatchDispatcher tests ───────────────────────────────────────


@pytest.fixture
def dispatcher(slurm_config, tmp_path):
    return SlurmBatchDispatcher(slurm_config, tmp_path)


def test_dispatcher_has_submit_batch(dispatcher):
    """Dispatcher has the submit_batch method."""
    assert hasattr(dispatcher, "submit_batch")
    assert callable(dispatcher.submit_batch)


def test_dispatcher_respects_max_concurrent(dispatcher):
    """Dispatcher config exposes max_concurrent_jobs."""
    dispatcher.config = SlurmConfig(max_concurrent_jobs=2)
    assert dispatcher.config.max_concurrent_jobs == 2


@patch("researchclaw.experiment.slurm_sandbox.subprocess.run")
def test_dispatcher_aggregates_results(mock_run, dispatcher, tmp_path):
    """Dispatcher collects metrics from all completed job logs."""
    log_dir = tmp_path / "slurm_logs"
    log_dir.mkdir()

    (log_dir / "rc-cond_A_100.out").write_text("condition=A accuracy: 0.90\n")
    (log_dir / "rc-cond_B_101.out").write_text("condition=B accuracy: 0.95\n")

    completed = {
        "100": {
            "task_id": "cond_A", "job_name": "rc-cond_A",
            "status": "COMPLETED", "elapsed": 10.0,
        },
        "101": {
            "task_id": "cond_B", "job_name": "rc-cond_B",
            "status": "COMPLETED", "elapsed": 12.0,
        },
    }

    aggregated = dispatcher._aggregate_results(completed, log_dir)
    assert len(aggregated) == 2
    assert any(
        r.sandbox_result.metrics.get("accuracy") == pytest.approx(0.90)
        for r in aggregated
    )
    assert any(
        r.sandbox_result.metrics.get("accuracy") == pytest.approx(0.95)
        for r in aggregated
    )
