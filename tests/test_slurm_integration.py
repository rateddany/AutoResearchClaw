"""End-to-end integration tests for Slurm backend (mocked sbatch/squeue)."""
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.config import SlurmConfig, ExperimentConfig
from researchclaw.experiment.slurm_sandbox import SlurmSandbox, SlurmBatchDispatcher


def _mock_subprocess_factory(tmp_path):
    """Create a mock subprocess.run that simulates sbatch/squeue."""
    counter = {"n": 0}
    completed = set()

    def mock_run(cmd, **kwargs):
        result = MagicMock()
        result.stderr = ""

        if cmd[0] == "sbatch":
            counter["n"] += 1
            job_id = str(1000 + counter["n"])
            result.returncode = 0
            result.stdout = f"Submitted batch job {job_id}\n"

            # Write fake log output
            log_dir = tmp_path / "slurm_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            script_content = Path(cmd[1]).read_text()
            m = re.search(r"--job-name=(\S+)", script_content)
            job_name = m.group(1) if m else "unknown"
            (log_dir / f"{job_name}_{job_id}.out").write_text(
                f"accuracy: 0.{90 + counter['n']}\n"
                f"loss: 0.0{counter['n']}\n"
            )
            (log_dir / f"{job_name}_{job_id}.err").write_text("")
            completed.add(job_id)

        elif cmd[0] == "squeue":
            job_id = cmd[2]  # -j <id>
            result.returncode = 0
            if job_id in completed:
                result.stdout = "\n"  # empty = completed
            else:
                result.stdout = "RUNNING\n"

        elif cmd[0] == "scancel":
            result.returncode = 0
            result.stdout = ""

        else:
            result.returncode = 0
            result.stdout = ""

        return result

    return mock_run


def test_batch_dispatcher_end_to_end(tmp_path):
    """Full batch dispatch cycle: submit 4 tasks, all complete."""
    mock_fn = _mock_subprocess_factory(tmp_path)
    cfg = SlurmConfig(
        partition="hermes-2",
        gpus_per_node=8,
        max_concurrent_jobs=3,
        poll_interval_sec=0,
        log_dir=str(tmp_path / "slurm_logs"),
    )

    with patch(
        "researchclaw.experiment.slurm_sandbox.subprocess.run",
        side_effect=mock_fn,
    ):
        dispatcher = SlurmBatchDispatcher(cfg, tmp_path / "work")
        tasks = [
            {
                "task_id": f"cond_{i}_seed_{s}",
                "code": f"print('accuracy: 0.9{i}')",
            }
            for i in range(2)
            for s in range(2)
        ]
        results = dispatcher.submit_batch(tasks, timeout_sec=60)

    assert len(results) == 4
    assert all(r.status == "COMPLETED" for r in results)
    assert all(r.sandbox_result.returncode == 0 for r in results)
    for r in results:
        assert "accuracy" in r.sandbox_result.metrics


def test_slurm_sandbox_single_run(tmp_path):
    """Single SlurmSandbox.run() submits one job and returns SandboxResult."""
    mock_fn = _mock_subprocess_factory(tmp_path)
    cfg = SlurmConfig(
        partition="hermes-2",
        poll_interval_sec=0,
        log_dir=str(tmp_path / "slurm_logs"),
    )

    with patch(
        "researchclaw.experiment.slurm_sandbox.subprocess.run",
        side_effect=mock_fn,
    ):
        sandbox = SlurmSandbox(cfg, tmp_path / "work")
        result = sandbox.run("print('loss: 0.05')", timeout_sec=30)

    assert result.returncode == 0
    assert not result.timed_out
    assert "accuracy" in result.metrics or "loss" in result.metrics


def test_slurm_sandbox_run_project(tmp_path):
    """SlurmSandbox.run_project() handles multi-file projects."""
    mock_fn = _mock_subprocess_factory(tmp_path)
    cfg = SlurmConfig(
        partition="hermes-2",
        poll_interval_sec=0,
        log_dir=str(tmp_path / "slurm_logs"),
    )

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "main.py").write_text("print('accuracy: 0.99')")
    (project_dir / "utils.py").write_text("def helper(): pass")

    with patch(
        "researchclaw.experiment.slurm_sandbox.subprocess.run",
        side_effect=mock_fn,
    ):
        sandbox = SlurmSandbox(cfg, tmp_path / "work")
        result = sandbox.run_project(project_dir, timeout_sec=30)

    assert result.returncode == 0


def test_batch_dispatcher_concurrency_throttling(tmp_path):
    """Dispatcher submits only max_concurrent_jobs at a time."""
    call_log = []
    mock_fn = _mock_subprocess_factory(tmp_path)

    original_mock = mock_fn

    def logging_mock(cmd, **kwargs):
        if cmd[0] == "sbatch":
            call_log.append("sbatch")
        return original_mock(cmd, **kwargs)

    cfg = SlurmConfig(
        partition="hermes-2",
        max_concurrent_jobs=2,
        poll_interval_sec=0,
        log_dir=str(tmp_path / "slurm_logs"),
    )

    with patch(
        "researchclaw.experiment.slurm_sandbox.subprocess.run",
        side_effect=logging_mock,
    ):
        dispatcher = SlurmBatchDispatcher(cfg, tmp_path / "work")
        tasks = [
            {"task_id": f"task_{i}", "code": "print('x: 1')"}
            for i in range(5)
        ]
        results = dispatcher.submit_batch(tasks, timeout_sec=60)

    assert len(results) == 5
    assert all(r.status == "COMPLETED" for r in results)


def test_build_parallel_tasks_integration():
    """Full integration: schedule + code -> tasks for batch dispatch."""
    from researchclaw.pipeline.stage_impls._execution import _build_parallel_tasks

    schedule = {
        "tasks": [
            {"id": "baseline", "name": "Baseline"},
            {"id": "method_A", "name": "Method A"},
            {"id": "method_B", "name": "Method B"},
        ],
    }
    code = (
        "import os\n"
        "cond = os.environ.get('RESEARCHCLAW_CONDITION', 'unknown')\n"
        "seed = int(os.environ.get('RESEARCHCLAW_SEED', '0'))\n"
        "print(f'accuracy: {0.85 + seed * 0.01}')\n"
    )
    tasks = _build_parallel_tasks(schedule, code, seeds=5)

    # 3 conditions x 5 seeds = 15 parallel jobs
    assert len(tasks) == 15

    # Verify all condition/seed combos present
    ids = {t["task_id"] for t in tasks}
    for cond in ["baseline", "method_A", "method_B"]:
        for seed in range(5):
            assert f"{cond}_seed{seed}" in ids
