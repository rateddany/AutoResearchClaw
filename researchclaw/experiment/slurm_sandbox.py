"""Slurm-based experiment execution backend.

Submits experiment code as Slurm batch jobs via sbatch, polls squeue
for completion, and collects results from log files.

SlurmSandbox: single-job SandboxProtocol implementation.
SlurmBatchDispatcher: parallel fan-out/fan-in for multiple experiment tasks.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from researchclaw.config import SlurmConfig
from researchclaw.experiment.sandbox import (
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)

logger = logging.getLogger(__name__)

_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


class SlurmSandbox:
    """Execute experiment code via Slurm sbatch jobs.

    Implements the same public interface as ExperimentSandbox so the
    pipeline can use it as a drop-in backend.
    """

    def __init__(self, config: SlurmConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    # ------------------------------------------------------------------
    # SandboxProtocol interface
    # ------------------------------------------------------------------

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        """Submit a single Python code string as a Slurm job."""
        self._run_counter += 1
        staging = self.workdir / f"_slurm_run_{self._run_counter}"
        staging.mkdir(parents=True, exist_ok=True)

        code_path = staging / "experiment.py"
        code_path.write_text(code, encoding="utf-8")

        return self._submit_and_wait(
            code_path=code_path,
            job_name=f"rc-exp-{self._run_counter}",
            staging=staging,
            timeout_sec=timeout_sec,
        )

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
    ) -> SandboxResult:
        """Submit a multi-file experiment project as a Slurm job."""
        import shutil

        self._run_counter += 1
        staging = self.workdir / f"_slurm_project_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        err = validate_entry_point(entry_point)
        if err:
            return SandboxResult(
                returncode=-1, stdout="", stderr=err,
                elapsed_sec=0.0, metrics={},
            )

        for src_file in project_dir.iterdir():
            if src_file.is_file():
                (staging / src_file.name).write_bytes(src_file.read_bytes())

        err = validate_entry_point_resolved(staging, entry_point)
        if err:
            return SandboxResult(
                returncode=-1, stdout="", stderr=err,
                elapsed_sec=0.0, metrics={},
            )

        entry = staging / entry_point
        if not entry.exists():
            return SandboxResult(
                returncode=-1, stdout="",
                stderr=f"Entry point {entry_point} not found in project",
                elapsed_sec=0.0, metrics={},
            )

        return self._submit_and_wait(
            code_path=entry,
            job_name=f"rc-proj-{self._run_counter}",
            staging=staging,
            timeout_sec=timeout_sec,
        )

    # ------------------------------------------------------------------
    # sbatch script generation
    # ------------------------------------------------------------------

    def _generate_sbatch_script(
        self,
        code_path: Path,
        job_name: str,
        log_dir: Path,
    ) -> str:
        """Generate an sbatch script for the experiment."""
        cfg = self.config
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={cfg.partition}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            f"#SBATCH --cpus-per-task={cfg.cpus_per_task}",
            f"#SBATCH --gpus-per-node={cfg.gpus_per_node}",
            f"#SBATCH --time={cfg.time_limit}",
        ]
        if cfg.exclusive:
            lines.append("#SBATCH --exclusive")
        lines.extend([
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
        ])
        for arg in cfg.extra_sbatch_args:
            lines.append(f"#SBATCH {arg}")

        lines.append("")
        lines.append("set -euo pipefail")
        lines.append("")

        for cmd in cfg.setup_commands:
            lines.append(cmd)
        if cfg.conda_env:
            lines.append(f"conda activate {cfg.conda_env}")
        lines.append("")

        lines.append(f"cd {code_path.parent}")
        lines.append(f"python -u {code_path.name}")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Job submission and monitoring
    # ------------------------------------------------------------------

    def _submit_job(self, script_path: Path) -> str:
        """Submit sbatch script and return the job ID."""
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        match = _JOB_ID_RE.search(result.stdout)
        if not match:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: "
                f"{result.stdout.strip()}"
            )
        return match.group(1)

    def _check_job_status(self, job_id: str) -> str:
        """Check job status via squeue. Returns status string or 'COMPLETED'."""
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=15,
        )
        status = result.stdout.strip()
        if not status:
            return "COMPLETED"
        return status

    def _wait_for_job(
        self, job_id: str, *, timeout_sec: int,
    ) -> tuple[str, float]:
        """Poll squeue until job completes or times out."""
        start = time.monotonic()
        poll_interval = self.config.poll_interval_sec

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout_sec:
                self._cancel_job(job_id)
                return "TIMEOUT", elapsed

            status = self._check_job_status(job_id)
            if status in (
                "COMPLETED", "FAILED", "CANCELLED",
                "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            ):
                return status, time.monotonic() - start

            time.sleep(poll_interval)

    def _cancel_job(self, job_id: str) -> None:
        """Cancel a running job via scancel."""
        try:
            subprocess.run(
                ["scancel", job_id],
                capture_output=True, timeout=15,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to cancel job %s", job_id)

    def _parse_log_metrics(self, log_path: Path) -> dict[str, float]:
        """Parse metrics from a Slurm stdout log file."""
        if not log_path.exists():
            return {}
        content = log_path.read_text(encoding="utf-8", errors="replace")
        return parse_metrics(content)

    def _find_log_file(
        self, log_dir: Path, job_name: str, job_id: str, suffix: str = ".out",
    ) -> Path | None:
        """Find the Slurm log file for a given job."""
        expected = log_dir / f"{job_name}_{job_id}{suffix}"
        if expected.exists():
            return expected
        candidates = list(log_dir.glob(f"*_{job_id}{suffix}"))
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # High-level submit-and-wait
    # ------------------------------------------------------------------

    def _submit_and_wait(
        self,
        code_path: Path,
        job_name: str,
        staging: Path,
        timeout_sec: int,
    ) -> SandboxResult:
        """Generate script, submit, wait, collect results."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.is_absolute():
            log_dir = self.workdir / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        script_content = self._generate_sbatch_script(
            code_path=code_path,
            job_name=job_name,
            log_dir=log_dir,
        )
        script_path = staging / f"{job_name}.sh"
        script_path.write_text(script_content, encoding="utf-8")

        try:
            job_id = self._submit_job(script_path)
        except RuntimeError as exc:
            return SandboxResult(
                returncode=-1, stdout="", stderr=str(exc),
                elapsed_sec=0.0, metrics={},
            )

        logger.info("Submitted Slurm job %s (%s)", job_id, job_name)

        final_status, elapsed = self._wait_for_job(
            job_id, timeout_sec=timeout_sec,
        )

        stdout_log = self._find_log_file(log_dir, job_name, job_id, ".out")
        stderr_log = self._find_log_file(log_dir, job_name, job_id, ".err")

        stdout = (
            stdout_log.read_text(encoding="utf-8", errors="replace")
            if stdout_log else ""
        )
        stderr = (
            stderr_log.read_text(encoding="utf-8", errors="replace")
            if stderr_log else ""
        )

        metrics = parse_metrics(stdout)
        timed_out = final_status == "TIMEOUT"
        returncode = 0 if final_status == "COMPLETED" else 1

        return SandboxResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed,
            metrics={k: v for k, v in metrics.items()},
            timed_out=timed_out,
        )


# ======================================================================
# Batch dispatcher — parallel fan-out/fan-in
# ======================================================================


@dataclass
class BatchJobResult:
    """Result from a single job in a batch submission."""

    task_id: str
    job_id: str
    sandbox_result: SandboxResult
    status: str  # COMPLETED, FAILED, TIMEOUT, etc.


class SlurmBatchDispatcher:
    """Submit multiple experiment tasks as parallel Slurm jobs.

    Fan-out: submit all tasks as sbatch jobs (respecting max_concurrent).
    Fan-in: poll squeue until all complete, collect results from logs.
    """

    def __init__(self, config: SlurmConfig, workdir: Path) -> None:
        self.config = config
        self.sandbox = SlurmSandbox(config, workdir)
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    def submit_batch(
        self,
        tasks: list[dict[str, str]],
        *,
        timeout_sec: int = 3600,
    ) -> list[BatchJobResult]:
        """Submit all tasks as parallel Slurm jobs and wait for completion.

        Args:
            tasks: List of dicts with keys: task_id, code.
            timeout_sec: Total wall-clock timeout for the entire batch.

        Returns:
            List of BatchJobResult, one per task.
        """
        log_dir = Path(self.config.log_dir)
        if not log_dir.is_absolute():
            log_dir = self.workdir / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Stage all tasks and generate scripts
        staged: list[dict] = []
        for task in tasks:
            task_id = task["task_id"]
            code = task["code"]
            staging = self.workdir / f"_batch_{task_id}"
            staging.mkdir(parents=True, exist_ok=True)

            code_path = staging / "experiment.py"
            code_path.write_text(code, encoding="utf-8")

            job_name = f"rc-{task_id}"
            script = self.sandbox._generate_sbatch_script(
                code_path=code_path,
                job_name=job_name,
                log_dir=log_dir,
            )
            script_path = staging / f"{job_name}.sh"
            script_path.write_text(script, encoding="utf-8")

            staged.append({
                "task_id": task_id,
                "job_name": job_name,
                "script_path": script_path,
            })

        # Submit with concurrency throttling
        active_jobs: dict[str, dict] = {}
        pending = list(staged)
        completed_jobs: dict[str, dict] = {}
        start = time.monotonic()

        while pending or active_jobs:
            elapsed = time.monotonic() - start
            if elapsed > timeout_sec:
                for jid in list(active_jobs):
                    self.sandbox._cancel_job(jid)
                    info = active_jobs.pop(jid)
                    completed_jobs[jid] = {
                        **info, "status": "TIMEOUT",
                        "elapsed": time.monotonic() - start,
                    }
                for task_info in pending:
                    key = f"not-submitted-{task_info['task_id']}"
                    completed_jobs[key] = {
                        **task_info, "status": "NOT_SUBMITTED",
                        "elapsed": 0.0,
                    }
                pending.clear()
                break

            # Submit pending tasks up to concurrency limit
            while (
                pending
                and len(active_jobs) < self.config.max_concurrent_jobs
            ):
                task_info = pending.pop(0)
                try:
                    job_id = self.sandbox._submit_job(
                        task_info["script_path"],
                    )
                    active_jobs[job_id] = {**task_info, "job_id": job_id}
                    logger.info(
                        "Submitted %s as job %s (%d active, %d pending)",
                        task_info["task_id"], job_id,
                        len(active_jobs), len(pending),
                    )
                except RuntimeError as exc:
                    logger.error(
                        "Failed to submit %s: %s",
                        task_info["task_id"], exc,
                    )
                    key = f"failed-{task_info['task_id']}"
                    completed_jobs[key] = {
                        **task_info, "status": "SUBMIT_FAILED",
                        "elapsed": 0.0, "error": str(exc),
                    }

            # Poll active jobs
            for jid in list(active_jobs):
                status = self.sandbox._check_job_status(jid)
                if status in (
                    "COMPLETED", "FAILED", "CANCELLED",
                    "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
                ):
                    info = active_jobs.pop(jid)
                    completed_jobs[jid] = {
                        **info, "status": status,
                        "elapsed": time.monotonic() - start,
                    }
                    logger.info(
                        "Job %s (%s) finished: %s",
                        jid, info["task_id"], status,
                    )

            if active_jobs:
                time.sleep(self.config.poll_interval_sec)

        return self._aggregate_results(completed_jobs, log_dir)

    def _aggregate_results(
        self,
        completed_jobs: dict[str, dict],
        log_dir: Path,
    ) -> list[BatchJobResult]:
        """Collect stdout/stderr/metrics from all completed job logs."""
        results: list[BatchJobResult] = []
        for job_id, info in completed_jobs.items():
            task_id = info.get("task_id", job_id)
            job_name = info.get("job_name", f"rc-{task_id}")
            status = info.get("status", "UNKNOWN")
            elapsed = float(info.get("elapsed", 0.0))

            stdout_log = self.sandbox._find_log_file(
                log_dir, job_name, job_id, ".out",
            )
            stderr_log = self.sandbox._find_log_file(
                log_dir, job_name, job_id, ".err",
            )

            stdout = ""
            stderr = ""
            if stdout_log and stdout_log.exists():
                stdout = stdout_log.read_text(
                    encoding="utf-8", errors="replace",
                )
            if stderr_log and stderr_log.exists():
                stderr = stderr_log.read_text(
                    encoding="utf-8", errors="replace",
                )

            metrics = parse_metrics(stdout) if stdout else {}
            timed_out = status == "TIMEOUT"
            returncode = 0 if status == "COMPLETED" else 1

            results.append(BatchJobResult(
                task_id=str(task_id),
                job_id=str(job_id),
                status=status,
                sandbox_result=SandboxResult(
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    elapsed_sec=elapsed,
                    metrics={k: v for k, v in metrics.items()},
                    timed_out=timed_out,
                ),
            ))
        return results
