"""Tests for Stage 12 Slurm parallel dispatch."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from researchclaw.experiment.slurm_sandbox import BatchJobResult
from researchclaw.experiment.sandbox import SandboxResult


def test_build_parallel_tasks_from_schedule():
    """_build_parallel_tasks creates one task per condition x seed."""
    from researchclaw.pipeline.stage_impls._execution import _build_parallel_tasks

    schedule = {
        "tasks": [
            {"id": "baseline", "name": "baseline run"},
            {"id": "proposed", "name": "proposed method"},
        ],
    }
    code = "print('accuracy: 0.9')"
    tasks = _build_parallel_tasks(schedule, code, seeds=3)
    # 2 conditions x 3 seeds = 6 tasks
    assert len(tasks) == 6
    assert all("task_id" in t and "code" in t for t in tasks)
    # Each task_id includes condition and seed
    ids = [t["task_id"] for t in tasks]
    assert "baseline_seed0" in ids
    assert "proposed_seed2" in ids
    # Code injects seed and condition
    for t in tasks:
        assert "RESEARCHCLAW_CONDITION" in t["code"]
        assert "RESEARCHCLAW_SEED" in t["code"]


def test_build_parallel_tasks_single_condition():
    """With no schedule tasks, creates seed-only tasks."""
    from researchclaw.pipeline.stage_impls._execution import _build_parallel_tasks

    tasks = _build_parallel_tasks({}, "print('x: 1')", seeds=5)
    assert len(tasks) == 5
    for t in tasks:
        assert "seed" in t["task_id"]


def test_build_parallel_tasks_preserves_code():
    """Original experiment code is included after seed injection."""
    from researchclaw.pipeline.stage_impls._execution import _build_parallel_tasks

    original_code = "result = compute_metrics()\nprint(f'accuracy: {result}')"
    tasks = _build_parallel_tasks(
        {"tasks": [{"id": "test"}]}, original_code, seeds=1,
    )
    assert len(tasks) == 1
    assert original_code in tasks[0]["code"]


def test_build_parallel_tasks_sets_random_seeds():
    """Each task sets random, numpy, and torch seeds."""
    from researchclaw.pipeline.stage_impls._execution import _build_parallel_tasks

    tasks = _build_parallel_tasks(
        {"tasks": [{"id": "exp"}]}, "pass", seeds=2,
    )
    # seed0 task should set random.seed(0)
    assert "random.seed(0)" in tasks[0]["code"]
    # seed1 task should set random.seed(1)
    assert "random.seed(1)" in tasks[1]["code"]
