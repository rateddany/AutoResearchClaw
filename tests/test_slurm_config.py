"""Tests for Slurm configuration parsing."""
import pytest
from researchclaw.config import _parse_experiment_config, ExperimentConfig


def test_slurm_config_defaults():
    """SlurmConfig has sensible defaults when section is absent."""
    cfg = _parse_experiment_config({"mode": "slurm"})
    assert cfg.mode == "slurm"
    assert cfg.slurm.partition == "hermes-2"
    assert cfg.slurm.gpus_per_node == 8
    assert cfg.slurm.cpus_per_task == 24
    assert cfg.slurm.max_concurrent_jobs == 50
    assert cfg.slurm.time_limit == "02:00:00"
    assert cfg.slurm.exclusive is True
    assert cfg.slurm.conda_env == ""
    assert cfg.slurm.setup_commands == ()
    assert cfg.slurm.poll_interval_sec == 15
    assert cfg.slurm.log_dir == "slurm_logs"


def test_slurm_config_from_yaml():
    """SlurmConfig parses all fields from YAML dict."""
    cfg = _parse_experiment_config({
        "mode": "slurm",
        "slurm": {
            "partition": "hermes-1",
            "gpus_per_node": 4,
            "cpus_per_task": 12,
            "max_concurrent_jobs": 100,
            "time_limit": "04:00:00",
            "exclusive": False,
            "conda_env": "research",
            "setup_commands": ["module load rocm", "source activate research"],
            "poll_interval_sec": 10,
            "log_dir": "/mnt/vast01/users/dani.bouch/slurm_logs",
            "extra_sbatch_args": ["--exclude=auh7-4b-gpu-156"],
        },
    })
    assert cfg.slurm.partition == "hermes-1"
    assert cfg.slurm.gpus_per_node == 4
    assert cfg.slurm.cpus_per_task == 12
    assert cfg.slurm.max_concurrent_jobs == 100
    assert cfg.slurm.time_limit == "04:00:00"
    assert cfg.slurm.exclusive is False
    assert cfg.slurm.conda_env == "research"
    assert cfg.slurm.setup_commands == ("module load rocm", "source activate research")
    assert cfg.slurm.poll_interval_sec == 10
    assert cfg.slurm.extra_sbatch_args == ("--exclude=auh7-4b-gpu-156",)


def test_slurm_mode_in_experiment_modes():
    """'slurm' is a valid experiment mode."""
    from researchclaw.config import EXPERIMENT_MODES
    assert "slurm" in EXPERIMENT_MODES


def test_factory_creates_slurm_sandbox(tmp_path):
    """create_sandbox returns SlurmSandbox for mode='slurm'."""
    from researchclaw.config import ExperimentConfig, SlurmConfig
    from researchclaw.experiment.factory import create_sandbox

    cfg = ExperimentConfig(mode="slurm", slurm=SlurmConfig())
    sandbox = create_sandbox(cfg, tmp_path)
    from researchclaw.experiment.slurm_sandbox import SlurmSandbox
    assert isinstance(sandbox, SlurmSandbox)
