from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from config import ExperimentConfig, apply_experiment_preset


ACTIVE_FLOW_GRID = tuple(range(18, 51, 4))
EDGE_NODE_GRID = tuple(range(18, 51, 4))
CONVERGENCE_LOADS = (18, 34, 50)
TOPOLOGY_REPEATS = 5
TRAJECTORY_REPEATS_PER_TOPOLOGY = 4
MAIN_METHODS = (
    "GDM-HMAPPO",
    "MA-SAC",
    "MA-P-DQN",
    "GraphPR",
    "FedRoute",
    "DALBH",
)


def build_v2_config(**overrides: Any) -> ExperimentConfig:
    config = apply_experiment_preset(ExperimentConfig(), "v2")
    if not overrides:
        return config
    values = asdict(config)
    values.update(overrides)
    return ExperimentConfig(**values)


def write_run_manifest(
    output_dir: str | Path,
    config: ExperimentConfig,
    command: Sequence[str],
    started_at: str,
    completed_at: str | None = None,
    checkpoint: str | None = None,
    artifacts: Sequence[str] = (),
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"

    payload = {
        "git_commit": commit,
        "command": list(command),
        "started_at_utc": started_at,
        "completed_at_utc": completed_at or datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "artifacts": list(artifacts),
        "seed": config.seed,
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    manifest_path = output_path / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path
