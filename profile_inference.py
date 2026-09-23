from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from benchmark_sweeps import write_csv
from env import SimplifiedRoutingEnv
from evaluation_tools import load_checkpoint, load_run_config, select_slot_actions
from mapdqn import load_mapdqn_policy, load_mapdqn_train_config
from masac import batch_obs_to_torch, load_masac_actor, load_masac_train_config
from experiment_protocol import write_run_manifest


PROFILE_POINTS = (18, 34, 50)


def observation_fingerprint(obs: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = np.ascontiguousarray(obs[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_scaled_config(base_config, sweep_type: str, x_value: int):
    if sweep_type == "active_flows":
        return replace(base_config, active_flow_range=(x_value, x_value))
    if sweep_type != "edge_nodes":
        raise ValueError(f"Unsupported profiling sweep: {sweep_type}")
    return replace(
        base_config,
        num_edges=x_value,
        access_edge_degree=min(x_value, max(base_config.access_edge_degree, 2)),
        edge_neighbor_degree=min(
            max(x_value - 1, 1), max(base_config.edge_neighbor_degree, 2)
        ),
        edge_placement_mode="nested_grid",
        edge_layout_reference_count=max(PROFILE_POINTS),
    )


def profile_slot_actions(policy_name, obs, config, model, rng):
    if policy_name == "masac":
        with torch.inference_mode():
            action = model.actor.sample(
                batch_obs_to_torch(obs, model.device),
                deterministic=True,
                log_std_min=model.log_std_min,
                log_std_max=model.log_std_max,
                gumbel_tau=model.gumbel_tau,
            )
        return (
            action["route_idx"].cpu().numpy().astype(np.int64),
            action["allocation"].cpu().numpy().astype(np.float32),
        )
    if policy_name == "mapdqn":
        with torch.inference_mode():
            obs_t = batch_obs_to_torch(obs, model.device)
            allocations = model.actor(obs_t)
            routes = model.critic(obs_t, allocations).argmax(dim=-1)
            selected = allocations[
                torch.arange(routes.size(0), device=routes.device), routes
            ]
        return (
            routes.cpu().numpy().astype(np.int64),
            selected.cpu().numpy().astype(np.float32),
        )
    return select_slot_actions(policy_name, obs, config, model, rng)


def profile_policy(
    method: str,
    policy_name: str,
    model: Any,
    base_config,
    warmup_slots: int,
    measured_slots: int,
    seed_base: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    rng = np.random.default_rng(seed_base + 999)
    reference_rng = np.random.default_rng(seed_base + 1999)
    for sweep_type in ("active_flows", "edge_nodes"):
        for point_idx, x_value in enumerate(PROFILE_POINTS):
            config = build_scaled_config(base_config, sweep_type, x_value)
            env = SimplifiedRoutingEnv(config)
            obs = env.reset_slot(seed=seed_base + point_idx + 10_000 * (sweep_type == "edge_nodes"))
            for slot_idx in range(warmup_slots + measured_slots):
                fingerprint = observation_fingerprint(obs) if slot_idx >= warmup_slots else ""
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    if slot_idx >= warmup_slots:
                        torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                routes, allocations = profile_slot_actions(
                    policy_name, obs, config, model, rng
                )
                if config.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1_000.0
                if slot_idx >= warmup_slots:
                    peak_memory_mb = (
                        torch.cuda.max_memory_allocated() / (1024.0**2)
                        if config.device.startswith("cuda")
                        else 0.0
                    )
                    rows.append(
                        {
                            "method": method,
                            "sweep_type": sweep_type,
                            "x_value": float(x_value),
                            "repeat_id": float(slot_idx - warmup_slots),
                            "num_tasks": float(obs["task_features"].shape[0]),
                            "observation_sha256": fingerprint,
                            "inference_latency_ms": elapsed_ms,
                            "peak_gpu_memory_mb": peak_memory_mb,
                        }
                    )
                reference_routes, reference_allocations = select_slot_actions(
                    "heuristic", obs, config, None, reference_rng
                )
                obs, _, done, _ = env.step_slot(
                    {"route_idx": reference_routes, "allocation": reference_allocations}
                )
                if done or obs is None:
                    obs = env.reset_slot(
                        seed=seed_base + point_idx + slot_idx + 1
                    )
    return rows


def aggregate_profile_rows(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str, int], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), str(row["sweep_type"]), int(float(row["x_value"])))].append(row)
    summary: list[dict[str, float | str]] = []
    for (method, sweep_type, x_value), items in sorted(groups.items()):
        latency = np.asarray([float(item["inference_latency_ms"]) for item in items])
        memory = np.asarray([float(item["peak_gpu_memory_mb"]) for item in items])
        summary.append(
            {
                "method": method,
                "sweep_type": sweep_type,
                "x_value": float(x_value),
                "num_samples": float(len(items)),
                "inference_latency_ms_mean": float(latency.mean()),
                "inference_latency_ms_std": float(latency.std()),
                "peak_gpu_memory_mb_mean": float(memory.mean()),
                "peak_gpu_memory_mb_max": float(memory.max()),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile online inference latency and peak GPU memory."
    )
    parser.add_argument("--proposed-run-dir", required=True)
    parser.add_argument("--masac-run-dir", required=True)
    parser.add_argument("--mapdqn-run-dir", required=True)
    parser.add_argument("--graphpr-run-dir")
    parser.add_argument("--fedroute-run-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proposed-checkpoint", default="best_routing_model.pt")
    parser.add_argument("--masac-checkpoint", default="best_masac_actor.pt")
    parser.add_argument("--mapdqn-checkpoint", default="best_mapdqn_actor.pt")
    parser.add_argument("--graphpr-checkpoint", default="best_routing_model.pt")
    parser.add_argument("--fedroute-checkpoint", default="best_fedroute_model.pt")
    parser.add_argument("--warmup-slots", type=int, default=10)
    parser.add_argument("--measured-slots", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=120_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = replace(
        load_run_config(Path(args.proposed_run_dir)),
        device=args.device,
        output_dir=str(output_dir),
    )

    loaders: list[tuple[str, str, Callable[[], Any]]] = [
        (
            "GDM-HMAPPO",
            "trained",
            lambda: load_checkpoint(
                base_config,
                Path(args.proposed_run_dir),
                args.proposed_checkpoint,
            ),
        ),
        (
            "MA-SAC",
            "masac",
            lambda: load_masac_actor(
                Path(args.masac_run_dir) / args.masac_checkpoint,
                base_config,
                load_masac_train_config(args.masac_run_dir, device=args.device),
            ),
        ),
        (
            "MA-P-DQN",
            "mapdqn",
            lambda: load_mapdqn_policy(
                Path(args.mapdqn_run_dir) / args.mapdqn_checkpoint,
                base_config,
                load_mapdqn_train_config(args.mapdqn_run_dir, device=args.device),
            ),
        ),
    ]
    for method, policy_name, run_dir_arg, checkpoint in (
        ("GraphPR", "graphpr", args.graphpr_run_dir, args.graphpr_checkpoint),
        ("FedRoute", "fedroute", args.fedroute_run_dir, args.fedroute_checkpoint),
    ):
        if run_dir_arg is not None:
            run_dir = Path(run_dir_arg)
            model_config = replace(load_run_config(run_dir), device=args.device)
            loaders.append(
                (
                    method,
                    policy_name,
                    lambda config=model_config, path=run_dir, name=checkpoint: load_checkpoint(
                        config, path, name
                    ),
                )
            )
    rows: list[dict[str, float | str]] = []
    for method, policy_name, loader in loaders:
        model = loader()
        rows.extend(
            profile_policy(
                method,
                policy_name,
                model,
                base_config,
                args.warmup_slots,
                args.measured_slots,
                args.seed_base,
            )
        )
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    state_hashes: dict[tuple[str, float, float], str] = {}
    for row in rows:
        key = (str(row["sweep_type"]), float(row["x_value"]), float(row["repeat_id"]))
        fingerprint = str(row["observation_sha256"])
        if key in state_hashes and state_hashes[key] != fingerprint:
            raise RuntimeError(f"Profiling methods saw different observations at {key}.")
        state_hashes[key] = fingerprint

    raw_path = output_dir / "inference_profile_raw.csv"
    summary_path = output_dir / "inference_profile_summary.csv"
    write_csv(rows, raw_path)
    write_csv(aggregate_profile_rows(rows), summary_path)
    meta = {
        "profile_points": PROFILE_POINTS,
        "warmup_slots": args.warmup_slots,
        "measured_slots": args.measured_slots,
        "timing_scope": "policy action generation for one complete decision slot",
        "inference_batching": "all profiled policies process the full decision slot in one model call",
        "environment_step_timed": False,
        "observation_trajectory": "same heuristic-driven states for every method; verified by SHA256",
        "state_hashes_verified": True,
        "memory_definition": "per-slot torch.cuda.max_memory_allocated, including loaded policy",
        "input_checkpoints": {
            "GDM-HMAPPO": str(
                Path(args.proposed_run_dir) / args.proposed_checkpoint
            ),
            "MA-SAC": str(Path(args.masac_run_dir) / args.masac_checkpoint),
            "MA-P-DQN": str(Path(args.mapdqn_run_dir) / args.mapdqn_checkpoint),
        },
    }
    if args.graphpr_run_dir is not None:
        meta["input_checkpoints"]["GraphPR"] = str(
            Path(args.graphpr_run_dir) / args.graphpr_checkpoint
        )
    if args.fedroute_run_dir is not None:
        meta["input_checkpoints"]["FedRoute"] = str(
            Path(args.fedroute_run_dir) / args.fedroute_checkpoint
        )
    meta_path = output_dir / "inference_profile_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_run_manifest(
        output_dir,
        base_config,
        command=[sys.executable, *sys.argv],
        started_at=started_at,
        checkpoint=args.proposed_checkpoint,
        artifacts=(raw_path.name, summary_path.name, meta_path.name),
    )
    print(f"Inference profile written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
