from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_sweeps import aggregate_rows, evaluate_policies, write_csv
from evaluation_tools import load_checkpoint, load_run_config
from profile_inference import aggregate_profile_rows, profile_policy
from experiment_protocol import ACTIVE_FLOW_GRID, write_run_manifest


FLOW_POINTS = (18, 34, 50)
STEPS = (5, 10, 15)


def validate_configs(configs: dict[int, object]) -> None:
    reference = asdict(configs[10])
    reference.pop("output_dir")
    reference.pop("alloc_refinement_steps")
    for steps in STEPS:
        config = configs[steps]
        if config.alloc_refinement_steps != steps:
            raise ValueError(f"Expected {steps} denoising steps in checkpoint config.")
        comparable = asdict(config)
        comparable.pop("output_dir")
        comparable.pop("alloc_refinement_steps")
        differences = [key for key in reference if reference[key] != comparable[key]]
        if differences:
            raise ValueError(f"Config mismatch for {steps} steps: {differences}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matched diffusion-step policies.")
    for steps in STEPS:
        parser.add_argument(f"--run-{steps}", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=60_000)
    parser.add_argument("--profile-seed-base", type=int, default=120_000)
    parser.add_argument("--num-repeats", type=int, default=4)
    parser.add_argument("--num-topologies", type=int, default=5)
    parser.add_argument("--warmup-slots", type=int, default=20)
    parser.add_argument("--measured-slots", type=int, default=100)
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {steps: Path(getattr(args, f"run_{steps}")) for steps in STEPS}
    configs = {
        steps: replace(load_run_config(run_dir), device=args.device)
        for steps, run_dir in run_dirs.items()
    }
    validate_configs(configs)
    base_config = configs[10]
    service_rows = []
    profile_rows = []

    for steps in STEPS:
        model = load_checkpoint(configs[steps], run_dirs[steps], "best_routing_model.pt")
        for flows in FLOW_POINTS:
            flow_config = replace(base_config, active_flow_range=(flows, flows))
            grid_index = ACTIVE_FLOW_GRID.index(flows)
            for topology_idx in range(args.num_topologies):
                topology_config = replace(
                    flow_config,
                    seed=base_config.seed + 10_000 * topology_idx,
                )
                for repeat_idx in range(args.num_repeats):
                    seed = (
                        args.seed_base
                        + 1_000 * (grid_index + 1)
                        + 10_000 * topology_idx
                        + 100_000 * repeat_idx
                    )
                    row = evaluate_policies(
                        topology_config,
                        model,
                        episodes=1,
                        seed_base=seed,
                        policy_names=["trained"],
                    )[0]
                    row.update(
                        policy=f"GDM-{steps}",
                        diffusion_steps=float(steps),
                        sweep_type="active_flows",
                        x_value=float(flows),
                        topology_repeat_id=float(topology_idx),
                        runtime_repeat_id=float(repeat_idx),
                        repeat_id=float(topology_idx * args.num_repeats + repeat_idx),
                    )
                    service_rows.append(row)

        profile_rows.extend(
            profile_policy(
                method=f"GDM-{steps}",
                policy_name="trained",
                model=model,
                base_config=base_config,
                warmup_slots=args.warmup_slots,
                measured_slots=args.measured_slots,
                seed_base=args.profile_seed_base,
            )
        )
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    hashes = {}
    for row in profile_rows:
        key = (row["sweep_type"], row["x_value"], row["repeat_id"])
        fingerprint = row["observation_sha256"]
        if key in hashes and hashes[key] != fingerprint:
            raise RuntimeError(f"Profiling observations differ at {key}.")
        hashes[key] = fingerprint

    artifacts = {
        "service_raw.csv": service_rows,
        "service_summary.csv": aggregate_rows(service_rows),
        "inference_raw.csv": profile_rows,
        "inference_summary.csv": aggregate_profile_rows(profile_rows),
    }
    for name, rows in artifacts.items():
        write_csv(rows, output_dir / name)
    meta = {
        "checkpoints": {
            str(steps): str(run_dirs[steps] / "best_routing_model.pt")
            for steps in STEPS
        },
        "flow_points": FLOW_POINTS,
        "num_repeats": args.num_repeats,
        "num_topologies": args.num_topologies,
        "seed_base": args.seed_base,
        "profile_seed_base": args.profile_seed_base,
        "warmup_slots": args.warmup_slots,
        "measured_slots": args.measured_slots,
        "common_profile_states_verified": True,
        "service_evaluation": "paired initial seeds and topology; trajectories may diverge with actions",
    }
    meta_path = output_dir / "sensitivity_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_run_manifest(
        output_dir,
        replace(base_config, output_dir=str(output_dir)),
        command=[sys.executable, *sys.argv],
        started_at=started_at,
        checkpoint=";".join(meta["checkpoints"].values()),
        artifacts=(*artifacts, meta_path.name),
    )
    print(f"Diffusion-step sensitivity results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
