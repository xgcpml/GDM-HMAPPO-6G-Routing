from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from env import SimplifiedRoutingEnv
from evaluation_tools import load_checkpoint, load_run_config, rollout_policy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run active-flow ablation sweeps for GDM-HMAPPO variants.")
    parser.add_argument("--full-run-dir", type=str, required=True, help="Run directory of the full GDM-HMAPPO model.")
    parser.add_argument("--wo-gnn-run-dir", type=str, required=True, help="Run directory of the w/o GNN ablation.")
    parser.add_argument(
        "--wo-transformer-run-dir",
        type=str,
        required=True,
        help="Run directory of the w/o Transformer ablation.",
    )
    parser.add_argument("--wo-gdm-run-dir", type=str, default=None, help="Optional run directory of the w/o GDM ablation.")
    parser.add_argument("--checkpoint", type=str, default="best_routing_model.pt")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=70_000)
    parser.add_argument("--num-repeats", type=int, default=20)
    parser.add_argument("--repeat-seed-stride", type=int, default=100_000)
    parser.add_argument(
        "--active-flows",
        type=int,
        nargs="*",
        default=[18, 22, 26, 30, 34, 38, 42, 46, 50],
    )
    parser.add_argument("--output-dir", type=str, required=True)
    return parser


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    group_keys = ("sweep_type", "policy", "x_value")
    metric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"repeat_id", "runtime_repeat_id", *group_keys}
    ]
    grouped: dict[tuple[str, str, float], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sweep_type"]), str(row["policy"]), float(row["x_value"]))].append(row)

    aggregated: list[dict[str, float | str]] = []
    for (sweep_type, policy, x_value), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
        record: dict[str, float | str] = {
            "sweep_type": sweep_type,
            "policy": policy,
            "x_value": float(x_value),
            "num_samples": float(len(items)),
        }
        for metric in metric_keys:
            values = [float(item[metric]) for item in items]
            mean = float(sum(values) / len(values))
            variance = sum((value - mean) ** 2 for value in values) / len(values) if len(values) > 1 else 0.0
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = float(variance ** 0.5)
        aggregated.append(record)
    return aggregated


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("full", Path(args.full_run_dir)),
        ("wo_gnn", Path(args.wo_gnn_run_dir)),
        ("wo_transformer", Path(args.wo_transformer_run_dir)),
    ]
    if args.wo_gdm_run_dir:
        variants.append(("wo_gdm", Path(args.wo_gdm_run_dir)))
    loaded = []
    for policy_name, run_dir in variants:
        config = load_run_config(run_dir)
        model = load_checkpoint(config, run_dir, args.checkpoint)
        loaded.append((policy_name, config, model))

    rows: list[dict[str, float | str]] = []
    for flow_idx, flows in enumerate(args.active_flows):
        for repeat_idx in range(args.num_repeats):
            seed_base = args.seed_base + 1_000 * (flow_idx + 1) + args.repeat_seed_stride * repeat_idx
            for policy_name, config, model in loaded:
                eval_config = replace(config, active_flow_range=(flows, flows))
                env = SimplifiedRoutingEnv(eval_config)
                metrics = rollout_policy(
                    env=env,
                    config=eval_config,
                    policy_name="trained",
                    episodes=args.episodes,
                    seed_base=seed_base,
                    model=model,
                )
                if "global_load_balancing_index" in metrics and "deadline_hit_ratio" in metrics:
                    metrics["qos_load_balancing_index"] = (
                        float(metrics["global_load_balancing_index"]) * float(metrics["deadline_hit_ratio"])
                    )
                row = {
                    "sweep_type": "active_flows",
                    "policy": policy_name,
                    "x_value": float(flows),
                    "runtime_repeat_id": float(repeat_idx),
                    "repeat_id": float(repeat_idx),
                    **metrics,
                }
                rows.append(row)

    aggregated = aggregate_rows(rows)
    write_csv(rows, output_dir / "active_flow_ablation.csv")
    write_csv(aggregated, output_dir / "active_flow_ablation_aggregated.csv")
    with (output_dir / "ablation_sweeps_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "full_run_dir": args.full_run_dir,
                "wo_gnn_run_dir": args.wo_gnn_run_dir,
                "wo_transformer_run_dir": args.wo_transformer_run_dir,
                "wo_gdm_run_dir": args.wo_gdm_run_dir,
                "checkpoint": args.checkpoint,
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "num_repeats": args.num_repeats,
                "repeat_seed_stride": args.repeat_seed_stride,
                "active_flows": args.active_flows,
            },
            handle,
            indent=2,
        )
    print("Ablation sweeps finished.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
