from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from env import SimplifiedRoutingEnv
from evaluation_tools import load_checkpoint, load_run_config, rollout_policy
from mapdqn import load_mapdqn_policy, load_mapdqn_train_config
from masac import load_masac_actor, load_masac_train_config


def _scaled_degree(base_degree: int, target_edges: int, reference_edges: int, minimum: int = 1) -> int:
    if reference_edges <= 0:
        return max(int(base_degree), minimum)
    scaled = int(round(float(base_degree) * float(target_edges) / float(reference_edges)))
    return max(scaled, minimum)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark sweeps for trained and heuristic policies.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing checkpoints and metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_routing_model.pt",
        help="Checkpoint filename for the trained policy.",
    )
    parser.add_argument("--episodes", type=int, default=16, help="Episodes per sweep point.")
    parser.add_argument("--seed-base", type=int, default=60_000, help="Base seed for deterministic evaluation.")
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of independent repeated evaluations per sweep point.",
    )
    parser.add_argument(
        "--repeat-seed-stride",
        type=int,
        default=100_000,
        help="Seed offset between repeated evaluations.",
    )
    parser.add_argument(
        "--num-topology-repeats",
        type=int,
        default=1,
        help="Number of topology seeds to average for each sweep point.",
    )
    parser.add_argument(
        "--topology-seed-stride",
        type=int,
        default=10_000,
        help="Seed offset between topology realizations.",
    )
    parser.add_argument(
        "--active-flows",
        type=int,
        nargs="*",
        default=[16, 20, 24, 28, 32],
        help="Active flow counts for the load sweep.",
    )
    parser.add_argument(
        "--edge-nodes",
        type=int,
        nargs="*",
        default=[18, 24, 30, 36, 42],
        help="Edge-node counts for the edge-resource sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for benchmark sweep artifacts.",
    )
    parser.add_argument(
        "--masac-run-dir",
        type=str,
        default=None,
        help="Optional run directory of the MA-SAC baseline to include in the sweep.",
    )
    parser.add_argument(
        "--masac-checkpoint",
        type=str,
        default="best_masac_actor.pt",
        help="Checkpoint filename for the MA-SAC baseline.",
    )
    parser.add_argument(
        "--mapdqn-run-dir",
        type=str,
        default=None,
        help="Optional run directory of the MA-P-DQN baseline to include in the sweep.",
    )
    parser.add_argument(
        "--mapdqn-checkpoint",
        type=str,
        default="best_mapdqn_actor.pt",
        help="Checkpoint filename for the MA-P-DQN baseline.",
    )
    return parser


def evaluate_policies(
    config,
    model,
    episodes: int,
    seed_base: int,
    masac_policy=None,
    mapdqn_policy=None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    policy_names = ["trained", "heuristic"]
    if masac_policy is not None:
        policy_names.append("masac")
    if mapdqn_policy is not None:
        policy_names.append("mapdqn")
    for policy_name in policy_names:
        env = SimplifiedRoutingEnv(config)
        metrics = rollout_policy(
            env=env,
            config=config,
            policy_name=policy_name,
            episodes=episodes,
            seed_base=seed_base,
            model=(
                model
                if policy_name == "trained"
                else masac_policy
                if policy_name == "masac"
                else mapdqn_policy
                if policy_name == "mapdqn"
                else None
            ),
        )
        if "global_load_balancing_index" in metrics and "deadline_hit_ratio" in metrics:
            metrics["qos_load_balancing_index"] = (
                float(metrics["global_load_balancing_index"]) * float(metrics["deadline_hit_ratio"])
            )
        rows.append({"policy": policy_name, **metrics})
    return rows


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
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not rows:
        return []

    group_keys = ("sweep_type", "policy", "x_value")
    metric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"repeat_id", "runtime_repeat_id", "topology_repeat_id", *group_keys}
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
            record[f"{metric}_mean"] = float(sum(values) / len(values))
            if len(values) > 1:
                mean = record[f"{metric}_mean"]
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                record[f"{metric}_std"] = float(variance ** 0.5)
            else:
                record[f"{metric}_std"] = 0.0
        aggregated.append(record)
    return aggregated


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "benchmark_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_run_config(run_dir)
    model = load_checkpoint(base_config, run_dir, args.checkpoint)
    masac_policy = None
    mapdqn_policy = None
    if args.masac_run_dir:
        masac_run_dir = Path(args.masac_run_dir)
        masac_train_config = load_masac_train_config(masac_run_dir, device=base_config.device)
        masac_policy = load_masac_actor(
            checkpoint_path=masac_run_dir / args.masac_checkpoint,
            env_config=base_config,
            train_config=masac_train_config,
        )
    if args.mapdqn_run_dir:
        mapdqn_run_dir = Path(args.mapdqn_run_dir)
        mapdqn_train_config = load_mapdqn_train_config(mapdqn_run_dir, device=base_config.device)
        mapdqn_policy = load_mapdqn_policy(
            checkpoint_path=mapdqn_run_dir / args.mapdqn_checkpoint,
            env_config=base_config,
            train_config=mapdqn_train_config,
        )

    default_rows: list[dict[str, float | str]] = []
    for topology_idx in range(args.num_topology_repeats):
        topology_config = replace(
            base_config,
            seed=base_config.seed + args.topology_seed_stride * topology_idx,
        )
        for repeat_idx in range(args.num_repeats):
            rows = evaluate_policies(
                topology_config,
                model,
                args.episodes,
                args.seed_base
                + args.topology_seed_stride * topology_idx
                + args.repeat_seed_stride * repeat_idx,
                masac_policy=masac_policy,
                mapdqn_policy=mapdqn_policy,
            )
            for row in rows:
                row["sweep_type"] = "default"
                row["x_value"] = 0.0
                row["topology_repeat_id"] = float(topology_idx)
                row["runtime_repeat_id"] = float(repeat_idx)
                row["repeat_id"] = float(topology_idx * args.num_repeats + repeat_idx)
            default_rows.extend(rows)

    active_flow_rows: list[dict[str, float | str]] = []
    for idx, flows in enumerate(args.active_flows):
        config = replace(base_config, active_flow_range=(flows, flows))
        for topology_idx in range(args.num_topology_repeats):
            topology_config = replace(
                config,
                seed=base_config.seed + args.topology_seed_stride * topology_idx,
            )
            for repeat_idx in range(args.num_repeats):
                rows = evaluate_policies(
                    topology_config,
                    model,
                    args.episodes,
                    args.seed_base
                    + 1_000 * (idx + 1)
                    + args.topology_seed_stride * topology_idx
                    + args.repeat_seed_stride * repeat_idx,
                    masac_policy=masac_policy,
                    mapdqn_policy=mapdqn_policy,
                )
                for row in rows:
                    row["sweep_type"] = "active_flows"
                    row["x_value"] = float(flows)
                    row["topology_repeat_id"] = float(topology_idx)
                    row["runtime_repeat_id"] = float(repeat_idx)
                    row["repeat_id"] = float(topology_idx * args.num_repeats + repeat_idx)
                active_flow_rows.extend(rows)

    edge_node_rows: list[dict[str, float | str]] = []
    edge_layout_reference_count = max(args.edge_nodes) if args.edge_nodes else int(base_config.num_edges)
    for idx, edges in enumerate(args.edge_nodes):
        access_edge_degree = min(
            edges,
            max(int(base_config.access_edge_degree), 2),
        )
        edge_neighbor_degree = min(
            max(edges - 1, 1),
            max(int(base_config.edge_neighbor_degree), 2),
        )
        config = replace(
            base_config,
            num_edges=edges,
            access_edge_degree=access_edge_degree,
            edge_neighbor_degree=edge_neighbor_degree,
            edge_placement_mode="nested_grid",
            edge_layout_reference_count=edge_layout_reference_count,
        )
        for topology_idx in range(args.num_topology_repeats):
            topology_config = replace(
                config,
                seed=base_config.seed + args.topology_seed_stride * topology_idx,
            )
            for repeat_idx in range(args.num_repeats):
                rows = evaluate_policies(
                    topology_config,
                    model,
                    args.episodes,
                    args.seed_base
                    + 10_000
                    + 1_000 * (idx + 1)
                    + args.topology_seed_stride * topology_idx
                    + args.repeat_seed_stride * repeat_idx,
                    masac_policy=masac_policy,
                    mapdqn_policy=mapdqn_policy,
                )
                for row in rows:
                    row["sweep_type"] = "edge_nodes"
                    row["x_value"] = float(edges)
                    row["topology_repeat_id"] = float(topology_idx)
                    row["runtime_repeat_id"] = float(repeat_idx)
                    row["repeat_id"] = float(topology_idx * args.num_repeats + repeat_idx)
                edge_node_rows.extend(rows)

    all_rows = default_rows + active_flow_rows + edge_node_rows
    default_rows_agg = aggregate_rows(default_rows)
    active_flow_rows_agg = aggregate_rows(active_flow_rows)
    edge_node_rows_agg = aggregate_rows(edge_node_rows)
    all_rows_agg = default_rows_agg + active_flow_rows_agg + edge_node_rows_agg
    write_csv(default_rows, output_dir / "default_comparison.csv")
    write_csv(active_flow_rows, output_dir / "active_flow_sweep.csv")
    write_csv(edge_node_rows, output_dir / "edge_node_sweep.csv")
    write_csv(all_rows, output_dir / "benchmark_sweeps_all.csv")
    write_csv(default_rows_agg, output_dir / "default_comparison_aggregated.csv")
    write_csv(active_flow_rows_agg, output_dir / "active_flow_sweep_aggregated.csv")
    write_csv(edge_node_rows_agg, output_dir / "edge_node_sweep_aggregated.csv")
    write_csv(all_rows_agg, output_dir / "benchmark_sweeps_all_aggregated.csv")

    payload = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "num_repeats": args.num_repeats,
        "repeat_seed_stride": args.repeat_seed_stride,
        "num_topology_repeats": args.num_topology_repeats,
        "topology_seed_stride": args.topology_seed_stride,
        "active_flows": args.active_flows,
        "edge_nodes": args.edge_nodes,
        "masac_run_dir": args.masac_run_dir,
        "masac_checkpoint": args.masac_checkpoint,
        "mapdqn_run_dir": args.mapdqn_run_dir,
        "mapdqn_checkpoint": args.mapdqn_checkpoint,
    }
    with (output_dir / "benchmark_sweeps_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Benchmark sweeps finished.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
