from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch

from config import ExperimentConfig, apply_experiment_preset
from fedroute import (
    FedRouteConfig,
    apply_fedroute_variant,
    train_fedroute,
)
from experiment_protocol import write_run_manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the adapted federated FedRoute baseline."
    )
    parser.add_argument("--preset", default="v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--federated-rounds", type=int, default=125)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--local-updates", type=int, default=1)
    parser.add_argument("--aggregation-interval", type=int, default=1)
    parser.add_argument("--client-rollout-episodes", type=int, default=8)
    parser.add_argument(
        "--aggregation-scope",
        choices=("route", "policy", "full"),
        default="policy",
    )
    parser.add_argument(
        "--client-load-mode",
        choices=("partitioned", "shared"),
        default="partitioned",
    )
    parser.add_argument(
        "--client-participation-rate",
        type=float,
        default=1.0,
        help="Fraction of clients selected per aggregation window.",
    )
    parser.add_argument(
        "--participation-schedule",
        choices=("window", "round_robin"),
        default="window",
    )
    parser.add_argument(
        "--local-graph-context",
        action="store_true",
        help=(
            "Restrict each federated client to candidate-local observations "
            "instead of the globally pooled graph context."
        ),
    )
    parser.add_argument(
        "--reset-optimizer-on-aggregation",
        action="store_true",
        help="Reset Adam state for parameters overwritten by aggregation.",
    )
    parser.add_argument(
        "--decoupled-allocation-adapter",
        action="store_true",
        help="Pretrain and freeze a separate Direct-MLP allocation adapter.",
    )
    parser.add_argument("--adapter-pretrain-updates", type=int, default=0)
    parser.add_argument("--adapter-pretrain-batch-size", type=positive_int, default=256)
    parser.add_argument(
        "--adapter-pretrain-active-flows",
        type=positive_int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--hidden-dim", type=positive_int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--deadline-budget-factor", type=float, default=None)
    parser.add_argument("--topology-seed-stride", type=int, default=None)
    parser.add_argument("--exploration-anneal-updates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if (
        args.adapter_pretrain_active_flows is not None
        and args.adapter_pretrain_active_flows[0]
        > args.adapter_pretrain_active_flows[1]
    ):
        raise SystemExit("adapter pretraining flow range must satisfy LOW <= HIGH")
    started_at = datetime.now(timezone.utc).isoformat()
    env_config = apply_experiment_preset(ExperimentConfig(), args.preset)
    overrides = {
        "output_dir": args.output_dir,
        "device": args.device
        or ("cuda" if torch.cuda.is_available() else env_config.device),
    }
    for argument, field in (
        (args.episode_length, "episode_length"),
        (args.hidden_dim, "hidden_dim"),
        (args.eval_episodes, "eval_episodes"),
        (args.validation_episodes, "train_eval_episodes"),
        (args.eval_interval, "eval_interval"),
        (args.ppo_epochs, "ppo_epochs"),
        (args.deadline_budget_factor, "deadline_budget_factor"),
        (args.topology_seed_stride, "topology_seed_stride"),
        (args.exploration_anneal_updates, "exploration_anneal_updates"),
        (args.seed, "seed"),
    ):
        if argument is not None:
            overrides[field] = argument
    env_config = apply_fedroute_variant(replace(env_config, **overrides))
    if args.local_graph_context:
        env_config = replace(env_config, use_global_graph_context=False)
    fed_config = FedRouteConfig(
        num_clients=args.num_clients,
        federated_rounds=args.federated_rounds,
        local_updates=args.local_updates,
        aggregation_interval=args.aggregation_interval,
        client_rollout_episodes=args.client_rollout_episodes,
        aggregation_scope=args.aggregation_scope,
        client_load_mode=args.client_load_mode,
        client_participation_rate=args.client_participation_rate,
        participation_schedule=args.participation_schedule,
        reset_optimizer_on_aggregation=args.reset_optimizer_on_aggregation,
        decoupled_allocation_adapter=args.decoupled_allocation_adapter,
        adapter_pretrain_updates=args.adapter_pretrain_updates,
        adapter_pretrain_batch_size=args.adapter_pretrain_batch_size,
        adapter_pretrain_flow_range=(
            tuple(args.adapter_pretrain_active_flows)
            if args.adapter_pretrain_active_flows is not None
            else None
        ),
    )
    output_dir = Path(args.output_dir)
    _, _, final_metrics = train_fedroute(env_config, fed_config, output_dir)
    write_run_manifest(
        output_dir,
        env_config,
        command=[sys.executable, *sys.argv],
        started_at=started_at,
        checkpoint=fed_config.best_checkpoint_name,
        artifacts=(
            fed_config.checkpoint_name,
            fed_config.best_checkpoint_name,
            "federated_history.csv",
            "client_history.csv",
            "evaluation_metrics.json",
        ),
    )
    print("FedRoute training finished.")
    print(f"Output directory: {output_dir.resolve()}")
    print(
        f"Final metrics: latency={final_metrics['latency']:.4f}, "
        f"deadline_hit_ratio={final_metrics['deadline_hit_ratio']:.4f}, "
        f"ALVR={final_metrics['latency_violation_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
