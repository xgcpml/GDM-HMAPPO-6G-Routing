from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from evaluation_tools import load_checkpoint, load_run_config, rollout_policy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained routing policy checkpoint.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing checkpoints and metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_routing_model.pt",
        help="Checkpoint filename inside the run directory.",
    )
    parser.add_argument("--episodes", type=int, default=8, help="Number of evaluation episodes.")
    parser.add_argument("--seed-base", type=int, default=40_000, help="Base seed for deterministic evaluation.")
    parser.add_argument(
        "--alloc-search-candidates",
        type=int,
        default=None,
        help="Override the number of deterministic allocation candidates at inference time.",
    )
    parser.add_argument(
        "--route-search-candidates",
        type=int,
        default=None,
        help="Override the number of joint route candidates at inference time.",
    )
    parser.add_argument(
        "--route-search-prior",
        type=float,
        default=None,
        help="Override the route-policy prior weight used in joint deterministic search.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir)

    config = load_run_config(run_dir)
    if args.alloc_search_candidates is not None:
        config = replace(config, allocation_search_candidates=args.alloc_search_candidates)
    if args.route_search_candidates is not None:
        config = replace(config, route_search_candidates=args.route_search_candidates)
    if args.route_search_prior is not None:
        config = replace(config, route_search_prior_coef=args.route_search_prior)
    env = SimplifiedRoutingEnv(config)
    model = load_checkpoint(config, run_dir, args.checkpoint)

    metrics = rollout_policy(
        env=env,
        config=config,
        policy_name="trained",
        episodes=args.episodes,
        seed_base=args.seed_base,
        model=model,
    )

    output_payload = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        **metrics,
    }

    output_path = Path(args.output) if args.output else run_dir / "standalone_evaluation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print("Standalone evaluation finished.")
    print(f"Run directory: {run_dir.resolve()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Latency: {metrics['latency']:.4f}")
    print(f"Deadline hit ratio: {metrics['deadline_hit_ratio']:.4f}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
