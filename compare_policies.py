from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from evaluation_tools import load_checkpoint, load_run_config, rollout_policy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trained and baseline routing policies.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directory containing checkpoints and metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_routing_model.pt",
        help="Checkpoint filename to evaluate as the trained policy.",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Number of evaluation episodes.")
    parser.add_argument("--seed-base", type=int, default=50_000, help="Base seed for evaluation runs.")
    parser.add_argument(
        "--alloc-search-candidates",
        type=int,
        default=None,
        help="Override the number of deterministic allocation candidates for the trained policy.",
    )
    parser.add_argument(
        "--route-search-candidates",
        type=int,
        default=None,
        help="Override the number of joint route candidates for the trained policy.",
    )
    parser.add_argument(
        "--route-search-prior",
        type=float,
        default=None,
        help="Override the route-policy prior weight used for the trained policy search.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path.")
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

    model = load_checkpoint(config, run_dir, args.checkpoint)

    policy_names = ("trained", "heuristic", "random")
    results = []
    for policy_name in policy_names:
        env = SimplifiedRoutingEnv(config)
        metrics = rollout_policy(
            env=env,
            config=config,
            policy_name=policy_name,
            episodes=args.episodes,
            seed_base=args.seed_base,
            model=model if policy_name == "trained" else None,
        )
        results.append({"policy": policy_name, **metrics})

    json_path = Path(args.output_json) if args.output_json else run_dir / "policy_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_dir": str(run_dir.resolve()),
                "checkpoint": args.checkpoint,
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "results": results,
            },
            handle,
            indent=2,
        )

    csv_path = Path(args.output_csv) if args.output_csv else run_dir / "policy_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Policy comparison finished.")
    print(f"Run directory: {run_dir.resolve()}")
    for row in results:
        print(
            f"{row['policy']}: latency={row['latency']:.4f}, "
            f"deadline_hit_ratio={row['deadline_hit_ratio']:.4f}"
        )
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
