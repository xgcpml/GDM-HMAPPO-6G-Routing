from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from config import ExperimentConfig, apply_experiment_preset
from masac import MASACConfig, train_masac


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimum viable MA-SAC benchmark for the 6G routing task.")
    parser.add_argument("--preset", type=str, default="paper_curve_faithful_long4000")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-episodes", type=int, default=4000)
    parser.add_argument("--eval-interval-episodes", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--actor-lr", type=float, default=2.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--route-alpha", type=float, default=0.08)
    parser.add_argument("--alloc-alpha", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    env_config = ExperimentConfig()
    if args.preset:
        env_config = apply_experiment_preset(env_config, args.preset)
    if args.seed is not None:
        env_config = replace(env_config, seed=args.seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env_config = replace(env_config, device=device, output_dir=args.output_dir)

    train_config = MASACConfig(
        train_episodes=args.train_episodes,
        eval_interval_episodes=args.eval_interval_episodes,
        eval_episodes=args.eval_episodes,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        route_alpha=args.route_alpha,
        alloc_alpha=args.alloc_alpha,
        updates_per_env_step=args.updates_per_env_step,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, final_metrics = train_masac(env_config=env_config, train_config=train_config, output_dir=output_dir)
    print("MA-SAC training finished.")
    print(f"Output directory: {output_dir.resolve()}")
    print(
        f"Final metrics: latency={final_metrics['latency']:.4f}, "
        f"deadline_hit_ratio={final_metrics['deadline_hit_ratio']:.4f}, "
        f"ALVR={final_metrics['latency_violation_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
