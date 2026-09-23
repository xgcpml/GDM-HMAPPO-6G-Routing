from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch

from config import ExperimentConfig, apply_experiment_preset
from mapdqn import MAPDQNConfig, apply_mapdqn_variant, train_mapdqn
from experiment_protocol import write_run_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimum viable MA-P-DQN benchmark for the 6G routing task.")
    parser.add_argument("--preset", type=str, default="v2")
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
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=600)
    parser.add_argument("--updates-per-env-step", type=int, default=1)
    parser.add_argument("--update-interval-slots", type=int, default=1)
    parser.add_argument("--deadline-budget-factor", type=float, default=None)
    parser.add_argument("--topology-seed-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    env_config = ExperimentConfig()
    if args.preset:
        env_config = apply_experiment_preset(env_config, args.preset)
    if args.seed is not None:
        env_config = replace(env_config, seed=args.seed)
    if args.deadline_budget_factor is not None:
        env_config = replace(
            env_config,
            deadline_budget_factor=args.deadline_budget_factor,
        )
    if args.topology_seed_stride is not None:
        env_config = replace(env_config, topology_seed_stride=args.topology_seed_stride)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env_config = apply_mapdqn_variant(
        replace(env_config, device=device, output_dir=args.output_dir)
    )

    train_config = MAPDQNConfig(
        train_episodes=args.train_episodes,
        eval_interval_episodes=args.eval_interval_episodes,
        eval_episodes=args.eval_episodes,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        updates_per_env_step=args.updates_per_env_step,
        update_interval_slots=args.update_interval_slots,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, final_metrics = train_mapdqn(env_config=env_config, train_config=train_config, output_dir=output_dir)
    write_run_manifest(
        output_dir,
        env_config,
        command=[sys.executable, *sys.argv],
        started_at=started_at,
        checkpoint=train_config.best_checkpoint_name,
        artifacts=(
            train_config.checkpoint_name,
            train_config.best_checkpoint_name,
            "training_history.csv",
            "rollout_episode_history.csv",
            "evaluation_metrics.json",
        ),
    )
    print("MA-P-DQN training finished.")
    print(f"Output directory: {output_dir.resolve()}")
    print(
        f"Final metrics: latency={final_metrics['latency']:.4f}, "
        f"deadline_hit_ratio={final_metrics['deadline_hit_ratio']:.4f}, "
        f"ALVR={final_metrics['latency_violation_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
