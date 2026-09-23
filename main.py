from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch

from config import ExperimentConfig, apply_experiment_preset
from env import SimplifiedRoutingEnv
from model import HybridRoutingPolicy
from ppo import PPOTrainer
from experiment_protocol import write_run_manifest
from utils import (
    save_history_csv,
    save_metrics_json,
    set_seed,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simplified PyTorch routing simulator for the proposed 6G edge-cloud policy."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Experiment preset. Use v2 for the complete default setting.",
    )
    parser.add_argument("--updates", type=int, default=None, help="Number of PPO updates.")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Initialize policy weights from an existing checkpoint.",
    )
    parser.add_argument(
        "--route-head-refinement",
        action="store_true",
        help="Freeze encoders and allocation head while refining route/value heads.",
    )
    parser.add_argument(
        "--variant",
        choices=("proposed", "graphpr"),
        default="proposed",
        help="Policy architecture variant.",
    )
    parser.add_argument("--warmup-updates", type=int, default=None, help="Warm-up updates before PPO training.")
    parser.add_argument(
        "--search-bootstrap-updates",
        type=int,
        default=None,
        help="Search-bootstrap updates before PPO training.",
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=None,
        help="Episodes collected per PPO update.",
    )
    parser.add_argument("--episode-length", type=int, default=None, help="Steps per rollout episode.")
    parser.add_argument(
        "--active-flows",
        type=positive_int,
        default=None,
        help="Fix the number of active task flows in every slot.",
    )
    parser.add_argument(
        "--active-flow-range",
        nargs=2,
        type=positive_int,
        metavar=("MIN", "MAX"),
        default=None,
        help="Sample active task flows from a shared training range.",
    )
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes.")
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=None,
        help="Validation episodes used during training for convergence curves and checkpoint selection.",
    )
    parser.add_argument(
        "--parallel-rollout-envs",
        type=int,
        default=None,
        help="Number of rollout environment workers.",
    )
    parser.add_argument(
        "--parallel-eval-envs",
        type=int,
        default=None,
        help="Number of evaluation environment workers.",
    )
    parser.add_argument("--eval-interval", type=int, default=None, help="Validation interval during training.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--shared-lr-factor", type=float, default=None, help="Shared encoder learning-rate multiplier.")
    parser.add_argument("--actor-lr-factor", type=float, default=None, help="Actor learning-rate multiplier.")
    parser.add_argument("--critic-lr-factor", type=float, default=None, help="Critic learning-rate multiplier.")
    parser.add_argument("--hidden-dim", type=positive_int, default=None, help="Policy hidden dimension override.")
    parser.add_argument("--graph-layers", type=positive_int, default=None, help="Number of graph layers override.")
    parser.add_argument(
        "--alloc-refinement-steps",
        type=positive_int,
        default=None,
        help="Number of diffusion denoising steps for training and inference.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=None, help="PPO epochs override.")
    parser.add_argument("--clip-eps", type=float, default=None, help="PPO policy clipping threshold.")
    parser.add_argument("--target-kl", type=float, default=None, help="PPO early-stop KL threshold.")
    parser.add_argument(
        "--disable-plateau-control",
        action="store_true",
        help="Disable validation-triggered rollback and parameter freezing.",
    )
    parser.add_argument(
        "--local-graph-context",
        action="store_true",
        help="Use source/path graph embeddings without a pooled global graph summary.",
    )
    parser.add_argument(
        "--graph-native-candidate-context",
        action="store_true",
        help="Infer compute-resource context from graph nodes instead of engineered path summaries.",
    )
    parser.add_argument(
        "--stochastic-allocation-rollout",
        dest="rollout_deterministic_allocation",
        action="store_false",
        default=None,
        help="Sample GDM allocation actions during training while keeping evaluation deterministic.",
    )
    parser.add_argument("--route-entropy", type=float, default=None, help="Route entropy coefficient override.")
    parser.add_argument("--alloc-noise-scale", type=float, default=None, help="Initial allocation exploration-noise scale.")
    parser.add_argument("--alloc-noise-final-scale", type=float, default=None, help="Final allocation exploration-noise scale.")
    parser.add_argument("--adv-weight-eta", type=float, default=None, help="Advantage-weight temperature for allocation learning.")
    parser.add_argument("--adv-weight-min", type=float, default=None, help="Minimum allocation advantage weight.")
    parser.add_argument("--adv-weight-max", type=float, default=None, help="Maximum allocation advantage weight.")
    parser.add_argument(
        "--exploration-anneal-updates",
        type=int,
        default=None,
        help="Complete exploration annealing after this many PPO updates.",
    )
    parser.add_argument(
        "--learning-rate-anneal-updates",
        type=int,
        default=None,
        help="Complete learning-rate annealing after this many PPO updates; zero follows exploration.",
    )
    parser.add_argument(
        "--learning-rate-warmup-updates",
        type=int,
        default=None,
        help="Linearly warm up the optimizer learning rate over the initial PPO updates.",
    )
    parser.add_argument(
        "--warmup-route-coef",
        type=float,
        default=None,
        help="Route supervised weight used during warmup updates.",
    )
    parser.add_argument(
        "--search-bootstrap-route-coef",
        type=float,
        default=None,
        help="Route supervised weight used during search-bootstrap updates.",
    )
    parser.add_argument(
        "--topology-prior-initial",
        type=float,
        default=None,
        help="Initial topology prior scale injected into route logits.",
    )
    parser.add_argument(
        "--topology-prior-final",
        type=float,
        default=None,
        help="Final topology prior scale injected into route logits.",
    )
    parser.add_argument(
        "--topology-distill",
        type=float,
        default=None,
        help="Initial topology distillation weight.",
    )
    parser.add_argument(
        "--topology-distill-final",
        type=float,
        default=None,
        help="Final topology distillation weight.",
    )
    parser.add_argument(
        "--deadline-budget-factor",
        type=float,
        default=None,
        help="Global scaling factor applied to sampled task deadlines.",
    )
    parser.add_argument(
        "--topology-seed-stride",
        type=int,
        default=None,
        help="Use fixed, distinct topology realizations across parallel train/eval environments.",
    )
    parser.add_argument(
        "--alloc-search-candidates",
        type=int,
        default=None,
        help="Number of deterministic allocation candidates evaluated during inference.",
    )
    parser.add_argument(
        "--route-search-candidates",
        type=int,
        default=None,
        help="Number of route candidates jointly refined during deterministic inference.",
    )
    parser.add_argument(
        "--route-search-prior",
        type=float,
        default=None,
        help="Weight of the route-policy prior inside joint deterministic search.",
    )
    parser.add_argument(
        "--gdm-only-search",
        action="store_true",
        help="Select only among policy routes and GDM-generated allocations.",
    )
    parser.add_argument(
        "--search-distill-route",
        type=float,
        default=None,
        help="Auxiliary route distillation weight from deterministic joint search targets.",
    )
    parser.add_argument(
        "--search-distill-diffusion",
        type=float,
        default=None,
        help="Auxiliary diffusion distillation weight from deterministic joint search targets.",
    )
    parser.add_argument(
        "--search-distill-relative-threshold",
        type=float,
        default=None,
        help="Minimum relative gain required before search-guided distillation becomes active.",
    )
    parser.add_argument(
        "--search-distill-max-weight",
        type=float,
        default=None,
        help="Maximum per-sample weight used by search-guided distillation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for checkpoints and metrics.")
    return parser


def apply_policy_variant(config: ExperimentConfig, variant: str) -> ExperimentConfig:
    if variant == "proposed":
        return config
    common = dict(
        use_topology_route_prior=False,
        use_topology_distillation=False,
        use_search_distillation=False,
        use_search_guided_inference=False,
        topology_route_prior_initial_scale=0.0,
        topology_route_prior_final_scale=0.0,
        topology_distill_coef=0.0,
        topology_distill_final_coef=0.0,
        search_distill_route_coef=0.0,
        search_distill_diffusion_coef=0.0,
        allocation_search_candidates=1,
        route_search_candidates=1,
        route_search_prior_coef=0.0,
    )
    if variant == "graphpr":
        return replace(
            config,
            method_name="GraphPR",
            use_gnn_encoder=True,
            use_graph_attention_encoder=True,
            use_transformer_context=False,
            use_gdm_allocator=False,
            allocation_policy_family="direct",
            **common,
        )
    raise ValueError(f"Unsupported policy variant: {variant}")


def initialize_model_from_checkpoint(
    model: HybridRoutingPolicy, checkpoint_path: str, device: str
) -> None:
    payload = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)


def configure_route_head_refinement(model: HybridRoutingPolicy) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.route_head, model.value_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    config = ExperimentConfig()
    if args.preset is not None:
        config = apply_experiment_preset(config, args.preset)
    if args.updates is not None:
        config = replace(config, train_updates=args.updates)
    if args.warmup_updates is not None:
        config = replace(config, warmup_updates=args.warmup_updates)
    if args.search_bootstrap_updates is not None:
        config = replace(config, search_bootstrap_updates=args.search_bootstrap_updates)
    if args.rollout_episodes is not None:
        config = replace(config, rollout_episodes_per_update=args.rollout_episodes)
    if args.episode_length is not None:
        config = replace(config, episode_length=args.episode_length)
    if args.active_flows is not None and args.active_flow_range is not None:
        parser.error("--active-flows and --active-flow-range are mutually exclusive")
    if args.active_flows is not None:
        config = replace(config, active_flow_range=(args.active_flows, args.active_flows))
    if args.active_flow_range is not None:
        flow_min, flow_max = args.active_flow_range
        if flow_min > flow_max:
            parser.error("--active-flow-range requires MIN <= MAX")
        config = replace(config, active_flow_range=(flow_min, flow_max))
    if args.eval_episodes is not None:
        config = replace(config, eval_episodes=args.eval_episodes)
    if args.validation_episodes is not None:
        config = replace(config, train_eval_episodes=args.validation_episodes)
    if args.parallel_rollout_envs is not None:
        config = replace(config, parallel_rollout_envs=args.parallel_rollout_envs)
    if args.parallel_eval_envs is not None:
        config = replace(config, parallel_eval_envs=args.parallel_eval_envs)
    if args.eval_interval is not None:
        config = replace(config, eval_interval=args.eval_interval)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    if args.shared_lr_factor is not None:
        config = replace(config, shared_lr_factor=args.shared_lr_factor)
    if args.actor_lr_factor is not None:
        config = replace(config, actor_lr_factor=args.actor_lr_factor)
    if args.critic_lr_factor is not None:
        config = replace(config, critic_lr_factor=args.critic_lr_factor)
    if args.hidden_dim is not None:
        config = replace(config, hidden_dim=args.hidden_dim)
    if args.graph_layers is not None:
        config = replace(config, graph_layers=args.graph_layers)
    if args.alloc_refinement_steps is not None:
        config = replace(config, alloc_refinement_steps=args.alloc_refinement_steps)
    if args.ppo_epochs is not None:
        config = replace(config, ppo_epochs=args.ppo_epochs)
    if args.clip_eps is not None:
        config = replace(config, clip_eps=args.clip_eps)
    if args.target_kl is not None:
        config = replace(config, target_kl=args.target_kl)
    if args.disable_plateau_control:
        config = replace(
            config,
            plateau_patience_updates=0,
            max_rollbacks=0,
            freeze_after_rollbacks=False,
        )
    if args.local_graph_context:
        config = replace(config, use_global_graph_context=False)
    if args.graph_native_candidate_context:
        config = replace(config, use_candidate_resource_summaries=False)
    if args.rollout_deterministic_allocation is not None:
        config = replace(
            config,
            rollout_deterministic_allocation=args.rollout_deterministic_allocation,
        )
    if args.route_entropy is not None:
        config = replace(config, route_entropy_coef=args.route_entropy)
    if args.alloc_noise_scale is not None:
        config = replace(config, alloc_noise_scale=args.alloc_noise_scale)
    if args.alloc_noise_final_scale is not None:
        config = replace(config, alloc_noise_final_scale=args.alloc_noise_final_scale)
    if args.adv_weight_eta is not None:
        config = replace(config, adv_weight_eta=args.adv_weight_eta)
    if args.adv_weight_min is not None:
        config = replace(config, adv_weight_min=args.adv_weight_min)
    if args.adv_weight_max is not None:
        config = replace(config, adv_weight_max=args.adv_weight_max)
    if args.exploration_anneal_updates is not None:
        if args.exploration_anneal_updates < 0:
            parser.error("--exploration-anneal-updates must be non-negative")
        config = replace(
            config,
            exploration_anneal_updates=args.exploration_anneal_updates,
        )
    if args.learning_rate_anneal_updates is not None:
        if args.learning_rate_anneal_updates < 0:
            parser.error("--learning-rate-anneal-updates must be non-negative")
        config = replace(
            config,
            learning_rate_anneal_updates=args.learning_rate_anneal_updates,
        )
    if args.learning_rate_warmup_updates is not None:
        if args.learning_rate_warmup_updates < 0:
            parser.error("--learning-rate-warmup-updates must be non-negative")
        config = replace(
            config,
            learning_rate_warmup_updates=args.learning_rate_warmup_updates,
        )
    if args.warmup_route_coef is not None:
        config = replace(config, warmup_route_coef=args.warmup_route_coef)
    if args.search_bootstrap_route_coef is not None:
        config = replace(config, search_bootstrap_route_coef=args.search_bootstrap_route_coef)
    if args.topology_prior_initial is not None:
        config = replace(config, topology_route_prior_initial_scale=args.topology_prior_initial)
    if args.topology_prior_final is not None:
        config = replace(config, topology_route_prior_final_scale=args.topology_prior_final)
    if args.topology_distill is not None:
        config = replace(config, topology_distill_coef=args.topology_distill)
    if args.topology_distill_final is not None:
        config = replace(config, topology_distill_final_coef=args.topology_distill_final)
    if args.deadline_budget_factor is not None:
        config = replace(config, deadline_budget_factor=args.deadline_budget_factor)
    if args.topology_seed_stride is not None:
        if args.topology_seed_stride < 0:
            parser.error("--topology-seed-stride must be non-negative")
        config = replace(config, topology_seed_stride=args.topology_seed_stride)
    if args.alloc_search_candidates is not None:
        config = replace(config, allocation_search_candidates=args.alloc_search_candidates)
    if args.route_search_candidates is not None:
        config = replace(config, route_search_candidates=args.route_search_candidates)
    if args.route_search_prior is not None:
        config = replace(config, route_search_prior_coef=args.route_search_prior)
    if args.gdm_only_search:
        config = replace(config, use_heuristic_search_candidates=False)
    if args.search_distill_route is not None:
        config = replace(config, search_distill_route_coef=args.search_distill_route)
    if args.search_distill_diffusion is not None:
        config = replace(config, search_distill_diffusion_coef=args.search_distill_diffusion)
    if args.search_distill_relative_threshold is not None:
        config = replace(config, search_distill_relative_threshold=args.search_distill_relative_threshold)
    if args.search_distill_max_weight is not None:
        config = replace(config, search_distill_max_weight=args.search_distill_max_weight)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)

    config = apply_policy_variant(config, args.variant)

    if args.device is None and config.device == "cpu" and torch.cuda.is_available():
        config = replace(config, device="cuda")

    config.output_path.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    if config.device.startswith("cuda") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    env = SimplifiedRoutingEnv(config)
    model = HybridRoutingPolicy(
        node_feature_dim=config.node_feature_dim,
        task_feature_dim=config.task_feature_dim,
        candidate_feature_dim=config.candidate_feature_dim,
        hidden_dim=config.hidden_dim,
        num_graph_layers=config.graph_layers,
        num_transformer_layers=config.transformer_layers,
        attention_heads=config.attention_heads,
        alloc_dim=config.alloc_dim,
        alloc_refinement_steps=config.alloc_refinement_steps,
        alloc_noise_scale=config.alloc_noise_scale,
        diffusion_beta_start=config.diffusion_beta_start,
        diffusion_beta_end=config.diffusion_beta_end,
        diffusion_latent_clip=config.diffusion_latent_clip,
        allocation_search_candidates=config.allocation_search_candidates,
        route_search_candidates=config.route_search_candidates,
        route_search_prior_coef=config.route_search_prior_coef,
        surrogate_penalty_coef=config.penalty_coeff,
        use_topology_route_prior=config.use_topology_route_prior,
        use_search_guided_inference=config.use_search_guided_inference,
        use_heuristic_search_candidates=config.use_heuristic_search_candidates,
        use_gnn_encoder=config.use_gnn_encoder,
        use_graph_attention_encoder=config.use_graph_attention_encoder,
        use_transformer_context=config.use_transformer_context,
        use_global_graph_context=config.use_global_graph_context,
        use_candidate_resource_summaries=config.use_candidate_resource_summaries,
        use_gdm_allocator=config.use_gdm_allocator,
        allocation_policy_family=config.allocation_policy_family,
    )
    if args.init_checkpoint is not None:
        initialize_model_from_checkpoint(model, args.init_checkpoint, config.device)
    if args.route_head_refinement:
        configure_route_head_refinement(model)

    trainer = PPOTrainer(config, env, model)
    try:
        warmup_history = trainer.warmup()
        history = trainer.train()
        final_evaluation = trainer.evaluate(config.eval_episodes)
    finally:
        trainer.close()
    validation_history = []
    paper_episode_history = []
    combined_history = warmup_history + history
    for stage_idx, row in enumerate(combined_history, start=1):
        if "eval_latency" not in row:
            continue
        validation_history.append(
            {
                "stage": float(stage_idx),
                "phase": row.get("phase", "ppo"),
                "eval_reward": row.get("eval_reward", float("nan")),
                "eval_latency": row.get("eval_latency", float("nan")),
                "eval_deadline_hit_ratio": row.get("eval_deadline_hit_ratio", float("nan")),
            }
        )
    for row in history:
        eval_reward = row.get("eval_reward", float("nan"))
        if eval_reward != eval_reward:
            continue
        update = float(row.get("update", len(paper_episode_history) + 1))
        paper_episode_history.append(
            {
                "update": update,
                "episode_in_update": 1.0,
                "episode_index": update * float(config.rollout_episodes_per_update),
                "episode_reward": float(eval_reward),
                "episode_length": float(config.rollout_episodes_per_update),
                "episode_latency": float(row.get("eval_latency", float("nan"))),
                "episode_deadline_hit_ratio": float(row.get("eval_deadline_hit_ratio", float("nan"))),
                "episode_violation_mean": float(row.get("violation_mean", float("nan"))),
                "episode_path_available_ratio": float(row.get("path_available_ratio", float("nan"))),
            }
        )

    torch.save(model.state_dict(), config.output_path / "routing_model.pt")
    torch.save(model.state_dict(), config.output_path / "routing_model_final.pt")
    if warmup_history:
        save_history_csv(warmup_history, config.output_path / "warmup_history.csv")
    save_history_csv(history, config.output_path / "training_history.csv")
    if paper_episode_history:
        save_history_csv(paper_episode_history, config.output_path / "episode_history.csv")
    if validation_history:
        save_history_csv(validation_history, config.output_path / "full_validation_history.csv")
    if trainer.episode_history:
        save_history_csv(trainer.episode_history, config.output_path / "rollout_episode_history.csv")
    if trainer.eval_episode_history:
        save_history_csv(trainer.eval_episode_history, config.output_path / "evaluation_episode_history.csv")
    best_checkpoint_metrics = trainer.best_checkpoint_info.copy()
    best_model_path = config.output_path / "best_routing_model.pt"
    best_evaluation = {}
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_evaluation = trainer.evaluate(config.eval_episodes)

    save_metrics_json(
        {
            **asdict(config),
            **final_evaluation,
            **{f"best_{k}": v for k, v in best_evaluation.items()},
            **best_checkpoint_metrics,
        },
        config.output_path / "evaluation_metrics.json",
    )
    artifact_names = [
        "routing_model.pt",
        "routing_model_final.pt",
        "training_history.csv",
        "rollout_episode_history.csv",
        "evaluation_metrics.json",
    ]
    if best_model_path.exists():
        artifact_names.append("best_routing_model.pt")
    write_run_manifest(
        config.output_path,
        config,
        command=[sys.executable, *sys.argv],
        started_at=started_at,
        checkpoint="best_routing_model.pt" if best_model_path.exists() else "routing_model.pt",
        artifacts=artifact_names,
    )
    final_row = history[-1] if history else {}
    best_eval_latency = float(best_checkpoint_metrics.get("best_eval_latency", float("nan")))
    print("Training finished.")
    print(f"Output directory: {config.output_path.resolve()}")
    if warmup_history:
        print(f"Heuristic warmup updates: {config.warmup_updates}")
        print(f"Search bootstrap updates: {config.search_bootstrap_updates}")
    print(f"Final training reward: {final_row.get('reward_mean', float('nan')):.4f}")
    print(f"Final training latency: {final_row.get('latency_mean', float('nan')):.4f}")
    print(f"Best validation reward: {trainer.best_checkpoint_info.get('best_eval_reward', float('nan')):.4f}")
    print(f"Best validation latency: {best_eval_latency:.4f}")
    print(f"Final model latency: {final_evaluation['eval_latency']:.4f}")
    print(f"Final model deadline hit ratio: {final_evaluation['eval_deadline_hit_ratio']:.4f}")
    if best_evaluation:
        print(f"Best checkpoint update: {best_checkpoint_metrics['best_update']}")
        print(f"Best checkpoint reward: {best_evaluation['eval_reward']:.4f}")
        print(f"Best checkpoint latency: {best_evaluation['eval_latency']:.4f}")
        print(f"Best checkpoint deadline hit ratio: {best_evaluation['eval_deadline_hit_ratio']:.4f}")


if __name__ == "__main__":
    main()
