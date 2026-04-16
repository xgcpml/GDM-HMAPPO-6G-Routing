from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from baselines import heuristic_action, random_action
from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from model import HybridRoutingPolicy


MetricDict = dict[str, float]


def load_run_config(run_dir: Path) -> ExperimentConfig:
    metrics_path = run_dir / "evaluation_metrics.json"
    config = ExperimentConfig(output_dir=str(run_dir))
    if not metrics_path.exists():
        return config

    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    valid_keys = {field.name for field in fields(ExperimentConfig)}
    filtered = {key: payload[key] for key in valid_keys if key in payload}
    filtered["output_dir"] = str(run_dir)
    return ExperimentConfig(**filtered)


def build_model(config: ExperimentConfig) -> HybridRoutingPolicy:
    return HybridRoutingPolicy(
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
        use_gnn_encoder=config.use_gnn_encoder,
        use_transformer_context=config.use_transformer_context,
        use_gdm_allocator=config.use_gdm_allocator,
    )


def load_checkpoint(
    config: ExperimentConfig,
    run_dir: Path,
    checkpoint_name: str,
) -> HybridRoutingPolicy:
    model = build_model(config).to(config.device)
    checkpoint_path = run_dir / checkpoint_name

    payload = torch.load(checkpoint_path, map_location=config.device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict)
    model.eval()
    return model


def obs_to_tensor(obs: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key in ("node_features", "adjacency", "task_features", "candidate_features", "candidate_mask"):
        converted[key] = torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
    converted["candidate_nodes"] = torch.as_tensor(obs["candidate_nodes"], dtype=torch.long, device=device).unsqueeze(0)
    return converted


def rollout_policy(
    env: SimplifiedRoutingEnv,
    config: ExperimentConfig,
    policy_name: str,
    episodes: int,
    seed_base: int,
    model: HybridRoutingPolicy | Any | None = None,
) -> MetricDict:
    rng = np.random.default_rng(config.seed + 99)
    rewards = []
    latencies = []
    comm_latencies = []
    comp_latencies = []
    deadline_hits = []
    violations = []
    violation_ratios = []
    timely_throughputs = []
    load_balancing_indices = []
    global_load_balancing_indices = []
    edge_queue_ratios = []
    resource_utilizations = []
    peak_resource_utilizations = []
    edge_allocs = []
    cloud_allocs = []

    for episode in range(episodes):
        obs = env.reset(seed=seed_base + episode)
        while True:
            route_idx, allocation = select_action(policy_name, obs, config, model, rng)
            obs, reward, done, info = env.step({"route_idx": route_idx, "allocation": allocation})

            rewards.append(float(reward))
            latencies.append(info["latency"])
            comm_latencies.append(info["comm_latency"])
            comp_latencies.append(info["comp_latency"])
            deadline_hits.append(info["deadline_hit"])
            violations.append(info["violation"])
            violation_ratios.append(info["latency_violation_ratio"])
            timely_throughputs.append(info["timely_throughput"])
            load_balancing_indices.append(info["load_balancing_index"])
            global_load_balancing_indices.append(info.get("global_load_balancing_index", info["step_load_balancing_index"]))
            edge_queue_ratios.append(info["avg_edge_queue_ratio"])
            resource_utilizations.append(info["avg_resource_utilization"])
            peak_resource_utilizations.append(info["peak_resource_utilization"])
            edge_allocs.append(info["edge_allocation_ratio"])
            cloud_allocs.append(info["cloud_allocation_ratio"])

            if done:
                break

    return {
        "reward_mean": float(np.mean(rewards)),
        "latency": float(np.mean(latencies)),
        "comm_latency": float(np.mean(comm_latencies)),
        "comp_latency": float(np.mean(comp_latencies)),
        "deadline_hit_ratio": float(np.mean(deadline_hits)),
        "violation_mean": float(np.mean(violations)),
        "latency_violation_ratio": float(np.mean(violation_ratios)),
        "timely_throughput": float(np.mean(timely_throughputs)),
        "load_balancing_index": float(np.mean(load_balancing_indices)),
        "global_load_balancing_index": float(np.mean(global_load_balancing_indices)),
        "avg_edge_queue_ratio": float(np.mean(edge_queue_ratios)),
        "resource_utilization": float(np.mean(resource_utilizations)),
        "peak_resource_utilization": float(np.mean(peak_resource_utilizations)),
        "edge_allocation_ratio": float(np.mean(edge_allocs)),
        "cloud_allocation_ratio": float(np.mean(cloud_allocs)),
    }


def select_action(
    policy_name: str,
    obs: dict[str, np.ndarray],
    config: ExperimentConfig,
    model: HybridRoutingPolicy | Any | None,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    if policy_name == "trained":
        if model is None:
            raise ValueError("A trained model must be provided for policy_name='trained'.")
        with torch.no_grad():
            action = model.act_deterministic(obs_to_tensor(obs, config.device))
        route_idx = int(action["route"].item())
        allocation = action["allocation"].squeeze(0).cpu().numpy()
        return route_idx, allocation

    if policy_name == "heuristic":
        return heuristic_action(obs, config)

    if policy_name == "masac":
        if model is None or not hasattr(model, "act_numpy"):
            raise ValueError("A MASAC policy adapter must be provided for policy_name='masac'.")
        return model.act_numpy(obs, deterministic=True)

    if policy_name == "mapdqn":
        if model is None or not hasattr(model, "act_numpy"):
            raise ValueError("A MAPDQN policy adapter must be provided for policy_name='mapdqn'.")
        return model.act_numpy(obs, deterministic=True)

    if policy_name == "random":
        return random_action(obs, config, rng)

    raise ValueError(f"Unsupported policy_name: {policy_name}")
