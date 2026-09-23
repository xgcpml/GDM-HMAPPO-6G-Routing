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
        use_heuristic_search_candidates=config.use_heuristic_search_candidates,
        use_gnn_encoder=config.use_gnn_encoder,
        use_graph_attention_encoder=config.use_graph_attention_encoder,
        use_transformer_context=config.use_transformer_context,
        use_global_graph_context=config.use_global_graph_context,
        use_candidate_resource_summaries=config.use_candidate_resource_summaries,
        use_gdm_allocator=config.use_gdm_allocator,
        allocation_policy_family=config.allocation_policy_family,
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
    for key in (
        "node_features",
        "adjacency",
        "task_features",
        "candidate_features",
        "candidate_mask",
        "candidate_node_mask",
        "candidate_link_mask",
        "candidate_alloc_mask",
    ):
        converted[key] = torch.as_tensor(obs[key], dtype=torch.float32, device=device).unsqueeze(0)
    for key in ("candidate_nodes", "candidate_compute_nodes"):
        converted[key] = torch.as_tensor(
            obs[key], dtype=torch.long, device=device
        ).unsqueeze(0)
    converted["source_node"] = torch.as_tensor(
        obs["source_node"], dtype=torch.long, device=device
    ).reshape(1)
    return converted


def slot_obs_to_tensor(obs: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    converted = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in obs.items()
        if key not in {"candidate_nodes", "candidate_compute_nodes", "source_node"}
    }
    for key in ("candidate_nodes", "candidate_compute_nodes"):
        converted[key] = torch.as_tensor(obs[key], dtype=torch.long, device=device)
    converted["source_node"] = torch.as_tensor(
        obs["source_node"], dtype=torch.long, device=device
    )
    return converted


def rollout_policy(
    env: SimplifiedRoutingEnv,
    config: ExperimentConfig,
    policy_name: str,
    episodes: int,
    seed_base: int,
    model: HybridRoutingPolicy | Any | None = None,
) -> MetricDict:
    if config.use_multi_task_slots:
        return rollout_slot_policy(
            env=env,
            config=config,
            policy_name=policy_name,
            episodes=episodes,
            seed_base=seed_base,
            model=model,
        )

    rng = np.random.default_rng(config.seed + 99)
    rewards = []
    latencies = []
    comm_latencies = []
    comp_latencies = []
    deadline_hits = []
    violations = []
    violation_ratios = []
    timely_throughputs = []
    timely_completed_workloads = []
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
            timely_completed_workloads.append(info["timely_completed_workload"])
            load_balancing_indices.append(info["load_balancing_index"])
            global_load_balancing_indices.append(info.get("global_load_balancing_index", info["step_load_balancing_index"]))
            edge_queue_ratios.append(info["avg_edge_queue_ratio"])
            resource_utilizations.append(info["avg_resource_utilization"])
            peak_resource_utilizations.append(info["peak_resource_utilization"])
            edge_allocs.append(info["edge_allocation_ratio"])
            cloud_allocs.append(info["cloud_allocation_ratio"])

            if done:
                break

    deadline_hit_ratio = float(np.mean(deadline_hits))
    resource_utilization = float(np.mean(resource_utilizations))
    return {
        "reward_mean": float(np.mean(rewards)),
        "latency": float(np.mean(latencies)),
        "comm_latency": float(np.mean(comm_latencies)),
        "comp_latency": float(np.mean(comp_latencies)),
        "deadline_hit_ratio": deadline_hit_ratio,
        "violation_mean": float(np.mean(violations)),
        "latency_violation_ratio": float(np.mean(violation_ratios)),
        "timely_throughput": float(np.mean(timely_throughputs)),
        "timely_completed_workload": float(np.mean(timely_completed_workloads)),
        "load_balancing_index": float(np.mean(load_balancing_indices)),
        "global_load_balancing_index": float(np.mean(global_load_balancing_indices)),
        "avg_edge_queue_ratio": float(np.mean(edge_queue_ratios)),
        "resource_utilization": resource_utilization,
        "qos_resource_efficiency": deadline_hit_ratio * resource_utilization,
        "peak_resource_utilization": float(np.mean(peak_resource_utilizations)),
        "edge_allocation_ratio": float(np.mean(edge_allocs)),
        "cloud_allocation_ratio": float(np.mean(cloud_allocs)),
    }


def rollout_slot_policy(
    env: SimplifiedRoutingEnv,
    config: ExperimentConfig,
    policy_name: str,
    episodes: int,
    seed_base: int,
    model: HybridRoutingPolicy | Any | None = None,
) -> MetricDict:
    rng = np.random.default_rng(config.seed + 99)
    metric_keys = (
        "latency",
        "comm_latency",
        "comp_latency",
        "deadline_hit",
        "violation",
        "latency_violation_ratio",
        "timely_throughput",
        "timely_completed_workload",
        "avg_edge_queue_ratio",
        "avg_resource_utilization",
        "edge_allocation_ratio",
        "cloud_allocation_ratio",
    )
    values = {key: [] for key in metric_keys}
    rewards: list[float] = []
    episode_load_balancing: list[float] = []
    episode_qos_load_balancing: list[float] = []
    episode_peak_load_ratio: list[float] = []

    for episode in range(episodes):
        obs = env.reset_slot(seed=seed_base + episode)
        assigned_total = np.zeros(len(env.compute_ids), dtype=np.float64)
        service_total = np.zeros(len(env.compute_ids), dtype=np.float64)
        reachable = np.zeros(len(env.compute_ids), dtype=bool)
        episode_hits: list[float] = []
        while True:
            route_indices, allocations = select_slot_actions(
                policy_name,
                obs,
                config,
                model,
                rng,
            )
            obs, reward, done, info = env.step_slot(
                {"route_idx": route_indices, "allocation": allocations}
            )
            rewards.append(float(reward))
            for key in metric_keys:
                values[key].append(float(info[key]))
            assigned_total += np.asarray(info["assigned_compute_workload"], dtype=np.float64)
            service_total += np.asarray(info["available_compute_service"], dtype=np.float64)
            reachable |= np.asarray(info["reachable_compute_mask"]) > 0.5
            episode_hits.append(float(info["deadline_hit"]))
            if done:
                break

        jain_index, peak_load_ratio = compute_episode_load_metrics(
            assigned_total,
            service_total,
            reachable,
            config.resource_peak_percentile,
        )
        episode_load_balancing.append(jain_index)
        episode_qos_load_balancing.append(jain_index * float(np.mean(episode_hits)))
        episode_peak_load_ratio.append(peak_load_ratio)

    deadline_hit_ratio = float(np.mean(values["deadline_hit"]))
    resource_utilization = float(np.mean(values["avg_resource_utilization"]))
    return {
        "reward_mean": float(np.mean(rewards)),
        "latency": float(np.mean(values["latency"])),
        "comm_latency": float(np.mean(values["comm_latency"])),
        "comp_latency": float(np.mean(values["comp_latency"])),
        "deadline_hit_ratio": deadline_hit_ratio,
        "violation_mean": float(np.mean(values["violation"])),
        "latency_violation_ratio": float(np.mean(values["latency_violation_ratio"])),
        "timely_throughput": float(np.mean(values["timely_throughput"])),
        "timely_completed_workload": float(
            np.mean(values["timely_completed_workload"])
        ),
        "load_balancing_index": float(np.mean(episode_load_balancing)),
        "global_load_balancing_index": float(np.mean(episode_load_balancing)),
        "qos_load_balancing_index": float(np.mean(episode_qos_load_balancing)),
        "avg_edge_queue_ratio": float(np.mean(values["avg_edge_queue_ratio"])),
        "resource_utilization": resource_utilization,
        "qos_resource_efficiency": deadline_hit_ratio * resource_utilization,
        "peak_resource_utilization": float(np.mean(episode_peak_load_ratio)),
        "peak_computation_load_ratio": float(np.mean(episode_peak_load_ratio)),
        "edge_allocation_ratio": float(np.mean(values["edge_allocation_ratio"])),
        "cloud_allocation_ratio": float(np.mean(values["cloud_allocation_ratio"])),
    }


def compute_episode_load_metrics(
    assigned_workload: np.ndarray,
    available_service: np.ndarray,
    reachable_mask: np.ndarray,
    peak_percentile: float,
) -> tuple[float, float]:
    """Return episode-level Jain balance and peak normalized compute load."""
    reachable = np.asarray(reachable_mask, dtype=bool)
    load_ratio = np.asarray(assigned_workload, dtype=np.float64)[reachable] / np.clip(
        np.asarray(available_service, dtype=np.float64)[reachable], 1e-9, None
    )
    squared_sum = float(np.square(load_ratio).sum())
    if load_ratio.size == 0 or squared_sum <= 1e-12:
        return 0.0, 0.0
    jain_index = float(load_ratio.sum() ** 2 / (load_ratio.size * squared_sum))
    return jain_index, float(np.percentile(load_ratio, peak_percentile))


def select_slot_actions(
    policy_name: str,
    obs: dict[str, np.ndarray],
    config: ExperimentConfig,
    model: HybridRoutingPolicy | Any | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if policy_name in {"trained", "graphpr", "fedroute"}:
        if model is None:
            raise ValueError(f"A trained model must be provided for {policy_name!r}.")
        with torch.inference_mode():
            action = model.act_deterministic(slot_obs_to_tensor(obs, config.device))
        return (
            action["route"].detach().cpu().numpy().astype(np.int64),
            action["allocation"].detach().cpu().numpy().astype(np.float32),
        )

    routes: list[int] = []
    allocations: list[np.ndarray] = []
    for task_idx in range(int(obs["task_features"].shape[0])):
        task_obs = task_observation_from_slot(obs, task_idx)
        route_idx, allocation = select_action(
            policy_name,
            task_obs,
            config,
            model,
            rng,
        )
        routes.append(int(route_idx))
        allocations.append(np.asarray(allocation, dtype=np.float32))
    return np.asarray(routes, dtype=np.int64), np.stack(allocations, axis=0)


def task_observation_from_slot(
    slot_obs: dict[str, np.ndarray], task_idx: int
) -> dict[str, np.ndarray]:
    """Extract one exchangeable task-agent observation from a shared slot batch."""
    return {key: value[task_idx] for key, value in slot_obs.items()}


def select_action(
    policy_name: str,
    obs: dict[str, np.ndarray],
    config: ExperimentConfig,
    model: HybridRoutingPolicy | Any | None,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    if policy_name in {"trained", "graphpr", "fedroute"}:
        if model is None:
            raise ValueError(f"A trained model must be provided for {policy_name!r}.")
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
