from __future__ import annotations

import numpy as np

from config import ExperimentConfig


DALBH_MAX_CLOUD_SHARE = 0.15


def heuristic_action(obs: dict[str, np.ndarray], config: ExperimentConfig) -> tuple[int, np.ndarray]:
    path_nodes = obs["candidate_nodes"]
    path_link_mask = obs.get(
        "candidate_link_mask",
        np.ones((path_nodes.shape[0], path_nodes.shape[1] - 1), dtype=np.float32),
    )
    node_features = obs["node_features"]
    observed_rates = obs["adjacency"]
    candidate_mask = obs["candidate_mask"]
    compute_nodes = obs.get(
        "candidate_compute_nodes",
        path_nodes[:, -config.alloc_dim :],
    )
    compute_masks = obs.get(
        "candidate_alloc_mask",
        np.ones((path_nodes.shape[0], config.alloc_dim), dtype=np.float32),
    )

    data_size = float(obs["task_features"][1]) * config.task_data_range[1]
    compute_load = float(obs["task_features"][2]) * config.task_compute_range[1]
    route_scores = []
    for path_idx, nodes in enumerate(path_nodes):
        if candidate_mask[path_idx] < 0.5:
            route_scores.append(float("inf"))
            continue
        comm_delay = 0.05 * float(obs["candidate_features"][path_idx, 8])
        for src, dst, valid in zip(nodes[:-1], nodes[1:], path_link_mask[path_idx]):
            if valid < 0.5:
                continue
            rate = max(float(observed_rates[src, dst]) * config.max_rate_scale, 1.0)
            comm_delay += data_size / rate

        allocation = heuristic_allocation(obs, path_idx, config)
        processing_delay = 0.0
        for node_id, alpha, valid in zip(
            compute_nodes[path_idx],
            allocation,
            compute_masks[path_idx],
        ):
            if valid < 0.5:
                continue
            capacity = max(float(node_features[node_id, 4]) * config.max_capacity_scale, 1.0)
            queue_delay = (
                float(node_features[node_id, 5])
                * config.slot_duration_s
                * config.queue_backlog_threshold_slots
            )
            processing_delay += queue_delay + float(alpha) * compute_load / capacity
        route_scores.append(comm_delay + processing_delay)

    route_score = np.asarray(route_scores, dtype=np.float32)
    route_idx = int(np.argmin(route_score))
    allocation = heuristic_allocation(obs, route_idx, config)
    return route_idx, allocation


def heuristic_allocation(
    obs: dict[str, np.ndarray],
    route_idx: int,
    config: ExperimentConfig,
) -> np.ndarray:
    compute_nodes = obs.get(
        "candidate_compute_nodes",
        obs["candidate_nodes"][:, -config.alloc_dim :],
    )[route_idx]
    allocation_mask = obs.get(
        "candidate_alloc_mask",
        np.ones((obs["candidate_nodes"].shape[0], config.alloc_dim), dtype=np.float32),
    )[route_idx]
    node_features = obs["node_features"]

    compute_load = float(obs["task_features"][2]) * config.task_compute_range[1]
    estimated_delays = np.full(config.alloc_dim, np.inf, dtype=np.float32)
    edge_positions: list[int] = []
    cloud_positions: list[int] = []
    for idx, (node_id, valid) in enumerate(zip(compute_nodes, allocation_mask)):
        if valid < 0.5:
            continue
        capacity = max(float(node_features[node_id, 4]) * config.max_capacity_scale, 1.0)
        queue_delay = (
            max(float(node_features[node_id, 5]), 0.0)
            * config.slot_duration_s
            * config.queue_backlog_threshold_slots
        )
        estimated_delays[idx] = queue_delay + compute_load / capacity

        if float(node_features[node_id, 2]) > 0.5:
            edge_positions.append(idx)
        elif float(node_features[node_id, 3]) > 0.5:
            cloud_positions.append(idx)

    allocation = np.zeros(config.alloc_dim, dtype=np.float32)
    if not edge_positions:
        allocation[int(np.argmin(estimated_delays))] = 1.0
        return allocation

    best_edge = min(edge_positions, key=lambda idx: float(estimated_delays[idx]))
    allocation[best_edge] = 1.0
    if cloud_positions:
        best_cloud = min(cloud_positions, key=lambda idx: float(estimated_delays[idx]))
        edge_delay = max(float(estimated_delays[best_edge]), 1e-6)
        cloud_advantage = np.clip(
            (edge_delay - float(estimated_delays[best_cloud])) / edge_delay,
            0.0,
            1.0,
        )
        cloud_share = float(DALBH_MAX_CLOUD_SHARE * cloud_advantage)
        allocation[best_edge] = 1.0 - cloud_share
        allocation[best_cloud] = cloud_share
    return allocation


def random_action(
    obs: dict[str, np.ndarray],
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    candidate_mask = (obs["candidate_mask"] > 0.5).astype(np.float32)
    route_probs = candidate_mask / candidate_mask.sum()
    route_idx = int(rng.choice(len(route_probs), p=route_probs))
    allocation_mask = obs.get(
        "candidate_alloc_mask",
        np.ones((len(route_probs), config.alloc_dim), dtype=np.float32),
    )[route_idx]
    allocation = rng.random(config.alloc_dim, dtype=np.float32) * allocation_mask
    allocation = np.clip(allocation, 1e-6, None) * allocation_mask
    allocation = allocation / allocation.sum()
    return route_idx, allocation
