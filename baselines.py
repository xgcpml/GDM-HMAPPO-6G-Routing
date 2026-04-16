from __future__ import annotations

import numpy as np

from config import ExperimentConfig


def heuristic_action(obs: dict[str, np.ndarray], config: ExperimentConfig) -> tuple[int, np.ndarray]:
    path_nodes = obs["candidate_nodes"]
    node_features = obs["node_features"]
    observed_rates = obs["adjacency"]
    candidate_confidence = obs["candidate_mask"]

    route_scores = []
    for path_idx, nodes in enumerate(path_nodes):
        comm_score = 0.0
        for src, dst in zip(nodes[:-1], nodes[1:]):
            rate = max(float(observed_rates[src, dst]), 1e-3)
            comm_score += 1.0 / rate

        edge_nodes = nodes[2:4]
        edge_queue = float(np.mean([max(float(node_features[node_id, 5]), 0.0) for node_id in edge_nodes]))
        edge_pressure = float(np.mean([max(float(node_features[node_id, 11]), 0.0) for node_id in edge_nodes]))
        confidence_penalty = 1.0 - float(candidate_confidence[path_idx])
        route_score = comm_score + 0.35 * edge_queue + 0.20 * edge_pressure + 0.10 * confidence_penalty
        route_scores.append(route_score)

    route_score = np.asarray(route_scores, dtype=np.float32)
    route_idx = int(np.argmin(route_score))
    allocation = heuristic_allocation(obs, route_idx, config)
    return route_idx, allocation


def heuristic_allocation(
    obs: dict[str, np.ndarray],
    route_idx: int,
    config: ExperimentConfig,
) -> np.ndarray:
    path_nodes = obs["candidate_nodes"][route_idx]
    compute_nodes = path_nodes[-3:]
    node_features = obs["node_features"]

    edge_scores = []
    edge_stress = []
    for node_id in compute_nodes[:2]:
        capacity = max(float(node_features[node_id, 4]), 1e-6)
        queue_level = max(float(node_features[node_id, 5]), 0.0)
        node_pressure = max(float(node_features[node_id, 11]), 0.0)
        edge_stress.append(queue_level + 0.6 * node_pressure)
        edge_score = capacity / (1.0 + 1.80 * queue_level + 1.00 * node_pressure)
        edge_scores.append(edge_score)

    edge_scores = np.asarray(edge_scores, dtype=np.float32)
    mean_edge_stress = float(np.mean(edge_stress))
    edge_weights = np.clip(edge_scores, 1e-6, None)
    edge_weights = edge_weights / max(float(edge_weights.sum()), 1e-6)

    allocation = np.zeros(config.alloc_dim, dtype=np.float32)
    cloud_capacity = max(float(node_features[compute_nodes[2], 4]), 1e-6)
    cloud_queue = max(float(node_features[compute_nodes[2], 5]), 0.0)
    cloud_pressure = max(float(node_features[compute_nodes[2], 11]), 0.0)
    cloud_score = 0.35 * cloud_capacity / (1.0 + 1.20 * cloud_queue + 0.75 * cloud_pressure)
    edge_score_ref = float(np.max(edge_scores))

    cloud_share = 0.0
    if mean_edge_stress > 1.05:
        stress_ratio = np.clip((mean_edge_stress - 1.05) / 0.75, 0.0, 1.0)
        cloud_share = max(cloud_share, 0.08 + 0.22 * float(stress_ratio))
    if cloud_score > 1.08 * edge_score_ref:
        cloud_share = max(cloud_share, 0.18)

    cloud_share = float(np.clip(cloud_share, 0.0, 0.35))
    allocation[:2] = (1.0 - cloud_share) * edge_weights
    allocation[2] = cloud_share
    return allocation


def random_action(
    obs: dict[str, np.ndarray],
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    candidate_confidence = np.clip(obs["candidate_mask"], 1e-3, None)
    route_probs = candidate_confidence / candidate_confidence.sum()
    route_idx = int(rng.choice(len(route_probs), p=route_probs))
    allocation = rng.random(config.alloc_dim, dtype=np.float32)
    allocation = np.clip(allocation, 1e-6, None)
    allocation = allocation / allocation.sum()
    return route_idx, allocation
