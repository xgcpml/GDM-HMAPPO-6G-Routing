from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from config import ExperimentConfig


SPEED_OF_LIGHT = 3.0e8


@dataclass(frozen=True)
class CandidatePath:
    nodes: tuple[int, int, int, int, int]
    compute_nodes: tuple[int, int, int]
    propagation_delay: float


class SimplifiedRoutingEnv:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.user_ids = list(range(config.num_users))
        self.access_ids = list(range(config.num_users, config.num_users + config.num_access))
        self.edge_ids = list(
            range(
                config.num_users + config.num_access,
                config.num_users + config.num_access + config.num_edges,
            )
        )
        self.cloud_ids = list(
            range(
                config.num_users + config.num_access + config.num_edges,
                config.num_users + config.num_access + config.num_edges + config.num_clouds,
            )
        )
        self.compute_ids = self.edge_ids + self.cloud_ids
        self.num_nodes = config.num_users + config.num_access + config.num_edges + config.num_clouds
        self.node_ids = np.arange(self.num_nodes, dtype=np.int64)
        self.user_ids_array = np.asarray(self.user_ids, dtype=np.int64)
        self.access_ids_array = np.asarray(self.access_ids, dtype=np.int64)
        self.edge_ids_array = np.asarray(self.edge_ids, dtype=np.int64)
        self.cloud_ids_array = np.asarray(self.cloud_ids, dtype=np.int64)
        self.compute_ids_array = np.asarray(self.compute_ids, dtype=np.int64)

        self.access_index = {node_id: idx for idx, node_id in enumerate(self.access_ids)}
        self.edge_index = {node_id: idx for idx, node_id in enumerate(self.edge_ids)}
        self.cloud_index = {node_id: idx for idx, node_id in enumerate(self.cloud_ids)}

        self.node_types = np.zeros(self.num_nodes, dtype=np.int64)
        self.node_types[self.access_ids] = 1
        self.node_types[self.edge_ids] = 2
        self.node_types[self.cloud_ids] = 3

        self.node_positions = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.base_rates = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.edge_kinds = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int64)
        self.base_capacity = np.zeros(self.num_nodes, dtype=np.float32)
        self.base_prop_delay = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.link_distances = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.base_bandwidth_hz = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        self.user_access_neighbors: Dict[int, List[int]] = {user: [] for user in self.user_ids}
        self.access_edge_neighbors: Dict[int, List[int]] = {access: [] for access in self.access_ids}
        self.edge_edge_neighbors: Dict[int, List[int]] = {edge: [] for edge in self.edge_ids}
        self.edge_cloud_neighbors: Dict[int, List[int]] = {edge: [] for edge in self.edge_ids}

        self.user_access_affinity = np.zeros((config.num_users, config.num_access), dtype=np.float32)
        self.access_edge_affinity = np.zeros((config.num_access, config.num_edges), dtype=np.float32)
        self.edge_cloud_affinity = np.zeros((config.num_edges, config.num_clouds), dtype=np.float32)
        self.compute_local_index = {node_id: idx for idx, node_id in enumerate(self.compute_ids)}

        self._build_static_topology()
        self.candidate_paths = self._build_candidate_paths()
        self._build_runtime_caches()

        self.queues = np.zeros(self.num_nodes, dtype=np.float32)
        self.current_rates = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.current_available = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.current_capacity = np.zeros(self.num_nodes, dtype=np.float32)
        self.observed_rates = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.observed_available = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.observed_capacity = np.zeros(self.num_nodes, dtype=np.float32)
        self.observed_queue_pressure = np.zeros(self.num_nodes, dtype=np.float32)
        self.observed_node_pressure = np.zeros(self.num_nodes, dtype=np.float32)
        self.prev_observed_rates = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.prev_observed_available = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.prev_observed_capacity = np.zeros(self.num_nodes, dtype=np.float32)
        self.prev_observed_queue_pressure = np.zeros(self.num_nodes, dtype=np.float32)
        self.prev_observed_node_pressure = np.zeros(self.num_nodes, dtype=np.float32)
        self.background_queue_arrivals = np.zeros(self.num_nodes, dtype=np.float32)
        self.slot_user_pressure = np.ones(config.num_users, dtype=np.float32)
        self.slot_access_pressure = np.ones(config.num_access, dtype=np.float32)
        self.slot_edge_pressure = np.ones(config.num_edges, dtype=np.float32)
        self.slot_cloud_pressure = np.ones(config.num_clouds, dtype=np.float32)
        self.active_flow_count = 0
        self.current_task: Dict[str, float] = {}
        self.episode_regime = "balanced"
        self.regime_hot_users: list[int] = []
        self.regime_hot_edges: list[int] = []
        self.regime_hot_clouds: list[int] = []
        self.hot_user_mask = np.zeros(self.num_nodes, dtype=np.float32)
        self.hot_edge_mask = np.zeros(self.num_nodes, dtype=np.float32)
        self.hot_cloud_mask = np.zeros(self.num_nodes, dtype=np.float32)
        self.regime_intensity = 1.0
        self.rate_multipliers = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.capacity_multipliers = np.ones(self.num_nodes, dtype=np.float32)
        self.t = 0
        self.episode_flow_baseline = max(int(sum(config.active_flow_range) / 2), 1)
        self.episode_edge_normalized_load = np.zeros(config.num_edges, dtype=np.float32)

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.queues.fill(0.0)
        self.episode_edge_normalized_load.fill(0.0)
        self.t = 0
        self._sample_episode_regime()
        self._initialize_episode_dynamics()
        self._sample_network_state()
        self._initialize_observations()
        self._refresh_observations()
        self.current_task = self._sample_task()
        return self._build_observation()

    def step(self, action: Dict[str, np.ndarray | int]):
        route_idx = int(action["route_idx"])
        allocation = self._normalize_allocation(np.asarray(action["allocation"], dtype=np.float32))
        action_eval = self._evaluate_action(route_idx, allocation)
        reward = float(action_eval["raw_reward"])
        resource_metrics = self._estimate_resource_metrics(action_eval["path"], allocation)
        self._update_episode_edge_load(action_eval["path"], allocation)
        episode_load_balancing_index = self._episode_load_balancing_index()

        self._update_queues(action_eval["path"], allocation)

        info = {
            "latency": float(action_eval["latency"]),
            "comm_latency": float(action_eval["comm_latency"]),
            "comp_latency": float(action_eval["comp_latency"]),
            "deadline": float(action_eval["deadline"]),
            "violation": float(action_eval["violation"]),
            "deadline_hit": float(action_eval["deadline_hit"]),
            "route_idx": route_idx,
            "allocation": allocation.copy(),
            "path_available": float(action_eval["path_available"]),
            "edge_allocation_ratio": float(allocation[:-1].sum()),
            "cloud_allocation_ratio": float(allocation[-1]),
            "mean_compute_queue_ratio": float(action_eval["mean_queue_ratio"]),
            "latency_ratio": float(action_eval["latency_ratio"]),
            "latency_violation_ratio": float(action_eval["violation_ratio"]),
            "active_flows": float(self.active_flow_count),
            "timely_throughput": float(action_eval["deadline_hit"] * self.active_flow_count),
            "avg_resource_utilization": float(resource_metrics["avg_resource_utilization"]),
            "peak_resource_utilization": float(resource_metrics["peak_resource_utilization"]),
            "step_load_balancing_index": float(resource_metrics["load_balancing_index"]),
            "global_load_balancing_index": float(resource_metrics["global_load_balancing_index"]),
            "load_balancing_index": float(episode_load_balancing_index),
            "avg_edge_queue_ratio": float(resource_metrics["avg_edge_queue_ratio"]),
            "raw_reward": float(action_eval["raw_reward"]),
            "reward": float(reward),
            "reward_slack_term": float(action_eval["reward_slack_term"]),
            "reward_violation_term": float(action_eval["reward_violation_term"]),
            "reward_queue_term": float(action_eval["reward_queue_term"]),
            "reward_unavailable_term": float(action_eval["reward_unavailable_term"]),
        }

        self.t += 1
        done = self.t >= self.config.episode_length

        if done:
            next_obs = None
        else:
            self._sample_network_state()
            self._refresh_observations()
            self.current_task = self._sample_task()
            next_obs = self._build_observation()

        return next_obs, float(reward), done, info

    def _update_episode_edge_load(self, path: CandidatePath, allocation: np.ndarray) -> None:
        compute_load = float(self.current_task["compute_load"])
        for node_id, alpha in zip(path.compute_nodes, allocation):
            if node_id not in self.edge_ids:
                continue
            local_idx = int(node_id - self.edge_ids[0])
            capacity = max(float(self.current_capacity[node_id]) * self.config.queue_drain_ratio, 1e-6)
            self.episode_edge_normalized_load[local_idx] += float(alpha) * compute_load / capacity

    def _episode_load_balancing_index(self) -> float:
        if self.episode_edge_normalized_load.size == 0:
            return 0.0
        active_load = self.episode_edge_normalized_load[self.episode_edge_normalized_load > 1e-6]
        if active_load.size == 0:
            return 0.0
        load_sum = float(np.sum(active_load))
        if load_sum <= 1e-8:
            return 0.0
        dominant_share = float(np.max(active_load) / load_sum)
        return float(np.clip(1.0 - dominant_share, 0.0, 1.0))

    def _normalize_allocation(self, allocation: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(allocation, dtype=np.float32), 1e-6, None)
        normalized = normalized / normalized.sum()
        return normalized.astype(np.float32)

    def _evaluate_action(self, route_idx: int, allocation: np.ndarray) -> Dict[str, float | int | CandidatePath]:
        source = int(self.current_task["source"])
        chosen_path = self.candidate_paths[source][route_idx]

        comm_latency = self._compute_communication_latency(chosen_path)
        comp_latency = self._compute_computation_latency(chosen_path, allocation)
        total_latency = comm_latency + comp_latency

        deadline = float(self.current_task["deadline"])
        violation = max(total_latency - deadline, 0.0)
        deadline_hit = float(violation <= 1e-8)
        path_available = float(self._path_availability(chosen_path))
        mean_queue_ratio = float(self._mean_queue_ratio(chosen_path.compute_nodes))
        deadline_safe = max(deadline, 1e-6)
        latency_ratio = total_latency / deadline_safe
        violation_ratio = float(
            np.clip(violation / deadline_safe, 0.0, self.config.reward_violation_clip)
        )
        unavailable_ratio = 1.0 - path_available
        latency_cost = self.config.reward_latency_scale * latency_ratio
        violation_cost = self.config.reward_violation_scale * violation_ratio
        queue_cost = self.config.reward_queue_penalty_scale * mean_queue_ratio
        unavailable_cost = self.config.reward_unavailability_scale * unavailable_ratio
        reward_slack_term = -latency_cost
        reward_violation_term = -violation_cost
        reward_queue_term = -queue_cost
        reward_unavailable_term = -unavailable_cost
        raw_reward = reward_slack_term + reward_violation_term + reward_queue_term + reward_unavailable_term
        return {
            "path": chosen_path,
            "route_idx": int(route_idx),
            "comm_latency": float(comm_latency),
            "comp_latency": float(comp_latency),
            "latency": float(total_latency),
            "deadline": float(deadline),
            "violation": float(violation),
            "deadline_hit": float(deadline_hit),
            "path_available": float(path_available),
            "mean_queue_ratio": float(mean_queue_ratio),
            "latency_ratio": float(latency_ratio),
            "violation_ratio": float(violation_ratio),
            "reward_slack_term": float(reward_slack_term),
            "reward_violation_term": float(reward_violation_term),
            "reward_queue_term": float(reward_queue_term),
            "reward_unavailable_term": float(reward_unavailable_term),
            "raw_reward": float(raw_reward),
        }

    def _variation_factor(self, factor: float) -> float:
        scale = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
        return float(1.0 + (float(factor) - 1.0) * scale)

    def _beta_shape_from_mean(self, mean: float, concentration: float) -> tuple[float, float]:
        clipped_mean = float(np.clip(mean, 0.18, 0.82))
        clipped_concentration = max(float(concentration), 2.2)
        alpha = max(clipped_mean * clipped_concentration, 1.05)
        beta = max((1.0 - clipped_mean) * clipped_concentration, 1.05)
        return float(alpha), float(beta)

    def _build_static_topology(self) -> None:
        self._assign_node_positions()

        for user in self.user_ids:
            access_positions = self.node_positions[self.access_ids]
            distances = np.linalg.norm(access_positions - self.node_positions[user], axis=1)
            for local_idx in np.argsort(distances)[: max(int(self.config.user_access_degree), 1)]:
                access = self.access_ids[int(local_idx)]
                distance = float(distances[local_idx])
                rate, bandwidth_hz = self._wireless_base_rate(distance)
                self._set_link(
                    src=user,
                    dst=access,
                    rate=rate,
                    edge_kind=1,
                    propagation_delay=distance / SPEED_OF_LIGHT,
                    distance=distance,
                    bandwidth_hz=bandwidth_hz,
                )
                self.user_access_neighbors[user].append(access)

        for access in self.access_ids:
            edge_positions = self.node_positions[self.edge_ids]
            distances = np.linalg.norm(edge_positions - self.node_positions[access], axis=1)
            for local_idx in np.argsort(distances)[: max(int(self.config.access_edge_degree), 1)]:
                edge = self.edge_ids[int(local_idx)]
                distance = float(distances[local_idx])
                self._set_link(
                    src=access,
                    dst=edge,
                    rate=float(self.rng.uniform(*self.config.access_edge_rate)),
                    edge_kind=2,
                    propagation_delay=self._wired_delay_from_distance(distance),
                    distance=distance,
                )
                self.access_edge_neighbors[access].append(edge)

        for edge in self.edge_ids:
            neighbor_positions = self.node_positions[self.edge_ids]
            distances = np.linalg.norm(neighbor_positions - self.node_positions[edge], axis=1)
            selected = 0
            for local_idx in np.argsort(distances):
                neighbor = self.edge_ids[int(local_idx)]
                if neighbor == edge:
                    continue
                distance = float(distances[local_idx])
                self._set_link(
                    src=edge,
                    dst=neighbor,
                    rate=float(self.rng.uniform(*self.config.edge_edge_rate)),
                    edge_kind=2,
                    propagation_delay=self._wired_delay_from_distance(distance),
                    distance=distance,
                )
                self.edge_edge_neighbors[edge].append(neighbor)
                selected += 1
                if selected >= max(int(self.config.edge_neighbor_degree), 1):
                    break

        for edge in self.edge_ids:
            cloud_positions = self.node_positions[self.cloud_ids]
            distances = np.linalg.norm(cloud_positions - self.node_positions[edge], axis=1)
            for local_idx in np.argsort(distances)[: max(int(self.config.edge_cloud_degree), 1)]:
                cloud = self.cloud_ids[int(local_idx)]
                distance = float(distances[local_idx])
                self._set_link(
                    src=edge,
                    dst=cloud,
                    rate=float(self.rng.uniform(*self.config.edge_cloud_rate)),
                    edge_kind=2,
                    propagation_delay=self._wired_delay_from_distance(distance),
                    distance=distance,
                )
                self.edge_cloud_neighbors[edge].append(cloud)

        for edge in self.edge_ids:
            self.base_capacity[edge] = self.rng.uniform(*self.config.edge_capacity)
        for cloud in self.cloud_ids:
            self.base_capacity[cloud] = self.rng.uniform(*self.config.cloud_capacity)

        self._sort_neighbors()
        self._build_affinity_matrices()

    def _assign_node_positions(self) -> None:
        width = self.config.area_width_m
        height = self.config.area_height_m

        if self.user_ids:
            self.node_positions[self.user_ids] = self.rng.uniform(
                low=(0.0, 0.0),
                high=(width, height),
                size=(len(self.user_ids), 2),
            ).astype(np.float32)

        access_positions = self._grid_positions(len(self.access_ids), 0.12, 0.88, 0.12, 0.88)
        if len(self.access_ids) > 0:
            access_positions += self.rng.normal(0.0, 25.0, size=access_positions.shape)
            access_positions[:, 0] = np.clip(access_positions[:, 0], 0.0, width)
            access_positions[:, 1] = np.clip(access_positions[:, 1], 0.0, height)
            self.node_positions[self.access_ids] = access_positions.astype(np.float32)

        if self.edge_ids:
            edge_placement_mode = getattr(self.config, "edge_placement_mode", "access_anchored_random")
            if edge_placement_mode == "nested_grid":
                reference_count = max(
                    int(getattr(self.config, "edge_layout_reference_count", 0)),
                    len(self.edge_ids),
                )
                reference_positions = self._grid_positions(reference_count, 0.16, 0.84, 0.22, 0.78)
                reference_positions += self.rng.normal(0.0, 12.0, size=reference_positions.shape)
                reference_positions[:, 0] = np.clip(reference_positions[:, 0], 0.0, width)
                reference_positions[:, 1] = np.clip(reference_positions[:, 1], 0.0, height)
                subset = self._select_nested_grid_subset(reference_positions, len(self.edge_ids))
                self.node_positions[self.edge_ids] = subset.astype(np.float32)
            elif edge_placement_mode == "uniform_grid":
                edge_positions = self._grid_positions(len(self.edge_ids), 0.16, 0.84, 0.22, 0.78)
                edge_positions += self.rng.normal(0.0, 12.0, size=edge_positions.shape)
                edge_positions[:, 0] = np.clip(edge_positions[:, 0], 0.0, width)
                edge_positions[:, 1] = np.clip(edge_positions[:, 1], 0.0, height)
                self.node_positions[self.edge_ids] = edge_positions.astype(np.float32)
            else:
                edge_positions = []
                for _ in self.edge_ids:
                    if self.access_ids:
                        anchor = self.node_positions[self.access_ids[int(self.rng.integers(0, len(self.access_ids)))]]
                    else:
                        anchor = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
                    jitter = self.rng.normal(0.0, 85.0, size=2)
                    position = np.clip(anchor + jitter, [0.0, 0.0], [width, height])
                    edge_positions.append(position.astype(np.float32))
                self.node_positions[self.edge_ids] = np.asarray(edge_positions, dtype=np.float32)

        cloud_positions = self._grid_positions(len(self.cloud_ids), 0.12, 0.88, 0.76, 0.94)
        if len(self.cloud_ids) > 0:
            cloud_positions += self.rng.normal(0.0, 20.0, size=cloud_positions.shape)
            cloud_positions[:, 0] = np.clip(cloud_positions[:, 0], 0.0, width)
            cloud_positions[:, 1] = np.clip(cloud_positions[:, 1], 0.0, height)
            self.node_positions[self.cloud_ids] = cloud_positions.astype(np.float32)

    def _grid_positions(
        self,
        count: int,
        x_low_frac: float,
        x_high_frac: float,
        y_low_frac: float,
        y_high_frac: float,
    ) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        xs = np.linspace(self.config.area_width_m * x_low_frac, self.config.area_width_m * x_high_frac, cols)
        ys = np.linspace(self.config.area_height_m * y_low_frac, self.config.area_height_m * y_high_frac, rows)
        points = []
        for y in ys:
            for x in xs:
                points.append((x, y))
                if len(points) >= count:
                    break
            if len(points) >= count:
                break
        return np.asarray(points, dtype=np.float32)

    def _select_nested_grid_subset(self, positions: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if count >= len(positions):
            return positions.astype(np.float32)

        center = np.asarray(
            [self.config.area_width_m * 0.5, self.config.area_height_m * 0.5],
            dtype=np.float32,
        )
        remaining = list(range(len(positions)))
        first = min(remaining, key=lambda idx: float(np.linalg.norm(positions[idx] - center)))
        order = [first]
        remaining.remove(first)

        while remaining:
            next_idx = max(
                remaining,
                key=lambda idx: min(
                    float(np.linalg.norm(positions[idx] - positions[selected])) for selected in order
                ),
            )
            order.append(next_idx)
            remaining.remove(next_idx)

        chosen = np.asarray(order[:count], dtype=np.int64)
        chosen_positions = positions[chosen]
        return chosen_positions.astype(np.float32)

    def _wireless_base_rate(self, distance: float) -> tuple[float, float]:
        distance = max(distance, 10.0)
        bandwidth_ghz = float(self.rng.uniform(*self.config.wireless_bandwidth_ghz))
        bandwidth_hz = bandwidth_ghz * 1.0e9
        path_loss_db = 61.4 + 20.0 * np.log10(distance) + 20.0 * np.log10(self.config.carrier_frequency_ghz)
        shadowing_db = float(self.rng.normal(0.0, self.config.wireless_shadowing_sigma_db))
        received_power_dbm = (
            self.config.tx_power_dbm
            + self.config.beamforming_gain_db
            - path_loss_db
            - shadowing_db
        )
        noise_dbm = (
            self.config.noise_psd_dbm_hz
            + 10.0 * np.log10(bandwidth_hz)
            + self.config.noise_figure_db
        )
        snr_linear = 10.0 ** ((received_power_dbm - noise_dbm) / 10.0)
        spectral_efficiency = float(np.clip(np.log2(1.0 + snr_linear), 0.4, 7.2))
        rate_mbps = bandwidth_hz / 1.0e6 * spectral_efficiency * 0.68
        return float(np.clip(rate_mbps, *self.config.user_access_rate)), bandwidth_hz

    def _wired_delay_from_distance(self, distance: float) -> float:
        min_ms, max_ms = self.config.wired_propagation_delay_ms
        diagonal = np.hypot(self.config.area_width_m, self.config.area_height_m)
        ratio = float(np.clip(distance / max(diagonal, 1.0), 0.0, 1.0))
        delay_ms = min_ms + (max_ms - min_ms) * (0.25 + 0.75 * ratio)
        return delay_ms / 1_000.0

    def _set_link(
        self,
        src: int,
        dst: int,
        rate: float,
        edge_kind: int,
        propagation_delay: float,
        distance: float,
        bandwidth_hz: float = 0.0,
    ) -> None:
        current_rate = float(self.base_rates[src, dst])
        if current_rate > 0.0 and current_rate >= rate:
            return

        self.base_rates[src, dst] = rate
        self.edge_kinds[src, dst] = edge_kind
        self.base_prop_delay[src, dst] = propagation_delay
        self.link_distances[src, dst] = distance
        self.base_bandwidth_hz[src, dst] = bandwidth_hz

    def _sort_neighbors(self) -> None:
        for user, neighbors in self.user_access_neighbors.items():
            neighbors.sort(key=lambda access: float(self.base_rates[user, access]), reverse=True)
        for access, neighbors in self.access_edge_neighbors.items():
            neighbors.sort(key=lambda edge: float(self.base_rates[access, edge]), reverse=True)
        for edge, neighbors in self.edge_edge_neighbors.items():
            neighbors.sort(key=lambda other: float(self.base_rates[edge, other]), reverse=True)
        for edge, neighbors in self.edge_cloud_neighbors.items():
            neighbors.sort(key=lambda cloud: float(self.base_rates[edge, cloud]), reverse=True)

    def _build_affinity_matrices(self) -> None:
        for user in self.user_ids:
            weights = np.asarray([self.base_rates[user, access] for access in self.access_ids], dtype=np.float32)
            total = float(weights.sum())
            if total > 0.0:
                self.user_access_affinity[user] = weights / total

        for local_access, access in enumerate(self.access_ids):
            weights = np.asarray([self.base_rates[access, edge] for edge in self.edge_ids], dtype=np.float32)
            total = float(weights.sum())
            if total > 0.0:
                self.access_edge_affinity[local_access] = weights / total

        for local_edge, edge in enumerate(self.edge_ids):
            weights = np.asarray([self.base_rates[edge, cloud] for cloud in self.cloud_ids], dtype=np.float32)
            total = float(weights.sum())
            if total > 0.0:
                self.edge_cloud_affinity[local_edge] = weights / total

    def _build_runtime_caches(self) -> None:
        self.existing_links_mask = self.base_rates > 0.0
        diagonal = max(np.hypot(self.config.area_width_m, self.config.area_height_m), 1.0)
        self.normalized_x = (self.node_positions[:, 0] / max(self.config.area_width_m, 1.0)).astype(np.float32)
        self.normalized_y = (self.node_positions[:, 1] / max(self.config.area_height_m, 1.0)).astype(np.float32)

        wireless_src, wireless_dst = np.where(self.edge_kinds == 1)
        self.wireless_src = wireless_src.astype(np.int64)
        self.wireless_dst = wireless_dst.astype(np.int64)
        self.wireless_base_rates = self.base_rates[self.wireless_src, self.wireless_dst].astype(np.float32)
        self.wireless_distance_ratio = np.clip(
            self.link_distances[self.wireless_src, self.wireless_dst] / diagonal,
            0.0,
            1.0,
        ).astype(np.float32)
        self.wireless_dst_access_idx = np.asarray(
            [self.access_index[int(dst)] for dst in self.wireless_dst],
            dtype=np.int64,
        )

        wired_src, wired_dst = np.where(self.edge_kinds == 2)
        self.wired_src = wired_src.astype(np.int64)
        self.wired_dst = wired_dst.astype(np.int64)
        self.wired_base_rates = self.base_rates[self.wired_src, self.wired_dst].astype(np.float32)

        self.candidate_nodes_cache: Dict[int, np.ndarray] = {}
        self.candidate_compute_nodes_cache: Dict[int, np.ndarray] = {}
        self.candidate_link_src_cache: Dict[int, np.ndarray] = {}
        self.candidate_link_dst_cache: Dict[int, np.ndarray] = {}
        self.candidate_static_prior_cache: Dict[int, np.ndarray] = {}
        self.candidate_propagation_norm_cache: Dict[int, np.ndarray] = {}

        for user, paths in self.candidate_paths.items():
            nodes = np.asarray([path.nodes for path in paths], dtype=np.int64)
            compute_nodes = np.asarray([path.compute_nodes for path in paths], dtype=np.int64)
            static_prior = np.asarray(
                [
                    np.clip(self._path_static_cost(path.nodes, path.compute_nodes) / 1.4, 0.0, 1.0)
                    for path in paths
                ],
                dtype=np.float32,
            )
            propagation_norm = np.asarray(
                [np.clip(path.propagation_delay / 0.05, 0.0, 1.0) for path in paths],
                dtype=np.float32,
            )
            self.candidate_nodes_cache[user] = nodes
            self.candidate_compute_nodes_cache[user] = compute_nodes
            self.candidate_link_src_cache[user] = nodes[:, :-1]
            self.candidate_link_dst_cache[user] = nodes[:, 1:]
            self.candidate_static_prior_cache[user] = static_prior
            self.candidate_propagation_norm_cache[user] = propagation_norm

    def _build_candidate_paths(self) -> Dict[int, List[CandidatePath]]:
        candidate_paths: Dict[int, List[CandidatePath]] = {}
        for user in self.user_ids:
            ranked_paths: list[tuple[float, CandidatePath]] = []
            seen_paths: set[tuple[int, int, int, int, int]] = set()

            for access in self.user_access_neighbors[user]:
                for edge_1 in self.access_edge_neighbors[access]:
                    for edge_2 in self.edge_edge_neighbors[edge_1]:
                        if edge_2 == edge_1:
                            continue
                        for cloud in self.edge_cloud_neighbors[edge_2]:
                            nodes = (user, access, edge_1, edge_2, cloud)
                            if nodes in seen_paths or not self._path_exists(nodes):
                                continue
                            seen_paths.add(nodes)
                            compute_nodes = (edge_1, edge_2, cloud)
                            propagation_delay = float(
                                sum(self.base_prop_delay[src, dst] for src, dst in zip(nodes[:-1], nodes[1:]))
                            )
                            path = CandidatePath(
                                nodes=nodes,
                                compute_nodes=compute_nodes,
                                propagation_delay=propagation_delay,
                            )
                            ranked_paths.append((self._path_static_cost(nodes, compute_nodes), path))

            if not ranked_paths:
                raise RuntimeError(f"No candidate path could be constructed for user {user}.")

            ranked_paths.sort(key=lambda item: item[0])
            selected = self._select_diverse_candidate_subset(ranked_paths)
            while len(selected) < self.config.num_candidate_paths:
                selected.append(selected[len(selected) % len(selected)])
            candidate_paths[user] = selected[: self.config.num_candidate_paths]
        return candidate_paths

    def _select_diverse_candidate_subset(
        self,
        ranked_paths: list[tuple[float, CandidatePath]],
    ) -> list[CandidatePath]:
        target = max(int(self.config.num_candidate_paths), 1)
        if len(ranked_paths) <= target:
            return [path for _, path in ranked_paths]

        best_keep = min(2, target)
        selected_indices = list(range(best_keep))
        selected_access = {ranked_paths[idx][1].nodes[1] for idx in selected_indices}
        selected_edge_1 = {ranked_paths[idx][1].nodes[2] for idx in selected_indices}
        selected_edge_2 = {ranked_paths[idx][1].nodes[3] for idx in selected_indices}
        selected_cloud = {ranked_paths[idx][1].nodes[4] for idx in selected_indices}
        selected_compute_sets = {tuple(ranked_paths[idx][1].compute_nodes) for idx in selected_indices}

        pool_limit = min(
            len(ranked_paths),
            max(target * 4, int(np.ceil(len(ranked_paths) * 0.80))),
        )
        pool_costs = np.asarray([cost for cost, _ in ranked_paths[:pool_limit]], dtype=np.float32)
        cost_min = float(pool_costs.min())
        cost_span = max(float(pool_costs.max() - cost_min), 1e-6)

        while len(selected_indices) < target:
            best_idx = None
            best_score = float("-inf")
            for idx in range(pool_limit):
                if idx in selected_indices:
                    continue
                cost, path = ranked_paths[idx]
                access, edge_1, edge_2, cloud = path.nodes[1:]
                compute_key = tuple(path.compute_nodes)

                novelty_score = 0.0
                novelty_score += 1.20 if access not in selected_access else 0.0
                novelty_score += 1.00 if edge_1 not in selected_edge_1 else 0.0
                novelty_score += 0.85 if edge_2 not in selected_edge_2 else 0.0
                novelty_score += 0.55 if cloud not in selected_cloud else 0.0
                novelty_score += 0.65 if compute_key not in selected_compute_sets else 0.0

                rank_penalty = idx / max(pool_limit - 1, 1)
                normalized_cost = (float(cost) - cost_min) / cost_span
                candidate_score = novelty_score - 0.65 * rank_penalty - 0.55 * normalized_cost

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_idx = idx

            if best_idx is None:
                break

            selected_indices.append(best_idx)
            selected_path = ranked_paths[best_idx][1]
            selected_access.add(selected_path.nodes[1])
            selected_edge_1.add(selected_path.nodes[2])
            selected_edge_2.add(selected_path.nodes[3])
            selected_cloud.add(selected_path.nodes[4])
            selected_compute_sets.add(tuple(selected_path.compute_nodes))

        if len(selected_indices) < target:
            for idx in range(pool_limit):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                if len(selected_indices) >= target:
                    break

        selected_indices = sorted(selected_indices[:target])
        return [ranked_paths[idx][1] for idx in selected_indices]

    def _path_exists(self, nodes: tuple[int, int, int, int, int]) -> bool:
        for src, dst in zip(nodes[:-1], nodes[1:]):
            if self.base_rates[src, dst] <= 0.0:
                return False
        return True

    def _path_static_cost(
        self,
        nodes: tuple[int, int, int, int, int],
        compute_nodes: tuple[int, int, int],
    ) -> float:
        mean_data = 0.5 * sum(self.config.task_data_range)
        mean_compute = 0.5 * sum(self.config.task_compute_range)
        comm_cost = 0.0
        for src, dst in zip(nodes[:-1], nodes[1:]):
            comm_cost += float(self.base_prop_delay[src, dst]) + mean_data / max(float(self.base_rates[src, dst]), 1e-6)

        compute_cost = 0.0
        for node_id in compute_nodes:
            compute_cost += (mean_compute / self.config.alloc_dim) / max(float(self.base_capacity[node_id]), 1e-6)

        return float(comm_cost + 0.45 * compute_cost)

    def _sample_episode_regime(self) -> None:
        regimes = (
            "balanced",
            "wireless_stress",
            "edge_hotspot",
            "backhaul_stress",
            "cloud_burst",
        )
        probabilities = np.asarray([0.72, 0.08, 0.08, 0.07, 0.05], dtype=np.float32)
        variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
        if variation < 0.999:
            stress_probabilities = probabilities[1:] * variation
            probabilities = np.concatenate(
                (
                    np.asarray([1.0 - float(stress_probabilities.sum())], dtype=np.float32),
                    stress_probabilities.astype(np.float32),
                )
            )
        self.episode_regime = str(self.rng.choice(regimes, p=probabilities))

        if self.episode_regime == "balanced":
            self.regime_intensity = self._variation_factor(self.rng.uniform(0.995, 1.005))
        elif self.episode_regime == "wireless_stress":
            self.regime_intensity = self._variation_factor(self.rng.uniform(1.01, 1.04))
        elif self.episode_regime == "edge_hotspot":
            self.regime_intensity = self._variation_factor(self.rng.uniform(1.02, 1.05))
        elif self.episode_regime == "backhaul_stress":
            self.regime_intensity = self._variation_factor(self.rng.uniform(1.01, 1.04))
        else:
            self.regime_intensity = self._variation_factor(self.rng.uniform(1.02, 1.05))

        base_hot_user_count = min(max(2, self.config.num_users // 10), self.config.num_users)
        base_hot_edge_count = min(max(2, self.config.num_edges // 8), self.config.num_edges)
        base_hot_cloud_count = min(max(1, self.config.num_clouds // 4), self.config.num_clouds)
        hot_user_count = int(np.clip(round(1 + (base_hot_user_count - 1) * variation), 1, self.config.num_users))
        hot_edge_count = int(np.clip(round(1 + (base_hot_edge_count - 1) * variation), 1, self.config.num_edges))
        hot_cloud_count = int(np.clip(round(1 + (base_hot_cloud_count - 1) * variation), 1, self.config.num_clouds))
        self.regime_hot_users = list(self.rng.choice(self.user_ids, size=hot_user_count, replace=False))
        self.regime_hot_edges = list(self.rng.choice(self.edge_ids, size=hot_edge_count, replace=False))
        self.regime_hot_clouds = list(self.rng.choice(self.cloud_ids, size=hot_cloud_count, replace=False))
        self.hot_user_mask.fill(0.0)
        self.hot_edge_mask.fill(0.0)
        self.hot_cloud_mask.fill(0.0)
        self.hot_user_mask[self.regime_hot_users] = 1.0
        self.hot_edge_mask[self.regime_hot_edges] = 1.0
        self.hot_cloud_mask[self.regime_hot_clouds] = 1.0

    def _initialize_episode_dynamics(self) -> None:
        self.current_available.fill(0.0)
        self.rate_multipliers.fill(1.0)
        self.capacity_multipliers.fill(1.0)
        self.background_queue_arrivals.fill(0.0)
        self.slot_user_pressure.fill(1.0)
        self.slot_access_pressure.fill(1.0)
        self.slot_edge_pressure.fill(1.0)
        self.slot_cloud_pressure.fill(1.0)
        low, high = self.config.active_flow_range
        regime_ratio = {
            "balanced": 0.50,
            "wireless_stress": 0.56,
            "edge_hotspot": 0.58,
            "backhaul_stress": 0.56,
            "cloud_burst": 0.57,
        }[self.episode_regime]
        variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
        baseline_ratio = float(
            np.clip(regime_ratio + self.rng.uniform(-0.04, 0.04) * variation, 0.18, 0.88)
        )
        self.episode_flow_baseline = int(
            np.clip(round(low + (high - low) * baseline_ratio), low, high)
        )
        self.active_flow_count = self.episode_flow_baseline

        self.current_available[self.existing_links_mask] = 1.0

    def _sample_slot_profile(self) -> None:
        low, high = self.config.active_flow_range
        target_flow = self.episode_flow_baseline + int(self.rng.integers(-1, 2))
        if self.episode_regime in {"edge_hotspot", "cloud_burst"}:
            target_flow += 1
        target_flow = int(np.clip(target_flow, low, high))
        self.active_flow_count = int(
            np.clip(
                round(0.74 * float(self.active_flow_count) + 0.26 * float(target_flow)),
                low,
                high,
            )
        )
        flow_ratio = self.active_flow_count / self._flow_reference()
        variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
        load_intensity = float(
            np.clip(
                flow_ratio ** float(getattr(self.config, "load_pressure_power", 1.0)),
                0.55,
                2.75,
            )
        )
        edge_relief = float(
            np.clip(
                (self._edge_reference() / max(float(self.config.num_edges), 1.0))
                ** float(getattr(self.config, "edge_relief_power", 1.0)),
                0.60,
                1.90,
            )
        )
        cloud_relief = float(
            np.clip(
                (self._edge_reference() / max(float(self.config.num_edges), 1.0))
                ** float(getattr(self.config, "cloud_relief_power", 0.0)),
                0.80,
                1.20,
            )
        )

        user_pref = np.ones(self.config.num_users, dtype=np.float32)
        edge_pref = np.ones(self.config.num_edges, dtype=np.float32)
        cloud_pref = np.ones(self.config.num_clouds, dtype=np.float32)
        user_pref[self.regime_hot_users] *= (
            self._variation_factor(1.28) if self.episode_regime == "wireless_stress" else self._variation_factor(1.12)
        )
        for node_id in self.regime_hot_edges:
            edge_pref[self.edge_index[node_id]] *= (
                self._variation_factor(1.30) if self.episode_regime == "edge_hotspot" else self._variation_factor(1.14)
            )
        for node_id in self.regime_hot_clouds:
            cloud_pref[self.cloud_index[node_id]] *= (
                self._variation_factor(1.26) if self.episode_regime == "cloud_burst" else self._variation_factor(1.10)
            )

        user_pressure_target = self.rng.dirichlet(user_pref).astype(np.float32)
        user_pressure_target *= flow_ratio * max(self.config.num_users, 1)

        access_pressure_target = user_pressure_target @ self.user_access_affinity
        edge_pressure_target = access_pressure_target @ self.access_edge_affinity
        edge_pressure_target = 0.65 * edge_pressure_target + 0.35 * (
            self.rng.dirichlet(edge_pref).astype(np.float32) * flow_ratio * max(self.config.num_edges, 1)
        )
        cloud_pressure_target = edge_pressure_target @ self.edge_cloud_affinity
        cloud_pressure_target = 0.70 * cloud_pressure_target + 0.30 * (
            self.rng.dirichlet(cloud_pref).astype(np.float32) * flow_ratio * max(self.config.num_clouds, 1)
        )

        momentum = self.config.task_pressure_momentum
        self.slot_user_pressure = (
            momentum * self.slot_user_pressure
            + (1.0 - momentum) * (load_intensity * self._normalize_pressure(user_pressure_target))
        ).astype(np.float32)
        self.slot_access_pressure = (
            momentum * self.slot_access_pressure
            + (1.0 - momentum) * (load_intensity * self._normalize_pressure(access_pressure_target))
        ).astype(np.float32)
        self.slot_edge_pressure = (
            momentum * self.slot_edge_pressure
            + (1.0 - momentum) * (load_intensity * edge_relief * self._normalize_pressure(edge_pressure_target))
        ).astype(np.float32)
        self.slot_cloud_pressure = (
            momentum * self.slot_cloud_pressure
            + (1.0 - momentum) * (load_intensity * cloud_relief * self._normalize_pressure(cloud_pressure_target))
        ).astype(np.float32)

    def _normalize_pressure(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        mean_value = float(np.mean(values))
        if mean_value <= 1e-6:
            return np.ones_like(values, dtype=np.float32)
        normalized = values / mean_value
        return np.clip(normalized, 0.25, 4.0).astype(np.float32)

    def _flow_reference(self) -> float:
        configured = float(getattr(self.config, "flow_load_reference", 0.0))
        if configured > 1e-6:
            return configured
        return max(0.5 * float(sum(self.config.active_flow_range)), 1.0)

    def _edge_reference(self) -> float:
        configured = int(getattr(self.config, "edge_count_reference", 0))
        if configured > 0:
            return float(configured)
        return float(max(self.config.num_edges, 1))

    def _sample_network_state(self) -> None:
        self._sample_slot_profile()
        full_node_pressure = self._compose_full_node_pressure()
        self.current_rates.fill(0.0)

        if self.wireless_src.size > 0:
            access_pressure = self.slot_access_pressure[self.wireless_dst_access_idx]
            user_pressure = self.slot_user_pressure[self.wireless_src]
            variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
            blockage_prob = 1.0 - self.config.wireless_up_prob + 0.08 * self.wireless_distance_ratio
            if self.episode_regime == "wireless_stress":
                blockage_prob = blockage_prob + 0.03 * variation
            blockage_prob = np.clip(
                blockage_prob + 0.004 * variation * self.hot_user_mask[self.wireless_src],
                0.02,
                0.09,
            ).astype(np.float32)

            prev_available = self.current_available[self.wireless_src, self.wireless_dst] >= 0.5
            blocked = self.rng.random(self.wireless_src.size) < np.where(
                prev_available,
                0.35 * blockage_prob,
                blockage_prob,
            )
            target_scale = self.rng.uniform(
                self.config.wireless_rate_jitter[0],
                self.config.wireless_rate_jitter[1],
                size=self.wireless_src.size,
            ).astype(np.float32)
            target_scale *= self.rng.lognormal(
                mean=0.0,
                sigma=0.12 * (0.55 + 0.45 * variation),
                size=self.wireless_src.size,
            ).astype(np.float32)
            target_scale *= 1.0 / (
                1.0
                + self.config.access_contention_scale * access_pressure
                + 0.06 * user_pressure
            )
            if self.episode_regime == "wireless_stress":
                target_scale *= self._variation_factor(0.96)
            blocked_scale = self.rng.uniform(
                self._variation_factor(0.78),
                self._variation_factor(0.92),
                size=self.wireless_src.size,
            ).astype(np.float32)
            target_scale = np.where(blocked, target_scale * blocked_scale, target_scale)

            prev_scale = self.rate_multipliers[self.wireless_src, self.wireless_dst]
            updated_scale = (
                self.config.temporal_rate_momentum * prev_scale
                + (1.0 - self.config.temporal_rate_momentum) * target_scale
            )
            updated_scale = np.clip(updated_scale, 0.08, 1.25).astype(np.float32)
            self.rate_multipliers[self.wireless_src, self.wireless_dst] = updated_scale
            self.current_rates[self.wireless_src, self.wireless_dst] = self.wireless_base_rates * updated_scale
            self.current_available[self.wireless_src, self.wireless_dst] = (
                self.current_rates[self.wireless_src, self.wireless_dst] >= 0.18 * self.wireless_base_rates
            ).astype(np.float32)

        if self.wired_src.size > 0:
            pressure_sum = 0.5 * (full_node_pressure[self.wired_src] + full_node_pressure[self.wired_dst])
            variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
            target_scale = self.rng.uniform(
                self.config.wired_rate_jitter[0],
                self.config.wired_rate_jitter[1],
                size=self.wired_src.size,
            ).astype(np.float32)
            target_scale *= 1.0 / (
                1.0 + self.config.backhaul_contention_scale * pressure_sum
            )
            if self.episode_regime == "backhaul_stress":
                target_scale *= self._variation_factor(0.94)
            if self.episode_regime == "cloud_burst":
                target_scale *= 1.0 - 0.025 * variation * self.hot_cloud_mask[self.wired_dst]

            prev_scale = self.rate_multipliers[self.wired_src, self.wired_dst]
            updated_scale = (
                self.config.temporal_rate_momentum * prev_scale
                + (1.0 - self.config.temporal_rate_momentum) * target_scale
            )
            updated_scale = np.clip(updated_scale, 0.55, 1.08).astype(np.float32)
            self.rate_multipliers[self.wired_src, self.wired_dst] = updated_scale
            self.current_rates[self.wired_src, self.wired_dst] = self.wired_base_rates * updated_scale
            self.current_available[self.wired_src, self.wired_dst] = 1.0

        self.current_capacity.fill(0.0)
        if self.compute_ids_array.size > 0:
            pressure = full_node_pressure[self.compute_ids_array]
            variation = float(np.clip(self.config.episode_variation_scale, 0.0, 1.0))
            target_scale = self.rng.uniform(
                self.config.compute_jitter[0],
                self.config.compute_jitter[1],
                size=self.compute_ids_array.size,
            ).astype(np.float32)
            target_scale *= 1.0 / (1.0 + self.config.compute_pressure_scale * pressure)
            if self.episode_regime == "edge_hotspot":
                target_scale *= 1.0 - 0.06 * variation * self.hot_edge_mask[self.compute_ids_array]
            if self.episode_regime == "cloud_burst":
                target_scale *= 1.0 - 0.04 * variation * self.hot_cloud_mask[self.compute_ids_array]

            prev_scale = self.capacity_multipliers[self.compute_ids_array]
            updated_scale = (
                self.config.temporal_capacity_momentum * prev_scale
                + (1.0 - self.config.temporal_capacity_momentum) * target_scale
            )
            updated_scale = np.clip(updated_scale, 0.35, 1.08).astype(np.float32)
            self.capacity_multipliers[self.compute_ids_array] = updated_scale
            self.current_capacity[self.compute_ids_array] = (
                self.base_capacity[self.compute_ids_array] * updated_scale
            )

        self.background_queue_arrivals.fill(0.0)
        if self.edge_ids_array.size > 0:
            self.background_queue_arrivals[self.edge_ids_array] = (
                self.current_capacity[self.edge_ids_array]
                * self.config.background_queue_scale
                * self.slot_edge_pressure
            )
        if self.cloud_ids_array.size > 0:
            self.background_queue_arrivals[self.cloud_ids_array] = (
                self.current_capacity[self.cloud_ids_array]
                * self.config.background_queue_scale
                * 0.65
                * self.slot_cloud_pressure
            )

    def _compose_full_node_pressure(self) -> np.ndarray:
        pressure = np.ones(self.num_nodes, dtype=np.float32)
        pressure[self.user_ids] = self.slot_user_pressure
        pressure[self.access_ids] = self.slot_access_pressure
        pressure[self.edge_ids] = self.slot_edge_pressure
        pressure[self.cloud_ids] = self.slot_cloud_pressure
        return pressure

    def _sample_task(self) -> Dict[str, float]:
        source_probs = self.slot_user_pressure / max(float(self.slot_user_pressure.sum()), 1e-6)
        source = int(self.rng.choice(self.user_ids, p=source_probs))

        data_low, data_high = self.config.task_data_range
        comp_low, comp_high = self.config.task_compute_range
        active_ratio = self.active_flow_count / self._flow_reference()
        load_bias = float(getattr(self.config, "task_load_bias_scale", 0.025)) * max(active_ratio - 1.0, 0.0)
        regime_profile = {
            "balanced": (0.47, 0.48, 1.01, 1.03),
            "wireless_stress": (0.54, 0.50, 0.995, 1.018),
            "edge_hotspot": (0.49, 0.59, 1.00, 1.022),
            "backhaul_stress": (0.53, 0.53, 0.995, 1.015),
            "cloud_burst": (0.50, 0.58, 1.00, 1.022),
        }[self.episode_regime]
        data_mean = regime_profile[0] + 0.35 * load_bias
        compute_mean = regime_profile[1] + 0.50 * load_bias
        deadline_scale = (
            self._variation_factor(regime_profile[2]),
            self._variation_factor(regime_profile[3]),
        )
        if self.episode_regime == "wireless_stress":
            data_mean += 0.008
        elif self.episode_regime == "edge_hotspot":
            compute_mean += 0.008
        elif self.episode_regime == "backhaul_stress":
            data_mean += 0.004
        elif self.episode_regime == "cloud_burst":
            compute_mean += 0.006

        if source in self.regime_hot_users:
            data_mean += 0.010
            compute_mean += 0.008

        shared_mean = 0.5 * (data_mean + compute_mean)
        shared_alpha, shared_beta = self._beta_shape_from_mean(shared_mean, concentration=8.0)
        data_alpha, data_beta = self._beta_shape_from_mean(data_mean, concentration=10.0)
        compute_alpha, compute_beta = self._beta_shape_from_mean(compute_mean, concentration=9.0)
        shared_mix = float(self.rng.beta(shared_alpha, shared_beta))
        data_mix = float(
            np.clip(0.65 * shared_mix + 0.35 * float(self.rng.beta(data_alpha, data_beta)), 0.12, 0.88)
        )
        compute_mix = float(
            np.clip(0.65 * shared_mix + 0.35 * float(self.rng.beta(compute_alpha, compute_beta)), 0.12, 0.88)
        )
        data_size = float(
            (data_low + data_mix * (data_high - data_low)) * (0.99 + 0.03 * self.regime_intensity)
        )
        compute_load = float(
            (comp_low + compute_mix * (comp_high - comp_low)) * (0.99 + 0.04 * self.regime_intensity)
        )
        difficulty_mix = 0.5 * (data_mix + compute_mix)
        deadline = 0.020 + 0.00018 * data_size + 0.00075 * compute_load
        deadline *= float(self.rng.uniform(*deadline_scale))
        deadline *= 1.01 + 0.05 * difficulty_mix
        deadline *= self.config.deadline_budget_factor
        deadline /= 1.0 + float(getattr(self.config, "deadline_load_tightening", 0.010)) * max(active_ratio - 1.0, 0.0)
        deadline = float(np.clip(deadline, 0.08, self.config.max_deadline_scale * 0.98))
        return {
            "source": source,
            "data_size": data_size,
            "compute_load": compute_load,
            "deadline": deadline,
            "active_flows": float(self.active_flow_count),
        }

    def _initialize_observations(self) -> None:
        self.observed_rates = np.clip(0.82 * self.base_rates / self.config.max_rate_scale, 0.0, 1.0)
        self.observed_capacity = np.clip(0.84 * self.base_capacity / self.config.max_capacity_scale, 0.0, 1.0)
        self.observed_queue_pressure.fill(0.0)
        self.observed_node_pressure.fill(0.25)
        self.observed_available.fill(0.0)

        existing_links = self.base_rates > 0.0
        self.observed_available[existing_links] = 1.0
        wireless_links = self.edge_kinds == 1
        self.observed_available[wireless_links] = float(np.clip(self.config.wireless_up_prob - 0.05, 0.45, 0.98))

        self.observed_rates = self._coarsen_signal(self.observed_rates)
        self.observed_capacity = self._coarsen_signal(self.observed_capacity)
        self.observed_available = self._coarsen_signal(self.observed_available)
        self.prev_observed_rates = self.observed_rates.copy()
        self.prev_observed_available = self.observed_available.copy()
        self.prev_observed_capacity = self.observed_capacity.copy()
        self.prev_observed_queue_pressure = self.observed_queue_pressure.copy()
        self.prev_observed_node_pressure = self.observed_node_pressure.copy()

    def _refresh_observations(self) -> None:
        prev_observed_rates = self.observed_rates.copy()
        prev_observed_available = self.observed_available.copy()
        prev_observed_capacity = self.observed_capacity.copy()
        prev_observed_queue = self.observed_queue_pressure.copy()
        prev_observed_pressure = self.observed_node_pressure.copy()

        rate_target = np.clip(self.current_rates / self.config.max_rate_scale, 0.0, 1.0)
        availability_target = np.clip(self.current_available, 0.0, 1.0)
        capacity_target = np.clip(self.current_capacity / self.config.max_capacity_scale, 0.0, 1.0)
        pressure_target = np.clip(self._compose_full_node_pressure() / 4.0, 0.0, 1.0)

        queue_pressure = np.zeros(self.num_nodes, dtype=np.float32)
        if self.compute_ids_array.size > 0:
            capacity = np.clip(self.current_capacity[self.compute_ids_array], 1e-6, None)
            queue_pressure[self.compute_ids_array] = np.clip(
                self.queues[self.compute_ids_array] / (capacity * 2.4),
                0.0,
                1.0,
            ).astype(np.float32)

        observed_rate = self._coarsen_signal(self._noisy_signal(rate_target))
        observed_availability = self._coarsen_signal(self._noisy_signal(availability_target, scale=0.45))
        observed_capacity = self._coarsen_signal(self._noisy_signal(capacity_target))
        observed_queue = self._coarsen_signal(self._noisy_signal(queue_pressure, scale=0.65))
        observed_pressure = self._coarsen_signal(self._noisy_signal(pressure_target, scale=0.55))

        self.observed_rates = (
            self.config.observation_rate_momentum * self.observed_rates
            + (1.0 - self.config.observation_rate_momentum) * observed_rate
        ).astype(np.float32)
        self.observed_available = (
            self.config.observation_rate_momentum * self.observed_available
            + (1.0 - self.config.observation_rate_momentum) * observed_availability
        ).astype(np.float32)
        self.observed_capacity = (
            self.config.observation_capacity_momentum * self.observed_capacity
            + (1.0 - self.config.observation_capacity_momentum) * observed_capacity
        ).astype(np.float32)
        self.observed_queue_pressure = (
            self.config.observation_queue_momentum * self.observed_queue_pressure
            + (1.0 - self.config.observation_queue_momentum) * observed_queue
        ).astype(np.float32)
        self.observed_node_pressure = (
            self.config.observation_queue_momentum * self.observed_node_pressure
            + (1.0 - self.config.observation_queue_momentum) * observed_pressure
        ).astype(np.float32)

        link_visibility = 0.40 + 0.60 * self.observed_available
        self.observed_rates = np.clip(self.observed_rates * link_visibility, 0.0, 1.0)
        self.observed_available = np.clip(self.observed_available, 0.0, 1.0)
        self.observed_capacity = np.clip(self.observed_capacity, 0.0, 1.0)
        self.observed_queue_pressure = np.clip(self.observed_queue_pressure, 0.0, 1.0)
        self.observed_node_pressure = np.clip(self.observed_node_pressure, 0.0, 1.0)
        self.prev_observed_rates = prev_observed_rates
        self.prev_observed_available = prev_observed_available
        self.prev_observed_capacity = prev_observed_capacity
        self.prev_observed_queue_pressure = prev_observed_queue
        self.prev_observed_node_pressure = prev_observed_pressure

    def _noisy_signal(self, signal: np.ndarray, scale: float = 1.0) -> np.ndarray:
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.observation_noise_scale * scale,
            size=signal.shape,
        )
        return np.clip(signal + noise.astype(np.float32), 0.0, 1.0)

    def _coarsen_signal(self, signal: np.ndarray) -> np.ndarray:
        bins = max(int(self.config.observation_bins), 2)
        return np.round(np.clip(signal, 0.0, 1.0) * (bins - 1)) / float(bins - 1)

    @staticmethod
    def _encode_trend(current: np.ndarray | float, previous: np.ndarray | float, gain: float = 2.5):
        return np.clip(0.5 + gain * (np.asarray(current) - np.asarray(previous)), 0.0, 1.0).astype(np.float32)

    def _compute_communication_latency(self, path: CandidatePath) -> float:
        data_size = float(self.current_task["data_size"])
        latency = 0.0
        for src, dst in zip(path.nodes[:-1], path.nodes[1:]):
            base_rate = max(float(self.base_rates[src, dst]), 1e-6)
            rate = max(float(self.current_rates[src, dst]), 0.05 * base_rate)
            latency += float(self.base_prop_delay[src, dst]) + data_size / rate
            if self.current_available[src, dst] < 0.5:
                latency += self.config.invalid_path_penalty
        return latency

    def _compute_computation_latency(self, path: CandidatePath, allocation: np.ndarray) -> float:
        compute_load = float(self.current_task["compute_load"])
        latency = 0.0
        for node_id, alpha in zip(path.compute_nodes, allocation):
            capacity = max(float(self.current_capacity[node_id]), 1e-6)
            queue_delay = float(self.queues[node_id]) / capacity
            service_delay = float(alpha) * compute_load / capacity
            latency += queue_delay + service_delay
        return latency

    def _update_queues(self, path: CandidatePath, allocation: np.ndarray) -> None:
        if self.compute_ids_array.size > 0:
            drain = self.current_capacity[self.compute_ids_array] * self.config.queue_drain_ratio
            self.queues[self.compute_ids_array] = np.maximum(
                self.queues[self.compute_ids_array] - drain,
                0.0,
            )
            self.queues[self.compute_ids_array] += self.background_queue_arrivals[self.compute_ids_array]

        compute_load = float(self.current_task["compute_load"])
        path_compute_nodes = np.asarray(path.compute_nodes, dtype=np.int64)
        self.queues[path_compute_nodes] += allocation.astype(np.float32) * compute_load

    def _estimate_resource_metrics(self, path: CandidatePath, allocation: np.ndarray) -> Dict[str, float]:
        def compute_load_balance(node_ids: np.ndarray) -> float:
            if node_ids.size == 0:
                return 0.0
            service_capacity = np.clip(
                self.current_capacity[node_ids] * self.config.queue_drain_ratio,
                1e-6,
                None,
            )
            demand = self.queues[node_ids].astype(np.float32).copy()
            if getattr(self.config, "resource_metrics_include_background", False):
                demand += self.background_queue_arrivals[node_ids].astype(np.float32)
            local_index = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
            compute_load = float(self.current_task["compute_load"])
            for node_id, alpha in zip(path.compute_nodes, allocation):
                idx = local_index.get(int(node_id))
                if idx is None:
                    continue
                demand[idx] += float(alpha) * compute_load

            utilization = np.clip(demand / service_capacity, 0.0, 1.0)
            eval_utilization = utilization
            if getattr(self.config, "resource_metrics_active_only", False):
                active_mask = demand > 1e-6
                if np.any(active_mask):
                    eval_utilization = utilization[active_mask]

            utilization_sum = float(np.sum(eval_utilization))
            utilization_sq_sum = float(np.sum(np.square(eval_utilization)))
            if utilization_sq_sum <= 1e-8:
                return 0.0
            return float((utilization_sum * utilization_sum) / (float(eval_utilization.size) * utilization_sq_sum))

        metric_node_ids = self.compute_ids_array
        if getattr(self.config, "resource_metrics_edge_only", False) and self.edge_ids_array.size > 0:
            metric_node_ids = self.edge_ids_array

        if metric_node_ids.size == 0:
            return {
                "avg_resource_utilization": 0.0,
                "peak_resource_utilization": 0.0,
                "load_balancing_index": 0.0,
                "global_load_balancing_index": 0.0,
                "avg_edge_queue_ratio": 0.0,
            }

        service_capacity = np.clip(
            self.current_capacity[metric_node_ids] * self.config.queue_drain_ratio,
            1e-6,
            None,
        )
        demand = self.queues[metric_node_ids].astype(np.float32).copy()
        if getattr(self.config, "resource_metrics_include_background", False):
            demand += self.background_queue_arrivals[metric_node_ids].astype(np.float32)
        metric_local_index = {int(node_id): idx for idx, node_id in enumerate(metric_node_ids.tolist())}

        compute_load = float(self.current_task["compute_load"])
        for node_id, alpha in zip(path.compute_nodes, allocation):
            local_idx = metric_local_index.get(int(node_id))
            if local_idx is None:
                continue
            demand[local_idx] += float(alpha) * compute_load

        utilization = np.clip(demand / service_capacity, 0.0, 1.0)
        eval_utilization = utilization
        eval_capacity = service_capacity
        if getattr(self.config, "resource_metrics_active_only", False):
            active_mask = demand > 1e-6
            if np.any(active_mask):
                eval_utilization = utilization[active_mask]
                eval_capacity = service_capacity[active_mask]

        utilization_sum = float(np.sum(eval_utilization))
        utilization_sq_sum = float(np.sum(np.square(eval_utilization)))
        if utilization_sq_sum <= 1e-8:
            load_balancing_index = 0.0
        else:
            load_balancing_index = (utilization_sum * utilization_sum) / (
                float(eval_utilization.size) * utilization_sq_sum
            )

        edge_queue_ratio = 0.0
        if self.edge_ids_array.size > 0:
            edge_service_capacity = np.clip(
                self.current_capacity[self.edge_ids_array] * self.config.queue_drain_ratio,
                1e-6,
                None,
            )
            edge_demand = self.queues[self.edge_ids_array].astype(np.float32).copy()
            if getattr(self.config, "resource_metrics_include_background", False):
                edge_demand += self.background_queue_arrivals[self.edge_ids_array].astype(np.float32)
            edge_local_index = {int(node_id): idx for idx, node_id in enumerate(self.edge_ids_array.tolist())}
            for node_id, alpha in zip(path.compute_nodes, allocation):
                local_idx = edge_local_index.get(int(node_id))
                if local_idx is None:
                    continue
                edge_demand[local_idx] += float(alpha) * compute_load
            edge_queue_ratio = float(np.mean(np.clip(edge_demand / edge_service_capacity, 0.0, 3.0)))

        return {
            "avg_resource_utilization": float(
                np.average(eval_utilization, weights=eval_capacity)
                if getattr(self.config, "resource_utilization_weighted", False)
                else np.mean(eval_utilization)
            ),
            "peak_resource_utilization": float(np.max(eval_utilization)) if eval_utilization.size > 0 else 0.0,
            "load_balancing_index": float(load_balancing_index),
            "global_load_balancing_index": float(compute_load_balance(self.compute_ids_array)),
            "avg_edge_queue_ratio": float(edge_queue_ratio),
        }

    def _path_availability(self, path: CandidatePath) -> bool:
        for src, dst in zip(path.nodes[:-1], path.nodes[1:]):
            if self.current_available[src, dst] < 0.5:
                return False
        return True

    def _mean_queue_ratio(self, compute_nodes: tuple[int, int, int]) -> float:
        nodes = np.asarray(compute_nodes, dtype=np.int64)
        capacity = np.clip(self.current_capacity[nodes], 1e-6, None)
        ratios = self.queues[nodes] / (capacity * 2.0)
        return float(np.mean(ratios))

    def _build_observation(self) -> Dict[str, np.ndarray]:
        node_features = self._build_node_features()
        task_features = np.asarray(
            [
                self.current_task["source"] / max(self.config.num_users - 1, 1),
                self.current_task["data_size"] / self.config.task_data_range[1],
                self.current_task["compute_load"] / self.config.task_compute_range[1],
                self.current_task["deadline"] / self.config.max_deadline_scale,
                self.active_flow_count / max(float(self.config.active_flow_range[1]), 1.0),
                np.clip(self.regime_intensity / 1.2, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        source = int(self.current_task["source"])
        path_nodes, candidate_features, candidate_mask = self._build_candidate_feature_bundle(source)

        return {
            "node_features": node_features.astype(np.float32),
            "adjacency": self.observed_rates.astype(np.float32),
            "task_features": task_features,
            "candidate_features": candidate_features.astype(np.float32),
            "candidate_mask": candidate_mask.astype(np.float32),
            "candidate_nodes": path_nodes.astype(np.int64),
        }

    def _build_node_features(self) -> np.ndarray:
        node_features = np.zeros((self.num_nodes, self.config.node_feature_dim), dtype=np.float32)
        source = int(self.current_task.get("source", 0))
        node_features[self.node_ids, self.node_types] = 1.0
        node_features[:, 4] = self.observed_capacity
        node_features[:, 5] = self.observed_queue_pressure
        node_features[:, 7] = self.observed_rates.mean(axis=1)
        node_features[:, 8] = self.observed_rates.mean(axis=0)
        node_features[:, 9] = self.normalized_x
        node_features[:, 10] = self.normalized_y
        node_features[:, 11] = self.observed_node_pressure
        node_features[:, 12] = self.prev_observed_capacity
        node_features[:, 13] = self.prev_observed_queue_pressure
        node_features[:, 14] = self._encode_trend(self.observed_capacity, self.prev_observed_capacity)
        node_features[:, 15] = self._encode_trend(
            self.observed_queue_pressure,
            self.prev_observed_queue_pressure,
        )
        node_features[source, 6] = 1.0

        return node_features

    def _build_candidate_feature_bundle(self, source: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_nodes = self.candidate_nodes_cache[source]
        compute_nodes = self.candidate_compute_nodes_cache[source]
        link_src = self.candidate_link_src_cache[source]
        link_dst = self.candidate_link_dst_cache[source]

        observed_availability = self.observed_available[link_src, link_dst]
        observed_rates = self.observed_rates[link_src, link_dst]
        wireless_confidence = observed_availability[:, 0]
        wireless_quality = observed_rates[:, 0] * (0.35 + 0.65 * wireless_confidence)
        wired_bottleneck = np.clip(np.percentile(observed_rates[:, 1:], 25, axis=1), 0.0, 1.0).astype(np.float32)
        path_confidence = observed_availability.mean(axis=1).astype(np.float32)
        prev_observed_availability = self.prev_observed_available[link_src, link_dst]
        prev_observed_rates = self.prev_observed_rates[link_src, link_dst]
        prev_wireless_confidence = prev_observed_availability[:, 0]
        prev_wireless_quality = prev_observed_rates[:, 0] * (0.35 + 0.65 * prev_wireless_confidence)
        prev_path_confidence = prev_observed_availability.mean(axis=1).astype(np.float32)

        queue_pressures = self.observed_queue_pressure[compute_nodes]
        capacities = self.observed_capacity[compute_nodes]
        node_pressures = self.observed_node_pressure[compute_nodes]
        prev_queue_pressures = self.prev_observed_queue_pressure[compute_nodes]
        edge_queue_mean = queue_pressures[:, :2].mean(axis=1)
        cloud_queue = queue_pressures[:, 2]
        edge_capacity_mean = capacities[:, :2].mean(axis=1)
        cloud_capacity = capacities[:, 2]
        path_pressure = node_pressures.mean(axis=1)
        prev_edge_queue_mean = prev_queue_pressures[:, :2].mean(axis=1)
        prev_cloud_queue = prev_queue_pressures[:, 2]
        propagation_delay_norm = self.candidate_propagation_norm_cache[source]
        static_prior = self.candidate_static_prior_cache[source]
        wireless_quality_trend = self._encode_trend(wireless_quality, prev_wireless_quality)
        path_confidence_trend = self._encode_trend(path_confidence, prev_path_confidence)
        edge_queue_trend = self._encode_trend(edge_queue_mean, prev_edge_queue_mean)
        cloud_queue_trend = self._encode_trend(cloud_queue, prev_cloud_queue)

        candidate_features = np.stack(
            [
                static_prior,
                wireless_quality.astype(np.float32),
                wired_bottleneck,
                path_confidence,
                edge_queue_mean.astype(np.float32),
                cloud_queue.astype(np.float32),
                edge_capacity_mean.astype(np.float32),
                cloud_capacity.astype(np.float32),
                propagation_delay_norm,
                path_pressure.astype(np.float32),
                wireless_quality_trend,
                path_confidence_trend,
                edge_queue_trend,
                cloud_queue_trend,
            ],
            axis=1,
        )
        candidate_mask = np.clip(path_confidence, self.config.path_confidence_floor, 1.0).astype(np.float32)
        return path_nodes, candidate_features, candidate_mask

    def _path_features(self, path: CandidatePath) -> np.ndarray:
        static_prior = float(np.clip(self._path_static_cost(path.nodes, path.compute_nodes) / 1.4, 0.0, 1.0))
        wireless_src, wireless_dst = path.nodes[0], path.nodes[1]
        wireless_confidence = float(self.observed_available[wireless_src, wireless_dst])
        wireless_quality = float(
            self.observed_rates[wireless_src, wireless_dst] * (0.35 + 0.65 * wireless_confidence)
        )
        prev_wireless_confidence = float(self.prev_observed_available[wireless_src, wireless_dst])
        prev_wireless_quality = float(
            self.prev_observed_rates[wireless_src, wireless_dst] * (0.35 + 0.65 * prev_wireless_confidence)
        )

        wired_strengths = []
        confidences = []
        prev_confidences = []
        for src, dst in zip(path.nodes[:-1], path.nodes[1:]):
            confidences.append(float(self.observed_available[src, dst]))
            prev_confidences.append(float(self.prev_observed_available[src, dst]))
            if self.edge_kinds[src, dst] == 2:
                wired_strengths.append(float(self.observed_rates[src, dst]))
        wired_bottleneck = float(np.clip(np.percentile(wired_strengths, 25), 0.0, 1.0)) if wired_strengths else 0.0
        path_confidence = float(np.mean(confidences))
        prev_path_confidence = float(np.mean(prev_confidences)) if prev_confidences else path_confidence

        queue_pressures = np.asarray(
            [self.observed_queue_pressure[node_id] for node_id in path.compute_nodes],
            dtype=np.float32,
        )
        prev_queue_pressures = np.asarray(
            [self.prev_observed_queue_pressure[node_id] for node_id in path.compute_nodes],
            dtype=np.float32,
        )
        capacities = np.asarray(
            [self.observed_capacity[node_id] for node_id in path.compute_nodes],
            dtype=np.float32,
        )
        node_pressures = np.asarray(
            [self.observed_node_pressure[node_id] for node_id in path.compute_nodes],
            dtype=np.float32,
        )
        edge_queue_mean = float(np.mean(queue_pressures[:2]))
        cloud_queue = float(queue_pressures[2])
        edge_capacity_mean = float(np.mean(capacities[:2]))
        cloud_capacity = float(capacities[2])
        propagation_delay_norm = float(np.clip(path.propagation_delay / 0.05, 0.0, 1.0))
        path_pressure = float(np.mean(node_pressures))
        prev_edge_queue_mean = float(np.mean(prev_queue_pressures[:2]))
        prev_cloud_queue = float(prev_queue_pressures[2])

        return np.asarray(
            [
                static_prior,
                wireless_quality,
                wired_bottleneck,
                path_confidence,
                edge_queue_mean,
                cloud_queue,
                edge_capacity_mean,
                cloud_capacity,
                propagation_delay_norm,
                path_pressure,
                float(self._encode_trend(wireless_quality, prev_wireless_quality)),
                float(self._encode_trend(path_confidence, prev_path_confidence)),
                float(self._encode_trend(edge_queue_mean, prev_edge_queue_mean)),
                float(self._encode_trend(cloud_queue, prev_cloud_queue)),
            ],
            dtype=np.float32,
        )

    def _path_confidence(self, path: CandidatePath) -> float:
        link_confidences = [float(self.observed_available[src, dst]) for src, dst in zip(path.nodes[:-1], path.nodes[1:])]
        if not link_confidences:
            return 1.0
        path_confidence = float(np.mean(link_confidences))
        return float(np.clip(path_confidence, self.config.path_confidence_floor, 1.0))
