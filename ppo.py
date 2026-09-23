from __future__ import annotations

import atexit
import copy
import multiprocessing as mp
from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import trange

from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from model import HybridRoutingPolicy
from baselines import heuristic_action


OBS_FLOAT_KEYS = (
    "node_features",
    "adjacency",
    "task_features",
    "candidate_features",
    "candidate_mask",
    "candidate_node_mask",
    "candidate_link_mask",
    "candidate_alloc_mask",
)
OBS_LONG_KEYS = ("candidate_nodes", "candidate_compute_nodes", "source_node")


@dataclass
class RolloutBatch:
    observations: Dict[str, Tensor]
    route_actions: Tensor
    allocations: Tensor
    old_route_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor


def _env_worker(config: ExperimentConfig, conn) -> None:
    env = SimplifiedRoutingEnv(config)
    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                conn.send(env.reset(seed=payload))
            elif command == "step":
                conn.send(env.step(payload))
            elif command == "close":
                break
            else:
                raise ValueError(f"Unsupported env worker command: {command}")
    finally:
        conn.close()


class _ParallelEnvPool:
    def __init__(self, config: ExperimentConfig, size: int):
        self.ctx = mp.get_context("spawn")
        self.parents = []
        self.processes = []
        self.size = max(int(size), 0)

        for _ in range(self.size):
            parent_conn, child_conn = self.ctx.Pipe()
            process = self.ctx.Process(target=_env_worker, args=(config, child_conn), daemon=True)
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)

    def reset_many(self, seeds: List[int]) -> List[Dict[str, np.ndarray]]:
        for conn, seed in zip(self.parents, seeds):
            conn.send(("reset", seed))
        return [conn.recv() for conn in self.parents[: len(seeds)]]

    def step_many(self, actions: List[Dict[str, np.ndarray | int]]):
        for conn, action in zip(self.parents, actions):
            conn.send(("step", action))
        return [conn.recv() for conn in self.parents[: len(actions)]]

    def close(self) -> None:
        for conn in self.parents:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for conn in self.parents:
            try:
                conn.close()
            except OSError:
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


class PPOTrainer:
    def __init__(self, config: ExperimentConfig, env: SimplifiedRoutingEnv, model: HybridRoutingPolicy):
        self.config = config
        self.env = env
        self.rollout_envs = [
            SimplifiedRoutingEnv(self._topology_config(index, evaluation=False))
            for index in range(max(int(config.rollout_episodes_per_update), 1))
        ]
        self.eval_envs = [
            SimplifiedRoutingEnv(self._topology_config(index, evaluation=True))
            for index in range(max(int(config.train_eval_episodes), int(config.eval_episodes), 1))
        ]
        self.rollout_env_pool = None
        self.eval_env_pool = None
        if not config.use_multi_task_slots and config.parallel_rollout_envs > 1:
            self.rollout_env_pool = _ParallelEnvPool(
                config, min(int(config.parallel_rollout_envs), int(config.rollout_episodes_per_update))
            )
        if not config.use_multi_task_slots and config.parallel_eval_envs > 1:
            self.eval_env_pool = _ParallelEnvPool(
                config, min(int(config.parallel_eval_envs), max(int(config.train_eval_episodes), int(config.eval_episodes)))
            )
        self.model = model.to(config.device)
        shared_params = list(self.model.graph_encoder.parameters()) + list(self.model.transformer_blocks.parameters())
        actor_params = list(self.model.route_head.parameters()) + list(self.model.allocation_head.parameters())
        critic_params = list(self.model.value_head.parameters())
        self.optimizer_group_scales = [
            float(config.shared_lr_factor),
            float(config.actor_lr_factor),
            float(config.critic_lr_factor),
        ]
        self.optimizer = Adam(
            [
                {"params": shared_params, "lr": config.lr * self.optimizer_group_scales[0]},
                {"params": actor_params, "lr": config.lr * self.optimizer_group_scales[1]},
                {"params": critic_params, "lr": config.lr * self.optimizer_group_scales[2]},
            ]
        )
        self.episode_history: List[Dict[str, float]] = []
        self.eval_episode_history: List[Dict[str, float]] = []
        self.best_model_path = config.output_path / "best_routing_model.pt"
        self.best_checkpoint_info = {
            "best_update": None,
            "best_eval_reward": float("-inf"),
            "best_eval_latency": float("inf"),
            "best_eval_deadline_hit_ratio": 0.0,
        }
        self.validation_history: List[Dict[str, float]] = []
        self.best_eval_reward = float("-inf")
        self.best_reward_state_dict = copy.deepcopy(self.model.state_dict())
        self.current_plateau_window_reward = float("nan")
        self.current_plateau_window_spread = float("nan")
        self.current_plateau_window_ready = 0.0
        self.no_improve_updates = 0
        self.rollback_count = 0
        self.freeze_updates = False
        self.current_lr = config.lr
        self.current_route_entropy_coef = config.route_entropy_coef
        self.current_route_sampling_temperature = config.route_sampling_temperature_initial
        self.current_alloc_noise_scale = config.alloc_noise_scale
        self.current_topology_prior_scale = config.topology_route_prior_initial_scale
        self.current_topology_distill_coef = config.topology_distill_coef
        self.current_search_distill_scale = 1.0
        self.current_greedy_rollout_prob = 0.0
        self.rollout_seed_pool = self._build_rollout_seed_pool()
        self.model.set_route_sampling_temperature(self.current_route_sampling_temperature)
        self.model.set_allocation_noise_scale(self.current_alloc_noise_scale)
        self.model.set_topology_prior_scale(self.current_topology_prior_scale)
        self._closed = False
        atexit.register(self.close)

    def _topology_config(self, index: int, evaluation: bool) -> ExperimentConfig:
        stride = max(int(self.config.topology_seed_stride), 0)
        if stride == 0:
            return self.config
        held_out_offset = int(self.config.eval_topology_seed_offset) if evaluation else 0
        topology_seed = int(self.config.seed) + held_out_offset + (index + 1) * stride
        return replace(self.config, seed=topology_seed)

    def warmup(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for update in trange(max(self.config.warmup_updates, 0), desc="Warmup", leave=False):
            obs_batch, route_target, alloc_target = self.collect_warmup_batch(update)
            metrics = self._run_supervised_update(
                obs_batch=obs_batch,
                route_target=route_target,
                alloc_target=alloc_target,
                route_coef=self.config.warmup_route_coef,
                diffusion_coef=self.config.warmup_diffusion_coef,
                prefix="warmup",
                update_idx=update + 1,
            )
            metrics.update(
                {
                    "phase": "warmup",
                    "reward_mean": float("nan"),
                    "latency_mean": float("nan"),
                    "deadline_hit_ratio": float("nan"),
                    **self.evaluate(
                        episodes=self.config.train_eval_episodes,
                        seed_base=self.config.validation_seed_base,
                    ),
                }
            )
            history.append(metrics)

        for update in trange(max(self.config.search_bootstrap_updates, 0), desc="SearchBootstrap", leave=False):
            obs_batch = self.collect_search_bootstrap_batch(update)
            with torch.no_grad():
                teacher = self.model.act_deterministic(obs_batch)
            metrics = self._run_supervised_update(
                obs_batch=obs_batch,
                route_target=teacher["route"].detach(),
                alloc_target=teacher["allocation"].detach(),
                route_coef=self.config.search_bootstrap_route_coef,
                diffusion_coef=self.config.search_bootstrap_diffusion_coef,
                prefix="search_bootstrap",
                update_idx=update + 1,
            )
            metrics.update(
                {
                    "phase": "search_bootstrap",
                    "reward_mean": float("nan"),
                    "latency_mean": float("nan"),
                    "deadline_hit_ratio": float("nan"),
                    **self.evaluate(
                        episodes=self.config.train_eval_episodes,
                        seed_base=self.config.validation_seed_base,
                    ),
                }
            )
            history.append(metrics)

        return history

    def train(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []

        for update in trange(self.config.train_updates, desc="Training", leave=False):
            self._set_training_schedule(update)
            rollout, rollout_metrics, episode_metrics = self.collect_rollout(update)
            self.episode_history.extend(episode_metrics)
            train_metrics = self.update_policy(rollout) if not self.freeze_updates else self._frozen_metrics()
            metrics = {
                "phase": "ppo",
                "update": float(update + 1),
                **rollout_metrics,
                **train_metrics,
            }

            if (update + 1) % self.config.eval_interval == 0:
                validation, eval_episode_metrics = self.evaluate(
                    episodes=self.config.train_eval_episodes,
                    seed_base=self.config.validation_seed_base,
                    record_episode_history=True,
                    update_idx=update + 1,
                )
                self.eval_episode_history.extend(eval_episode_metrics)
                metrics.update(validation)
                self._maybe_save_best(update + 1, validation)
                self._apply_plateau_control(validation)
            else:
                metrics.update(
                    {
                        "eval_reward": float("nan"),
                        "eval_latency": float("nan"),
                        "eval_deadline_hit_ratio": float("nan"),
                    }
                )

            metrics.update(
                {
                    "lr": float(self.current_lr),
                    "route_entropy_coef": float(self.current_route_entropy_coef),
                    "route_sampling_temperature": float(self.current_route_sampling_temperature),
                    "alloc_noise_scale": float(self.current_alloc_noise_scale),
                    "topology_prior_scale": float(self.current_topology_prior_scale),
                    "topology_distill_coef": float(self.current_topology_distill_coef),
                    "search_distill_scale": float(self.current_search_distill_scale),
                    "greedy_rollout": float(self.current_greedy_rollout_prob),
                    "freeze_updates": float(self.freeze_updates),
                    "rollback_count": float(self.rollback_count),
                    "no_improve_updates": float(self.no_improve_updates),
                    "plateau_window_ready": float(self.current_plateau_window_ready),
                    "plateau_window_reward": float(self.current_plateau_window_reward),
                    "plateau_window_spread": float(self.current_plateau_window_spread),
                }
            )

            history.append(metrics)

        return history

    def collect_warmup_batch(self, update_idx: int) -> tuple[Dict[str, Tensor], Tensor, Tensor]:
        obs_list = []
        route_targets = []
        allocation_targets = []

        seed_cursor = self.config.warmup_seed_base + update_idx * 1_000
        obs = self.env.reset(seed=seed_cursor)

        while len(obs_list) < self.config.warmup_batch_size:
            route_idx, allocation = self._heuristic_action(obs)

            obs_list.append(self._copy_observation(obs))
            route_targets.append(route_idx)
            allocation_targets.append(allocation.astype(np.float32))

            next_obs, _, done, _ = self.env.step({"route_idx": route_idx, "allocation": allocation})
            if done:
                seed_cursor += 1
                obs = self.env.reset(seed=seed_cursor)
            else:
                obs = next_obs

        obs_batch = self._stack_observations(obs_list)
        route_tensor = torch.as_tensor(route_targets, dtype=torch.long, device=self.config.device)
        allocation_tensor = torch.as_tensor(
            np.asarray(allocation_targets), dtype=torch.float32, device=self.config.device
        )
        return obs_batch, route_tensor, allocation_tensor

    def collect_search_bootstrap_batch(self, update_idx: int) -> Dict[str, Tensor]:
        obs_list = []
        seed_cursor = self.config.warmup_seed_base + 50_000 + update_idx * 1_000
        obs = self.env.reset(seed=seed_cursor)

        while len(obs_list) < self.config.search_bootstrap_batch_size:
            obs_list.append(self._copy_observation(obs))
            route_idx, allocation = self._heuristic_action(obs)
            next_obs, _, done, _ = self.env.step({"route_idx": route_idx, "allocation": allocation})
            if done:
                seed_cursor += 1
                obs = self.env.reset(seed=seed_cursor)
            else:
                obs = next_obs

        return self._stack_observations(obs_list)

    def collect_rollout(self, update_idx: int) -> tuple[RolloutBatch, Dict[str, float], List[Dict[str, float]]]:
        if self.config.use_multi_task_slots:
            return self._collect_multi_task_rollout(update_idx)

        observation_store = {key: [] for key in OBS_FLOAT_KEYS + OBS_LONG_KEYS}
        route_actions = []
        allocations = []
        old_route_log_probs = []
        rewards = []
        dones = []
        values = []
        metrics: List[Dict[str, float]] = []
        advantages_all = []
        returns_all = []
        episode_summaries: List[Dict[str, float]] = []
        episode_counter = len(self.episode_history)
        num_envs = self.config.rollout_episodes_per_update
        rollout_seeds = self._select_rollout_seeds(update_idx, num_envs)
        observations = self._reset_rollout_envs(rollout_seeds)
        greedy_episode_mask: np.ndarray | None = None
        if not self.freeze_updates:
            greedy_prob = float(np.clip(self.current_greedy_rollout_prob, 0.0, 1.0))
            if greedy_prob > 0.0:
                rng = np.random.default_rng(self.config.seed + 1_000_003 * (update_idx + 1))
                greedy_episode_mask = rng.random(num_envs) < greedy_prob
        episode_rewards = [[] for _ in range(num_envs)]
        episode_dones = [[] for _ in range(num_envs)]
        episode_values = [[] for _ in range(num_envs)]
        episode_step_metrics: List[List[Dict[str, float]]] = [[] for _ in range(num_envs)]

        for _ in range(self.config.episode_length):
            obs_batch = self._stack_observations(observations)
            with torch.inference_mode():
                if self.freeze_updates:
                    action = self.model.act_deterministic(obs_batch)
                else:
                    action = self.model.act(
                        obs_batch,
                        deterministic_allocation=self.config.rollout_deterministic_allocation,
                        deterministic_route=False,
                    )
                    if greedy_episode_mask is not None and bool(greedy_episode_mask.any()):
                        deterministic_action = self.model.act(
                            obs_batch,
                            deterministic_allocation=True,
                            deterministic_route=True,
                        )
                        route_mask = torch.as_tensor(
                            greedy_episode_mask,
                            dtype=torch.bool,
                            device=action["route"].device,
                        )
                        allocation_mask = route_mask.unsqueeze(-1)
                        action = {
                            **action,
                            "route": torch.where(route_mask, deterministic_action["route"], action["route"]),
                            "allocation": torch.where(
                                allocation_mask,
                                deterministic_action["allocation"],
                                action["allocation"],
                            ),
                            "route_log_prob": torch.where(
                                route_mask,
                                deterministic_action["route_log_prob"],
                                action["route_log_prob"],
                            ),
                            "value": torch.where(route_mask, deterministic_action["value"], action["value"]),
                        }

            route_batch = action["route"].detach().cpu().tolist()
            allocation_batch = action["allocation"].detach().cpu().numpy()
            log_prob_batch = action["route_log_prob"].detach().cpu().tolist()
            value_batch = action["value"].detach().cpu().tolist()
            step_results = self._step_rollout_envs(
                [
                    {"route_idx": int(route_batch[env_idx]), "allocation": allocation_batch[env_idx]}
                    for env_idx in range(num_envs)
                ]
            )
            next_observations: List[Dict[str, np.ndarray] | None] = [None] * num_envs

            for env_idx, (next_obs, reward, done, info) in enumerate(step_results):
                obs = observations[env_idx]
                allocation = allocation_batch[env_idx]
                route_idx = int(route_batch[env_idx])

                for key in OBS_FLOAT_KEYS:
                    observation_store[key].append(obs[key].copy())
                for key in OBS_LONG_KEYS:
                    observation_store[key].append(obs[key].copy())

                route_actions.append(route_idx)
                allocations.append(allocation.astype(np.float32))
                old_route_log_probs.append(float(log_prob_batch[env_idx]))
                rewards.append(float(reward))
                dones.append(float(done))
                values.append(float(value_batch[env_idx]))
                metrics.append(info)

                episode_rewards[env_idx].append(float(reward))
                episode_dones[env_idx].append(float(done))
                episode_values[env_idx].append(float(value_batch[env_idx]))
                episode_step_metrics[env_idx].append(info)

                if not done:
                    next_observations[env_idx] = next_obs

            if greedy_episode_mask is not None:
                greedy_episode_mask = np.asarray(
                    [greedy_episode_mask[idx] for idx, (_, _, done, _) in enumerate(step_results) if not done],
                    dtype=np.bool_,
                )
                if greedy_episode_mask.size == 0:
                    greedy_episode_mask = None
            observations = [obs for obs in next_observations if obs is not None]
            if not observations:
                break

        for episode_idx in range(num_envs):
            ep_advantages, ep_returns = self._compute_gae(
                np.asarray(episode_rewards[episode_idx], dtype=np.float32),
                np.asarray(episode_dones[episode_idx], dtype=np.float32),
                np.asarray(episode_values[episode_idx], dtype=np.float32),
                last_value=0.0,
            )
            advantages_all.append(ep_advantages)
            returns_all.append(ep_returns)
            episode_summaries.append(
                self._build_episode_summary(
                    update_idx=update_idx,
                    episode_in_group=episode_idx,
                    episode_index=episode_counter + episode_idx,
                    episode_seed=rollout_seeds[episode_idx],
                    step_metrics=episode_step_metrics[episode_idx],
                    step_rewards=episode_rewards[episode_idx],
                )
            )

        advantages = np.concatenate(advantages_all, axis=0)
        returns = np.concatenate(returns_all, axis=0)

        obs_tensors = {
            key: torch.as_tensor(np.asarray(values_list), dtype=torch.float32, device=self.config.device)
            for key, values_list in observation_store.items()
            if key in OBS_FLOAT_KEYS
        }
        obs_tensors.update(
            {
                key: torch.as_tensor(np.asarray(values_list), dtype=torch.long, device=self.config.device)
                for key, values_list in observation_store.items()
                if key in OBS_LONG_KEYS
            }
        )

        batch = RolloutBatch(
            observations=obs_tensors,
            route_actions=torch.as_tensor(route_actions, dtype=torch.long, device=self.config.device),
            allocations=torch.as_tensor(np.asarray(allocations), dtype=torch.float32, device=self.config.device),
            old_route_log_probs=torch.as_tensor(
                old_route_log_probs, dtype=torch.float32, device=self.config.device
            ),
            old_values=torch.as_tensor(values, dtype=torch.float32, device=self.config.device),
            returns=torch.as_tensor(returns, dtype=torch.float32, device=self.config.device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=self.config.device),
        )

        rollout_metrics = self._summarize_metrics(metrics, rewards)
        return batch, rollout_metrics, episode_summaries

    def _collect_multi_task_rollout(
        self,
        update_idx: int,
    ) -> tuple[RolloutBatch, Dict[str, float], List[Dict[str, float]]]:
        observation_store = {key: [] for key in OBS_FLOAT_KEYS + OBS_LONG_KEYS}
        route_actions: list[int] = []
        allocations: list[np.ndarray] = []
        old_route_log_probs: list[float] = []
        old_values: list[float] = []
        returns: list[float] = []
        advantages: list[float] = []
        slot_metrics: List[Dict[str, float]] = []
        slot_rewards: list[float] = []
        episode_summaries: List[Dict[str, float]] = []

        num_envs = max(int(self.config.rollout_episodes_per_update), 1)
        rollout_seeds = self._select_rollout_seeds(update_idx, num_envs)
        active_envs = self.rollout_envs[:num_envs]
        observations = [
            env.reset_slot(seed=seed)
            for env, seed in zip(active_envs, rollout_seeds)
        ]
        episode_step_metrics: List[List[Dict[str, float]]] = [
            [] for _ in range(num_envs)
        ]
        episode_step_rewards: List[List[float]] = [
            [] for _ in range(num_envs)
        ]
        episode_counter = len(self.episode_history)

        for _ in range(self.config.episode_length):
            task_counts = [int(obs["task_features"].shape[0]) for obs in observations]
            obs_batch = self._concatenate_slot_observations(observations)
            with torch.inference_mode():
                if self.freeze_updates:
                    action = self.model.act_deterministic(obs_batch)
                else:
                    action = self.model.act(
                        obs_batch,
                        deterministic_allocation=self.config.rollout_deterministic_allocation,
                        deterministic_route=False,
                    )

            route_batch = action["route"].detach().cpu().numpy()
            allocation_batch = action["allocation"].detach().cpu().numpy()
            log_prob_batch = action["route_log_prob"].detach().cpu().numpy()
            value_batch = action["value"].detach().cpu().numpy()

            step_results = []
            offset = 0
            for env, count in zip(active_envs, task_counts):
                next_offset = offset + count
                step_results.append(
                    env.step_slot(
                        {
                            "route_idx": route_batch[offset:next_offset],
                            "allocation": allocation_batch[offset:next_offset],
                        }
                    )
                )
                offset = next_offset

            next_observations = [
                result[0] for result in step_results if result[0] is not None
            ]
            next_value_means = np.zeros(num_envs, dtype=np.float32)
            if next_observations:
                next_counts = [
                    int(obs["task_features"].shape[0])
                    for obs in next_observations
                ]
                next_batch = self._concatenate_slot_observations(next_observations)
                with torch.inference_mode():
                    next_values = self.model.value(next_batch).detach().cpu().numpy()
                offset = 0
                for env_idx, count in enumerate(next_counts):
                    next_value_means[env_idx] = float(
                        np.mean(next_values[offset : offset + count])
                    )
                    offset += count

            offset = 0
            for env_idx, (next_obs, slot_reward, done, info) in enumerate(step_results):
                count = task_counts[env_idx]
                next_offset = offset + count
                current_obs = observations[env_idx]
                for key in OBS_FLOAT_KEYS + OBS_LONG_KEYS:
                    observation_store[key].extend(
                        np.asarray(current_obs[key]).copy()
                    )

                task_rewards = np.asarray(
                    [
                        float(task_info["reward"])
                        for task_info in info["task_infos"]
                    ],
                    dtype=np.float32,
                )
                bootstrap = 0.0 if done else self.config.gamma * next_value_means[env_idx]
                task_returns = task_rewards + bootstrap
                task_values = value_batch[offset:next_offset]

                route_actions.extend(route_batch[offset:next_offset].astype(np.int64).tolist())
                allocations.extend(allocation_batch[offset:next_offset].astype(np.float32))
                old_route_log_probs.extend(
                    log_prob_batch[offset:next_offset].astype(np.float32).tolist()
                )
                old_values.extend(task_values.astype(np.float32).tolist())
                returns.extend(task_returns.astype(np.float32).tolist())
                advantages.extend((task_returns - task_values).astype(np.float32).tolist())

                numeric_info = {
                    key: float(value)
                    for key, value in info.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
                slot_metrics.append(numeric_info)
                slot_rewards.append(float(slot_reward))
                episode_step_metrics[env_idx].append(numeric_info)
                episode_step_rewards[env_idx].append(float(slot_reward))
                offset = next_offset

            observations = [
                result[0] for result in step_results if result[0] is not None
            ]
            if not observations:
                break

        for episode_idx in range(num_envs):
            episode_summaries.append(
                self._build_episode_summary(
                    update_idx=update_idx,
                    episode_in_group=episode_idx,
                    episode_index=episode_counter + episode_idx,
                    episode_seed=rollout_seeds[episode_idx],
                    step_metrics=episode_step_metrics[episode_idx],
                    step_rewards=episode_step_rewards[episode_idx],
                )
            )

        obs_tensors = {
            key: torch.as_tensor(
                np.asarray(values),
                dtype=torch.float32,
                device=self.config.device,
            )
            for key, values in observation_store.items()
            if key in OBS_FLOAT_KEYS
        }
        obs_tensors.update(
            {
                key: torch.as_tensor(
                    np.asarray(values),
                    dtype=torch.long,
                    device=self.config.device,
                )
                for key, values in observation_store.items()
                if key in OBS_LONG_KEYS
            }
        )
        batch = RolloutBatch(
            observations=obs_tensors,
            route_actions=torch.as_tensor(
                route_actions, dtype=torch.long, device=self.config.device
            ),
            allocations=torch.as_tensor(
                np.asarray(allocations),
                dtype=torch.float32,
                device=self.config.device,
            ),
            old_route_log_probs=torch.as_tensor(
                old_route_log_probs,
                dtype=torch.float32,
                device=self.config.device,
            ),
            old_values=torch.as_tensor(
                old_values, dtype=torch.float32, device=self.config.device
            ),
            returns=torch.as_tensor(
                returns, dtype=torch.float32, device=self.config.device
            ),
            advantages=torch.as_tensor(
                advantages, dtype=torch.float32, device=self.config.device
            ),
        )
        return (
            batch,
            self._summarize_metrics(slot_metrics, slot_rewards),
            episode_summaries,
        )

    def _concatenate_slot_observations(
        self,
        observations: List[Dict[str, np.ndarray]],
    ) -> Dict[str, Tensor]:
        batch: Dict[str, Tensor] = {}
        for key in OBS_FLOAT_KEYS:
            batch[key] = torch.as_tensor(
                np.concatenate([obs[key] for obs in observations], axis=0),
                dtype=torch.float32,
                device=self.config.device,
            )
        for key in OBS_LONG_KEYS:
            batch[key] = torch.as_tensor(
                np.concatenate([obs[key] for obs in observations], axis=0),
                dtype=torch.long,
                device=self.config.device,
            )
        return batch

    def update_policy(self, batch: RolloutBatch) -> Dict[str, float]:
        num_samples = batch.route_actions.size(0)
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))
        if self.config.advantage_clip > 0.0:
            advantages = advantages.clamp(-self.config.advantage_clip, self.config.advantage_clip)

        last_loss = 0.0
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_route_entropy = 0.0
        last_alloc_refinement_norm = 0.0
        last_diffusion_loss = 0.0
        last_diffusion_noise_loss = 0.0
        last_diffusion_recon_loss = 0.0
        last_diffusion_prior_loss = 0.0
        last_adv_weight = 0.0
        last_pred_noise_norm = 0.0
        last_allocation_shift = 0.0
        last_search_route_loss = 0.0
        last_search_diffusion_loss = 0.0
        last_search_route_agreement = 0.0
        last_search_improvement = 0.0
        last_search_relative_gain = 0.0
        last_search_active_ratio = 0.0
        last_search_weight_mean = 0.0
        last_topology_route_loss = 0.0
        last_topology_route_agreement = 0.0
        last_value_scale = 1.0
        last_approx_kl = 0.0

        search_route_targets = batch.route_actions.detach()
        search_alloc_targets = batch.allocations.detach()
        search_improvement = torch.zeros_like(batch.advantages)
        search_relative_gain = torch.zeros_like(batch.advantages)
        search_weights = torch.zeros_like(batch.advantages)
        topology_target_probs: Tensor | None = None
        if self.config.use_search_distillation:
            with torch.inference_mode():
                search_teacher = self.model.act_deterministic(batch.observations)
                rollout_action_stats = self.model.evaluate_joint_action(
                    batch.observations,
                    batch.route_actions,
                    batch.allocations,
                )
            search_route_targets = search_teacher["route"].detach()
            search_alloc_targets = search_teacher["allocation"].detach()
            search_joint_scores = search_teacher["route_search_score"].detach()
            rollout_joint_scores = rollout_action_stats["joint_score"].detach()
            search_improvement = (rollout_joint_scores - search_joint_scores).clamp_min(0.0)
            search_relative_gain = search_improvement / rollout_joint_scores.abs().clamp_min(1e-6)
            search_weights = torch.clamp(
                (search_relative_gain - self.config.search_distill_relative_threshold).clamp_min(0.0)
                / max(self.config.search_distill_relative_threshold, 1e-6),
                max=self.config.search_distill_max_weight,
            )
        if self.config.use_topology_distillation:
            with torch.inference_mode():
                topology_prior_logits = self.model.topology_route_prior(batch.observations)
            topology_target_probs = torch.softmax(
                topology_prior_logits / max(self.config.topology_route_temperature, 1e-6),
                dim=-1,
            )

        stop_early = False
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(num_samples, device=self.config.device)
            for start in range(0, num_samples, self.config.mini_batch_size):
                idx = permutation[start : start + self.config.mini_batch_size]
                obs_mb = {key: value[idx] for key, value in batch.observations.items()}
                route_mb = batch.route_actions[idx]
                alloc_mb = batch.allocations[idx]
                old_route_log_prob_mb = batch.old_route_log_probs[idx]
                old_value_mb = batch.old_values[idx]
                return_mb = batch.returns[idx]
                advantage_mb = advantages[idx]
                search_route_mb = search_route_targets[idx]
                search_alloc_mb = search_alloc_targets[idx]
                search_weight_mb = search_weights[idx]
                search_improvement_mb = search_improvement[idx]
                search_relative_gain_mb = search_relative_gain[idx]
                topology_target_probs_mb = topology_target_probs[idx] if topology_target_probs is not None else None

                route_eval = self.model.evaluate_route(obs_mb, route_mb)
                ratio = torch.exp(route_eval["route_log_prob"] - old_route_log_prob_mb)
                approx_kl = (old_route_log_prob_mb - route_eval["route_log_prob"]).mean()

                unclipped = ratio * advantage_mb
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantage_mb
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_scale = return_mb.std().detach().clamp_min(1.0)
                scaled_value = route_eval["value"] / value_scale
                scaled_old_value = old_value_mb / value_scale
                scaled_return = return_mb / value_scale
                if self.config.value_clip_eps > 0.0:
                    scaled_value_clipped = scaled_old_value + torch.clamp(
                        scaled_value - scaled_old_value,
                        -self.config.value_clip_eps,
                        self.config.value_clip_eps,
                    )
                    value_loss_unclipped = (scaled_value - scaled_return).pow(2)
                    value_loss_clipped = (scaled_value_clipped - scaled_return).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(scaled_value, scaled_return)
                route_entropy = route_eval["route_entropy"].mean()
                zero_term = route_entropy.new_zeros(())
                if topology_target_probs_mb is not None:
                    policy_route_log_probs = F.log_softmax(route_eval["policy_route_logits"], dim=-1)
                    topology_route_loss = -(topology_target_probs_mb * policy_route_log_probs).sum(dim=-1).mean()
                    topology_route_agreement = (
                        route_eval["policy_route_logits"].argmax(dim=-1) == topology_target_probs_mb.argmax(dim=-1)
                    ).float().mean()
                else:
                    topology_route_loss = zero_term
                    topology_route_agreement = zero_term
                diffusion_stats = self.model.diffusion_training_loss(
                    obs=obs_mb,
                    route=route_mb,
                    target_allocation=alloc_mb,
                    advantage=advantage_mb,
                    eta=self.config.adv_weight_eta,
                    wmin=self.config.adv_weight_min,
                    wmax=self.config.adv_weight_max,
                    recon_coef=self.config.diffusion_recon_coef,
                    prior_coef=self.config.diffusion_prior_coef,
                )
                if self.config.use_search_distillation:
                    search_route_loss_per_sample = torch.nn.functional.cross_entropy(
                        route_eval["policy_route_logits"],
                        search_route_mb,
                        reduction="none",
                    )
                    search_route_weight_sum = search_weight_mb.sum().clamp_min(1e-6)
                    search_route_loss = (search_route_loss_per_sample * search_weight_mb).sum() / search_route_weight_sum
                    zero_advantage = torch.zeros_like(advantage_mb)
                    search_diffusion_stats = self.model.diffusion_training_loss(
                        obs=obs_mb,
                        route=search_route_mb,
                        target_allocation=search_alloc_mb,
                        advantage=zero_advantage,
                        eta=0.0,
                        wmin=1.0,
                        wmax=1.0,
                        recon_coef=self.config.diffusion_recon_coef,
                        prior_coef=self.config.diffusion_prior_coef,
                        sample_weight=search_weight_mb,
                    )
                    search_route_agreement = (
                        route_eval["route_logits"].argmax(dim=-1) == search_route_mb
                    ).float().mean()
                    search_diffusion_loss = search_diffusion_stats["diffusion_loss"]
                else:
                    search_route_loss = zero_term
                    search_route_agreement = zero_term
                    search_diffusion_loss = zero_term

                loss = (
                    self.config.policy_coef * policy_loss
                    + self.config.value_coef * value_loss
                    + self.current_topology_distill_coef * topology_route_loss
                    + self.config.diffusion_loss_coef * diffusion_stats["diffusion_loss"]
                    + self.current_search_distill_scale * self.config.search_distill_route_coef * search_route_loss
                    + self.current_search_distill_scale
                    * self.config.search_distill_diffusion_coef
                    * search_diffusion_loss
                    - self.current_route_entropy_coef * route_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                last_loss = float(loss.item())
                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_route_entropy = float(route_entropy.item())
                last_alloc_refinement_norm = float(diffusion_stats["denoised_allocation_shift"].item())
                last_diffusion_loss = float(diffusion_stats["diffusion_loss"].item())
                last_diffusion_noise_loss = float(diffusion_stats["noise_loss"].item())
                last_diffusion_recon_loss = float(diffusion_stats["recon_loss"].item())
                last_diffusion_prior_loss = float(diffusion_stats["prior_loss"].item())
                last_adv_weight = float(diffusion_stats["adv_weight"].item())
                last_pred_noise_norm = float(diffusion_stats["pred_noise_norm"].item())
                last_allocation_shift = float(diffusion_stats["denoised_allocation_shift"].item())
                last_search_route_loss = float(search_route_loss.item())
                last_search_diffusion_loss = float(search_diffusion_loss.item())
                last_search_route_agreement = float(search_route_agreement.item())
                last_search_improvement = float(search_improvement_mb.mean().item())
                last_search_relative_gain = float(search_relative_gain_mb.mean().item())
                last_search_active_ratio = float((search_weight_mb > 0.0).float().mean().item())
                last_search_weight_mean = float(search_weight_mb.mean().item())
                last_topology_route_loss = float(topology_route_loss.item())
                last_topology_route_agreement = float(topology_route_agreement.item())
                last_value_scale = float(value_scale.item())
                last_approx_kl = float(approx_kl.item())

                if self.config.target_kl > 0.0 and last_approx_kl > self.config.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        return {
            "loss": last_loss,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "value_scale": last_value_scale,
            "approx_kl": last_approx_kl,
            "route_entropy": last_route_entropy,
            "topology_route_loss": last_topology_route_loss,
            "topology_route_agreement": last_topology_route_agreement,
            "alloc_refinement_norm": last_alloc_refinement_norm,
            "diffusion_loss": last_diffusion_loss,
            "diffusion_noise_loss": last_diffusion_noise_loss,
            "diffusion_recon_loss": last_diffusion_recon_loss,
            "diffusion_prior_loss": last_diffusion_prior_loss,
            "adv_weight_mean": last_adv_weight,
            "pred_noise_norm": last_pred_noise_norm,
            "allocation_shift": last_allocation_shift,
            "search_route_loss": last_search_route_loss,
            "search_diffusion_loss": last_search_diffusion_loss,
            "search_route_agreement": last_search_route_agreement,
            "search_improvement_mean": last_search_improvement,
            "search_relative_gain_mean": last_search_relative_gain,
            "search_active_ratio": last_search_active_ratio,
            "search_weight_mean": last_search_weight_mean,
        }

    def _set_training_schedule(self, update_idx: int) -> None:
        configured_updates = int(self.config.exploration_anneal_updates)
        schedule_updates = configured_updates if configured_updates > 0 else int(self.config.train_updates)
        if schedule_updates <= 1:
            progress = 1.0
        else:
            progress = float(np.clip(update_idx / float(schedule_updates - 1), 0.0, 1.0))

        hold_fraction = min(max(self.config.exploration_hold_fraction, 0.0), 0.95)
        if progress <= hold_fraction:
            anneal_progress = 0.0
        else:
            anneal_progress = (progress - hold_fraction) / max(1.0 - hold_fraction, 1e-6)
        anneal_progress = float(np.clip(anneal_progress, 0.0, 1.0))
        anneal = anneal_progress ** max(self.config.exploration_decay_power, 1e-6)

        lr_configured_updates = int(self.config.learning_rate_anneal_updates)
        lr_schedule_updates = lr_configured_updates if lr_configured_updates > 0 else schedule_updates
        if lr_schedule_updates <= 1:
            lr_progress = 1.0
        else:
            lr_progress = float(np.clip(update_idx / float(lr_schedule_updates - 1), 0.0, 1.0))
        if lr_progress <= hold_fraction:
            lr_anneal_progress = 0.0
        else:
            lr_anneal_progress = (lr_progress - hold_fraction) / max(1.0 - hold_fraction, 1e-6)
        lr_anneal_progress = float(np.clip(lr_anneal_progress, 0.0, 1.0))
        lr_anneal = lr_anneal_progress ** max(self.config.exploration_decay_power, 1e-6)

        scheduled_lr = self.config.lr * (
            (1.0 - lr_anneal) + lr_anneal * self.config.lr_final_factor
        )
        warmup_updates = max(int(self.config.learning_rate_warmup_updates), 0)
        if warmup_updates > 1 and update_idx < warmup_updates:
            initial_factor = float(np.clip(self.config.learning_rate_warmup_initial_factor, 0.0, 1.0))
            warmup_progress = float(update_idx) / float(warmup_updates - 1)
            warmup_factor = initial_factor + (1.0 - initial_factor) * warmup_progress
        else:
            warmup_factor = 1.0
        self.current_lr = scheduled_lr * warmup_factor
        for param_group, group_scale in zip(self.optimizer.param_groups, self.optimizer_group_scales):
            param_group["lr"] = self.current_lr * group_scale

        self.current_route_entropy_coef = (
            self.config.route_entropy_coef * (1.0 - anneal)
            + self.config.route_entropy_final_coef * anneal
        )
        self.current_route_sampling_temperature = (
            self.config.route_sampling_temperature_initial * (1.0 - anneal)
            + self.config.route_sampling_temperature_final * anneal
        )
        if self.config.use_topology_route_prior:
            self.current_topology_prior_scale = (
                self.config.topology_route_prior_initial_scale * (1.0 - anneal)
                + self.config.topology_route_prior_final_scale * anneal
            )
        else:
            self.current_topology_prior_scale = 0.0
        if self.config.use_topology_distillation:
            self.current_topology_distill_coef = (
                self.config.topology_distill_coef * (1.0 - anneal)
                + self.config.topology_distill_final_coef * anneal
            )
        else:
            self.current_topology_distill_coef = 0.0
        self.current_alloc_noise_scale = (
            self.config.alloc_noise_scale * (1.0 - anneal)
            + self.config.alloc_noise_final_scale * anneal
        )
        self.model.set_route_sampling_temperature(self.current_route_sampling_temperature)
        self.model.set_allocation_noise_scale(self.current_alloc_noise_scale)
        self.model.set_topology_prior_scale(self.current_topology_prior_scale)

        delay_fraction = min(max(self.config.search_distill_delay_fraction, 0.0), 1.0)
        ramp_fraction = max(self.config.search_distill_ramp_fraction, 0.0)
        if not self.config.use_search_distillation:
            self.current_search_distill_scale = 0.0
        elif progress <= delay_fraction:
            self.current_search_distill_scale = 0.0
        elif ramp_fraction <= 1e-6:
            self.current_search_distill_scale = 1.0
        else:
            ramp_progress = (progress - delay_fraction) / ramp_fraction
            self.current_search_distill_scale = float(np.clip(ramp_progress, 0.0, 1.0))

        greedy_start = float(np.clip(self.config.greedy_rollout_fraction, 0.0, 1.0))
        greedy_ramp = max(float(self.config.greedy_rollout_ramp_fraction), 1e-6)
        greedy_initial_prob = float(np.clip(self.config.greedy_rollout_initial_prob, 0.0, 1.0))
        greedy_ramp_power = max(float(self.config.greedy_rollout_ramp_power), 1e-6)
        if progress <= greedy_start:
            self.current_greedy_rollout_prob = greedy_initial_prob
        else:
            greedy_progress = (progress - greedy_start) / greedy_ramp
            greedy_progress = float(np.clip(greedy_progress, 0.0, 1.0)) ** greedy_ramp_power
            self.current_greedy_rollout_prob = float(
                np.clip(
                    greedy_initial_prob + (1.0 - greedy_initial_prob) * greedy_progress,
                    greedy_initial_prob,
                    1.0,
                )
            )

    def _run_supervised_update(
        self,
        obs_batch: Dict[str, Tensor],
        route_target: Tensor,
        alloc_target: Tensor,
        route_coef: float,
        diffusion_coef: float,
        prefix: str,
        update_idx: int,
    ) -> Dict[str, float]:
        route_eval = self.model.evaluate_route(obs_batch, route_target)
        route_loss = F.cross_entropy(route_eval["policy_route_logits"], route_target)
        route_accuracy = (route_eval["policy_route_logits"].argmax(dim=-1) == route_target).float().mean()

        zero_advantage = torch.zeros_like(route_target, dtype=torch.float32, device=self.config.device)
        diffusion_stats = self.model.diffusion_training_loss(
            obs=obs_batch,
            route=route_target,
            target_allocation=alloc_target,
            advantage=zero_advantage,
            eta=0.0,
            wmin=1.0,
            wmax=1.0,
            recon_coef=self.config.diffusion_recon_coef,
            prior_coef=self.config.diffusion_prior_coef,
        )

        loss = route_coef * route_loss + diffusion_coef * diffusion_stats["diffusion_loss"]

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            f"{prefix}_update": float(update_idx),
            f"{prefix}_loss": float(loss.item()),
            f"{prefix}_route_loss": float(route_loss.item()),
            f"{prefix}_route_accuracy": float(route_accuracy.item()),
            f"{prefix}_diffusion_loss": float(diffusion_stats["diffusion_loss"].item()),
            f"{prefix}_noise_loss": float(diffusion_stats["noise_loss"].item()),
            f"{prefix}_recon_loss": float(diffusion_stats["recon_loss"].item()),
            f"{prefix}_prior_loss": float(diffusion_stats["prior_loss"].item()),
            f"{prefix}_pred_noise_norm": float(diffusion_stats["pred_noise_norm"].item()),
            f"{prefix}_allocation_shift": float(diffusion_stats["denoised_allocation_shift"].item()),
        }

    def _frozen_metrics(self) -> Dict[str, float]:
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "value_scale": float("nan"),
            "approx_kl": float("nan"),
            "route_entropy": float("nan"),
            "topology_route_loss": float("nan"),
            "topology_route_agreement": float("nan"),
            "alloc_refinement_norm": float("nan"),
            "diffusion_loss": float("nan"),
            "diffusion_noise_loss": float("nan"),
            "diffusion_recon_loss": float("nan"),
            "diffusion_prior_loss": float("nan"),
            "adv_weight_mean": float("nan"),
            "pred_noise_norm": float("nan"),
            "allocation_shift": float("nan"),
            "search_route_loss": float("nan"),
            "search_diffusion_loss": float("nan"),
            "search_route_agreement": float("nan"),
            "search_improvement_mean": float("nan"),
            "search_relative_gain_mean": float("nan"),
            "search_active_ratio": float("nan"),
            "search_weight_mean": float("nan"),
        }

    def evaluate(
        self,
        episodes: int,
        seed_base: int | None = None,
        record_episode_history: bool = False,
        update_idx: int | None = None,
    ) -> Dict[str, float] | tuple[Dict[str, float], List[Dict[str, float]]]:
        if self.config.use_multi_task_slots:
            return self._evaluate_multi_task(
                episodes=episodes,
                seed_base=seed_base,
                record_episode_history=record_episode_history,
                update_idx=update_idx,
            )
        episode_metrics = []
        if seed_base is None:
            seed_base = self.config.final_eval_seed_base
        eval_episode_metrics: List[Dict[str, float]] = []
        eval_episode_offset = len(self.eval_episode_history)
        observations = self._reset_eval_envs([seed_base + episode for episode in range(episodes)])
        step_metrics_per_episode: List[List[Dict[str, float]]] = [[] for _ in range(episodes)]
        step_rewards_per_episode: List[List[float]] = [[] for _ in range(episodes)]

        for _ in range(self.config.episode_length):
            obs_batch = self._stack_observations(observations)
            with torch.inference_mode():
                action = self.model.act_deterministic(obs_batch)
            route_batch = action["route"].detach().cpu().tolist()
            allocation_batch = action["allocation"].detach().cpu().numpy()
            step_results = self._step_eval_envs(
                [
                    {"route_idx": int(route_batch[episode_idx]), "allocation": allocation_batch[episode_idx]}
                    for episode_idx in range(episodes)
                ]
            )
            next_observations: List[Dict[str, np.ndarray] | None] = [None] * episodes

            for episode_idx, (next_obs, reward, done, info) in enumerate(step_results):
                step_metrics_per_episode[episode_idx].append(info)
                step_rewards_per_episode[episode_idx].append(float(reward))
                if not done:
                    next_observations[episode_idx] = next_obs

            observations = [obs for obs in next_observations if obs is not None]
            if not observations:
                break

        for episode in range(episodes):
            step_metrics = step_metrics_per_episode[episode]
            step_rewards = step_rewards_per_episode[episode]
            avg_latency = float(np.mean([metric["latency"] for metric in step_metrics]))
            avg_reward = float(np.mean(step_rewards))
            episode_reward = float(np.sum(step_rewards))
            deadline_hit_ratio = float(np.mean([metric["deadline_hit"] for metric in step_metrics]))
            episode_metrics.append(
                {
                    "eval_reward": avg_reward,
                    "eval_latency": avg_latency,
                    "eval_deadline_hit_ratio": deadline_hit_ratio,
                }
            )
            if record_episode_history:
                episode_raw_reward = (
                    float(np.sum([metric["raw_reward"] for metric in step_metrics]))
                    if step_metrics and "raw_reward" in step_metrics[0]
                    else float("nan")
                )
                episode_baseline_reward = (
                    float(np.sum([metric["baseline_reward"] for metric in step_metrics]))
                    if step_metrics and "baseline_reward" in step_metrics[0]
                    else float("nan")
                )
                eval_episode_metrics.append(
                    {
                        "update": float(update_idx if update_idx is not None else 0),
                        "episode_in_eval": float(episode + 1),
                        "episode_index": float(eval_episode_offset + episode + 1),
                        "episode_reward": episode_reward,
                        "episode_raw_reward": episode_raw_reward,
                        "episode_baseline_reward": episode_baseline_reward,
                        "episode_reward_mean": avg_reward,
                        "episode_length": float(len(step_rewards)),
                        "episode_latency": avg_latency,
                        "episode_deadline_hit_ratio": deadline_hit_ratio,
                        "episode_violation_mean": float(np.mean([metric["violation"] for metric in step_metrics])),
                        "episode_path_available_ratio": float(
                            np.mean([metric["path_available"] for metric in step_metrics])
                        ),
                    }
                )

        summary = {
            key: float(np.mean([metric[key] for metric in episode_metrics]))
            for key in episode_metrics[0]
        }
        if record_episode_history:
            return summary, eval_episode_metrics
        return summary

    def _evaluate_multi_task(
        self,
        episodes: int,
        seed_base: int | None,
        record_episode_history: bool,
        update_idx: int | None,
    ) -> Dict[str, float] | tuple[Dict[str, float], List[Dict[str, float]]]:
        if seed_base is None:
            seed_base = self.config.final_eval_seed_base
        episode_metrics: List[Dict[str, float]] = []
        eval_episode_metrics: List[Dict[str, float]] = []
        eval_episode_offset = len(self.eval_episode_history)

        for episode_idx in range(episodes):
            env = self.eval_envs[episode_idx]
            observation = env.reset_slot(seed=seed_base + episode_idx)
            step_metrics: List[Dict[str, float]] = []
            step_rewards: List[float] = []

            while True:
                obs_batch = self._concatenate_slot_observations([observation])
                with torch.inference_mode():
                    action = self.model.act_deterministic(obs_batch)
                next_obs, reward, done, info = env.step_slot(
                    {
                        "route_idx": action["route"].detach().cpu().numpy(),
                        "allocation": action["allocation"].detach().cpu().numpy(),
                    }
                )
                numeric_info = {
                    key: float(value)
                    for key, value in info.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                }
                step_metrics.append(numeric_info)
                step_rewards.append(float(reward))
                if done:
                    break
                observation = next_obs

            avg_reward = float(np.mean(step_rewards))
            avg_latency = float(np.mean([item["latency"] for item in step_metrics]))
            hit_ratio = float(np.mean([item["deadline_hit"] for item in step_metrics]))
            episode_metrics.append(
                {
                    "eval_reward": avg_reward,
                    "eval_latency": avg_latency,
                    "eval_deadline_hit_ratio": hit_ratio,
                }
            )
            if record_episode_history:
                eval_episode_metrics.append(
                    {
                        "update": float(update_idx if update_idx is not None else 0),
                        "episode_in_eval": float(episode_idx + 1),
                        "episode_index": float(eval_episode_offset + episode_idx + 1),
                        "episode_reward": float(np.sum(step_rewards)),
                        "episode_raw_reward": float(np.sum(step_rewards)),
                        "episode_baseline_reward": float("nan"),
                        "episode_reward_mean": avg_reward,
                        "episode_length": float(len(step_rewards)),
                        "episode_latency": avg_latency,
                        "episode_deadline_hit_ratio": hit_ratio,
                        "episode_violation_mean": float(
                            np.mean([item["violation"] for item in step_metrics])
                        ),
                        "episode_path_available_ratio": 1.0,
                    }
                )

        summary = {
            key: float(np.mean([metric[key] for metric in episode_metrics]))
            for key in episode_metrics[0]
        }
        if record_episode_history:
            return summary, eval_episode_metrics
        return summary

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.rollout_env_pool is not None:
            self.rollout_env_pool.close()
        if self.eval_env_pool is not None:
            self.eval_env_pool.close()

    def _maybe_save_best(self, update: int, validation: Dict[str, float]) -> None:
        eval_reward = float(validation.get("eval_reward", float("-inf")))
        eval_latency = float(validation["eval_latency"])
        eval_hit = float(validation["eval_deadline_hit_ratio"])
        best_reward = float(self.best_checkpoint_info["best_eval_reward"])
        best_latency = float(self.best_checkpoint_info["best_eval_latency"])
        best_hit = float(self.best_checkpoint_info["best_eval_deadline_hit_ratio"])

        improved = eval_reward > best_reward + 1e-8
        if not improved and abs(eval_reward - best_reward) <= 1e-8 and eval_hit > best_hit + 1e-8:
            improved = True
        if (
            not improved
            and abs(eval_reward - best_reward) <= 1e-8
            and abs(eval_hit - best_hit) <= 1e-8
            and eval_latency < best_latency - 1e-8
        ):
            improved = True

        if improved:
            self.best_checkpoint_info = {
                "best_update": int(update),
                "best_eval_reward": eval_reward,
                "best_eval_latency": eval_latency,
                "best_eval_deadline_hit_ratio": eval_hit,
            }
            self.best_reward_state_dict = copy.deepcopy(self.model.state_dict())
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "update": int(update),
                    "validation": validation,
                },
                self.best_model_path,
            )

        self.validation_history.append(
            {
                "update": float(update),
                "eval_reward": eval_reward,
                "eval_latency": eval_latency,
                "eval_deadline_hit_ratio": eval_hit,
            }
        )
        self._update_plateau_tracking(update)

    def _update_plateau_tracking(self, update: int) -> None:
        window = max(int(self.config.plateau_eval_window), 1)
        min_update = max(int(self.config.plateau_min_update), 0)
        self.current_plateau_window_reward = float("nan")
        self.current_plateau_window_spread = float("nan")
        self.current_plateau_window_ready = 0.0

        if update < min_update or len(self.validation_history) < window:
            self.no_improve_updates = 0
            return

        recent = self.validation_history[-window:]
        reward_values = [float(item["eval_reward"]) for item in recent]
        self.current_plateau_window_reward = float(np.mean(reward_values))
        self.current_plateau_window_spread = float(max(reward_values) - min(reward_values))
        self.current_plateau_window_ready = 1.0

        stability_tolerance = float(self.config.plateau_stability_tolerance)
        if stability_tolerance > 0.0 and self.current_plateau_window_spread > stability_tolerance:
            self.no_improve_updates = 0
            return

        meaningful_delta = max(float(self.config.plateau_improve_threshold), 1e-8)
        if self.current_plateau_window_reward > self.best_eval_reward + meaningful_delta:
            self.best_eval_reward = self.current_plateau_window_reward

        reference_reward = float(self.best_checkpoint_info["best_eval_reward"])
        if not np.isfinite(reference_reward):
            reference_reward = self.best_eval_reward

        if self.current_plateau_window_reward >= reference_reward - meaningful_delta:
            self.no_improve_updates = 0
        else:
            self.no_improve_updates += 1

    def _apply_plateau_control(self, validation: Dict[str, float]) -> None:
        patience = max(int(self.config.plateau_patience_updates), 0)
        if patience <= 0 or self.freeze_updates:
            return
        if self.no_improve_updates < patience:
            return
        max_rollbacks = max(int(self.config.max_rollbacks), 0)
        if max_rollbacks <= 0:
            if self.config.freeze_after_rollbacks:
                self.model.load_state_dict(self.best_reward_state_dict)
                self.freeze_updates = True
            self.no_improve_updates = 0
            return
        if self.rollback_count >= max_rollbacks:
            if self.config.freeze_after_rollbacks:
                self.model.load_state_dict(self.best_reward_state_dict)
                self.freeze_updates = True
            self.no_improve_updates = 0
            return

        self.model.load_state_dict(self.best_reward_state_dict)
        self.rollback_count += 1
        self.no_improve_updates = 0
        self.validation_history.clear()
        self.current_plateau_window_reward = float("nan")
        self.current_plateau_window_spread = float("nan")
        self.current_plateau_window_ready = 0.0

        self.current_lr = max(self.current_lr * self.config.rollback_lr_factor, self.config.lr * 0.02)
        for param_group, group_scale in zip(self.optimizer.param_groups, self.optimizer_group_scales):
            param_group["lr"] = self.current_lr * group_scale

        self.current_route_entropy_coef = max(self.current_route_entropy_coef * 0.5, self.config.route_entropy_final_coef)
        self.current_route_sampling_temperature = max(
            self.current_route_sampling_temperature * 0.8,
            self.config.route_sampling_temperature_final,
        )
        self.current_alloc_noise_scale = max(
            self.current_alloc_noise_scale * 0.5,
            self.config.alloc_noise_final_scale,
        )
        self.model.set_route_sampling_temperature(self.current_route_sampling_temperature)
        self.model.set_allocation_noise_scale(self.current_alloc_noise_scale)

        if self.rollback_count >= max_rollbacks and self.config.freeze_after_rollbacks:
            self.freeze_updates = True

    def _build_rollout_seed_pool(self) -> list[int]:
        pool_size = max(int(self.config.training_seed_pool_size), 0)
        if pool_size <= 0:
            return []

        candidate_multiplier = max(int(self.config.training_seed_pool_candidate_multiplier), 1)
        candidate_count = max(pool_size * candidate_multiplier, pool_size)
        trim_fraction = float(np.clip(self.config.training_seed_pool_trim_fraction, 0.0, 0.45))
        candidate_seed_base = self.config.seed + 100_000
        strategy = str(self.config.training_seed_pool_strategy).strip().lower()
        pool_rng = np.random.default_rng(self.config.seed + 54_321)

        if strategy == "random":
            candidate_seeds = np.arange(candidate_seed_base, candidate_seed_base + candidate_count, dtype=np.int64)
            if len(candidate_seeds) > pool_size:
                selected_indices = pool_rng.choice(len(candidate_seeds), size=pool_size, replace=False)
                candidate_seeds = candidate_seeds[np.sort(selected_indices)]
            random_pool = [int(seed) for seed in candidate_seeds.tolist()]
            pool_rng.shuffle(random_pool)
            return random_pool

        scored_candidates: list[tuple[int, float]] = []

        for offset in range(candidate_count):
            seed = candidate_seed_base + offset
            obs = self.env.reset(seed=seed)
            rewards: list[float] = []
            for _ in range(self.config.episode_length):
                route_idx, allocation = self._heuristic_action(obs)
                next_obs, reward, done, _ = self.env.step({"route_idx": route_idx, "allocation": allocation})
                rewards.append(float(reward))
                if done:
                    break
                obs = next_obs
            scored_candidates.append((seed, float(np.sum(rewards))))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        trim_count = min(
            int(round(candidate_count * trim_fraction)),
            max((candidate_count - pool_size) // 2, 0),
        )
        trimmed_candidates = scored_candidates[trim_count : candidate_count - trim_count]
        if len(trimmed_candidates) < pool_size:
            trimmed_candidates = scored_candidates

        candidate_seeds = [seed for seed, _ in trimmed_candidates]
        if len(candidate_seeds) > pool_size:
            selected_indices = pool_rng.choice(len(candidate_seeds), size=pool_size, replace=False)
            selected_set = {int(idx) for idx in selected_indices.tolist()}
            candidate_seeds = [seed for idx, seed in enumerate(candidate_seeds) if idx in selected_set]
        return candidate_seeds

    def _current_seed_pool_fraction(self, update_idx: int) -> float:
        initial_fraction = float(np.clip(self.config.training_seed_curriculum_initial_fraction, 0.05, 1.0))
        ramp_fraction = float(np.clip(self.config.training_seed_curriculum_ramp_fraction, 0.0, 1.0))
        power = max(float(self.config.training_seed_curriculum_power), 1e-6)
        if ramp_fraction <= 1e-6 or self.config.train_updates <= 1:
            return initial_fraction
        progress = float(np.clip(float(update_idx) / float(self.config.train_updates - 1), 0.0, 1.0))
        ramp_progress = float(np.clip(progress / ramp_fraction, 0.0, 1.0)) ** power
        return float(np.clip(initial_fraction + (1.0 - initial_fraction) * ramp_progress, initial_fraction, 1.0))

    def _select_rollout_seeds(self, update_idx: int, num_envs: int) -> List[int]:
        hold_updates = max(int(self.config.training_seed_hold_updates), 1)
        window_index = max(int(update_idx), 0) // hold_updates
        strategy = str(self.config.training_seed_pool_strategy).strip().lower()

        if strategy == "random_stream":
            rng = np.random.default_rng(self.config.seed + 7_919 * (window_index + 1))
            random_seeds = rng.integers(
                self.config.seed + 100_000,
                self.config.seed + 10_000_000,
                size=num_envs,
                endpoint=False,
            )
            return [int(seed) for seed in random_seeds.tolist()]

        if self.rollout_seed_pool:
            pool_size = len(self.rollout_seed_pool)
            accessible_fraction = self._current_seed_pool_fraction(update_idx)
            accessible_count = int(np.clip(round(pool_size * accessible_fraction), num_envs, pool_size))
            shift = max(int(self.config.training_seed_window_shift), 1)
            start = (window_index * shift) % accessible_count
            return [self.rollout_seed_pool[(start + env_idx) % accessible_count] for env_idx in range(num_envs)]

        shift = int(self.config.training_seed_window_shift)
        if shift <= 0:
            shift = max(num_envs, 1)
        base_seed = self.config.seed + window_index * shift
        return [base_seed + env_idx for env_idx in range(num_envs)]

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray], batch: bool) -> Dict[str, Tensor]:
        converted: Dict[str, Tensor] = {}
        for key in OBS_FLOAT_KEYS:
            tensor = torch.as_tensor(obs[key], dtype=torch.float32, device=self.config.device)
            converted[key] = tensor.unsqueeze(0) if batch else tensor
        for key in OBS_LONG_KEYS:
            tensor = torch.as_tensor(obs[key], dtype=torch.long, device=self.config.device)
            converted[key] = tensor.unsqueeze(0) if batch else tensor
        return converted

    def _stack_observations(self, observations: List[Dict[str, np.ndarray]]) -> Dict[str, Tensor]:
        stacked: Dict[str, Tensor] = {}
        for key in OBS_FLOAT_KEYS:
            stacked[key] = torch.as_tensor(
                np.asarray([obs[key] for obs in observations]),
                dtype=torch.float32,
                device=self.config.device,
            )
        for key in OBS_LONG_KEYS:
            stacked[key] = torch.as_tensor(
                np.asarray([obs[key] for obs in observations]),
                dtype=torch.long,
                device=self.config.device,
            )
        return stacked

    def _copy_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {key: value.copy() for key, value in obs.items()}

    def _reset_rollout_envs(self, seeds: List[int]) -> List[Dict[str, np.ndarray]]:
        if self.rollout_env_pool is None:
            return [env.reset(seed=seed) for env, seed in zip(self.rollout_envs, seeds)]

        pool_size = min(len(seeds), self.rollout_env_pool.size)
        observations = self.rollout_env_pool.reset_many(seeds[:pool_size])
        if pool_size < len(seeds):
            observations.extend(
                env.reset(seed=seed)
                for env, seed in zip(self.rollout_envs[pool_size:], seeds[pool_size:])
            )
        return observations

    def _step_rollout_envs(self, actions: List[Dict[str, np.ndarray | int]]):
        if self.rollout_env_pool is None:
            return [env.step(action) for env, action in zip(self.rollout_envs, actions)]

        pool_size = min(len(actions), self.rollout_env_pool.size)
        results = self.rollout_env_pool.step_many(actions[:pool_size])
        if pool_size < len(actions):
            results.extend(
                env.step(action)
                for env, action in zip(self.rollout_envs[pool_size:], actions[pool_size:])
            )
        return results

    def _recreate_eval_env_pool(self) -> None:
        if self.eval_env_pool is not None:
            self.eval_env_pool.close()
        if self.config.parallel_eval_envs > 1:
            self.eval_env_pool = _ParallelEnvPool(
                self.config,
                min(int(self.config.parallel_eval_envs), max(int(self.config.train_eval_episodes), int(self.config.eval_episodes))),
            )
        else:
            self.eval_env_pool = None

    def _reset_eval_envs(self, seeds: List[int]) -> List[Dict[str, np.ndarray]]:
        if self.eval_env_pool is None:
            return [env.reset(seed=seed) for env, seed in zip(self.eval_envs, seeds)]

        pool_size = min(len(seeds), self.eval_env_pool.size)
        try:
            observations = self.eval_env_pool.reset_many(seeds[:pool_size])
        except (BrokenPipeError, EOFError, OSError):
            self._recreate_eval_env_pool()
            if self.eval_env_pool is None:
                return [env.reset(seed=seed) for env, seed in zip(self.eval_envs, seeds)]
            pool_size = min(len(seeds), self.eval_env_pool.size)
            observations = self.eval_env_pool.reset_many(seeds[:pool_size])
        if pool_size < len(seeds):
            observations.extend(
                env.reset(seed=seed)
                for env, seed in zip(self.eval_envs[pool_size:], seeds[pool_size:])
            )
        return observations

    def _step_eval_envs(self, actions: List[Dict[str, np.ndarray | int]]):
        if self.eval_env_pool is None:
            return [env.step(action) for env, action in zip(self.eval_envs, actions)]

        pool_size = min(len(actions), self.eval_env_pool.size)
        try:
            results = self.eval_env_pool.step_many(actions[:pool_size])
        except (BrokenPipeError, EOFError, OSError):
            self._recreate_eval_env_pool()
            if self.eval_env_pool is None:
                return [env.step(action) for env, action in zip(self.eval_envs, actions)]
            pool_size = min(len(actions), self.eval_env_pool.size)
            results = self.eval_env_pool.step_many(actions[:pool_size])
        if pool_size < len(actions):
            results.extend(
                env.step(action)
                for env, action in zip(self.eval_envs[pool_size:], actions[pool_size:])
            )
        return results

    def _build_episode_summary(
        self,
        update_idx: int,
        episode_in_group: int,
        episode_index: int,
        episode_seed: int,
        step_metrics: List[Dict[str, float]],
        step_rewards: List[float],
    ) -> Dict[str, float]:
        summary = {
            "update": float(update_idx + 1),
            "episode_in_update": float(episode_in_group + 1),
            "episode_index": float(episode_index + 1),
            "episode_seed": float(episode_seed),
            "episode_reward": float(np.sum(step_rewards)),
            "episode_length": float(len(step_rewards)),
            "episode_latency": float(np.mean([metric["latency"] for metric in step_metrics])),
            "episode_comm_latency": float(np.mean([metric["comm_latency"] for metric in step_metrics])),
            "episode_comp_latency": float(np.mean([metric["comp_latency"] for metric in step_metrics])),
            "episode_deadline_hit_ratio": float(np.mean([metric["deadline_hit"] for metric in step_metrics])),
            "episode_violation_mean": float(np.mean([metric["violation"] for metric in step_metrics])),
            "episode_latency_violation_ratio": float(
                np.mean([metric["latency_violation_ratio"] for metric in step_metrics])
            ),
            "episode_timely_throughput": float(np.mean([metric["timely_throughput"] for metric in step_metrics])),
            "episode_load_balancing_index": float(
                np.mean([metric["load_balancing_index"] for metric in step_metrics])
            ),
            "episode_resource_utilization": float(
                np.mean([metric["avg_resource_utilization"] for metric in step_metrics])
            ),
            "episode_path_available_ratio": float(np.mean([metric["path_available"] for metric in step_metrics])),
        }
        if step_metrics and "raw_reward" in step_metrics[0]:
            summary["episode_raw_reward"] = float(np.sum([metric["raw_reward"] for metric in step_metrics]))
        if step_metrics and "baseline_reward" in step_metrics[0]:
            summary["episode_baseline_reward"] = float(np.sum([metric["baseline_reward"] for metric in step_metrics]))
            summary["episode_reward_delta_vs_baseline"] = float(
                np.sum([metric["reward_delta_vs_baseline"] for metric in step_metrics])
            )
        return summary

    def _heuristic_action(self, obs: Dict[str, np.ndarray]) -> tuple[int, np.ndarray]:
        return heuristic_action(obs, self.config)

    def _heuristic_allocation(self, obs: Dict[str, np.ndarray], route_idx: int) -> np.ndarray:
        return heuristic_action(obs, self.config)[1]

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            next_value = last_value if step == len(rewards) - 1 else values[step + 1]
            delta = rewards[step] + self.config.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + values
        return advantages, returns

    def _summarize_metrics(self, metrics: List[Dict[str, float]], rewards: List[float]) -> Dict[str, float]:
        summary = {
            "reward_mean": float(np.mean(rewards)),
            "latency_mean": float(np.mean([metric["latency"] for metric in metrics])),
            "comm_latency_mean": float(np.mean([metric["comm_latency"] for metric in metrics])),
            "comp_latency_mean": float(np.mean([metric["comp_latency"] for metric in metrics])),
            "deadline_hit_ratio": float(np.mean([metric["deadline_hit"] for metric in metrics])),
            "violation_mean": float(np.mean([metric["violation"] for metric in metrics])),
            "latency_violation_ratio_mean": float(
                np.mean([metric["latency_violation_ratio"] for metric in metrics])
            ),
            "timely_throughput_mean": float(np.mean([metric["timely_throughput"] for metric in metrics])),
            "load_balancing_index_mean": float(np.mean([metric["load_balancing_index"] for metric in metrics])),
            "resource_utilization_mean": float(
                np.mean([metric["avg_resource_utilization"] for metric in metrics])
            ),
            "path_available_ratio": float(np.mean([metric["path_available"] for metric in metrics])),
            "edge_allocation_ratio": float(np.mean([metric["edge_allocation_ratio"] for metric in metrics])),
            "cloud_allocation_ratio": float(np.mean([metric["cloud_allocation_ratio"] for metric in metrics])),
            "mean_compute_queue_ratio": float(np.mean([metric["mean_compute_queue_ratio"] for metric in metrics])),
        }
        if metrics and "raw_reward" in metrics[0]:
            summary["raw_reward_mean"] = float(np.mean([metric["raw_reward"] for metric in metrics]))
        if metrics and "baseline_reward" in metrics[0]:
            summary["baseline_reward_mean"] = float(np.mean([metric["baseline_reward"] for metric in metrics]))
            summary["reward_delta_vs_baseline_mean"] = float(
                np.mean([metric["reward_delta_vs_baseline"] for metric in metrics])
            )
            summary["baseline_latency_mean"] = float(np.mean([metric["baseline_latency"] for metric in metrics]))
        return summary
