from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from evaluation_tools import rollout_policy, task_observation_from_slot
from masac import (
    MASAC_CANDIDATE_FEATURE_INDICES,
    batch_obs_to_torch,
    edge_preferred_allocation,
    gather_candidate_features,
    zero_observation_like,
)
from utils import save_history_csv, save_metrics_json, set_seed


@dataclass
class MAPDQNConfig:
    train_episodes: int = 4000
    eval_interval_episodes: int = 32
    eval_episodes: int = 16
    warmup_steps: int = 1024
    batch_size: int = 256
    replay_size: int = 200_000
    hidden_dim: int = 192
    actor_lr: float = 2.0e-4
    critic_lr: float = 3.0e-4
    gamma: float = 0.99
    tau: float = 0.01
    epsilon_start: float = 0.30
    epsilon_final: float = 0.03
    epsilon_decay_episodes: int = 600
    updates_per_env_step: int = 1
    update_interval_slots: int = 1
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    checkpoint_name: str = "mapdqn_actor.pt"
    best_checkpoint_name: str = "best_mapdqn_actor.pt"

    def __post_init__(self) -> None:
        if self.update_interval_slots < 1:
            raise ValueError("update_interval_slots must be at least 1")


def apply_mapdqn_variant(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, method_name="MA-P-DQN")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.task_features: np.ndarray | None = None
        self.candidate_features: np.ndarray | None = None
        self.candidate_mask: np.ndarray | None = None
        self.candidate_alloc_mask: np.ndarray | None = None
        self.route_idx: np.ndarray | None = None
        self.allocation: np.ndarray | None = None
        self.reward: np.ndarray | None = None
        self.next_task_features: np.ndarray | None = None
        self.next_candidate_features: np.ndarray | None = None
        self.next_candidate_mask: np.ndarray | None = None
        self.next_candidate_alloc_mask: np.ndarray | None = None
        self.done: np.ndarray | None = None

    def _ensure_arrays(self, obs: dict[str, np.ndarray], allocation_dim: int) -> None:
        if self.task_features is not None:
            return
        self.task_features = np.zeros((self.capacity, *obs["task_features"].shape), dtype=np.float32)
        self.candidate_features = np.zeros((self.capacity, *obs["candidate_features"].shape), dtype=np.float32)
        self.candidate_mask = np.zeros((self.capacity, *obs["candidate_mask"].shape), dtype=np.float32)
        self.candidate_alloc_mask = np.zeros(
            (self.capacity, *obs["candidate_alloc_mask"].shape), dtype=np.float32
        )
        self.route_idx = np.zeros((self.capacity,), dtype=np.int64)
        self.allocation = np.zeros((self.capacity, allocation_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.next_task_features = np.zeros((self.capacity, *obs["task_features"].shape), dtype=np.float32)
        self.next_candidate_features = np.zeros(
            (self.capacity, *obs["candidate_features"].shape), dtype=np.float32
        )
        self.next_candidate_mask = np.zeros((self.capacity, *obs["candidate_mask"].shape), dtype=np.float32)
        self.next_candidate_alloc_mask = np.zeros(
            (self.capacity, *obs["candidate_alloc_mask"].shape), dtype=np.float32
        )
        self.done = np.zeros((self.capacity,), dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        route_idx: int,
        allocation: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        self._ensure_arrays(obs, int(allocation.shape[0]))
        assert self.task_features is not None
        assert self.candidate_features is not None
        assert self.candidate_mask is not None
        assert self.candidate_alloc_mask is not None
        assert self.route_idx is not None
        assert self.allocation is not None
        assert self.reward is not None
        assert self.next_task_features is not None
        assert self.next_candidate_features is not None
        assert self.next_candidate_mask is not None
        assert self.next_candidate_alloc_mask is not None
        assert self.done is not None

        idx = self.position
        self.task_features[idx] = obs["task_features"]
        self.candidate_features[idx] = obs["candidate_features"]
        self.candidate_mask[idx] = obs["candidate_mask"]
        self.candidate_alloc_mask[idx] = obs["candidate_alloc_mask"]
        self.route_idx[idx] = int(route_idx)
        self.allocation[idx] = allocation.astype(np.float32)
        self.reward[idx] = float(reward)
        self.next_task_features[idx] = next_obs["task_features"]
        self.next_candidate_features[idx] = next_obs["candidate_features"]
        self.next_candidate_mask[idx] = next_obs["candidate_mask"]
        self.next_candidate_alloc_mask[idx] = next_obs["candidate_alloc_mask"]
        self.done[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str) -> dict[str, Tensor]:
        if self.size < batch_size:
            raise ValueError("Replay buffer does not contain enough samples.")
        assert self.task_features is not None
        assert self.candidate_features is not None
        assert self.candidate_mask is not None
        assert self.candidate_alloc_mask is not None
        assert self.route_idx is not None
        assert self.allocation is not None
        assert self.reward is not None
        assert self.next_task_features is not None
        assert self.next_candidate_features is not None
        assert self.next_candidate_mask is not None
        assert self.next_candidate_alloc_mask is not None
        assert self.done is not None

        indices = np.random.randint(0, self.size, size=batch_size)
        obs = batch_obs_to_torch(
            {
                "task_features": self.task_features[indices],
                "candidate_features": self.candidate_features[indices],
                "candidate_mask": self.candidate_mask[indices],
                "candidate_alloc_mask": self.candidate_alloc_mask[indices],
            },
            device,
        )
        next_obs = batch_obs_to_torch(
            {
                "task_features": self.next_task_features[indices],
                "candidate_features": self.next_candidate_features[indices],
                "candidate_mask": self.next_candidate_mask[indices],
                "candidate_alloc_mask": self.next_candidate_alloc_mask[indices],
            },
            device,
        )
        return {
            "obs": obs,
            "next_obs": next_obs,
            "route_idx": torch.as_tensor(self.route_idx[indices], dtype=torch.long, device=device),
            "allocation": torch.as_tensor(self.allocation[indices], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(self.reward[indices], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[indices], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.size


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MAPDQNActor(nn.Module):
    def __init__(self, task_feature_dim: int, candidate_feature_dim: int, num_routes: int, alloc_dim: int, hidden_dim: int):
        super().__init__()
        self.num_routes = num_routes
        self.alloc_dim = alloc_dim
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.alloc_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim),
        )

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        candidate_subset = obs["candidate_features"][..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
        task_context = self.task_encoder(obs["task_features"])
        repeated_task = task_context.unsqueeze(1).expand(-1, self.num_routes, -1)
        alloc_latent = self.alloc_head(torch.cat([repeated_task, candidate_subset], dim=-1))
        return edge_preferred_allocation(
            alloc_latent,
            candidate_subset,
            obs["candidate_alloc_mask"],
        )


class MAPDQNCritic(nn.Module):
    def __init__(self, task_feature_dim: int, candidate_feature_dim: int, num_routes: int, alloc_dim: int, hidden_dim: int):
        super().__init__()
        self.num_routes = num_routes
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feature_dim * 2 + alloc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict[str, Tensor], allocations: Tensor) -> Tensor:
        candidate_subset = obs["candidate_features"][..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
        weighted_mask = obs["candidate_mask"].unsqueeze(-1)
        mean_candidate = (candidate_subset * weighted_mask).sum(dim=1) / weighted_mask.sum(dim=1).clamp_min(1e-6)
        task_context = self.task_encoder(obs["task_features"])

        repeated_task = task_context.unsqueeze(1).expand(-1, self.num_routes, -1)
        repeated_mean = mean_candidate.unsqueeze(1).expand(-1, self.num_routes, -1)
        q_input = torch.cat([repeated_task, repeated_mean, candidate_subset, allocations], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)
        return q_values.masked_fill(obs["candidate_mask"] < 0.05, -1.0e9)


class MAPDQNPolicyAdapter:
    def __init__(self, actor: MAPDQNActor, critic: MAPDQNCritic, device: str, epsilon: float = 0.0):
        self.actor = actor
        self.critic = critic
        self.device = device
        self.epsilon = float(epsilon)

    def act_numpy(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> tuple[int, np.ndarray]:
        obs_t = {
            "task_features": torch.as_tensor(obs["task_features"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "candidate_features": torch.as_tensor(
                obs["candidate_features"], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "candidate_mask": torch.as_tensor(obs["candidate_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "candidate_alloc_mask": torch.as_tensor(
                obs["candidate_alloc_mask"], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        }
        with torch.inference_mode():
            allocations = self.actor(obs_t)
            q_values = self.critic(obs_t, allocations)
        route_idx = int(q_values.argmax(dim=-1).item())
        allocation = allocations[0, route_idx].detach().cpu().numpy().astype(np.float32)
        return route_idx, allocation


class MAPDQNTrainer:
    def __init__(self, env_config: ExperimentConfig, train_config: MAPDQNConfig, output_dir: Path):
        self.env_config = apply_mapdqn_variant(env_config)
        self.train_config = train_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = train_config.device
        self.env = SimplifiedRoutingEnv(env_config)
        self.eval_env = SimplifiedRoutingEnv(env_config)
        self.num_routes = env_config.num_candidate_paths
        self.alloc_dim = env_config.alloc_dim

        self.actor = MAPDQNActor(
            task_feature_dim=env_config.task_feature_dim,
            candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
            num_routes=self.num_routes,
            alloc_dim=self.alloc_dim,
            hidden_dim=train_config.hidden_dim,
        ).to(self.device)
        self.critic = MAPDQNCritic(
            task_feature_dim=env_config.task_feature_dim,
            candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
            num_routes=self.num_routes,
            alloc_dim=self.alloc_dim,
            hidden_dim=train_config.hidden_dim,
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=train_config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=train_config.critic_lr)

        self.buffer = ReplayBuffer(train_config.replay_size)
        self.total_steps = 0
        self.decision_slots = 0
        self.training_history: list[dict[str, float]] = []
        self.episode_history: list[dict[str, float]] = []
        self.best_eval_score = -float("inf")
        self.best_eval_metrics: dict[str, float] | None = None

    def current_epsilon(self, episode_idx: int) -> float:
        progress = min(max(float(episode_idx) / max(float(self.train_config.epsilon_decay_episodes), 1.0), 0.0), 1.0)
        return float(
            self.train_config.epsilon_start
            + (self.train_config.epsilon_final - self.train_config.epsilon_start) * progress
        )

    def policy_adapter(self, epsilon: float = 0.0) -> MAPDQNPolicyAdapter:
        return MAPDQNPolicyAdapter(actor=self.actor, critic=self.critic, device=self.device, epsilon=epsilon)

    def train(self) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
        recent_losses: list[dict[str, float]] = []
        rng = np.random.default_rng(self.env_config.seed + 9876)

        for episode_idx in range(self.train_config.train_episodes):
            obs = (
                self.env.reset_slot(seed=self.env_config.seed + episode_idx)
                if self.env_config.use_multi_task_slots
                else self.env.reset(seed=self.env_config.seed + episode_idx)
            )
            episode_rewards: list[float] = []
            episode_metrics: list[dict[str, float]] = []
            epsilon = self.current_epsilon(episode_idx)

            for _ in range(self.env_config.episode_length):
                if self.env_config.use_multi_task_slots:
                    task_observations = [
                        task_observation_from_slot(obs, task_idx)
                        for task_idx in range(int(obs["task_features"].shape[0]))
                    ]
                    routes: list[int] = []
                    allocations: list[np.ndarray] = []
                    for task_obs in task_observations:
                        if self.total_steps < self.train_config.warmup_steps:
                            route_probs = np.clip(
                                task_obs["candidate_mask"].astype(np.float64), 1e-6, None
                            )
                            route_probs /= route_probs.sum()
                            route_idx = int(rng.choice(self.num_routes, p=route_probs))
                            valid_alloc = task_obs["candidate_alloc_mask"][route_idx]
                            allocation = rng.random(self.alloc_dim, dtype=np.float32) * valid_alloc
                            allocation /= np.clip(allocation.sum(), 1e-6, None)
                        else:
                            route_idx, allocation = self._select_action(task_obs, epsilon, rng)
                        routes.append(route_idx)
                        allocations.append(allocation)

                    next_obs, reward, done, info = self.env.step_slot(
                        {
                            "route_idx": np.asarray(routes, dtype=np.int64),
                            "allocation": np.stack(allocations),
                        }
                    )
                    task_infos = info["task_infos"]
                    for task_idx, (task_obs, task_info) in enumerate(
                        zip(task_observations, task_infos)
                    ):
                        stored_next_obs = (
                            zero_observation_like(task_obs)
                            if done or next_obs is None
                            else task_observation_from_slot(
                                next_obs, task_idx % int(next_obs["task_features"].shape[0])
                            )
                        )
                        self.buffer.add(
                            task_obs,
                            int(task_info["route_idx"]),
                            np.asarray(task_info["allocation"], dtype=np.float32),
                            float(task_info["reward"]),
                            stored_next_obs,
                            done,
                        )
                    episode_rewards.append(float(reward))
                    episode_metrics.append(info)
                    self.total_steps += len(task_observations)
                    self.decision_slots += 1
                    if (
                        len(self.buffer) >= self.train_config.batch_size
                        and self.total_steps >= self.train_config.warmup_steps
                        and self.decision_slots % self.train_config.update_interval_slots == 0
                    ):
                        for _ in range(self.train_config.updates_per_env_step):
                            recent_losses.append(self.update_step())
                    obs = next_obs if next_obs is not None else obs
                    if done:
                        break
                    continue

                if self.total_steps < self.train_config.warmup_steps:
                    route_probs = np.clip(obs["candidate_mask"].astype(np.float64), 1e-6, None)
                    route_probs = route_probs / route_probs.sum()
                    route_idx = int(rng.choice(self.num_routes, p=route_probs))
                    allocation = rng.random(self.alloc_dim, dtype=np.float32)
                    allocation = np.clip(allocation, 1e-6, None)
                    allocation = allocation / allocation.sum()
                else:
                    route_idx, allocation = self._select_action(obs, epsilon, rng)

                next_obs, reward, done, info = self.env.step({"route_idx": route_idx, "allocation": allocation})
                stored_next_obs = zero_observation_like(obs) if done or next_obs is None else next_obs
                self.buffer.add(obs, route_idx, allocation, reward, stored_next_obs, done)
                episode_rewards.append(float(reward))
                episode_metrics.append(info)
                obs = next_obs if next_obs is not None else zero_observation_like(obs)
                self.total_steps += 1
                self.decision_slots += 1

                if (
                    len(self.buffer) >= self.train_config.batch_size
                    and self.total_steps >= self.train_config.warmup_steps
                    and self.decision_slots % self.train_config.update_interval_slots == 0
                ):
                    for _ in range(self.train_config.updates_per_env_step):
                        recent_losses.append(self.update_step())

                if done:
                    break

            self.episode_history.append(self._build_episode_summary(episode_idx, episode_rewards, episode_metrics))

            should_eval = (
                (episode_idx + 1) % self.train_config.eval_interval_episodes == 0
                or episode_idx + 1 == self.train_config.train_episodes
            )
            if should_eval:
                window = self.episode_history[-self.train_config.eval_interval_episodes :]
                train_row = self._build_training_row(episode_idx + 1, window, recent_losses, epsilon)
                eval_metrics = rollout_policy(
                    env=self.eval_env,
                    config=self.env_config,
                    policy_name="mapdqn",
                    episodes=self.train_config.eval_episodes,
                    seed_base=self.env_config.validation_seed_base,
                    model=self.policy_adapter(epsilon=0.0),
                )
                train_row["eval_reward"] = float(eval_metrics["reward_mean"])
                train_row["eval_latency"] = float(eval_metrics["latency"])
                train_row["eval_deadline_hit_ratio"] = float(eval_metrics["deadline_hit_ratio"])
                train_row["eval_latency_violation_ratio"] = float(eval_metrics["latency_violation_ratio"])
                train_row["eval_timely_throughput"] = float(eval_metrics["timely_throughput"])
                train_row["eval_load_balancing_index"] = float(eval_metrics["load_balancing_index"])
                train_row["eval_resource_utilization"] = float(eval_metrics["resource_utilization"])
                train_row["buffer_size"] = float(len(self.buffer))
                train_row["total_steps"] = float(self.total_steps)
                train_row["decision_slots"] = float(self.decision_slots)
                self.training_history.append(train_row)
                recent_losses = []

                eval_score = self._score_eval_metrics(eval_metrics)
                if eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    self.best_eval_metrics = dict(eval_metrics)
                    torch.save(
                        {
                            "actor_state_dict": self.actor.state_dict(),
                            "critic_state_dict": self.critic.state_dict(),
                            "env_config": asdict(self.env_config),
                            "mapdqn_config": asdict(self.train_config),
                        },
                        self.output_dir / self.train_config.best_checkpoint_name,
                    )

        if self.best_eval_metrics is None:
            self.best_eval_metrics = rollout_policy(
                env=self.eval_env,
                config=self.env_config,
                policy_name="mapdqn",
                episodes=self.train_config.eval_episodes,
                seed_base=self.env_config.validation_seed_base,
                model=self.policy_adapter(epsilon=0.0),
            )
            self.best_eval_score = self._score_eval_metrics(self.best_eval_metrics)

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "env_config": asdict(self.env_config),
                "mapdqn_config": asdict(self.train_config),
            },
            self.output_dir / self.train_config.checkpoint_name,
        )

        best_payload = torch.load(self.output_dir / self.train_config.best_checkpoint_name, map_location=self.device)
        self.actor.load_state_dict(best_payload["actor_state_dict"])
        self.critic.load_state_dict(best_payload["critic_state_dict"])
        final_metrics = rollout_policy(
            env=self.eval_env,
            config=self.env_config,
            policy_name="mapdqn",
            episodes=max(self.env_config.eval_episodes, self.train_config.eval_episodes),
            seed_base=self.env_config.final_eval_seed_base,
            model=self.policy_adapter(epsilon=0.0),
        )
        self._save_artifacts(final_metrics)
        return self.training_history, self.episode_history, final_metrics

    def _select_action(self, obs: dict[str, np.ndarray], epsilon: float, rng: np.random.Generator) -> tuple[int, np.ndarray]:
        if rng.random() < epsilon:
            route_probs = np.clip(obs["candidate_mask"].astype(np.float64), 1e-6, None)
            route_probs = route_probs / route_probs.sum()
            route_idx = int(rng.choice(self.num_routes, p=route_probs))
            with torch.inference_mode():
                obs_t = batch_obs_to_torch(
                    {
                        "task_features": obs["task_features"][None, ...],
                        "candidate_features": obs["candidate_features"][None, ...],
                        "candidate_mask": obs["candidate_mask"][None, ...],
                        "candidate_alloc_mask": obs["candidate_alloc_mask"][None, ...],
                    },
                    self.device,
                )
                allocations = self.actor(obs_t)
            allocation = allocations[0, route_idx].detach().cpu().numpy().astype(np.float32)
            return route_idx, allocation
        return self.policy_adapter(epsilon=0.0).act_numpy(obs, deterministic=True)

    def update_step(self) -> dict[str, float]:
        batch = self.buffer.sample(self.train_config.batch_size, self.device)

        with torch.no_grad():
            next_allocations = self.target_actor(batch["next_obs"])
            next_q_values = self.target_critic(batch["next_obs"], next_allocations)
            next_route_idx = next_q_values.argmax(dim=-1)
            next_q = next_q_values.gather(1, next_route_idx.unsqueeze(-1)).squeeze(-1)
            target_q = batch["reward"] + self.train_config.gamma * (1.0 - batch["done"]) * next_q

        current_allocations = self.actor(batch["obs"])
        current_q_values = self.critic(batch["obs"], current_allocations)
        chosen_q = current_q_values.gather(1, batch["route_idx"].unsqueeze(-1)).squeeze(-1)
        critic_loss = F.mse_loss(chosen_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.train_config.grad_clip_norm)
        self.critic_optimizer.step()

        actor_allocations = self.actor(batch["obs"])
        actor_q_values = self.critic(batch["obs"], actor_allocations)
        actor_loss = -actor_q_values.max(dim=-1).values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.train_config.grad_clip_norm)
        self.actor_optimizer.step()

        self._soft_update_targets()
        greedy_route = actor_q_values.argmax(dim=-1)
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(actor_q_values.max(dim=-1).values.mean().item()),
            "greedy_route_mean": float(greedy_route.to(torch.float32).mean().item()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.train_config.tau
        for target_param, source_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def _build_episode_summary(
        self,
        episode_idx: int,
        episode_rewards: list[float],
        episode_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        def mean_metric(key: str) -> float:
            if not episode_metrics:
                return 0.0
            return float(np.mean([row[key] for row in episode_metrics]))

        return {
            "episode_index": float(episode_idx + 1),
            "episode_reward": float(np.sum(episode_rewards)),
            "episode_latency": mean_metric("latency"),
            "episode_deadline_hit_ratio": mean_metric("deadline_hit"),
            "episode_latency_violation_ratio": mean_metric("latency_violation_ratio"),
            "episode_timely_throughput": mean_metric("timely_throughput"),
            "episode_load_balancing_index": mean_metric("load_balancing_index"),
            "episode_resource_utilization": mean_metric("avg_resource_utilization"),
        }

    def _build_training_row(
        self,
        episode_count: int,
        window: list[dict[str, float]],
        losses: list[dict[str, float]],
        epsilon: float,
    ) -> dict[str, float]:
        def mean_episode(key: str) -> float:
            if not window:
                return float("nan")
            return float(np.mean([row[key] for row in window]))

        def mean_loss(key: str) -> float:
            if not losses:
                return float("nan")
            return float(np.mean([row[key] for row in losses]))

        return {
            "update": float(len(self.training_history) + 1),
            "episode": float(episode_count),
            "reward_mean": mean_episode("episode_reward"),
            "latency_mean": mean_episode("episode_latency"),
            "deadline_hit_ratio": mean_episode("episode_deadline_hit_ratio"),
            "latency_violation_ratio_mean": mean_episode("episode_latency_violation_ratio"),
            "timely_throughput_mean": mean_episode("episode_timely_throughput"),
            "load_balancing_index_mean": mean_episode("episode_load_balancing_index"),
            "resource_utilization_mean": mean_episode("episode_resource_utilization"),
            "actor_loss": mean_loss("actor_loss"),
            "critic_loss": mean_loss("critic_loss"),
            "q_mean": mean_loss("q_mean"),
            "greedy_route_mean": mean_loss("greedy_route_mean"),
            "epsilon": float(epsilon),
        }

    def _score_eval_metrics(self, metrics: dict[str, float]) -> float:
        return float(
            2.5 * metrics["deadline_hit_ratio"]
            - 1.2 * metrics["latency"]
            - 1.8 * metrics["latency_violation_ratio"]
            + 0.05 * metrics["load_balancing_index"]
            + 0.03 * metrics["resource_utilization"]
        )

    def _save_artifacts(self, final_metrics: dict[str, float]) -> None:
        save_history_csv(self.training_history, self.output_dir / "training_history.csv")
        save_history_csv(self.episode_history, self.output_dir / "rollout_episode_history.csv")
        payload: dict[str, Any] = {
            **asdict(self.env_config),
            **{f"mapdqn_{key}": value for key, value in asdict(self.train_config).items()},
            **{f"final_{key}": value for key, value in final_metrics.items()},
            "best_eval_score": float(self.best_eval_score),
        }
        if self.best_eval_metrics is not None:
            payload.update({f"best_{key}": value for key, value in self.best_eval_metrics.items()})
        save_metrics_json(payload, self.output_dir / "evaluation_metrics.json")


def load_mapdqn_train_config(run_dir: Path | str, device: str | None = None) -> MAPDQNConfig:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "evaluation_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    valid_keys = set(MAPDQNConfig.__dataclass_fields__.keys())
    values: dict[str, Any] = {}
    for key in valid_keys:
        prefixed_key = f"mapdqn_{key}"
        if prefixed_key in payload:
            values[key] = payload[prefixed_key]
    if device is not None:
        values["device"] = device
    return MAPDQNConfig(**values)


def load_mapdqn_policy(
    checkpoint_path: Path | str,
    env_config: ExperimentConfig,
    train_config: MAPDQNConfig,
) -> MAPDQNPolicyAdapter:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=train_config.device)
    actor = MAPDQNActor(
        task_feature_dim=env_config.task_feature_dim,
        candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
        num_routes=env_config.num_candidate_paths,
        alloc_dim=env_config.alloc_dim,
        hidden_dim=train_config.hidden_dim,
    ).to(train_config.device)
    critic = MAPDQNCritic(
        task_feature_dim=env_config.task_feature_dim,
        candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
        num_routes=env_config.num_candidate_paths,
        alloc_dim=env_config.alloc_dim,
        hidden_dim=train_config.hidden_dim,
    ).to(train_config.device)
    actor.load_state_dict(payload["actor_state_dict"])
    critic.load_state_dict(payload["critic_state_dict"])
    actor.eval()
    critic.eval()
    return MAPDQNPolicyAdapter(actor=actor, critic=critic, device=train_config.device, epsilon=0.0)


def train_mapdqn(
    env_config: ExperimentConfig,
    train_config: MAPDQNConfig,
    output_dir: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    set_seed(env_config.seed)
    trainer = MAPDQNTrainer(env_config=env_config, train_config=train_config, output_dir=output_dir)
    return trainer.train()
