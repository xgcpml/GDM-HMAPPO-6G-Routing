from __future__ import annotations

import copy
import json
import math
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
from utils import save_history_csv, save_metrics_json, set_seed

MASAC_CANDIDATE_FEATURE_INDICES = (1, 2, 4, 5, 8)
MASAC_EDGE_QUEUE_OFFSET = 2
MASAC_CLOUD_QUEUE_OFFSET = 3


@dataclass
class MASACConfig:
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
    route_alpha: float = 0.08
    alloc_alpha: float = 0.02
    updates_per_env_step: int = 1
    update_interval_slots: int = 1
    log_std_min: float = -4.5
    log_std_max: float = 1.0
    gumbel_tau: float = 0.8
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    checkpoint_name: str = "masac_actor.pt"
    best_checkpoint_name: str = "best_masac_actor.pt"

    def __post_init__(self) -> None:
        if self.update_interval_slots < 1:
            raise ValueError("update_interval_slots must be at least 1")


def apply_masac_variant(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, method_name="MA-SAC")


def zero_observation_like(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in obs.items()}


def obs_to_torch(obs: dict[str, np.ndarray], device: str) -> dict[str, Tensor]:
    return {
        "task_features": torch.as_tensor(obs["task_features"], dtype=torch.float32, device=device).unsqueeze(0),
        "candidate_features": torch.as_tensor(
            obs["candidate_features"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "candidate_mask": torch.as_tensor(obs["candidate_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "candidate_alloc_mask": torch.as_tensor(
            obs["candidate_alloc_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    }


def batch_obs_to_torch(batch: dict[str, np.ndarray], device: str) -> dict[str, Tensor]:
    return {
        "task_features": torch.as_tensor(batch["task_features"], dtype=torch.float32, device=device),
        "candidate_features": torch.as_tensor(batch["candidate_features"], dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(batch["candidate_mask"], dtype=torch.float32, device=device),
        "candidate_alloc_mask": torch.as_tensor(
            batch["candidate_alloc_mask"], dtype=torch.float32, device=device
        ),
    }


def flatten_observation(obs: dict[str, Tensor]) -> Tensor:
    candidate_subset = obs["candidate_features"][..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
    masked_candidate_features = candidate_subset * obs["candidate_mask"].unsqueeze(-1)
    return torch.cat(
        [
            obs["task_features"],
            masked_candidate_features.flatten(start_dim=1),
            obs["candidate_mask"],
        ],
        dim=-1,
    )


def gather_candidate_features(candidate_features: Tensor, route_onehot: Tensor) -> Tensor:
    candidate_subset = candidate_features[..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
    return torch.einsum("br,brf->bf", route_onehot, candidate_subset)


def edge_preferred_allocation(
    alloc_latent: Tensor,
    selected_candidate: Tensor,
    allocation_mask: Tensor | None = None,
) -> Tensor:
    if alloc_latent.size(-1) != 3:
        raise ValueError("Edge-preferred allocation expects two edge nodes and one cloud node.")
    if allocation_mask is None:
        allocation_mask = torch.ones_like(alloc_latent)

    edge_logits = alloc_latent[..., :2]
    cloud_logit = alloc_latent[..., 2:3]
    edge_mask = allocation_mask[..., :2]
    cloud_mask = allocation_mask[..., 2:3]

    masked_edge_logits = edge_logits.masked_fill(edge_mask < 0.5, -1.0e9)
    edge_alloc = F.softmax(masked_edge_logits, dim=-1) * edge_mask
    edge_available = edge_mask.sum(dim=-1, keepdim=True) > 0.5
    edge_alloc = edge_alloc / edge_alloc.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    edge_queue = selected_candidate[
        ..., MASAC_EDGE_QUEUE_OFFSET : MASAC_EDGE_QUEUE_OFFSET + 1
    ]
    cloud_queue = selected_candidate[
        ..., MASAC_CLOUD_QUEUE_OFFSET : MASAC_CLOUD_QUEUE_OFFSET + 1
    ]

    # Keep cloud as a congestion-relief option rather than the default sink.
    adaptive_cloud_cap = torch.clamp(0.12 + 0.38 * edge_queue, min=0.10, max=0.55)
    cloud_gate = adaptive_cloud_cap * torch.sigmoid(cloud_logit - 1.35 - 0.65 * cloud_queue)
    cloud_gate = cloud_gate * cloud_mask
    cloud_gate = torch.where(edge_available, cloud_gate, cloud_mask)
    edge_scale = (1.0 - cloud_gate).clamp_min(1e-6)
    allocation = torch.cat([edge_scale * edge_alloc, cloud_gate], dim=-1)
    allocation = allocation * allocation_mask
    return allocation / allocation.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def masked_simplex(logits: Tensor, mask: Tensor) -> Tensor:
    """Map logits to a simplex while assigning exactly zero mass to padded nodes."""
    masked_logits = logits.masked_fill(mask < 0.5, -1.0e9)
    return F.softmax(masked_logits, dim=-1)


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

    def sample(self, batch_size: int, device: str, num_routes: int) -> dict[str, Tensor]:
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
        route_idx = torch.as_tensor(self.route_idx[indices], dtype=torch.long, device=device)
        route_onehot = F.one_hot(route_idx, num_classes=num_routes).to(torch.float32)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "route_idx": route_idx,
            "route_onehot": route_onehot,
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


class MASACActor(nn.Module):
    def __init__(self, task_feature_dim: int, candidate_feature_dim: int, num_routes: int, alloc_dim: int, hidden_dim: int):
        super().__init__()
        self.num_routes = num_routes
        self.alloc_dim = alloc_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.route_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.alloc_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim * 2),
        )

    def sample(
        self,
        obs: dict[str, Tensor],
        deterministic: bool,
        log_std_min: float,
        log_std_max: float,
        gumbel_tau: float,
    ) -> dict[str, Tensor]:
        candidate_subset = obs["candidate_features"][..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
        task_context = self.task_encoder(obs["task_features"])
        repeated_task = task_context.unsqueeze(1).expand(-1, self.num_routes, -1)
        raw_logits = self.route_head(torch.cat([repeated_task, candidate_subset], dim=-1)).squeeze(-1)
        masked_logits = raw_logits.masked_fill(obs["candidate_mask"] < 0.05, -1.0e9)
        log_route_probs = F.log_softmax(masked_logits, dim=-1)

        if deterministic:
            route_idx = masked_logits.argmax(dim=-1)
            route_onehot = F.one_hot(route_idx, num_classes=self.num_routes).to(torch.float32)
        else:
            route_onehot = F.gumbel_softmax(masked_logits, tau=gumbel_tau, hard=True, dim=-1)
            route_idx = route_onehot.argmax(dim=-1)
        route_log_prob = torch.sum(route_onehot * log_route_probs, dim=-1)

        selected_candidate = gather_candidate_features(obs["candidate_features"], route_onehot)
        selected_alloc_mask = torch.einsum(
            "br,bra->ba", route_onehot, obs["candidate_alloc_mask"]
        )
        alloc_stats = self.alloc_head(torch.cat([task_context, selected_candidate], dim=-1))
        alloc_mean, alloc_log_std = torch.chunk(alloc_stats, 2, dim=-1)
        alloc_log_std = alloc_log_std.clamp(log_std_min, log_std_max)

        if deterministic:
            alloc_latent = alloc_mean
            alloc_log_prob = torch.zeros_like(route_log_prob)
        else:
            alloc_std = alloc_log_std.exp()
            alloc_noise = torch.randn_like(alloc_std)
            alloc_latent = alloc_mean + alloc_std * alloc_noise
            alloc_log_prob_per_dim = (
                -0.5
                * (
                    ((alloc_latent - alloc_mean) / alloc_std.clamp_min(1e-6)) ** 2
                    + 2.0 * alloc_log_std
                    + math.log(2.0 * math.pi)
                )
            )
            alloc_log_prob = (alloc_log_prob_per_dim * selected_alloc_mask).sum(dim=-1)
        allocation = edge_preferred_allocation(
            alloc_latent,
            selected_candidate,
            selected_alloc_mask,
        )

        return {
            "route_idx": route_idx,
            "route_onehot": route_onehot,
            "route_log_prob": route_log_prob,
            "allocation": allocation,
            "alloc_log_prob": alloc_log_prob,
        }


class MASACCritic(nn.Module):
    def __init__(self, task_feature_dim: int, candidate_feature_dim: int, num_routes: int, alloc_dim: int, hidden_dim: int):
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_feature_dim * 2 + num_routes + alloc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict[str, Tensor], route_onehot: Tensor, allocation: Tensor) -> Tensor:
        task_context = self.task_encoder(obs["task_features"])
        selected_candidate = gather_candidate_features(obs["candidate_features"], route_onehot)
        candidate_subset = obs["candidate_features"][..., list(MASAC_CANDIDATE_FEATURE_INDICES)]
        weighted_mask = obs["candidate_mask"].unsqueeze(-1)
        mean_candidate = (candidate_subset * weighted_mask).sum(dim=1) / weighted_mask.sum(dim=1).clamp_min(1e-6)
        q_input = torch.cat([task_context, mean_candidate, selected_candidate, route_onehot, allocation], dim=-1)
        return self.q_head(q_input).squeeze(-1)


class MASACPolicyAdapter:
    def __init__(
        self,
        actor: MASACActor,
        device: str,
        log_std_min: float,
        log_std_max: float,
        gumbel_tau: float,
    ):
        self.actor = actor
        self.device = device
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gumbel_tau = gumbel_tau

    def act_numpy(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> tuple[int, np.ndarray]:
        with torch.inference_mode():
            action = self.actor.sample(
                obs_to_torch(obs, self.device),
                deterministic=deterministic,
                log_std_min=self.log_std_min,
                log_std_max=self.log_std_max,
                gumbel_tau=self.gumbel_tau,
            )
        return (
            int(action["route_idx"].item()),
            action["allocation"].squeeze(0).detach().cpu().numpy().astype(np.float32),
        )

    def state_dict(self) -> dict[str, Any]:
        return self.actor.state_dict()


class MASACTrainer:
    def __init__(self, env_config: ExperimentConfig, train_config: MASACConfig, output_dir: Path):
        self.env_config = apply_masac_variant(env_config)
        self.train_config = train_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = train_config.device
        self.env = SimplifiedRoutingEnv(env_config)
        self.eval_env = SimplifiedRoutingEnv(env_config)
        self.num_routes = env_config.num_candidate_paths
        self.alloc_dim = env_config.alloc_dim
        self.actor = MASACActor(
            task_feature_dim=env_config.task_feature_dim,
            candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
            num_routes=self.num_routes,
            alloc_dim=self.alloc_dim,
            hidden_dim=train_config.hidden_dim,
        ).to(self.device)
        self.critic_1 = MASACCritic(
            task_feature_dim=env_config.task_feature_dim,
            candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
            num_routes=self.num_routes,
            alloc_dim=self.alloc_dim,
            hidden_dim=train_config.hidden_dim,
        ).to(self.device)
        self.critic_2 = MASACCritic(
            task_feature_dim=env_config.task_feature_dim,
            candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
            num_routes=self.num_routes,
            alloc_dim=self.alloc_dim,
            hidden_dim=train_config.hidden_dim,
        ).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=train_config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=train_config.critic_lr,
        )

        self.buffer = ReplayBuffer(train_config.replay_size)
        self.total_steps = 0
        self.decision_slots = 0
        self.training_history: list[dict[str, float]] = []
        self.episode_history: list[dict[str, float]] = []
        self.best_eval_score = -float("inf")
        self.best_eval_metrics: dict[str, float] | None = None

    def policy_adapter(self) -> MASACPolicyAdapter:
        return MASACPolicyAdapter(
            actor=self.actor,
            device=self.device,
            log_std_min=self.train_config.log_std_min,
            log_std_max=self.train_config.log_std_max,
            gumbel_tau=self.train_config.gumbel_tau,
        )

    def train(self) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
        recent_losses: list[dict[str, float]] = []
        rng = np.random.default_rng(self.env_config.seed + 1234)

        for episode_idx in range(self.train_config.train_episodes):
            obs = (
                self.env.reset_slot(seed=self.env_config.seed + episode_idx)
                if self.env_config.use_multi_task_slots
                else self.env.reset(seed=self.env_config.seed + episode_idx)
            )
            episode_rewards: list[float] = []
            episode_metrics: list[dict[str, float]] = []

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
                            route_idx, allocation = self.policy_adapter().act_numpy(
                                task_obs, deterministic=False
                            )
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
                    route_idx, allocation = self.policy_adapter().act_numpy(obs, deterministic=False)

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
                train_row = self._build_training_row(episode_idx + 1, window, recent_losses)
                eval_metrics = rollout_policy(
                    env=self.eval_env,
                    config=self.env_config,
                    policy_name="masac",
                    episodes=self.train_config.eval_episodes,
                    seed_base=self.env_config.validation_seed_base,
                    model=self.policy_adapter(),
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
                            "env_config": asdict(self.env_config),
                            "masac_config": asdict(self.train_config),
                        },
                        self.output_dir / self.train_config.best_checkpoint_name,
                    )

        if self.best_eval_metrics is None:
            self.best_eval_metrics = rollout_policy(
                env=self.eval_env,
                config=self.env_config,
                policy_name="masac",
                episodes=self.train_config.eval_episodes,
                seed_base=self.env_config.validation_seed_base,
                model=self.policy_adapter(),
            )
            self.best_eval_score = self._score_eval_metrics(self.best_eval_metrics)

        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "env_config": asdict(self.env_config),
                "masac_config": asdict(self.train_config),
            },
            self.output_dir / self.train_config.checkpoint_name,
        )

        best_payload = torch.load(self.output_dir / self.train_config.best_checkpoint_name, map_location=self.device)
        self.actor.load_state_dict(best_payload["actor_state_dict"])
        final_metrics = rollout_policy(
            env=self.eval_env,
            config=self.env_config,
            policy_name="masac",
            episodes=max(self.env_config.eval_episodes, self.train_config.eval_episodes),
            seed_base=self.env_config.final_eval_seed_base,
            model=self.policy_adapter(),
        )
        self._save_artifacts(final_metrics)
        return self.training_history, self.episode_history, final_metrics

    def update_step(self) -> dict[str, float]:
        batch = self.buffer.sample(self.train_config.batch_size, self.device, self.num_routes)

        with torch.no_grad():
            next_action = self.actor.sample(
                batch["next_obs"],
                deterministic=False,
                log_std_min=self.train_config.log_std_min,
                log_std_max=self.train_config.log_std_max,
                gumbel_tau=self.train_config.gumbel_tau,
            )
            next_q1 = self.target_critic_1(batch["next_obs"], next_action["route_onehot"], next_action["allocation"])
            next_q2 = self.target_critic_2(batch["next_obs"], next_action["route_onehot"], next_action["allocation"])
            next_q = torch.minimum(next_q1, next_q2)
            next_entropy = (
                self.train_config.route_alpha * next_action["route_log_prob"]
                + self.train_config.alloc_alpha * next_action["alloc_log_prob"]
            )
            target_q = batch["reward"] + self.train_config.gamma * (1.0 - batch["done"]) * (next_q - next_entropy)

        current_q1 = self.critic_1(batch["obs"], batch["route_onehot"], batch["allocation"])
        current_q2 = self.critic_2(batch["obs"], batch["route_onehot"], batch["allocation"])
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.train_config.grad_clip_norm,
        )
        self.critic_optimizer.step()

        action = self.actor.sample(
            batch["obs"],
            deterministic=False,
            log_std_min=self.train_config.log_std_min,
            log_std_max=self.train_config.log_std_max,
            gumbel_tau=self.train_config.gumbel_tau,
        )
        actor_q1 = self.critic_1(batch["obs"], action["route_onehot"], action["allocation"])
        actor_q2 = self.critic_2(batch["obs"], action["route_onehot"], action["allocation"])
        actor_q = torch.minimum(actor_q1, actor_q2)
        entropy_term = (
            self.train_config.route_alpha * action["route_log_prob"]
            + self.train_config.alloc_alpha * action["alloc_log_prob"]
        )
        actor_loss = (entropy_term - actor_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.train_config.grad_clip_norm)
        self.actor_optimizer.step()

        self._soft_update_targets()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(actor_q.mean().item()),
            "route_log_prob": float(action["route_log_prob"].mean().item()),
            "alloc_log_prob": float(action["alloc_log_prob"].mean().item()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.train_config.tau
        for target_param, source_param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
        for target_param, source_param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
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
            "route_log_prob": mean_loss("route_log_prob"),
            "alloc_log_prob": mean_loss("alloc_log_prob"),
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
            **{f"masac_{key}": value for key, value in asdict(self.train_config).items()},
            **{f"final_{key}": value for key, value in final_metrics.items()},
            "best_eval_score": float(self.best_eval_score),
        }
        if self.best_eval_metrics is not None:
            payload.update({f"best_{key}": value for key, value in self.best_eval_metrics.items()})
        save_metrics_json(payload, self.output_dir / "evaluation_metrics.json")


def load_masac_actor(
    checkpoint_path: Path | str,
    env_config: ExperimentConfig,
    train_config: MASACConfig,
) -> MASACPolicyAdapter:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=train_config.device)
    actor = MASACActor(
        task_feature_dim=env_config.task_feature_dim,
        candidate_feature_dim=len(MASAC_CANDIDATE_FEATURE_INDICES),
        num_routes=env_config.num_candidate_paths,
        alloc_dim=env_config.alloc_dim,
        hidden_dim=train_config.hidden_dim,
    ).to(train_config.device)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return MASACPolicyAdapter(
        actor=actor,
        device=train_config.device,
        log_std_min=train_config.log_std_min,
        log_std_max=train_config.log_std_max,
        gumbel_tau=train_config.gumbel_tau,
    )


def load_masac_train_config(run_dir: Path | str, device: str | None = None) -> MASACConfig:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "evaluation_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    valid_keys = set(MASACConfig.__dataclass_fields__.keys())
    values: dict[str, Any] = {}
    for key in valid_keys:
        prefixed_key = f"masac_{key}"
        if prefixed_key in payload:
            values[key] = payload[prefixed_key]
    if device is not None:
        values["device"] = device
    return MASACConfig(**values)


def train_masac(
    env_config: ExperimentConfig,
    train_config: MASACConfig,
    output_dir: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    set_seed(env_config.seed)
    trainer = MASACTrainer(env_config=env_config, train_config=train_config, output_dir=output_dir)
    return trainer.train()
