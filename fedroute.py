from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import ExperimentConfig
from env import SimplifiedRoutingEnv
from evaluation_tools import build_model, rollout_policy
from ppo import PPOTrainer
from utils import save_history_csv, save_metrics_json, set_seed


@dataclass
class FedRouteConfig:
    num_clients: int = 4
    federated_rounds: int = 125
    local_updates: int = 1
    aggregation_interval: int = 1
    client_rollout_episodes: int = 8
    client_seed_stride: int = 100_000
    aggregation_scope: str = "policy"
    client_load_mode: str = "partitioned"
    client_participation_rate: float = 1.0
    participation_schedule: str = "window"
    participation_seed_offset: int = 700_000
    reset_optimizer_on_aggregation: bool = False
    decoupled_allocation_adapter: bool = False
    adapter_pretrain_updates: int = 0
    adapter_pretrain_batch_size: int = 256
    adapter_pretrain_flow_range: tuple[int, int] | None = None
    checkpoint_name: str = "fedroute_model.pt"
    best_checkpoint_name: str = "best_fedroute_model.pt"


def apply_fedroute_variant(config: ExperimentConfig) -> ExperimentConfig:
    return replace(
        config,
        method_name="FedRoute",
        use_gnn_encoder=False,
        use_graph_attention_encoder=False,
        use_transformer_context=False,
        use_gdm_allocator=False,
        allocation_policy_family="direct",
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


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("At least one client state dictionary is required.")
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        values = [state[key].detach() for state in state_dicts]
        if values[0].is_floating_point():
            averaged[key] = torch.stack(values, dim=0).mean(dim=0)
        else:
            averaged[key] = values[0].clone()
    return averaged


def select_federated_state_dict(
    state_dict: dict[str, torch.Tensor], aggregation_scope: str
) -> dict[str, torch.Tensor]:
    scope = aggregation_scope.strip().lower()
    if scope == "full":
        return state_dict
    if scope == "route":
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith("route_head.")
        }
    if scope != "policy":
        raise ValueError(f"Unsupported aggregation scope: {aggregation_scope}")
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("value_head.")
    }


def partition_client_flow_ranges(
    active_flow_range: tuple[int, int], num_clients: int, client_load_mode: str
) -> list[tuple[int, int]]:
    mode = client_load_mode.strip().lower()
    if mode == "shared":
        return [active_flow_range] * num_clients
    if mode != "partitioned":
        raise ValueError(f"Unsupported client load mode: {client_load_mode}")
    low, high = active_flow_range
    if num_clients == 1 or low == high:
        return [active_flow_range] * num_clients
    client_loads = np.rint(np.linspace(low, high, num_clients)).astype(int)
    return [(int(load), int(load)) for load in client_loads]


def select_participating_clients(
    num_clients: int,
    participation_rate: float,
    rng: np.random.Generator,
) -> list[int]:
    if not 0.0 < participation_rate <= 1.0:
        raise ValueError("client_participation_rate must be in (0, 1]")
    num_participants = max(1, int(np.ceil(num_clients * participation_rate)))
    if num_participants == num_clients:
        return list(range(num_clients))
    return sorted(
        int(client_id)
        for client_id in rng.choice(
            num_clients, size=num_participants, replace=False
        ).tolist()
    )


def reset_aggregated_optimizer_state(
    trainer: PPOTrainer,
    aggregated_parameter_names: set[str],
) -> None:
    named_parameters = dict(trainer.model.named_parameters())
    for name in aggregated_parameter_names:
        parameter = named_parameters.get(name)
        if parameter is not None:
            trainer.optimizer.state.pop(parameter, None)


class FedRouteTrainer:
    def __init__(
        self,
        env_config: ExperimentConfig,
        fed_config: FedRouteConfig,
        output_dir: Path,
    ):
        if fed_config.num_clients < 1:
            raise ValueError("num_clients must be positive")
        if fed_config.local_updates < 1:
            raise ValueError("local_updates must be positive")
        if fed_config.aggregation_interval < 1:
            raise ValueError("aggregation_interval must be positive")
        if not 0.0 < fed_config.client_participation_rate <= 1.0:
            raise ValueError("client_participation_rate must be in (0, 1]")
        if fed_config.participation_schedule not in {"window", "round_robin"}:
            raise ValueError("participation_schedule must be window or round_robin")
        if fed_config.adapter_pretrain_updates < 0:
            raise ValueError("adapter_pretrain_updates must be non-negative")
        if fed_config.adapter_pretrain_batch_size < 1:
            raise ValueError("adapter_pretrain_batch_size must be positive")

        client_flow_ranges = partition_client_flow_ranges(
            env_config.active_flow_range,
            fed_config.num_clients,
            fed_config.client_load_mode,
        )

        self.env_config = apply_fedroute_variant(env_config)
        self.fed_config = fed_config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.global_model = build_model(self.env_config).to(self.env_config.device)
        self.client_trainers: list[PPOTrainer] = []
        self.participation_rng = np.random.default_rng(
            self.env_config.seed + fed_config.participation_seed_offset
        )
        self.num_participating_clients = max(
            1,
            int(
                np.ceil(
                    fed_config.num_clients * fed_config.client_participation_rate
                )
            ),
        )
        # Keep the total rollout budget per training round unchanged when only a
        # subset of clients participates.
        self.effective_client_rollout_episodes = int(
            np.ceil(
                fed_config.client_rollout_episodes
                * fed_config.num_clients
                / self.num_participating_clients
            )
        )
        total_local_updates = fed_config.federated_rounds * fed_config.local_updates
        for client_id, client_flow_range in enumerate(client_flow_ranges):
            client_config = replace(
                self.env_config,
                seed=self.env_config.seed + fed_config.client_seed_stride * client_id,
                output_dir=str(self.output_dir / f"client_{client_id}"),
                active_flow_range=client_flow_range,
                train_updates=total_local_updates,
                rollout_episodes_per_update=self.effective_client_rollout_episodes,
                parallel_rollout_envs=1,
                train_eval_episodes=1,
                eval_episodes=1,
                warmup_updates=0,
                warmup_batch_size=fed_config.adapter_pretrain_batch_size,
                search_bootstrap_updates=0,
            )
            client_model = copy.deepcopy(self.global_model)
            self.client_trainers.append(
                PPOTrainer(
                    client_config,
                    SimplifiedRoutingEnv(client_config),
                    client_model,
                )
            )

        if self.fed_config.decoupled_allocation_adapter:
            self._prepare_decoupled_allocation_adapter()

        self.round_history: list[dict[str, float]] = []
        self.client_history: list[dict[str, float]] = []
        self.best_eval_reward = -float("inf")
        self.best_eval_metrics: dict[str, float] | None = None

    def _prepare_decoupled_allocation_adapter(self) -> None:
        source_trainer = self.client_trainers[0]
        for parameter in source_trainer.model.parameters():
            parameter.requires_grad_(False)
        for module in (
            source_trainer.model.graph_encoder,
            source_trainer.model.allocation_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

        original_env = source_trainer.env
        if self.fed_config.adapter_pretrain_flow_range is not None:
            source_trainer.env = SimplifiedRoutingEnv(
                replace(
                    source_trainer.config,
                    active_flow_range=self.fed_config.adapter_pretrain_flow_range,
                )
            )
        try:
            for update_idx in range(self.fed_config.adapter_pretrain_updates):
                obs_batch, route_target, alloc_target = source_trainer.collect_warmup_batch(
                    update_idx
                )
                source_trainer._run_supervised_update(
                    obs_batch=obs_batch,
                    route_target=route_target,
                    alloc_target=alloc_target,
                    route_coef=0.0,
                    diffusion_coef=1.0,
                    prefix="adapter_pretrain",
                    update_idx=update_idx + 1,
                )
        finally:
            source_trainer.env = original_env

        adapter_state = copy.deepcopy(
            source_trainer.model.allocation_head.state_dict()
        )
        adapter_encoder_state = copy.deepcopy(
            source_trainer.model.graph_encoder.state_dict()
        )
        self.global_model.allocation_head.load_state_dict(adapter_state)
        self.global_model.graph_encoder.load_state_dict(adapter_encoder_state)
        for trainer in self.client_trainers:
            trainer.model.allocation_head.load_state_dict(adapter_state)
            trainer.model.graph_encoder.load_state_dict(adapter_encoder_state)
            trainer.optimizer.state.clear()
            for parameter in trainer.model.parameters():
                parameter.requires_grad_(False)
            for module in (trainer.model.route_head, trainer.model.value_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def train(self) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
        participating_client_ids: list[int] = []
        aggregation_client_ids: set[int] = set()
        for round_idx in range(self.fed_config.federated_rounds):
            if self.fed_config.participation_schedule == "round_robin":
                start = (
                    round_idx * self.num_participating_clients
                ) % self.fed_config.num_clients
                participating_client_ids = [
                    (start + offset) % self.fed_config.num_clients
                    for offset in range(self.num_participating_clients)
                ]
            elif round_idx % self.fed_config.aggregation_interval == 0:
                participating_client_ids = select_participating_clients(
                    self.fed_config.num_clients,
                    self.fed_config.client_participation_rate,
                    self.participation_rng,
                )
            aggregation_client_ids.update(participating_client_ids)
            round_client_rows: list[dict[str, float]] = []
            for client_id in participating_client_ids:
                trainer = self.client_trainers[client_id]
                for local_idx in range(self.fed_config.local_updates):
                    update_idx = round_idx * self.fed_config.local_updates + local_idx
                    trainer._set_training_schedule(update_idx)
                    batch, rollout_metrics, episode_metrics = trainer.collect_rollout(
                        update_idx
                    )
                    trainer.episode_history.extend(episode_metrics)
                    update_metrics = trainer.update_policy(batch)
                    row = {
                        "round": float(round_idx + 1),
                        "client_id": float(client_id),
                        "local_update": float(local_idx + 1),
                        **rollout_metrics,
                        **update_metrics,
                    }
                    self.client_history.append(row)
                    round_client_rows.append(row)

            should_aggregate = (
                (round_idx + 1) % self.fed_config.aggregation_interval == 0
                or round_idx + 1 == self.fed_config.federated_rounds
            )
            if should_aggregate:
                aggregation_sources = (
                    sorted(aggregation_client_ids)
                    if self.fed_config.participation_schedule == "round_robin"
                    else participating_client_ids
                )
                averaged = average_state_dicts(
                    [
                        select_federated_state_dict(
                            self.client_trainers[client_id].model.state_dict(),
                            self.fed_config.aggregation_scope,
                        )
                        for client_id in aggregation_sources
                    ]
                )
                self.global_model.load_state_dict(averaged, strict=False)
                for trainer in self.client_trainers:
                    trainer.model.load_state_dict(averaged, strict=False)
                    if self.fed_config.reset_optimizer_on_aggregation:
                        reset_aggregated_optimizer_state(
                            trainer, set(averaged)
                        )
                aggregation_client_ids.clear()

            round_row = self._aggregate_client_rows(round_idx, round_client_rows)
            round_row["aggregated"] = float(should_aggregate)
            round_row["participating_clients"] = float(
                len(participating_client_ids)
            )
            round_row["rollout_episode_budget"] = float(
                len(participating_client_ids)
                * self.effective_client_rollout_episodes
            )
            round_row["aggregation_clients"] = float(
                len(aggregation_sources) if should_aggregate else 0
            )
            round_row.update(
                {
                    "eval_reward": float("nan"),
                    "eval_latency": float("nan"),
                    "eval_deadline_hit_ratio": float("nan"),
                }
            )
            if should_aggregate and (
                (round_idx + 1) % self.env_config.eval_interval == 0
                or round_idx + 1 == self.fed_config.federated_rounds
            ):
                evaluation = self.evaluate(
                    self.env_config.train_eval_episodes,
                    self.env_config.validation_seed_base,
                )
                round_row.update(
                    {
                        "eval_reward": evaluation["reward_mean"],
                        "eval_latency": evaluation["latency"],
                        "eval_deadline_hit_ratio": evaluation["deadline_hit_ratio"],
                    }
                )
                if evaluation["reward_mean"] > self.best_eval_reward:
                    self.best_eval_reward = evaluation["reward_mean"]
                    self.best_eval_metrics = dict(evaluation)
                    self._save_checkpoint(self.fed_config.best_checkpoint_name)
            self.round_history.append(round_row)

        if self.best_eval_metrics is None:
            self.best_eval_metrics = self.evaluate(
                self.env_config.train_eval_episodes,
                self.env_config.validation_seed_base,
            )
            self.best_eval_reward = self.best_eval_metrics["reward_mean"]
            self._save_checkpoint(self.fed_config.best_checkpoint_name)

        self._save_checkpoint(self.fed_config.checkpoint_name)
        best_payload = torch.load(
            self.output_dir / self.fed_config.best_checkpoint_name,
            map_location=self.env_config.device,
        )
        self.global_model.load_state_dict(best_payload["model_state_dict"])
        final_metrics = self.evaluate(
            self.env_config.eval_episodes,
            self.env_config.final_eval_seed_base,
        )
        self._save_artifacts(final_metrics)
        return self.round_history, self.client_history, final_metrics

    def evaluate(self, episodes: int, seed_base: int) -> dict[str, float]:
        self.global_model.eval()
        metrics = rollout_policy(
            SimplifiedRoutingEnv(self.env_config),
            self.env_config,
            "trained",
            episodes=episodes,
            seed_base=seed_base,
            model=self.global_model,
        )
        self.global_model.train()
        return metrics

    def close(self) -> None:
        for trainer in self.client_trainers:
            trainer.close()

    def _aggregate_client_rows(
        self,
        round_idx: int,
        rows: list[dict[str, float]],
    ) -> dict[str, float]:
        result = {"round": float(round_idx + 1)}
        excluded = {"round", "client_id", "local_update"}
        for key in rows[0]:
            if key in excluded:
                continue
            result[key] = float(np.mean([float(row[key]) for row in rows]))
        return result

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "model_state_dict": self.global_model.state_dict(),
                "env_config": asdict(self.env_config),
                "fedroute_config": asdict(self.fed_config),
            },
            self.output_dir / filename,
        )

    def _save_artifacts(self, final_metrics: dict[str, float]) -> None:
        save_history_csv(self.round_history, self.output_dir / "federated_history.csv")
        save_history_csv(self.client_history, self.output_dir / "client_history.csv")
        payload: dict[str, Any] = {
            **asdict(self.env_config),
            **{f"fedroute_{key}": value for key, value in asdict(self.fed_config).items()},
            **{f"final_{key}": value for key, value in final_metrics.items()},
            "best_eval_reward": self.best_eval_reward,
        }
        if self.best_eval_metrics is not None:
            payload.update(
                {f"best_{key}": value for key, value in self.best_eval_metrics.items()}
            )
        save_metrics_json(payload, self.output_dir / "evaluation_metrics.json")


def train_fedroute(
    env_config: ExperimentConfig,
    fed_config: FedRouteConfig,
    output_dir: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    set_seed(env_config.seed)
    trainer = FedRouteTrainer(env_config, fed_config, output_dir)
    try:
        return trainer.train()
    finally:
        trainer.close()
