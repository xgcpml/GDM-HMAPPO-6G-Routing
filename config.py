from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class ExperimentConfig:
    method_name: str = "GDM-HMAPPO"
    seed: int = 7
    device: str = "cpu"
    output_dir: str = "outputs"
    use_multi_task_slots: bool = True

    # Units used in the simulator:
    # - rates: Mbps
    # - task data sizes: Mb
    # - workloads: GFLOP
    # - capacities: GFLOPS
    # - latency / deadlines: seconds

    num_users: int = 6
    num_access: int = 3
    num_edges: int = 4
    num_clouds: int = 1
    num_candidate_paths: int = 4
    candidate_path_selection_mode: str = "completion_diverse"
    path_length: int = 5
    alloc_dim: int = 3

    episode_length: int = 18
    train_updates: int = 60
    eval_episodes: int = 8
    eval_interval: int = 1
    train_eval_episodes: int = 10
    rollout_episodes_per_update: int = 4
    parallel_rollout_envs: int = 1
    parallel_eval_envs: int = 1
    topology_seed_stride: int = 0
    eval_topology_seed_offset: int = 1_000_000
    warmup_updates: int = 6
    warmup_batch_size: int = 128
    search_bootstrap_updates: int = 2
    search_bootstrap_batch_size: int = 128
    warmup_seed_base: int = 12_000
    warmup_route_coef: float = 0.05
    warmup_diffusion_coef: float = 1.0
    search_bootstrap_route_coef: float = 0.18
    search_bootstrap_diffusion_coef: float = 1.0
    validation_seed_base: int = 20_000
    final_eval_seed_base: int = 30_000

    hidden_dim: int = 96
    graph_layers: int = 2
    transformer_layers: int = 2
    attention_heads: int = 4
    use_gnn_encoder: bool = True
    use_graph_attention_encoder: bool = False
    use_transformer_context: bool = True
    use_global_graph_context: bool = True
    use_candidate_resource_summaries: bool = True
    use_gdm_allocator: bool = True
    allocation_policy_family: str = "gdm"
    alloc_refinement_steps: int = 6
    alloc_min_concentration: float = 0.2
    alloc_noise_scale: float = 0.65
    alloc_noise_final_scale: float = 0.08
    rollout_deterministic_allocation: bool = False
    allocation_search_candidates: int = 24
    route_search_candidates: int = 3
    route_search_prior_coef: float = 0.08
    use_heuristic_search_candidates: bool = True

    lr: float = 2.5e-4
    shared_lr_factor: float = 1.0
    actor_lr_factor: float = 1.0
    critic_lr_factor: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    ppo_epochs: int = 5
    mini_batch_size: int = 64
    policy_coef: float = 1.0
    value_coef: float = 0.5
    advantage_clip: float = 0.0
    route_entropy_coef: float = 0.010
    route_entropy_final_coef: float = 0.002
    route_sampling_temperature_initial: float = 1.0
    route_sampling_temperature_final: float = 1.0
    greedy_rollout_fraction: float = 1.1
    greedy_rollout_ramp_fraction: float = 0.2
    greedy_rollout_initial_prob: float = 0.0
    greedy_rollout_ramp_power: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.0
    use_topology_route_prior: bool = True
    topology_route_prior_initial_scale: float = 1.15
    topology_route_prior_final_scale: float = 0.35
    topology_route_temperature: float = 0.65
    use_topology_distillation: bool = True
    topology_distill_coef: float = 0.30
    topology_distill_final_coef: float = 0.08
    diffusion_loss_coef: float = 1.0
    diffusion_recon_coef: float = 0.35
    diffusion_prior_coef: float = 0.25
    use_search_distillation: bool = True
    search_distill_route_coef: float = 0.06
    search_distill_diffusion_coef: float = 0.20
    search_distill_min_improvement: float = 0.01
    search_distill_relative_threshold: float = 0.08
    search_distill_max_weight: float = 2.5
    search_distill_delay_fraction: float = 0.0
    search_distill_ramp_fraction: float = 0.0
    use_search_guided_inference: bool = True
    lr_final_factor: float = 0.45
    exploration_anneal_updates: int = 0
    learning_rate_anneal_updates: int = 0
    learning_rate_warmup_updates: int = 0
    learning_rate_warmup_initial_factor: float = 0.2
    exploration_hold_fraction: float = 0.10
    exploration_decay_power: float = 1.0
    plateau_patience_updates: int = 10
    plateau_improve_threshold: float = 1e-8
    plateau_eval_window: int = 1
    plateau_min_update: int = 0
    plateau_stability_tolerance: float = 0.0
    rollback_lr_factor: float = 0.72
    max_rollbacks: int = 1
    freeze_after_rollbacks: bool = False
    diffusion_beta_start: float = 0.02
    diffusion_beta_end: float = 0.10
    adv_weight_eta: float = 0.4
    adv_weight_min: float = 0.35
    adv_weight_max: float = 1.80
    diffusion_latent_clip: float = 3.0

    penalty_coeff: float = 3.0
    invalid_path_penalty: float = 0.60
    slot_duration_s: float = 0.02
    queue_backlog_threshold_slots: float = 2.0
    reward_deadline_weight: float = 1.60
    reward_queue_weight: float = 0.1125
    queue_drain_ratio: float = 0.30
    flow_load_reference: float = 24.0
    edge_count_reference: int = 30
    load_pressure_power: float = 1.15
    edge_relief_power: float = 0.85
    cloud_relief_power: float = 0.30
    deadline_load_tightening: float = 0.055
    task_load_bias_scale: float = 0.060
    resource_metrics_edge_only: bool = True
    resource_utilization_weighted: bool = True
    resource_metrics_include_background: bool = False
    resource_metrics_active_only: bool = True
    resource_metrics_reachable_only: bool = False
    resource_peak_percentile: float = 95.0
    reward_latency_scale: float = 28.0
    reward_violation_scale: float = 18.0
    reward_queue_penalty_scale: float = 4.0
    reward_unavailability_scale: float = 3.0
    reward_violation_clip: float = 2.5
    training_seed_pool_size: int = 0
    training_seed_pool_candidate_multiplier: int = 3
    training_seed_pool_trim_fraction: float = 0.2
    training_seed_pool_strategy: str = "heuristic"
    training_seed_hold_updates: int = 1
    training_seed_window_shift: int = 0
    training_seed_curriculum_initial_fraction: float = 1.0
    training_seed_curriculum_ramp_fraction: float = 0.0
    training_seed_curriculum_power: float = 1.0

    wireless_up_prob: float = 0.93
    wireless_rate_jitter: tuple[float, float] = (0.88, 1.12)
    wired_rate_jitter: tuple[float, float] = (0.92, 1.08)
    compute_jitter: tuple[float, float] = (0.88, 1.15)
    temporal_rate_momentum: float = 0.76
    temporal_capacity_momentum: float = 0.82
    task_pressure_momentum: float = 0.72
    observation_rate_momentum: float = 0.86
    observation_capacity_momentum: float = 0.88
    observation_queue_momentum: float = 0.82
    observation_noise_scale: float = 0.06
    observation_bins: int = 7
    path_confidence_floor: float = 0.24

    area_width_m: float = 1_000.0
    area_height_m: float = 1_000.0
    carrier_frequency_ghz: float = 140.0
    wireless_bandwidth_ghz: tuple[float, float] = (1.0, 5.0)
    tx_power_dbm: float = 20.0
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 7.0
    beamforming_gain_db: float = 60.0
    wireless_shadowing_sigma_db: float = 3.0
    wireless_interference_to_noise: tuple[float, float] = (0.05, 0.30)
    blockage_residual_rate_ratio: float = 0.02
    wired_propagation_delay_ms: tuple[float, float] = (1.0, 10.0)
    user_access_degree: int = 3
    access_edge_degree: int = 4
    edge_neighbor_degree: int = 3
    edge_cloud_degree: int = 2
    edge_placement_mode: str = "access_anchored_random"
    edge_layout_reference_count: int = 0

    active_flow_range: tuple[int, int] = (6, 18)
    deadline_budget_factor: float = 1.0
    background_queue_scale: float = 0.08
    access_contention_scale: float = 0.22
    backhaul_contention_scale: float = 0.14
    compute_pressure_scale: float = 0.18
    episode_variation_scale: float = 1.0

    user_access_rate: tuple[float, float] = (1_200.0, 8_000.0)
    access_edge_rate: tuple[float, float] = (20_000.0, 80_000.0)
    edge_edge_rate: tuple[float, float] = (25_000.0, 100_000.0)
    edge_cloud_rate: tuple[float, float] = (40_000.0, 120_000.0)

    edge_capacity: tuple[float, float] = (4_000.0, 12_000.0)
    cloud_capacity: tuple[float, float] = (35_000.0, 90_000.0)

    task_data_range: tuple[float, float] = (240.0, 1_600.0)
    task_compute_range: tuple[float, float] = (80.0, 420.0)

    node_feature_dim: int = 16
    task_feature_dim: int = 6
    candidate_feature_dim: int = 14

    max_rate_scale: float = 120_000.0
    max_capacity_scale: float = 100_000.0
    max_queue_scale: float = 28_000.0
    max_deadline_scale: float = 1.8

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def apply_experiment_preset(config: ExperimentConfig, preset: str) -> ExperimentConfig:
    normalized = preset.strip().lower()
    if normalized in {"default", "base", "current"}:
        return config

    if normalized == "toy":
        return replace(
            config,
            num_users=6,
            num_access=3,
            num_edges=4,
            num_clouds=1,
            num_candidate_paths=4,
            episode_length=18,
            train_updates=60,
            eval_episodes=8,
            train_eval_episodes=10,
            rollout_episodes_per_update=4,
            warmup_updates=4,
            search_bootstrap_updates=1,
            active_flow_range=(4, 12),
        )

    if normalized == "paper_lite":
        return replace(
            config,
            num_users=20,
            num_access=5,
            num_edges=10,
            num_clouds=3,
            num_candidate_paths=8,
            episode_length=24,
            train_updates=120,
            eval_episodes=16,
            train_eval_episodes=16,
            rollout_episodes_per_update=6,
            parallel_rollout_envs=4,
            parallel_eval_envs=1,
            warmup_updates=3,
            search_bootstrap_updates=1,
            warmup_batch_size=160,
            search_bootstrap_batch_size=160,
            hidden_dim=112,
            graph_layers=3,
            transformer_layers=2,
            alloc_refinement_steps=8,
            alloc_noise_scale=0.58,
            alloc_noise_final_scale=0.05,
            lr=2.0e-4,
            ppo_epochs=4,
            mini_batch_size=96,
            route_entropy_coef=0.022,
            route_entropy_final_coef=0.004,
            search_distill_route_coef=0.025,
            search_distill_diffusion_coef=0.06,
            search_distill_relative_threshold=0.06,
            search_distill_max_weight=2.2,
            lr_final_factor=0.30,
            exploration_hold_fraction=0.06,
            observation_bins=8,
            path_confidence_floor=0.30,
            user_access_degree=4,
            access_edge_degree=5,
            edge_neighbor_degree=4,
            edge_cloud_degree=2,
            active_flow_range=(8, 24),
            background_queue_scale=0.09,
            access_contention_scale=0.24,
            backhaul_contention_scale=0.15,
            compute_pressure_scale=0.20,
            edge_capacity=(5_000.0, 15_000.0),
            cloud_capacity=(35_000.0, 90_000.0),
            task_data_range=(320.0, 2_000.0),
            task_compute_range=(100.0, 600.0),
        )

    if normalized == "paper_curve":
        return replace(
            config,
            num_users=60,
            num_access=15,
            num_edges=30,
            num_clouds=9,
            num_candidate_paths=14,
            episode_length=24,
            train_updates=260,
            eval_episodes=12,
            train_eval_episodes=12,
            rollout_episodes_per_update=8,
            parallel_rollout_envs=8,
            parallel_eval_envs=1,
            warmup_updates=1,
            search_bootstrap_updates=1,
            warmup_batch_size=256,
            search_bootstrap_batch_size=256,
            warmup_route_coef=0.08,
            search_bootstrap_route_coef=0.10,
            hidden_dim=128,
            graph_layers=3,
            transformer_layers=3,
            attention_heads=4,
            alloc_refinement_steps=10,
            alloc_noise_scale=0.60,
            alloc_noise_final_scale=0.01,
            rollout_deterministic_allocation=False,
            allocation_search_candidates=32,
            route_search_candidates=5,
            route_search_prior_coef=0.05,
            lr=1.6e-4,
            clip_eps=0.15,
            ppo_epochs=4,
            mini_batch_size=160,
            route_entropy_coef=0.005,
            route_entropy_final_coef=0.00002,
            route_sampling_temperature_initial=1.04,
            route_sampling_temperature_final=0.03,
            greedy_rollout_fraction=0.10,
            greedy_rollout_ramp_fraction=0.08,
            topology_route_prior_initial_scale=1.05,
            topology_route_prior_final_scale=0.30,
            topology_route_temperature=0.55,
            topology_distill_coef=0.40,
            topology_distill_final_coef=0.12,
            search_distill_route_coef=0.085,
            search_distill_diffusion_coef=0.070,
            search_distill_relative_threshold=0.10,
            search_distill_max_weight=2.0,
            search_distill_delay_fraction=0.0,
            search_distill_ramp_fraction=0.05,
            lr_final_factor=0.25,
            exploration_hold_fraction=0.08,
            exploration_decay_power=1.20,
            plateau_patience_updates=4,
            plateau_improve_threshold=0.60,
            rollback_lr_factor=0.72,
            max_rollbacks=1,
            freeze_after_rollbacks=True,
            invalid_path_penalty=0.45,
            queue_drain_ratio=0.24,
            reward_latency_scale=64.0,
            reward_violation_scale=68.0,
            reward_queue_penalty_scale=5.0,
            reward_unavailability_scale=3.5,
            wireless_up_prob=0.985,
            wireless_rate_jitter=(0.96, 1.07),
            wired_rate_jitter=(0.98, 1.04),
            compute_jitter=(0.96, 1.06),
            temporal_rate_momentum=0.87,
            temporal_capacity_momentum=0.90,
            task_pressure_momentum=0.86,
            observation_rate_momentum=0.90,
            observation_capacity_momentum=0.92,
            observation_queue_momentum=0.88,
            observation_noise_scale=0.025,
            observation_bins=8,
            path_confidence_floor=0.30,
            user_access_degree=4,
            access_edge_degree=6,
            edge_neighbor_degree=5,
            edge_cloud_degree=3,
            active_flow_range=(24, 34),
            deadline_budget_factor=1.34,
            background_queue_scale=0.06,
            access_contention_scale=0.18,
            backhaul_contention_scale=0.12,
            compute_pressure_scale=0.16,
            training_seed_pool_size=0,
            access_edge_rate=(20_000.0, 80_000.0),
            edge_edge_rate=(30_000.0, 100_000.0),
            edge_cloud_rate=(40_000.0, 120_000.0),
            edge_capacity=(6_000.0, 18_000.0),
            cloud_capacity=(40_000.0, 100_000.0),
            task_data_range=(600.0, 1_800.0),
            task_compute_range=(180.0, 600.0),
        )

    if normalized == "paper_curve_faithful":
        return replace(
            apply_experiment_preset(config, "paper_curve"),
            num_candidate_paths=10,
            train_updates=240,
            rollout_episodes_per_update=8,
            parallel_rollout_envs=8,
            warmup_updates=0,
            search_bootstrap_updates=0,
            lr=7.5e-5,
            clip_eps=0.10,
            ppo_epochs=2,
            mini_batch_size=128,
            alloc_noise_scale=0.45,
            rollout_deterministic_allocation=False,
            alloc_noise_final_scale=0.02,
            route_entropy_coef=0.0035,
            route_entropy_final_coef=0.00005,
            route_sampling_temperature_initial=1.00,
            route_sampling_temperature_final=0.05,
            greedy_rollout_fraction=0.0,
            greedy_rollout_ramp_fraction=0.07,
            use_topology_route_prior=False,
            topology_route_prior_initial_scale=0.0,
            topology_route_prior_final_scale=0.0,
            use_topology_distillation=False,
            topology_distill_coef=0.0,
            topology_distill_final_coef=0.0,
            use_search_distillation=False,
            search_distill_route_coef=0.0,
            search_distill_diffusion_coef=0.0,
            search_distill_relative_threshold=1.0,
            search_distill_max_weight=0.0,
            use_search_guided_inference=False,
            route_search_prior_coef=0.0,
            plateau_patience_updates=12,
            plateau_improve_threshold=0.60,
            plateau_eval_window=1,
            plateau_min_update=0,
            plateau_stability_tolerance=0.0,
            rollback_lr_factor=0.65,
            max_rollbacks=2,
            freeze_after_rollbacks=True,
            lr_final_factor=0.25,
            exploration_hold_fraction=0.08,
            exploration_decay_power=1.20,
            wireless_up_prob=0.990,
            wireless_rate_jitter=(0.975, 1.045),
            wired_rate_jitter=(0.985, 1.035),
            compute_jitter=(0.975, 1.05),
            temporal_rate_momentum=0.89,
            temporal_capacity_momentum=0.92,
            task_pressure_momentum=0.88,
            observation_rate_momentum=0.92,
            observation_capacity_momentum=0.93,
            observation_queue_momentum=0.90,
            observation_noise_scale=0.020,
            active_flow_range=(21, 29),
            deadline_budget_factor=1.40,
            background_queue_scale=0.045,
            access_contention_scale=0.14,
            backhaul_contention_scale=0.10,
            compute_pressure_scale=0.12,
            episode_variation_scale=0.65,
            reward_latency_scale=64.0,
            reward_violation_scale=68.0,
            training_seed_pool_size=0,
            training_seed_pool_candidate_multiplier=4,
            training_seed_pool_trim_fraction=0.2,
            training_seed_hold_updates=1,
            training_seed_window_shift=0,
            training_seed_curriculum_initial_fraction=1.0,
            training_seed_curriculum_ramp_fraction=0.0,
            training_seed_curriculum_power=1.0,
        )

    if normalized == "paper_curve_faithful_long4000":
        return replace(
            apply_experiment_preset(config, "paper_curve_faithful"),
            episode_length=48,
            train_updates=125,
            rollout_episodes_per_update=32,
            parallel_rollout_envs=8,
            eval_episodes=24,
            train_eval_episodes=24,
            lr=5.0e-5,
            shared_lr_factor=0.55,
            actor_lr_factor=1.00,
            critic_lr_factor=1.40,
            clip_eps=0.06,
            value_clip_eps=0.10,
            target_kl=0.003,
            ppo_epochs=1,
            mini_batch_size=288,
            policy_coef=1.0,
            value_coef=0.20,
            advantage_clip=2.5,
            diffusion_loss_coef=0.50,
            alloc_noise_scale=0.18,
            alloc_noise_final_scale=0.00,
            rollout_deterministic_allocation=True,
            route_entropy_coef=0.0010,
            route_entropy_final_coef=0.00002,
            route_sampling_temperature_initial=0.45,
            route_sampling_temperature_final=0.03,
            greedy_rollout_ramp_fraction=0.08,
            greedy_rollout_initial_prob=0.0,
            greedy_rollout_ramp_power=1.0,
            exploration_hold_fraction=0.02,
            exploration_decay_power=1.0,
            plateau_patience_updates=2,
            plateau_eval_window=2,
            plateau_min_update=15,
            plateau_stability_tolerance=45.0,
            rollback_lr_factor=0.75,
            max_rollbacks=0,
            freeze_after_rollbacks=True,
            wireless_rate_jitter=(0.98, 1.035),
            wired_rate_jitter=(0.99, 1.025),
            compute_jitter=(0.98, 1.035),
            task_pressure_momentum=0.90,
            observation_noise_scale=0.015,
            active_flow_range=(22, 26),
            background_queue_scale=0.040,
            access_contention_scale=0.12,
            backhaul_contention_scale=0.09,
            compute_pressure_scale=0.10,
            episode_variation_scale=0.45,
            training_seed_pool_size=0,
            training_seed_pool_candidate_multiplier=4,
            training_seed_pool_trim_fraction=0.2,
            training_seed_pool_strategy="sequential",
            training_seed_hold_updates=1,
            training_seed_window_shift=0,
            training_seed_curriculum_initial_fraction=1.0,
            training_seed_curriculum_ramp_fraction=0.0,
            training_seed_curriculum_power=1.0,
            reward_latency_scale=40.0,
            reward_violation_scale=64.0,
            reward_queue_penalty_scale=4.5,
        )

    if normalized == "v2":
        return replace(
            apply_experiment_preset(config, "paper_curve_faithful_long4000"),
            use_multi_task_slots=True,
            slot_duration_s=0.08,
            deadline_budget_factor=1.10,
            candidate_path_selection_mode="cost_diverse",
            background_queue_scale=0.0,
            task_load_bias_scale=0.0,
            deadline_load_tightening=0.0,
            resource_metrics_edge_only=False,
            resource_metrics_active_only=False,
            resource_metrics_reachable_only=True,
        )

    raise ValueError(f"Unsupported experiment preset: {preset}")
