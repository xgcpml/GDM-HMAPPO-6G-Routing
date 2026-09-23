# GDM-HMAPPO v2 for THz-Enabled 6G Edge-Cloud Routing

This repository provides the clean PyTorch implementation of **GDM-HMAPPO v2** for topology-aware routing and workload allocation in dynamic THz-enabled 6G edge-cloud networks.

## File Guide

| File | Role |
| --- | --- |
| `config.py` | Simulation, THz channel, topology, task, PPO, and model parameters. |
| `env.py` | Dynamic communication-computation environment and service metrics. |
| `model.py` | GNN-Transformer policy with generative diffusion allocation. |
| `ppo.py` | PPO rollout collection, optimization, validation, and checkpointing. |
| `main.py` | Training entry for GDM-HMAPPO and GraphPR. |
| `evaluation_tools.py`, `evaluate.py` | Checkpoint loading and standalone policy evaluation. |
| `masac.py`, `run_masac_baseline.py` | MA-SAC implementation and training entry. |
| `mapdqn.py`, `run_mapdqn_baseline.py` | MA-P-DQN implementation and training entry. |
| `fedroute.py`, `run_fedroute_baseline.py` | FedRoute implementation and training entry. |
| `baselines.py` | Delay-aware load-balanced heuristic support. |
| `benchmark_sweeps.py` | Performance comparison over task-flow and edge-node grids. |
| `evaluate_diffusion_step_sensitivity.py` | Denoising-step comparison. |
| `profile_inference.py` | Online inference runtime and peak GPU-memory profiling. |
| `experiment_protocol.py` | Shared experiment grids and run-manifest utilities. |
| `utils.py` | Seed and metric-output utilities. |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU training is recommended. Use `--device cuda` when CUDA-enabled PyTorch is available.

## Convergence Experiment

The following commands train GDM-HMAPPO under three task-flow densities. Each run saves update-level and episode-level reward, latency, and deadline-hit metrics.

```bash
python main.py --preset v2 --variant proposed --active-flows 26 --device cuda --output-dir runs/convergence_flow26
python main.py --preset v2 --variant proposed --active-flows 34 --device cuda --output-dir runs/convergence_flow34
python main.py --preset v2 --variant proposed --active-flows 50 --device cuda --output-dir runs/convergence_flow50
```

The main output files are `training_history.csv`, `rollout_episode_history.csv`, `evaluation_metrics.json`, `best_routing_model.pt`, and `routing_model_final.pt` in each run directory.

## Performance Comparison

Train the learning-based methods:

```bash
python main.py --preset v2 --variant proposed --device cuda --output-dir runs/gdm_hmappo
python main.py --preset v2 --variant graphpr --device cuda --output-dir runs/graphpr
python run_fedroute_baseline.py --preset v2 --device cuda --output-dir runs/fedroute
python run_masac_baseline.py --preset v2 --device cuda --output-dir runs/masac
python run_mapdqn_baseline.py --preset v2 --device cuda --output-dir runs/mapdqn
```

Run the common evaluation grids. DALBH is evaluated directly and does not require training.

```bash
python benchmark_sweeps.py \
  --run-dir runs/gdm_hmappo \
  --graphpr-run-dir runs/graphpr \
  --fedroute-run-dir runs/fedroute \
  --masac-run-dir runs/masac \
  --mapdqn-run-dir runs/mapdqn \
  --episodes 1 \
  --num-repeats 4 \
  --num-topology-repeats 5 \
  --active-flows 18 22 26 30 34 38 42 46 50 \
  --edge-nodes 18 22 26 30 34 38 42 46 50 \
  --output-dir results/performance_comparison
```

The script saves raw and aggregated CSV files for the default setting, task-flow sweep, and edge-node sweep.

## Denoising-Step Comparison

Train matched GDM-HMAPPO policies with 5, 10, and 15 denoising steps:

```bash
python main.py --preset v2 --variant proposed --alloc-refinement-steps 5 --device cuda --output-dir runs/gdm_steps5
python main.py --preset v2 --variant proposed --alloc-refinement-steps 10 --device cuda --output-dir runs/gdm_steps10
python main.py --preset v2 --variant proposed --alloc-refinement-steps 15 --device cuda --output-dir runs/gdm_steps15
```

Evaluate service performance and inference cost:

```bash
python evaluate_diffusion_step_sensitivity.py \
  --run-5 runs/gdm_steps5 \
  --run-10 runs/gdm_steps10 \
  --run-15 runs/gdm_steps15 \
  --device cuda \
  --output-dir results/denoising_steps
```

## Runtime Test

Measure online inference runtime per decision slot and peak GPU memory:

```bash
python profile_inference.py \
  --proposed-run-dir runs/gdm_hmappo \
  --masac-run-dir runs/masac \
  --mapdqn-run-dir runs/mapdqn \
  --device cuda \
  --warmup-slots 20 \
  --measured-slots 100 \
  --output-dir results/runtime
```

The profiler saves raw measurements, aggregated summaries, and the complete run configuration. Random seeds and input checkpoints are recorded for reproducibility.

## Metrics

The evaluation scripts report average end-to-end latency, deadline hit ratio, timely throughput, average latency violation ratio, QoS-aware load balancing, peak computational resource utilization, reward, inference runtime, and peak GPU memory where applicable.

## Reproducibility

Experiment scripts expose deterministic seed controls and save a run manifest containing the command, configuration, software environment, Git commit, checkpoints, and generated artifacts.
