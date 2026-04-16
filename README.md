# GDM-HMAPPO for THz-Enabled 6G Edge-Cloud Routing

This repository contains the clean simulation code for the paper implementation of **Generative Diffusion Model enabled Hierarchical Multi-Agent PPO (GDM-HMAPPO)** for topology-aware routing and computation allocation in dynamic THz-enabled 6G edge-cloud networks.

## File Guide

| File | Role |
| --- | --- |
| `config.py` | Default simulation, THz channel, task, topology, PPO, and model hyperparameters. The final paper setting is `paper_curve_faithful_long4000`. |
| `env.py` | Dynamic THz-enabled 6G edge-cloud environment, including wireless/wired communication, computation resources, task generation, latency, reward, and QoS metrics. |
| `model.py` | Proposed GDM-HMAPPO policy network, including graph encoding, Transformer context modeling, and generative diffusion allocation. It also contains the ablation switches for removing GNN, Transformer, or GDM. |
| `ppo.py` | PPO rollout collection, optimization, validation, checkpointing, and metric logging for the proposed method. |
| `main.py` | Training entry for the proposed GDM-HMAPPO and its ablation variants. |
| `evaluation_tools.py` | Shared checkpoint loading and rollout evaluation utilities. |
| `evaluate.py` | Standalone evaluation of a trained GDM-HMAPPO checkpoint. |
| `compare_policies.py` | Simple comparison between the trained policy and lightweight non-learning baselines. |
| `baselines.py` | Heuristic baseline logic, including the delay-aware load-balanced heuristic used in comparison experiments. |
| `masac.py`, `run_masac_baseline.py` | MA-SAC benchmark implementation and training entry. |
| `mapdqn.py`, `run_mapdqn_baseline.py` | MA-P-DQN benchmark implementation and training entry. |
| `benchmark_sweeps.py` | Final performance-comparison experiment over active task flows and edge-node numbers. |
| `ablation_sweeps.py` | Final ablation experiment over active task flows for full GDM-HMAPPO, w/o GNN, w/o Transformer, and w/o GDM. |
| `utils.py` | Minimal utilities for random seeds and CSV/JSON logging. |

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

## Train the Proposed Method

```bash
python main.py \
  --preset paper_curve_faithful_long4000 \
  --device cuda \
  --output-dir runs/gdm_hmappo
```

Main outputs:

| Output | Description |
| --- | --- |
| `runs/gdm_hmappo/best_routing_model.pt` | Best validation checkpoint. |
| `runs/gdm_hmappo/routing_model_final.pt` | Final model checkpoint. |
| `runs/gdm_hmappo/training_history.csv` | Training metrics by PPO update. |
| `runs/gdm_hmappo/rollout_episode_history.csv` | Per-episode rollout metrics. |
| `runs/gdm_hmappo/evaluation_metrics.json` | Final and best-checkpoint evaluation summary. |

## Train Benchmark Methods

MA-SAC:

```bash
python run_masac_baseline.py \
  --preset paper_curve_faithful_long4000 \
  --device cuda \
  --output-dir runs/masac
```

MA-P-DQN:

```bash
python run_mapdqn_baseline.py \
  --preset paper_curve_faithful_long4000 \
  --device cuda \
  --output-dir runs/mapdqn
```

The delay-aware load-balanced heuristic does not require training and is evaluated directly inside `benchmark_sweeps.py`.

## Run Performance Comparison

After training GDM-HMAPPO, MA-SAC, and MA-P-DQN, run:

```bash
python benchmark_sweeps.py \
  --run-dir runs/gdm_hmappo \
  --masac-run-dir runs/masac \
  --mapdqn-run-dir runs/mapdqn \
  --episodes 8 \
  --num-repeats 20 \
  --active-flows 18 22 26 30 34 38 42 46 50 \
  --edge-nodes 18 22 26 30 34 38 42 46 50 \
  --output-dir results/performance_comparison
```

Important outputs:

| Output | Description |
| --- | --- |
| `default_comparison_aggregated.csv` | Default-setting comparison. |
| `active_flow_sweep_aggregated.csv` | Comparison under varying active task-flow numbers. |
| `edge_node_sweep_aggregated.csv` | Comparison under varying edge-node numbers. |
| `benchmark_sweeps_all_aggregated.csv` | All aggregated performance-comparison results. |

## Run Ablation Study

Train the three degraded variants:

```bash
python main.py --preset paper_curve_faithful_long4000 --device cuda --disable-gnn --output-dir runs/ablation_wo_gnn
python main.py --preset paper_curve_faithful_long4000 --device cuda --disable-transformer --output-dir runs/ablation_wo_transformer
python main.py --preset paper_curve_faithful_long4000 --device cuda --disable-gdm --output-dir runs/ablation_wo_gdm
```

Then evaluate all variants:

```bash
python ablation_sweeps.py \
  --full-run-dir runs/gdm_hmappo \
  --wo-gnn-run-dir runs/ablation_wo_gnn \
  --wo-transformer-run-dir runs/ablation_wo_transformer \
  --wo-gdm-run-dir runs/ablation_wo_gdm \
  --episodes 8 \
  --num-repeats 20 \
  --active-flows 18 22 26 30 34 38 42 46 50 \
  --output-dir results/ablation
```

Important outputs:

| Output | Description |
| --- | --- |
| `active_flow_ablation.csv` | Raw repeated ablation results. |
| `active_flow_ablation_aggregated.csv` | Aggregated ablation results used for paper analysis. |
| `ablation_sweeps_meta.json` | Ablation experiment metadata. |

## Metrics

The main service-oriented metrics saved by the experiment scripts include average end-to-end latency, deadline hit ratio, average latency violation ratio, timely throughput, QoS-aware load balancing index, and peak resource utilization.

The QoS-aware load balancing index is computed as the product of the global load balancing index and the deadline hit ratio.

## Reproducibility Notes

The scripts expose seed controls such as `--seed`, `--seed-base`, `--num-repeats`, and `--repeat-seed-stride`. For paper-style averaged results, use repeated evaluation with fixed checkpoints rather than a single random rollout.


