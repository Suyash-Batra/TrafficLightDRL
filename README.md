# SUMO DRL Traffic-Light Control

Adaptive traffic-light control using Deep Reinforcement Learning (DQN-style) and SUMO (Simulation of Urban Mobility). This repository contains training and baseline scripts, utilities for running domain-randomized SUMO experiments, and tools to evaluate and visualize per-run waiting-time statistics.

---

## Table of Contents

* Overview
* Features
* Requirements
* Installation
* File structure
* Quick start (train / test / static baseline)
* Outputs and interpretation
* Implementation details
* Tips and troubleshooting
* License

---

## Overview

This project trains a compact DRL agent to control traffic signals in SUMO to minimize vehicle waiting time. The code supports domain randomization by accepting multiple SUMO configuration files, automatically detects appropriate input dimensions, and saves trained weights and diagnostics for analysis.

## Features

* DQN-style agent with separate target network and epsilon-greedy exploration
* Per-junction replay memory (supports multiple intersections)
* Auto-detection of input dimensions across SUMO configs
* Reward shaping based on lane waiting-time reduction
* Robust SUMO handling (handles early termination/TeleTRACl errors)
* Static fixed-timing baseline for direct comparisons
* Automatic plotting of training/testing curves and tripinfo parsing

## Requirements

* SUMO (set `SUMO_HOME` environment variable and ensure `sumo`/`sumo-gui` available)
* Python 3.8+
* Python packages: `numpy`, `torch`, `matplotlib`, `sumolib`, `traci`

Install Python deps with:

```bash
pip install -r requirements.txt
# or
pip install numpy torch matplotlib sumolib traci
```

## File structure

* `train.py` — Main DRL training & testing loop (model, agent, training logic)
* `tets.py` — Static controller (fixed green/yellow/all-red cycle) for baseline comparisons
* `*.sumocfg`, routes, and network files — SUMO scenarios (domain-randomized configs)
* `models/` — Saved model weights
* `plots/` — Saved plots for training/test runs
* `maps/tripinfo.xml` — SUMO tripinfo output parsed by scripts

## Quick start

Ensure `SUMO_HOME` is set and SUMO binaries are on PATH.

Train (example):

```bash
python train.py --train -m model_name -e 50 -s 500 --train-cfgs pune.sumocfg,pune_alt.sumocfg --nogui
```

Test (example):

```bash
python train.py -m model_name --test-cfg pune_test.sumocfg -e 10 -s 500 --nogui
```

Run static baseline:

```bash
python tets.py -e 50 -s 500 --test-cfg pune_test.sumocfg --nogui
```

### Important options

* `--train` — enable training mode
* `-m/--model_name` — model filename (saved to `models/`)
* `-e/--epochs` — number of epochs/runs
* `-s/--steps` — SUMO simulation steps per epoch
* `--train-cfgs` — comma-separated train SUMO cfg files
* `--nogui` — run SUMO headless
* `--seed` — random seed
* `--fine-tune` — fine-tune on test environment after evaluation

## Outputs and interpretation

* `models/<model_name>.bin` — saved PyTorch weights
* `plots/avg_wait_vs_epoch_*.png` — training/testing curves
* `maps/tripinfo.xml` — SUMO tripinfo parsed for waiting-time statistics
* Log prints contain per-run metrics (mean/median/p90 waiting times, teleports)

## Implementation details

* Agent: two-layer feed-forward network (128, 64 units)
* Optimizer: Adam, learning rate `1e-4` and MSE loss
* Exploration: epsilon-greedy with linear decay
* Memory: separate circular buffer per-junction for stable learning

## Tips & troubleshooting

* If SUMO exits early, ensure `--end <steps>` is passed (the training script already adds it).
* If no traffic lights found in a cfg, check network/route files.
* Use `--nogui` on compute servers.

## License

MIT

---

