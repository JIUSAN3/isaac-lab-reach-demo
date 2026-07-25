# GPU setup notes

Laptop: Windows, AMD 780M — no local Isaac. Training was on **AutoDL RTX 3090**.

Official docs change often; re-check pins the day you install:

- [Isaac Lab install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Pip install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [RL scripts](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)

## What worked for me (2026-07)

| | |
|--|--|
| GPU | RTX 3090 24GB (need RT cores; avoid A100/H100 for Isaac Sim) |
| Python | 3.11 |
| Isaac Sim | `pip install "isaacsim[all,extscache]==5.1.0"` + pypi.nvidia.com |
| torch | 2.7.0 + cu128 wheels |
| Isaac Lab | git tag **`v2.3.2`** (matches Sim 5.1) |

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v2.3.2
./isaaclab.sh -i rsl_rl
```

Put big installs on a large data disk if the system disk is tiny.

## Train / play

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 --num_envs 64 --max_iterations 150 \
  --seed 42 --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-Franka-Play-v0 --num_envs 4 --headless
```

Helpers in this repo: `scripts/train_reach_plan_a.sh`, `scripts/play_reach.sh` (run from IsaacLab root).

## After a run

Copy into `results/`: curve, notes, a checkpoint if small enough.  
Then shut the instance down.
