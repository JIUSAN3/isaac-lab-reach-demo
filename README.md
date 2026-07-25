# isaac-lab-reach-demo

Small Franka **reach** experiment on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), plus a CPU mock so I can debug the control loop without a GPU.

I ran a short rsl_rl train on AutoDL (RTX 3090): Isaac Sim 5.1 + Lab v2.3.2, task `Isaac-Reach-Franka-v0`. Notes and a checkpoint are under [`results/`](results/).

## What’s in here

| Path | What |
|------|------|
| `src/` | `ActionInterface` + mock reach env (FK + dense reward) |
| `scripts/run_mock_reach.py` | laptop demo (no NVIDIA GPU) |
| `scripts/train_reach*.sh` | wrappers around official Isaac Lab train/play |
| `configs/reach_config.yaml` | shared knobs / task ids |
| `docs/` | install notes, design, troubleshooting |
| `results/` | mock plots + cloud train curve / notes / `model_149.pt` |

## Mock (laptop)

```bash
pip install -r requirements-mock.txt
python scripts/run_mock_reach.py --episodes 5 --seed 42
```

Compares a simple Jacobian-style controller vs random actions.

## Isaac Lab (GPU)

Needs a real NVIDIA GPU (I used a rented 3090). Rough flow:

1. Install Isaac Sim **5.1** + clone Isaac Lab **v2.3.2** (pair matters — default Lab tip was Sim 6 / Py 3.12).
2. From the IsaacLab repo root, train e.g.:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Franka-v0 \
  --num_envs 64 \
  --max_iterations 150 \
  --seed 42 \
  --headless
```

More detail: [`docs/cloud_setup.md`](docs/cloud_setup.md), [`docs/troubleshooting.md`](docs/troubleshooting.md).

## Results (cloud)

- Run notes: [`results/isaac_run_notes.md`](results/isaac_run_notes.md)
- Curve: [`results/isaac_reach_train_curve.png`](results/isaac_reach_train_curve.png)
- Checkpoint + configs: [`results/isaac_cloud/`](results/isaac_cloud/)

Position error dropped roughly **0.46 → 0.25** over ~150 iters (short train, not tuned for SOTA).

## Design sketch

Policy/controller → `ActionCommand` (7-D joint delta) → `ActionInterface` (scale/clamp/step).  
`make_backend("mock"|"isaac")` switches implementation. Mock is only for interface + reward shape; real physics is Isaac.

See [`docs/design.md`](docs/design.md).

## License

Mock/demo code in this repo is for learning/portfolio use.  
Isaac Sim / Lab / Franka assets stay under their own NVIDIA / vendor licenses.
