# Run notes — Franka Reach on AutoDL

**Date:** 2026-07-27  
**Box:** AutoDL (`connect.nmb2.seetacloud.com`), RTX 3090 24GB, data on `/root/autodl-fs`

## Stack

- Isaac Sim 5.1.0 (pip)
- Isaac Lab **v2.3.2** (not the default tip — that wanted Sim 6 / Python 3.12)
- Python 3.11, torch 2.7.0+cu128
- rsl-rl-lib 3.1.2

## Job

```text
task:   Isaac-Reach-Franka-v0
mode:   headless
seed:   42
envs:   64 (requested)
iters:  ~150
log:    logs/rsl_rl/franka_reach/2026-07-27_17-22-12
ckpts:  model_0 / 50 / 100 / 149.pt
```

Curve and `model_149.pt` are in this repo under `results/`.

## Things that broke (and fixes)

1. **Lab version** — bare `git clone` pulled a Sim-6 branch; checked out `v2.3.2` for Sim 5.1.
2. **`isaaclab` package missing** — `flatdict` failed to build; installed core with older setuptools + `pip install -e source/isaaclab --no-build-isolation`.
3. **`libXt.so.6` / GLU** — minimal image; `apt install libxt6` (and related X/GL libs).
4. **autodl-fs** — network disk made `extscache` extract very slow (hours), but finished.

Vulkan/display warnings showed up headless; train still ran.

## Quick takeaway

Short rsl_rl baseline on official reach: position error ~0.46 → ~0.25 over the run. Good enough to show the stack works end-to-end; not a paper-level tune.
