# Design notes

## Goal

1. Touch the official Isaac Lab reach stack (`Isaac-Reach-Franka-v0`).
2. Keep a thin **action interface** so higher-level code isn’t glued to one simulator.

Laptop has no NVIDIA GPU → mock backend first; Isaac on a rented 3090.

## Layout

```text
controller / policy
    └─ ActionCommand (joint delta, ~[-1,1])
           └─ ActionInterface.scale_and_clamp + step
                  ├─ MockReachEnv   (analytic FK)
                  └─ Isaac backend  (official env / train scripts)
```

```python
env = make_backend("mock", config)   # here
# train scripts hit Isaac Lab directly on the GPU box
```

## Mock obs / action / reward

| | dim | notes |
|--|-----|--------|
| joint pos / vel | 7 + 7 | |
| ee pos, target | 3 + 3 | |
| **obs** | **20** | concat |
| **action** | **7** | joint delta |
| reward | | `-dist - λ‖Δq‖² + success bonus` |
| success | | `‖ee − target‖ < 0.05 m` |

Isaac’s live manager config can differ — see dumped `results/isaac_cloud/env.yaml` from the real run.

## Safety-ish bits (even in mock)

- clip policy output, scale, per-step delta limit  
- joint limits  
- episode timeout  

Same ideas you’d want before streaming joints to hardware.
