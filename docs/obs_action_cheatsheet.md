# Obs / action cheat sheet

## Mock (this repo)

- obs 20 = q(7) + dq(7) + ee(3) + target(3)
- action 7 = joint delta
- reward ≈ −distance + success − action penalty

## Isaac run (from dumped config)

See `results/isaac_cloud/env.yaml` for the live manager setup.  
Don’t assume mock dims == official env dims in an interview — say you checked the dumped config.
