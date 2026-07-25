#!/usr/bin/env bash
# Train Franka Reach via Isaac Lab. From IsaacLab root:
#   NUM_ENVS=128 MAX_ITERS=300 bash path/to/train_reach.sh

set -euo pipefail

TASK="${TASK:-Isaac-Reach-Franka-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERS="${MAX_ITERS:-150}"
SEED="${SEED:-42}"

if [[ ! -f "./isaaclab.sh" ]]; then
  echo "run from IsaacLab root"
  exit 1
fi

echo "task=${TASK} num_envs=${NUM_ENVS} max_iterations=${MAX_ITERS} seed=${SEED}"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  --headless

echo "done — see logs/rsl_rl/"
