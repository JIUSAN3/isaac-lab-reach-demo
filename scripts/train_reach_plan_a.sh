#!/usr/bin/env bash
# Short Franka Reach train. Run from IsaacLab repo root.
#   NUM_ENVS=32 MAX_ITERS=100 bash path/to/train_reach_plan_a.sh

set -euo pipefail

TASK="${TASK:-Isaac-Reach-Franka-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERS="${MAX_ITERS:-150}"
SEED="${SEED:-42}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

if [[ ! -f "./isaaclab.sh" ]]; then
  echo "run this from the IsaacLab repo root (isaaclab.sh missing)"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found"
  exit 1
fi

echo "task=${TASK} num_envs=${NUM_ENVS} max_iterations=${MAX_ITERS} seed=${SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

set +e
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERS}" \
  --seed "${SEED}" \
  ${HEADLESS_FLAG}
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "train failed (exit ${status}). check train.py --help or lower NUM_ENVS"
  exit $status
fi

echo "done — logs under logs/rsl_rl/ usually"
exit 0
