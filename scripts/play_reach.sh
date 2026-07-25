#!/usr/bin/env bash
# Play Reach. From IsaacLab root. Optional: CHECKPOINT=... bash play_reach.sh

set -euo pipefail

TASK="${TASK:-Isaac-Reach-Franka-Play-v0}"
NUM_ENVS="${NUM_ENVS:-4}"

if [[ ! -f "./isaaclab.sh" ]]; then
  echo "run from IsaacLab root"
  exit 1
fi

EXTRA=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  EXTRA+=(--checkpoint "${CHECKPOINT}")
else
  EXTRA+=(--use_pretrained_checkpoint)
fi

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  "${EXTRA[@]}"
