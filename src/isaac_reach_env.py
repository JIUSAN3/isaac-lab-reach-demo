"""Optional Isaac Lab adapter (import deferred so mock works without GPU).

Most training uses Isaac Lab's own train.py; this class is only for the
shared ActionInterface shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.action_interface import (
    ActionCommand,
    ActionInterface,
    Backend,
    Observation,
    StepResult,
)


class IsaacReachEnv(ActionInterface):
    """Thin wrapper; prefer official train/play scripts for real runs."""

    backend = Backend.ISAAC

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.task_id = config["task"]["isaac_task_id"]
        self.dof = int(config["robot"]["dof"])
        self.action_scale = float(config["action"]["scale"])
        self._env = None
        self._closed = False
        self._last_obs_raw = None

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        try:
            # Deferred imports — only resolve on GPU/Isaac machines.
            import gymnasium as gym  # noqa: F401
            from isaaclab.app import AppLauncher
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Lab / Isaac Sim is not available in this Python env.\n"
                "This laptop has no NVIDIA GPU — use backend=mock here, or run\n"
                "on a cloud GPU following docs/cloud_setup.md.\n"
                f"Original import error: {exc}"
            ) from exc

        raise RuntimeError(
            "Inline IsaacReachEnv embedding needs a running Omniverse app "
            f"context for task '{self.task_id}'.\n"
            "For the portfolio demo, prefer the official entrypoints:\n"
            "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py "
            f"--task {self.task_id} --headless\n"
            "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py "
            f"--task {self.config['task']['isaac_play_task_id']} "
            "--num_envs 16\n"
            "See docs/cloud_setup.md and scripts/train_reach.sh."
        )

    def reset(self, seed: int | None = None) -> Observation:
        self._ensure_env()
        raise AssertionError("unreachable")

    def step(self, action: ActionCommand) -> StepResult:
        self._ensure_env()
        raise AssertionError("unreachable")

    def get_observation(self) -> Observation:
        self._ensure_env()
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._closed = True
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None

    @staticmethod
    def _to_observation(flat: np.ndarray, dof: int) -> Observation:
        """Best-effort split if you later wire a real env obs vector.

        Official Reach obs layout can differ by manager config; treat this
        as a placeholder mapping and adjust against the live env once on GPU.
        """
        flat = np.asarray(flat, dtype=np.float64).reshape(-1)
        joint_pos = flat[:dof]
        joint_vel = flat[dof : 2 * dof]
        ee_pos = flat[2 * dof : 2 * dof + 3]
        target_pos = flat[2 * dof + 3 : 2 * dof + 6]
        return Observation(
            vector=flat,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            ee_pos=ee_pos,
            target_pos=target_pos,
        )
