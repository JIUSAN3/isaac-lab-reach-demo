"""CPU-only mock of a Franka Reach task.

Runs on a laptop without NVIDIA GPU. Demonstrates:
  - observation / action / reward interface shape
  - action clamping and joint limits
  - episode termination on success / timeout
  - logging metrics for README plots

This is intentionally simple kinematics — not PhysX. When you move to a GPU
box, swap backend=isaac and use Isaac-Reach-Franka-v0.
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


def forward_kinematics(q: np.ndarray, link_lengths: np.ndarray | None = None) -> np.ndarray:
    """Simplified 7-DoF serial FK with a plausible tabletop workspace.

    Not real Panda DH — only needs continuous EE motion vs joints so a
    Jacobian controller and distance reward are meaningful.
    """
    q = np.asarray(q, dtype=np.float64).reshape(7)
    if link_lengths is None:
        # base lift, shoulder, upper arm, forearm-ish, wrist stack
        link_lengths = np.array([0.15, 0.10, 0.35, 0.05, 0.30, 0.05, 0.08], dtype=np.float64)
    else:
        link_lengths = np.asarray(link_lengths, dtype=np.float64)

    # Start above origin facing +x tabletop region
    p = np.array([0.0, 0.0, link_lengths[0]], dtype=np.float64)

    # Joint 0: yaw about z
    c0, s0 = np.cos(q[0]), np.sin(q[0])
    # Joint 1: shoulder pitch
    c1, s1 = np.cos(q[1]), np.sin(q[1])
    # Upper arm direction in horizontal plane after yaw, elevated by shoulder
    dir1 = np.array([c0 * c1, s0 * c1, s1], dtype=np.float64)
    p = p + link_lengths[2] * dir1

    # Joint 2: small roll-ish offset laterally
    p = p + link_lengths[3] * np.array([-s0, c0, 0.0]) * np.sin(q[2])

    # Joint 3: elbow pitch relative to shoulder
    elbow = q[1] + q[3]
    c3, s3 = np.cos(elbow), np.sin(elbow)
    dir3 = np.array([c0 * c3, s0 * c3, s3], dtype=np.float64)
    p = p + link_lengths[4] * dir3

    # Joints 4-6: wrist contribution (small)
    wrist = q[4] * 0.3 + q[5] * 0.2
    p = p + link_lengths[6] * np.array(
        [c0 * np.cos(elbow + wrist), s0 * np.cos(elbow + wrist), np.sin(elbow + wrist)],
        dtype=np.float64,
    )
    p = p + 0.02 * np.array([np.sin(q[6]), np.cos(q[6]), 0.0])
    return p


class MockReachEnv(ActionInterface):
    """Simplified 7-DoF reach env with analytic FK and dense reward."""

    backend = Backend.MOCK

    def __init__(self, config: dict[str, Any]) -> None:
        robot = config["robot"]
        env = config["env"]
        action = config["action"]
        reward = config["reward"]

        self.dof = int(robot["dof"])
        self.joint_low = np.asarray(robot["joint_limits_low"], dtype=np.float64)
        self.joint_high = np.asarray(robot["joint_limits_high"], dtype=np.float64)
        self.link_lengths = np.asarray(robot["link_lengths"], dtype=np.float64)
        self.max_joint_delta = float(robot["max_joint_delta"])
        self.action_scale = float(action["scale"])

        self.episode_length = int(env["episode_length"])
        self.success_threshold = float(env["success_threshold"])
        self.target_low = np.asarray(env["target_workspace"]["low"], dtype=np.float64)
        self.target_high = np.asarray(env["target_workspace"]["high"], dtype=np.float64)

        self.position_weight = float(reward["position_weight"])
        self.success_bonus = float(reward["success_bonus"])
        self.action_l2_weight = float(reward["action_l2_weight"])

        self._rng = np.random.default_rng()
        self._joint_pos = np.zeros(self.dof, dtype=np.float64)
        self._joint_vel = np.zeros(self.dof, dtype=np.float64)
        self._target = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._closed = False

    def fk(self, q: np.ndarray | None = None) -> np.ndarray:
        if q is None:
            q = self._joint_pos
        return forward_kinematics(q, self.link_lengths)

    def reset(self, seed: int | None = None) -> Observation:
        if self._closed:
            raise RuntimeError("Env is closed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reachable-ish home: arm stretched forward above table
        home = np.array([0.0, 0.4, 0.0, -1.0, 0.0, 0.6, 0.0], dtype=np.float64)
        self._joint_pos = home + 0.05 * self._rng.standard_normal(self.dof)
        self._joint_pos = np.clip(self._joint_pos, self.joint_low, self.joint_high)
        self._joint_vel[:] = 0.0

        # Sample target near current EE so episodes are usually solvable
        ee = self.fk()
        offset = self._rng.uniform(-0.12, 0.12, size=3)
        offset[2] = self._rng.uniform(-0.08, 0.08)
        self._target = np.clip(ee + offset, self.target_low, self.target_high)
        # Guarantee minimum challenge
        if np.linalg.norm(self._target - ee) < 0.04:
            self._target = np.clip(ee + np.array([0.08, 0.05, 0.03]), self.target_low, self.target_high)

        self._step_count = 0
        return self.get_observation()

    def step(self, action: ActionCommand) -> StepResult:
        if self._closed:
            raise RuntimeError("Env is closed")
        if action.joint_delta.shape != (self.dof,):
            raise ValueError(
                f"Expected action dim {self.dof}, got {action.joint_delta.shape}"
            )

        delta = self.scale_and_clamp(
            action, self.action_scale, self.joint_low, self.joint_high
        )
        delta = np.clip(delta, -self.max_joint_delta, self.max_joint_delta)

        prev = self._joint_pos.copy()
        self._joint_pos = np.clip(prev + delta, self.joint_low, self.joint_high)
        self._joint_vel = self._joint_pos - prev
        self._step_count += 1

        ee = self.fk()
        dist = float(np.linalg.norm(ee - self._target))
        reward = (
            -self.position_weight * dist
            - self.action_l2_weight * float(np.dot(delta, delta))
        )
        success = dist < self.success_threshold
        if success:
            reward += self.success_bonus

        terminated = success
        truncated = self._step_count >= self.episode_length
        obs = self.get_observation()
        info = {
            "distance": dist,
            "success": success,
            "step": self._step_count,
            "ee_pos": ee.copy(),
        }
        return StepResult(obs, reward, terminated, truncated, info)

    def get_observation(self) -> Observation:
        ee = self.fk()
        vector = np.concatenate(
            [self._joint_pos, self._joint_vel, ee, self._target]
        ).astype(np.float64)
        return Observation(
            vector=vector,
            joint_pos=self._joint_pos.copy(),
            joint_vel=self._joint_vel.copy(),
            ee_pos=ee,
            target_pos=self._target.copy(),
        )

    def close(self) -> None:
        self._closed = True
