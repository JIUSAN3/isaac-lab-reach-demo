"""Action / controller interface shared by mock and Isaac-facing code.

Policy or scripted controller outputs ActionCommand; backends execute it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Backend(str, Enum):
    MOCK = "mock"
    ISAAC = "isaac"


@dataclass
class ActionCommand:
    """Normalized action from a policy or scripted controller.

    Values are expected in roughly [-1, 1] before backend scaling/clamping.
    """

    joint_delta: np.ndarray  # shape (dof,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_delta = np.asarray(self.joint_delta, dtype=np.float64)
        if self.joint_delta.ndim != 1:
            raise ValueError(f"joint_delta must be 1-D, got shape {self.joint_delta.shape}")


@dataclass
class Observation:
    """Flat observation vector + named parts for debugging / logging."""

    vector: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    ee_pos: np.ndarray
    target_pos: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


@dataclass
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class ActionInterface(ABC):
    """Backend-agnostic robot action interface."""

    backend: Backend

    @abstractmethod
    def reset(self, seed: int | None = None) -> Observation:
        ...

    @abstractmethod
    def step(self, action: ActionCommand) -> StepResult:
        ...

    @abstractmethod
    def get_observation(self) -> Observation:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    # --- helpers shared by backends ---

    def scale_and_clamp(
        self,
        action: ActionCommand,
        scale: float,
        low: np.ndarray,
        high: np.ndarray,
    ) -> np.ndarray:
        """Map policy output -> joint delta, then clamp to limits.

        This is the kind of safety boundary you also want on real hardware
        before sending MoveJ / joint streams.
        """
        delta = np.clip(action.joint_delta, -1.0, 1.0) * scale
        # Caller applies delta to current joints; here we only bound the delta
        # magnitude already via scale. Absolute joint clamp happens in env.
        _ = low, high  # absolute clamp is env responsibility after integration
        return delta


class ControllerInterface(ABC):
    """Optional higher-level controller that produces ActionCommands.

    Keeps policy / scripted experts swappable without touching the backend.
    """

    @abstractmethod
    def act(self, obs: Observation) -> ActionCommand:
        ...


class ProportionalReachController(ControllerInterface):
    """Numerical Jacobian IK controller for the mock arm (demo only).

    Uses finite-difference J(q) so EE error maps to joint deltas. Good enough
    to show success curves on CPU; Isaac mode uses a trained RL policy instead.
    """

    def __init__(
        self,
        fk_fn=None,
        gain: float = 1.5,
        damping: float = 1e-4,
        noise: float = 0.0,
        eps: float = 1e-4,
    ) -> None:
        self.fk_fn = fk_fn
        self.gain = gain
        self.damping = damping
        self.noise = noise
        self.eps = eps

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        if self.fk_fn is None:
            raise RuntimeError("ProportionalReachController requires fk_fn for Jacobian IK")
        q = np.asarray(q, dtype=np.float64)
        j = np.zeros((3, q.size), dtype=np.float64)
        f0 = self.fk_fn(q)
        for i in range(q.size):
            dq = q.copy()
            dq[i] += self.eps
            j[:, i] = (self.fk_fn(dq) - f0) / self.eps
        return j

    def act(self, obs: Observation) -> ActionCommand:
        err = obs.target_pos - obs.ee_pos
        q = obs.joint_pos
        J = self._jacobian(q)
        # Damped least squares: dq = J^T (J J^T + λ I)^{-1} err
        jj = J @ J.T + self.damping * np.eye(3)
        dq = J.T @ np.linalg.solve(jj, self.gain * err)
        if self.noise > 0:
            dq = dq + self.noise * np.random.randn(*dq.shape)
        # Normalize into roughly [-1, 1] for ActionCommand contract
        joint_delta = np.clip(dq / 0.05, -1.0, 1.0)
        return ActionCommand(joint_delta=joint_delta, metadata={"controller": "jacobian_ik"})


def make_backend(name: str, config: dict[str, Any]) -> ActionInterface:
    """Factory — single switch between mock and isaac."""
    backend = Backend(name.lower())
    if backend is Backend.MOCK:
        from src.mock_reach_env import MockReachEnv

        return MockReachEnv(config)
    if backend is Backend.ISAAC:
        from src.isaac_reach_env import IsaacReachEnv

        return IsaacReachEnv(config)
    raise ValueError(f"Unknown backend: {name}")
