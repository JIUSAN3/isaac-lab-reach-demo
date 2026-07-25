#!/usr/bin/env python3
"""Run the CPU mock Reach demo end-to-end.

Examples (from repo root):
  python scripts/run_mock_reach.py
  python scripts/run_mock_reach.py --episodes 5 --seed 0
  python scripts/run_mock_reach.py --controller random --episodes 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow `python scripts/run_mock_reach.py` from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.action_interface import (  # noqa: E402
    ActionCommand,
    ProportionalReachController,
    make_backend,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(env, controller, max_steps: int, seed: int | None = None) -> dict:
    obs = env.reset(seed=seed)
    distances = []
    rewards = []
    success = False
    steps = 0

    for t in range(max_steps):
        steps = t + 1
        if controller == "random":
            action = ActionCommand(joint_delta=np.random.uniform(-1, 1, size=env.dof))
        else:
            action = controller.act(obs)

        result = env.step(action)
        obs = result.observation
        distances.append(float(result.info["distance"]))
        rewards.append(float(result.reward))
        if result.terminated:
            success = True
            break
        if result.truncated:
            break

    return {
        "steps": steps,
        "success": success,
        "final_distance": distances[-1] if distances else None,
        "return": float(np.sum(rewards)) if rewards else 0.0,
        "distances": distances,
        "rewards": rewards,
    }


def save_plot(all_distances: list[list[float]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed — skip plot. pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, d in enumerate(all_distances):
        ax.plot(d, label=f"ep{i}", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("EE–target distance (m)")
    ax.set_title("Mock Franka Reach — distance over episode")
    ax.grid(True, alpha=0.3)
    if len(all_distances) <= 8:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU mock Isaac-style Reach demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "reach_config.yaml",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controller",
        choices=["p_reach", "random"],
        default="p_reach",
        help="p_reach = heuristic controller; random = baseline",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "mock_reach_metrics.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_backend("mock", config)
    controller: object
    if args.controller == "p_reach":
        # Bind mock FK so the Jacobian controller can invert EE error -> joints
        controller = ProportionalReachController(fk_fn=env.fk, gain=1.8, noise=0.0)
    else:
        controller = "random"

    np.random.seed(args.seed)
    summaries = []
    all_distances = []
    print(f"backend=mock  task={config['task']['name']}  dof={env.dof}")
    print(f"controller={args.controller}  episodes={args.episodes}  seed={args.seed}")
    print("-" * 60)

    for ep in range(args.episodes):
        summary = run_episode(
            env,
            controller,
            config["env"]["episode_length"],
            seed=args.seed + ep,
        )
        summaries.append(
            {
                "episode": ep,
                "steps": summary["steps"],
                "success": summary["success"],
                "final_distance": summary["final_distance"],
                "return": summary["return"],
            }
        )
        all_distances.append(summary["distances"])
        flag = "OK" if summary["success"] else "--"
        print(
            f"ep{ep:02d}  steps={summary['steps']:3d}  "
            f"dist={summary['final_distance']:.4f}  "
            f"R={summary['return']:+7.3f}  success={flag}"
        )

    env.close()

    success_rate = float(np.mean([s["success"] for s in summaries]))
    mean_return = float(np.mean([s["return"] for s in summaries]))
    mean_dist = float(np.mean([s["final_distance"] for s in summaries]))

    payload = {
        "backend": "mock",
        "isaac_task_id": config["task"]["isaac_task_id"],
        "controller": args.controller,
        "seed": args.seed,
        "episodes": args.episodes,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_final_distance": mean_dist,
        "episodes_detail": summaries,
        "notes": (
            "Mock kinematics on CPU. Replace backend with Isaac Lab "
            f"{config['task']['isaac_task_id']} on a NVIDIA GPU machine."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 60)
    print(f"success_rate={success_rate:.0%}  mean_return={mean_return:.3f}  "
          f"mean_final_dist={mean_dist:.4f}")
    print(f"[ok] wrote {args.out}")

    plot_path = args.out.with_suffix(".png")
    save_plot(all_distances, plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
