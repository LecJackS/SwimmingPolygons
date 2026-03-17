"""Render scripted V5 simulator scenarios without loading a policy."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from triangles import OctopusEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scripted V5 simulator debug scenarios.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="forward",
        choices=(
            "neutral",
            "forward",
            "turn_left",
            "turn_right",
            "toward_target",
            "away_from_target",
            "orbit_failure",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def scenario_spec(name: str, steps: int) -> dict[str, object]:
    if name == "neutral":
        return {
            "title": "Neutral drift check",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": 0.0,
                "omega": 0.0,
                "food_position": [5.0, 0.0],
                "timestep": 0,
            },
            "actions": [np.array([1, 1], dtype=np.int64) for _ in range(steps)],
        }
    if name == "forward":
        return {
            "title": "Forward thrust",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": 0.0,
                "omega": 0.0,
                "food_position": [5.0, 0.0],
                "timestep": 0,
            },
            "actions": [np.array([1, 2], dtype=np.int64) for _ in range(steps)],
        }
    if name == "turn_left":
        return {
            "title": "Left-turn thrust arc",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": 0.0,
                "omega": 0.0,
                "food_position": [0.0, 5.0],
                "timestep": 0,
            },
            "actions": [np.array([2, 2], dtype=np.int64) for _ in range(steps)],
        }
    if name == "turn_right":
        return {
            "title": "Right-turn thrust arc",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": 0.0,
                "omega": 0.0,
                "food_position": [0.0, -5.0],
                "timestep": 0,
            },
            "actions": [np.array([0, 2], dtype=np.int64) for _ in range(steps)],
        }
    if name == "toward_target":
        return {
            "title": "Toward target",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": 0.0,
                "omega": 0.0,
                "food_position": [5.0, 0.0],
                "timestep": 0,
            },
            "actions": [np.array([1, 2], dtype=np.int64) for _ in range(steps)],
        }
    if name == "away_from_target":
        return {
            "title": "Away from target",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": math.pi,
                "omega": 0.0,
                "food_position": [5.0, 0.0],
                "timestep": 0,
            },
            "actions": [np.array([1, 2], dtype=np.int64) for _ in range(steps)],
        }
    if name == "orbit_failure":
        return {
            "title": "Orbit-like failure mode",
            "state": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "theta": math.pi / 2.0,
                "omega": 0.0,
                "food_position": [2.5, 0.0],
                "timestep": 0,
            },
            "actions": [np.array([2, 2], dtype=np.int64) for _ in range(steps)],
        }
    raise ValueError(f"Unsupported scenario: {name}")


def main() -> None:
    args = parse_args()
    spec = scenario_spec(args.scenario, args.steps)
    render_mode = None if args.no_render else "human"

    env = OctopusEnv(
        epsilon=0.0,
        render_mode=render_mode,
        enable_curriculum=False,
        fixed_food_distance=10.0,
        time_limit=max(args.steps, 100),
    )
    env.reset(seed=args.seed)
    env.set_debug_state(
        position=spec["state"]["position"],
        velocity=spec["state"]["velocity"],
        theta=spec["state"]["theta"],
        omega=spec["state"]["omega"],
        food_position=spec["state"]["food_position"],
        timestep=spec["state"]["timestep"],
    )

    print("V5 scripted visual debug")
    print(f"Scenario: {args.scenario}")
    print(f"Description: {spec['title']}")
    print(f"Render: {not args.no_render}")

    try:
        for step_idx, action in enumerate(spec["actions"], start=1):
            obs, reward, terminated, truncated, info = env.step(action)
            snapshot = env.get_debug_snapshot()
            if not args.no_render:
                env.render()
            if args.log_every > 0 and (step_idx == 1 or step_idx % args.log_every == 0):
                dynamics = snapshot["dynamics_breakdown"]
                print(
                    f"step={step_idx:04d} action={action.tolist()} reward={reward:.3f} "
                    f"dist={info['distance_to_food']:.3f} pos={np.round(snapshot['position'], 3)} "
                    f"vel={np.round(snapshot['velocity'], 3)} theta={snapshot['theta']:.3f} "
                    f"thrust={np.round(dynamics['thrust_vector'], 3)} accel={np.round(dynamics['acceleration'], 3)}"
                )
            if terminated or truncated:
                print(
                    f"episode_end step={step_idx:04d} terminated={terminated} truncated={truncated} "
                    f"dist={info['distance_to_food']:.3f}"
                )
                break
        if not args.no_render:
            time.sleep(0.5)
    finally:
        env.close()


if __name__ == "__main__":
    main()
