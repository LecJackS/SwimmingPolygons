"""Scripted visual runner for V6 continuous foraging scenarios."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np

from triangles import OctopusEnv


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scripted V6 foraging scenarios.")
    parser.add_argument("--scenario", type=str, default="sensor_overlay_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> OctopusEnv:
    env = OctopusEnv(
        render_mode=None if args.no_render else "human",
        epsilon=0.0,
        time_limit=max(args.steps, 200),
        show_sensor_overlay=not args.hide_sensor_overlay,
    )
    env.reset(seed=args.seed)
    return env


def pad_food_positions(env: OctopusEnv, lead_positions: list[list[float]]) -> np.ndarray:
    positions = [np.asarray(position, dtype=np.float32) for position in lead_positions]
    angle_count = max(env.food_count, 16)
    radius = max(env.sensor_radius + 1.0, 8.5)
    for idx in range(angle_count * 4):
        if len(positions) >= env.food_count:
            break
        angle = (2.0 * math.pi * idx) / float(angle_count)
        candidate = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)
        candidate = np.clip(candidate, -env.playable_half_extent, env.playable_half_extent).astype(np.float32)
        if all(float(np.linalg.norm(candidate - existing)) >= 0.6 for existing in positions):
            positions.append(candidate)
    while len(positions) < env.food_count:
        positions.append(np.array([env.playable_half_extent, env.playable_half_extent], dtype=np.float32))
    return np.asarray(positions[: env.food_count], dtype=np.float32)


def configure_scenario(env: OctopusEnv, scenario: str) -> Callable[[int], np.ndarray]:
    if scenario == "sensor_overlay_demo":
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            theta=0.0,
            omega=0.0,
            food_positions=pad_food_positions(
                env,
                [[1.0, 0.0], [0.0, 2.0], [-2.5, 0.0], [1.6, 1.4], [3.7, -0.5], [-3.8, 2.0]],
            ),
            timestep=0,
        )
        return lambda step: np.array([-1.0, 0.0], dtype=np.float32)

    if scenario == "dense_patch":
        cluster = []
        for x in np.linspace(1.0, 2.2, 4):
            for y in np.linspace(-0.9, 0.9, 3):
                cluster.append([x, y])
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            theta=0.0,
            omega=0.0,
            food_positions=pad_food_positions(env, cluster),
            timestep=0,
        )
        return lambda step: np.array([0.6, 0.15], dtype=np.float32)

    if scenario == "edge_sweep":
        ring = []
        for idx in range(12):
            angle = (2.0 * math.pi * idx) / 12.0
            ring.append([4.35 * math.cos(angle), 4.35 * math.sin(angle)])
        env.set_debug_state(
            position=[0.0, 0.0],
            velocity=[0.0, 0.0],
            theta=0.0,
            omega=0.0,
            food_positions=pad_food_positions(env, ring),
            timestep=0,
        )
        return lambda step: np.array([0.25, 0.7 * math.sin(step / 20.0)], dtype=np.float32)

    if scenario == "straight_swim":
        return lambda step: np.array([0.8, 0.0], dtype=np.float32)

    if scenario == "left_turn":
        return lambda step: np.array([0.7, 0.8], dtype=np.float32)

    if scenario == "right_turn":
        return lambda step: np.array([0.7, -0.8], dtype=np.float32)

    if scenario == "random_forage":
        rng = np.random.default_rng(1234)
        return lambda step: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    raise ValueError(f"Unknown scenario: {scenario}")


def main() -> None:
    args = parse_args()
    env = build_env(args)
    try:
        action_fn = configure_scenario(env, args.scenario)
        obs, info = env.reset(seed=args.seed)
        if args.scenario in {"sensor_overlay_demo", "dense_patch", "edge_sweep"}:
            action_fn = configure_scenario(env, args.scenario)
            obs = env.get_debug_snapshot()["observation"]

        print(f"Scenario: {args.scenario}")
        print(f"Render: {not args.no_render}")
        print(f"Sensor overlay: {not args.hide_sensor_overlay}")

        for step in range(1, args.steps + 1):
            action = np.asarray(action_fn(step), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            if step % 25 == 0 or info.get("food_eaten_this_step", 0):
                print(
                    f"step={step:04d} reward={reward:.3f} food_step={info.get('food_eaten_this_step', 0)} "
                    f"food_episode={info.get('food_eaten_episode', 0)} nearest={info.get('nearest_food_distance', float('nan')):.3f} "
                    f"visible={info.get('visible_food_count', 0)}"
                )
            if terminated or truncated:
                print(
                    f"episode_end step={step:04d} terminated={terminated} truncated={truncated} "
                    f"food_episode={info.get('food_eaten_episode', 0)}"
                )
                obs, info = env.reset(seed=args.seed + step)
                if args.scenario in {"sensor_overlay_demo", "dense_patch", "edge_sweep"}:
                    action_fn = configure_scenario(env, args.scenario)
                    obs = env.get_debug_snapshot()["observation"]
    finally:
        env.close()


if __name__ == "__main__":
    main()
