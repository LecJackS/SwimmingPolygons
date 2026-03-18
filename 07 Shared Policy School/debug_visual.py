"""Render scripted V7 shared-school scenarios."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from triangles import SchoolingFishEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scripted V7 shared-school scenarios.")
    parser.add_argument("--scenario", type=str, default="sensor_overlay_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> SchoolingFishEnv:
    return SchoolingFishEnv(
        render_mode=None if args.no_render else "human",
        epsilon=0.0,
        show_sensor_overlay=not args.hide_sensor_overlay,
        focus_agent_id=args.focus_agent_id,
    )


def dense_patch_food(env: SchoolingFishEnv) -> np.ndarray:
    positions = []
    for x in np.linspace(-3.0, 3.0, 8):
        for y in np.linspace(-3.0, 3.0, 6):
            positions.append(np.array([x, y], dtype=np.float32))
            if len(positions) >= env.food_count:
                return np.asarray(positions, dtype=np.float32)
    return np.asarray(positions[: env.food_count], dtype=np.float32)


def edge_food(env: SchoolingFishEnv) -> np.ndarray:
    positions = []
    for idx in range(env.food_count):
        edge = idx % 4
        frac = (idx / max(env.food_count - 1, 1)) * 2.0 - 1.0
        span = env.playable_half_extent
        if edge == 0:
            positions.append(np.array([span, frac * span], dtype=np.float32))
        elif edge == 1:
            positions.append(np.array([-span, frac * span], dtype=np.float32))
        elif edge == 2:
            positions.append(np.array([frac * span, span], dtype=np.float32))
        else:
            positions.append(np.array([frac * span, -span], dtype=np.float32))
    return np.asarray(positions, dtype=np.float32)


def configure_scenario(env: SchoolingFishEnv, scenario: str) -> Callable[[int], dict[str, np.ndarray]]:
    env.reset(seed=0)
    if scenario == "sensor_overlay_demo":
        states = {}
        for idx, agent_id in enumerate(env.get_agent_ids()):
            states[agent_id] = {"position": [2.0 * np.cos(idx), 2.0 * np.sin(idx)], "theta": 0.2 * idx}
        food = dense_patch_food(env)
        env.set_debug_state(agent_states=states, food_positions=food, focus_agent_id=env.focus_agent_id)
        return lambda step_idx: {
            agent_id: np.array([0.4, 0.2 if agent_id == env.focus_agent_id else 0.0], dtype=np.float32)
            for agent_id in env.get_agent_ids()
        }
    if scenario == "dense_patch":
        food = dense_patch_food(env)
        states = {
            agent_id: {"position": [(-2.0 + 0.6 * idx), (-2.0 + 0.3 * idx)], "theta": 0.0}
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
        env.set_debug_state(agent_states=states, food_positions=food, focus_agent_id=env.focus_agent_id)
        return lambda step_idx: {agent_id: np.array([0.6, 0.0], dtype=np.float32) for agent_id in env.get_agent_ids()}
    if scenario == "edge_sweep":
        food = edge_food(env)
        states = {
            agent_id: {"position": [0.0, 0.0], "theta": (2.0 * np.pi * idx) / float(env.num_fish)}
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
        env.set_debug_state(agent_states=states, food_positions=food, focus_agent_id=env.focus_agent_id)
        return lambda step_idx: {
            agent_id: np.array([0.6, 0.3 if idx % 2 == 0 else -0.3], dtype=np.float32)
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
    if scenario == "random_forage":
        env.reset(seed=0)
        return lambda step_idx: {
            agent_id: env.action_space.sample().astype(np.float32) for agent_id in env.get_agent_ids()
        }
    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    env = build_env(args)
    try:
        env.set_focus_agent(args.focus_agent_id)
        action_fn = configure_scenario(env, args.scenario)
        for step_idx in range(args.steps):
            _, rewards, terminateds, truncateds, infos = env.step(action_fn(step_idx))
            if not args.no_render:
                env.render()
            focus_info = infos[args.focus_agent_id]
            print(
                f"step={step_idx:04d} team_reward={rewards[args.focus_agent_id]:+.3f} "
                f"team_food={focus_info['team_food_eaten_episode']} "
                f"focus_visible_food={focus_info['visible_food_count']} "
                f"focus_visible_peers={focus_info['visible_peer_count']}"
            )
            if terminateds["__all__"] or truncateds["__all__"]:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
