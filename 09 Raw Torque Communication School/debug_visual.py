"""Render scripted V9 raw-torque communication scenarios."""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from triangles import CommunicatingSchoolEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scripted V9 raw-torque communication scenarios.")
    parser.add_argument("--scenario", type=str, default="sensor_overlay_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--render-profile", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--render-engine", type=str, choices=["auto", "blit", "safe"], default="auto")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> CommunicatingSchoolEnv:
    return CommunicatingSchoolEnv(
        render_mode=None if args.no_render else "human",
        render_profile=args.render_profile,
        render_engine=args.render_engine,
        epsilon=0.0,
        show_sensor_overlay=not args.hide_sensor_overlay,
        focus_agent_id=args.focus_agent_id,
    )


def sampled_food_field(env: CommunicatingSchoolEnv, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    return env.food_positions.astype(np.float32).copy(), env.food_team_indices.astype(np.int64).copy()


def configure_scenario(
    env: CommunicatingSchoolEnv,
    scenario: str,
    *,
    seed: int,
) -> Callable[[int], dict[str, dict[str, np.ndarray | int]]]:
    food_positions, food_teams = sampled_food_field(env, seed=seed)
    if scenario == "sensor_overlay_demo":
        states = {}
        for idx, agent_id in enumerate(env.get_agent_ids()):
            angle = (2.0 * np.pi * idx) / float(env.num_fish)
            states[agent_id] = {"position": [2.5 * np.cos(angle), 2.5 * np.sin(angle)], "theta": 0.2 * idx}
        env.set_debug_state(
            agent_states=states,
            food_positions=food_positions,
            food_team_indices=food_teams,
            focus_agent_id=env.focus_agent_id,
        )
        return lambda step_idx: {
            agent_id: {
                "motion": np.array(
                    [
                        0.7 * np.sin(0.18 * step_idx),
                        0.7 * np.sin((0.18 * step_idx) - (0.5 * np.pi)),
                    ],
                    dtype=np.float32,
                ),
                "message": (step_idx + idx) % 4,
            }
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
    if scenario == "message_demo":
        states = {agent_id: {"position": [8.0, 8.0], "theta": 0.0} for agent_id in env.get_agent_ids()}
        states["fish_0"] = {"position": [0.0, 0.0], "theta": 0.0}
        states["fish_1"] = {"position": [2.0, 0.0], "theta": 0.0}
        states["fish_5"] = {"position": [0.0, 2.0], "theta": 0.0}
        env.set_debug_state(
            agent_states=states,
            food_positions=food_positions,
            food_team_indices=food_teams,
            focus_agent_id=env.focus_agent_id,
        )
        return lambda step_idx: {
            agent_id: {
                "motion": np.zeros(2, dtype=np.float32),
                "message": (
                    3 if agent_id == "fish_1" else
                    2 if agent_id == "fish_5" else
                    0
                ),
            }
            for agent_id in env.get_agent_ids()
        }
    if scenario == "color_capture_demo":
        states = {
            agent_id: {
                "position": [(-4.0 + 0.8 * idx), 0.0 if idx < env.num_red_fish else 1.5],
                "theta": 0.0 if idx < env.num_red_fish else np.pi,
            }
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
        env.set_debug_state(
            agent_states=states,
            food_positions=food_positions,
            food_team_indices=food_teams,
            focus_agent_id=env.focus_agent_id,
        )
        return lambda step_idx: {
            agent_id: {
                "motion": np.array(
                    [
                        0.8 * np.sin(0.22 * step_idx),
                        0.8 * np.sin((0.22 * step_idx) - (0.5 * np.pi)),
                    ],
                    dtype=np.float32,
                ),
                "message": int(step_idx % 4),
            }
            for agent_id in env.get_agent_ids()
        }
    if scenario == "random_forage":
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        return lambda step_idx: {
            agent_id: {"motion": rng.uniform(-1.0, 1.0, size=2).astype(np.float32), "message": int(rng.integers(0, 4))}
            for agent_id in env.get_agent_ids()
        }
    if scenario == "blit_boundary_stress":
        states = {}
        for idx, agent_id in enumerate(env.get_agent_ids()):
            if idx == 0:
                states[agent_id] = {"position": [-8.5, 0.0], "theta": 0.0}
            elif idx == 1:
                states[agent_id] = {"position": [8.5, 0.0], "theta": np.pi}
            elif idx == 2:
                states[agent_id] = {"position": [0.0, -8.5], "theta": 0.5 * np.pi}
            elif idx == 3:
                states[agent_id] = {"position": [0.0, 8.5], "theta": -0.5 * np.pi}
            else:
                angle = (2.0 * np.pi * idx) / float(env.num_fish)
                states[agent_id] = {"position": [6.5 * np.cos(angle), 6.5 * np.sin(angle)], "theta": angle + np.pi}
        env.set_debug_state(
            agent_states=states,
            food_positions=food_positions,
            food_team_indices=food_teams,
            focus_agent_id=env.focus_agent_id,
        )
        return lambda step_idx: {
            agent_id: {
                "motion": np.array(
                    [
                        0.75 * np.sin((0.10 * step_idx) + (0.40 * idx)),
                        0.75 * np.sin((0.10 * step_idx) + (0.40 * idx) - (0.5 * np.pi)),
                    ],
                    dtype=np.float32,
                ),
                "message": int((step_idx + idx) % 4),
            }
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    env = build_env(args)
    try:
        env.set_focus_agent(args.focus_agent_id)
        action_fn = configure_scenario(env, args.scenario, seed=args.seed)
        for step_idx in range(args.steps):
            _, rewards, terminateds, truncateds, infos = env.step(action_fn(step_idx))
            if not args.no_render:
                env.render()
            focus_info = infos[args.focus_agent_id]
            print(
                f"step={step_idx:04d} reward={rewards[args.focus_agent_id]:+.3f} "
                f"food_episode={focus_info['food_eaten_episode']} "
                f"red_food={focus_info['food_eaten_episode_red']} blue_food={focus_info['food_eaten_episode_blue']} "
                f"visible_food={focus_info['visible_food_count']} "
                f"visible_teammates={focus_info['visible_teammate_count']} "
                f"visible_opponents={focus_info['visible_opponent_count']} "
                f"msg={focus_info['emitted_message_token']}"
            )
            if terminateds["__all__"] or truncateds["__all__"]:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
