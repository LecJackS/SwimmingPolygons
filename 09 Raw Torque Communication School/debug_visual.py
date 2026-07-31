"""Render scripted V9 muscle-activation communication scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from triangles import CommunicatingSchoolEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render scripted V9 muscle-activation communication scenarios.")
    parser.add_argument("--scenario", type=str, default="sensor_overlay_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--focus-agent-id", type=str, default="fish_0")
    parser.add_argument("--hide-sensor-overlay", action="store_true")
    parser.add_argument("--render-profile", type=str, choices=["fast", "full"], default="fast")
    parser.add_argument("--render-engine", type=str, choices=["auto", "blit", "safe"], default="auto")
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--gif-seconds", type=float, default=6.0)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def resolve_live_render_profile(requested_profile: str, *, save_gif: bool, no_render: bool) -> str:
    if requested_profile == "full" and not save_gif and not no_render:
        print("warning: live full mode is unreliable on Windows; using fast for live view.")
        print("warning: use --render-profile full together with --save-gif for diagnostic capture.")
        return "fast"
    return requested_profile


def build_env(args: argparse.Namespace, *, render_profile: str) -> CommunicatingSchoolEnv:
    if args.scenario == "scripted_wave_demo":
        return CommunicatingSchoolEnv(
            render_mode=None if args.no_render and not args.save_gif else "human",
            render_profile=render_profile,
            render_engine=args.render_engine,
            epsilon=0.0,
            reward_mode="locomotion_debug",
            num_red_fish=1,
            num_blue_fish=0,
            num_red_pellets=0,
            num_blue_pellets=0,
            show_sensor_overlay=not args.hide_sensor_overlay,
            focus_agent_id="fish_0",
        )
    return CommunicatingSchoolEnv(
        render_mode=None if args.no_render and not args.save_gif else "human",
        render_profile=render_profile,
        render_engine=args.render_engine,
        epsilon=0.0,
        show_sensor_overlay=not args.hide_sensor_overlay,
        focus_agent_id=args.focus_agent_id,
    )


def with_motion_dim(env: CommunicatingSchoolEnv, values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == env.num_joints:
        return array.astype(np.float32, copy=True)
    if array.size == 2 and env.num_joints > 0:
        motion = np.zeros(env.num_joints, dtype=np.float32)
        motion[: min(2, env.num_joints)] = array[: min(2, env.num_joints)]
        return motion
    raise ValueError(f"Expected {env.num_joints} motion values, got {array.size}.")


def traveling_wave_motion(
    env: CommunicatingSchoolEnv,
    step_idx: int,
    *,
    amplitude: float,
    phase_rate: float,
    phase_delta: float = 0.5 * np.pi,
    phase_offset: float = 0.0,
) -> np.ndarray:
    phase = (phase_rate * float(step_idx)) + float(phase_offset)
    offsets = np.arange(env.num_joints, dtype=np.float32) * float(phase_delta)
    motion = amplitude * np.sin(phase - offsets)
    return np.clip(np.asarray(motion, dtype=np.float32), -1.0, 1.0)


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")
    if args.gif_seconds <= 0.0:
        raise ValueError("--gif-seconds must be > 0.")
    if args.gif_fps <= 0:
        raise ValueError("--gif-fps must be > 0.")


def capture_frame_rgb(env: CommunicatingSchoolEnv) -> np.ndarray:
    if env.fig is None:
        raise RuntimeError("Render figure is not initialized; cannot capture frame.")
    frame = np.asarray(env.fig.canvas.buffer_rgba(), dtype=np.uint8)
    return frame[:, :, :3].copy()


def pump_live_window(seconds: float = 0.01) -> None:
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return
    plt.pause(seconds)


def write_gif(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write GIF with zero frames.")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", duration=(1.0 / float(fps)))


def sampled_food_field(env: CommunicatingSchoolEnv, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=seed)
    return env.food_positions.astype(np.float32).copy(), env.food_team_indices.astype(np.int64).copy()


def configure_scenario(
    env: CommunicatingSchoolEnv,
    scenario: str,
    *,
    seed: int,
) -> Callable[[int], dict[str, dict[str, np.ndarray | int]]]:
    if scenario == "scripted_wave_demo":
        env.set_debug_state(
            agent_states={"fish_0": {"position": [0.0, 0.0], "theta": 0.0}},
            food_positions=np.zeros((0, 2), dtype=np.float32),
            food_team_indices=np.zeros(0, dtype=np.int64),
            focus_agent_id="fish_0",
        )
        return lambda step_idx: {
            "fish_0": {
                "motion": traveling_wave_motion(env, step_idx, amplitude=0.95, phase_rate=0.34),
                "message": 0,
            }
        }

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
                "motion": traveling_wave_motion(env, step_idx, amplitude=0.7, phase_rate=0.18),
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
                "motion": np.zeros(env.num_joints, dtype=np.float32),
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
                "motion": traveling_wave_motion(env, step_idx, amplitude=0.8, phase_rate=0.22),
                "message": int(step_idx % 4),
            }
            for agent_id in env.get_agent_ids()
        }
    if scenario == "random_forage":
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        return lambda step_idx: {
            agent_id: {
                "motion": rng.uniform(-1.0, 1.0, size=env.num_joints).astype(np.float32),
                "message": int(rng.integers(0, 4)),
            }
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
                "motion": traveling_wave_motion(
                    env,
                    step_idx,
                    amplitude=0.75,
                    phase_rate=0.10,
                    phase_offset=(0.40 * idx),
                ),
                "message": int((step_idx + idx) % 4),
            }
            for idx, agent_id in enumerate(env.get_agent_ids())
        }
    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    gif_output_path = Path(args.save_gif).resolve() if args.save_gif else None
    if gif_output_path is not None:
        plt.switch_backend("Agg")
    render_profile = resolve_live_render_profile(
        args.render_profile,
        save_gif=gif_output_path is not None,
        no_render=args.no_render,
    )
    env = build_env(args, render_profile=render_profile)
    focus_agent_id = "fish_0" if args.scenario == "scripted_wave_demo" else args.focus_agent_id
    try:
        env.set_focus_agent(focus_agent_id)
        action_fn = configure_scenario(env, args.scenario, seed=args.seed)
        gif_frames: list[np.ndarray] = []
        gif_frame_limit = int(round(args.gif_seconds * float(args.gif_fps))) if gif_output_path is not None else 0
        if env.render_mode == "human":
            env.render()
            if gif_output_path is None:
                pump_live_window()
            elif len(gif_frames) < gif_frame_limit:
                gif_frames.append(capture_frame_rgb(env))
        for step_idx in range(args.steps):
            _, rewards, terminateds, truncateds, infos = env.step(action_fn(step_idx))
            if env.render_mode == "human":
                env.render()
                if gif_output_path is not None and len(gif_frames) < gif_frame_limit:
                    gif_frames.append(capture_frame_rgb(env))
                elif gif_output_path is None:
                    pump_live_window()
            focus_info = infos[focus_agent_id]
            print(
                f"step={step_idx:04d} reward={rewards[focus_agent_id]:+.3f} "
                f"food_episode={focus_info['food_eaten_episode']} "
                f"red_food={focus_info['food_eaten_episode_red']} blue_food={focus_info['food_eaten_episode_blue']} "
                f"visible_food={focus_info['visible_food_count']} "
                f"visible_teammates={focus_info['visible_teammate_count']} "
                f"visible_opponents={focus_info['visible_opponent_count']} "
                f"msg={focus_info['emitted_message_token']}"
            )
            if terminateds["__all__"] or truncateds["__all__"] or (gif_output_path is not None and len(gif_frames) >= gif_frame_limit):
                break
        if gif_output_path is not None:
            write_gif(gif_output_path, gif_frames, fps=args.gif_fps)
            print(f"saved_gif={gif_output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
