# Swimming Polygons

`Swimming Polygons` is a reinforcement learning project that iterates on a simple swimmer from a 1D hinge-propulsion toy setup to 2D target-seeking, then to more explicit control/physics environments, then to articulated fish bodies, and now to shared-policy schools. The repository is organized as seven chronological experiment stages. The top-level goal is to preserve the learning trajectory and keep each stage runnable.

## Project Evolution

The table below summarizes the current code behavior (source of truth), not just the historical notes in sub-READMEs.

| Version | Folder | Env class | Action space | Observation | Episode objective / done | Reward style |
|---|---|---|---|---|---|---|
| V1 | `01 Unidirectional Triangle` | `SwimmingAgentEnv` | `Box(shape=(1,), low=-0.2, high=0.2)` | 7D vector: position, velocity, orientation, hinge, previous action | Done when `x >= 1.0` or `timestep >= 1000` | Mostly `-1` per step (+ reversal penalty), terminal reward `+1` |
| V2 | `02 Multidirectional Triangle` | `SwimmingAgentEnv` | `Box(shape=(3,), low=-1, high=1)` | 12D vector: 2D position/velocity, orientation, hinge, previous action, food position, countdown | Done when close to food (`dist < 0.02`) or episode timeout (`100` steps) | Time penalty + action-shaping penalties + activity/thrust shaping + terminal adjustment |
| V3 | `03 2D Improved Physics` | `OctopusEnv` | `MultiDiscrete([3, 3])` (turn, push) | Flattened history (`nHist=2`) of normalized 11-feature state, then masked | Done when close to food (`dist < 0.5`) or timeout (`time_limit=100`) | Sparse-like baseline (`-1` per step, `0` at success), with curriculum on target distance |
| V4 | `04 Accelerated Physics` | `OctopusEnv` | `MultiDiscrete([3, 3])` (turn, push) | Flat normalized 11D vector with relative-target encoding | Done on success (`dist < 0.5`) or timeout (`100` steps), with staged curriculum target distance | Step penalty + bounded progress shaping + terminal success bonus |
| V5 | `05 Validated Physics` | `OctopusEnv` | `MultiDiscrete([3, 3])` (turn, push) | Flat normalized 11D vector with relative-target encoding | Done on success (`dist < 0.5`) or timeout, with staged curriculum inherited from V4 | Same current reward model as V4, plus expanded simulator validation and scripted visual debugging |
| V6 | `06 Articulated Fish` | `ArticulatedFishEnv` (`OctopusEnv` alias kept for script compatibility) | `Box(shape=(2,), low=-1, high=1)` drive + steer | 46D observation: 36-bin local polar food sensor + 10 proprioception features | Done on timeout only (`time_limit=600` by default) in a continuous multi-pellet foraging field | `+1` per pellet eaten and a small per-step time cost |
| V7 | `07 Shared Policy School` | `SchoolingFishEnv` | Per-fish `Box(shape=(2,), low=-1, high=1)` drive + steer | Per-fish 82D observation: 36 food bins + 36 peer bins + 10 proprioception features | Shared-world team foraging, timeout only, one shared pellet field, no fish-fish collisions in milestone 1 | Shared team reward: pellets eaten by the school minus a small step cost, broadcast to every fish |

## Repository Map

Each version folder is mostly self-contained:

- `triangles.py`: custom Gym environment dynamics + rendering.
- `agent.py`: PPO training script.
- `test_model.py`: load saved checkpoint and run policy in the environment.

Version-specific extras:

- `03 2D Improved Physics/plot_statistics.py`: helper to inspect `historical_data.csv` distributions if that file is generated.
- `04 Accelerated Physics/media/`: placeholder directory for future V4 visual artifacts.
- `05 Validated Physics/debug_env.py`: env-only validation harness with baseline, physics audit, and control contract probe sets.
- `05 Validated Physics/debug_visual.py`: scripted simulator-only visual debugging entrypoint.
- `06 Articulated Fish/debug_env.py`: articulated foraging env validation harness with locomotion, food-field, sensor, and reward-contract probes.
- `06 Articulated Fish/debug_visual.py`: scripted articulated-fish foraging viewer with sensor overlay demos.
- `07 Shared Policy School/debug_env.py`: multi-fish shared-world validation harness for food-field, peer-sensor, shared-reward, and simultaneity checks.
- `07 Shared Policy School/debug_visual.py`: scripted shared-school viewer with one focus fish sensor overlay.

## Reproducibility

This project is older and uses the legacy Gym API (`obs, reward, done, info`). Prefer a virtual environment and install dependencies first.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib gym stable-baselines3 torch imageio pandas keyboard click
```

### Run Version 1

```bash
cd "01 Unidirectional Triangle"
python agent.py        # train PPO and save ppo_swimming_agent.zip
python test_model.py   # load model and render rollout (optionally save GIF)
```

### Run Version 2

```bash
cd "02 Multidirectional Triangle"
python agent.py
python test_model.py
```

### Run Version 3

```bash
cd "03 2D Improved Physics"
python agent.py
python test_model.py
```

### Run Version 4

```bash
cd "04 Accelerated Physics"
python agent.py
python test_model.py
```

### Run Version 5

```bash
cd "05 Validated Physics"
python agent.py
python test_model.py
python debug_env.py --probe-set baseline
python debug_visual.py --scenario forward
```

V5 training also writes deterministic checkpoint-time evaluation reports under its checkpoint root:

- `eval_reports.jsonl`
- `eval_reports.csv`
- `run_summary.json`

You can score a checkpoint or latest run headlessly with:

```bash
cd "05 Validated Physics"
python test_model.py --checkpoint-root ./rllib_checkpoints --episodes 10 --distances 5.7,10.0 --no-render --summary-json ./rllib_checkpoints/eval_latest.json
```

### Run Version 6

```bash
cd "06 Articulated Fish"
python agent.py
python test_model.py
python debug_env.py --probe-set baseline
python debug_visual.py --scenario sensor_overlay_demo
```

V6 keeps the RLlib/reporting stack, but now uses an articulated 3-segment fish in a continuous-foraging task with a local polar food sensor and renderable sensor overlay.

### Run Version 7

```bash
cd "07 Shared Policy School"
python agent.py
python test_model.py
python debug_env.py --probe-set baseline
python debug_visual.py --scenario sensor_overlay_demo
```

V7 extends V6 into an 8-fish shared world with one shared trainable policy, a shared team reward, a local peer sensor, and a focus-fish render overlay.

Training scripts are long-running by default (`100k+` timesteps in V1/V2 and `1,000,000+` in V3).

## Known Caveats / Current State

- Early versions use old Gym-style signatures; V4, V5, and V6 are Gymnasium-native.
- Reward shaping is still experimental in V2/V3, with multiple commented alternatives in code.
- `03 2D Improved Physics/test_model.py` has a `save_animation` toggle path that calls `env.save_animation_file(...)`, but `OctopusEnv` does not currently implement that method.
- V5 is the frozen rigid-body validation baseline; V6 is the frozen articulated single-fish foraging baseline; V7 is the active shared-policy schooling branch.
- Subfolder READMEs are useful historical context, but the code is the source of truth when they differ.

## Artifacts

Available in-repo media/checkpoints:

- V1 GIF: [`01 Unidirectional Triangle/media/swimming_agent.gif`](01%20Unidirectional%20Triangle/media/swimming_agent.gif)
- V2 GIF: [`02 Multidirectional Triangle/media/swimming_agent.gif`](02%20Multidirectional%20Triangle/media/swimming_agent.gif)
- Saved PPO checkpoints: `01 Unidirectional Triangle/ppo_swimming_agent.zip`, `02 Multidirectional Triangle/ppo_swimming_agent.zip`, `03 2D Improved Physics/ppo_swimming_agent.zip`
