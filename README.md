# Swimming Polygons

`Swimming Polygons` is a reinforcement learning project that iterates on a simple swimmer from a 1D hinge-propulsion toy setup to 2D target-seeking and then to a more explicit control/physics environment. The repository is organized as four chronological experiment stages. The top-level goal is to preserve the learning trajectory and keep each stage runnable.

## Project Evolution

The table below summarizes the current code behavior (source of truth), not just the historical notes in sub-READMEs.

| Version | Folder | Env class | Action space | Observation | Episode objective / done | Reward style |
|---|---|---|---|---|---|---|
| V1 | `01 Unidirectional Triangle` | `SwimmingAgentEnv` | `Box(shape=(1,), low=-0.2, high=0.2)` | 7D vector: position, velocity, orientation, hinge, previous action | Done when `x >= 1.0` or `timestep >= 1000` | Mostly `-1` per step (+ reversal penalty), terminal reward `+1` |
| V2 | `02 Multidirectional Triangle` | `SwimmingAgentEnv` | `Box(shape=(3,), low=-1, high=1)` | 12D vector: 2D position/velocity, orientation, hinge, previous action, food position, countdown | Done when close to food (`dist < 0.02`) or episode timeout (`100` steps) | Time penalty + action-shaping penalties + activity/thrust shaping + terminal adjustment |
| V3 | `03 2D Improved Physics` | `OctopusEnv` | `MultiDiscrete([3, 3])` (turn, push) | Flattened history (`nHist=2`) of normalized 11-feature state, then masked | Done when close to food (`dist < 0.5`) or timeout (`time_limit=100`) | Sparse-like baseline (`-1` per step, `0` at success), with curriculum on target distance |
| V4 | `04 Accelerated Physics` | `OctopusEnv` | `MultiDiscrete([3, 3])` (turn, push) | Flat normalized 11D vector with relative-target encoding | Done on success (`dist < 0.5`) or timeout (`100` steps), with curriculum target distance | V3-style baseline (`-1` per step, `0` at success) |

## Repository Map

Each version folder is mostly self-contained:

- `triangles.py`: custom Gym environment dynamics + rendering.
- `agent.py`: PPO training script.
- `test_model.py`: load saved checkpoint and run policy in the environment.

Version-specific extras:

- `03 2D Improved Physics/plot_statistics.py`: helper to inspect `historical_data.csv` distributions if that file is generated.
- `04 Accelerated Physics/media/`: placeholder directory for future V4 visual artifacts.

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

Training scripts are long-running by default (`100k+` timesteps in V1/V2 and `1,000,000+` in V3).

## Known Caveats / Current State

- The code is written against old Gym-style signatures, so modern `gymnasium`-first stacks may require adaptation.
- Reward shaping is still experimental in V2/V3, with multiple commented alternatives in code.
- `03 2D Improved Physics/test_model.py` has a `save_animation` toggle path that calls `env.save_animation_file(...)`, but `OctopusEnv` does not currently implement that method.
- Subfolder READMEs are useful historical context, but the code is the source of truth when they differ.

## Artifacts

Available in-repo media/checkpoints:

- V1 GIF: [`01 Unidirectional Triangle/media/swimming_agent.gif`](01%20Unidirectional%20Triangle/media/swimming_agent.gif)
- V2 GIF: [`02 Multidirectional Triangle/media/swimming_agent.gif`](02%20Multidirectional%20Triangle/media/swimming_agent.gif)
- Saved PPO checkpoints: `01 Unidirectional Triangle/ppo_swimming_agent.zip`, `02 Multidirectional Triangle/ppo_swimming_agent.zip`, `03 2D Improved Physics/ppo_swimming_agent.zip`
