# V6 Continuous Foraging

`06 Articulated Fish` is now a continuous-foraging articulated fish environment.

This V6 no longer uses a single target, staged curriculum, or distance-based early stop. The fish swims in a full map with many pellets, receives `+1` for each pellet eaten, pays a small per-step cost, and observes food through a fish-local polar sensor that is also rendered as an overlay with an on-canvas legend.

Old V6 checkpoints from the target-seeking task are obsolete for this version and should not be reused.

## Environment Contract

- Env class: `ArticulatedFishEnv` with `OctopusEnv` kept as a compatibility alias.
- Action space: `Box(low=-1, high=1, shape=(2,))`
  - `action[0]`: drive
  - `action[1]`: steer
- Observation space: `46D`
  - `36` local food-sensor bins: `3` rings x `12` sectors
  - `10` proprioception features:
    - root forward velocity
    - root lateral velocity
    - root angular velocity
    - joint 0 angle
    - joint 1 angle
    - joint 0 velocity
    - joint 1 velocity
    - previous action 0
    - previous action 1
    - episode progress
- Episode objective:
  - forage continuously until `time_limit`
  - eating food does not end the episode
- Reward:
  - `+1.0 * pellets_eaten_this_step`
  - `-0.002` per step by default

Default task parameters:

- `food_count = 48`
- `food_capture_radius = 0.45`
- `time_limit = 600`
- `sensor_radius = 4.5`
- `sensor_ring_edges = [1.5, 3.0, 4.5]`
- `sensor_num_sectors = 12`

## Run Training

```powershell
cd "c:\Users\adminlcarreira\SwimmingPolygons"
.\.venv\Scripts\Activate.ps1
cd "06 Articulated Fish"

python agent.py `
  --device cuda `
  --train-iterations 100 `
  --checkpoint-every-iterations 5 `
  --checkpoint-root .\rllib_checkpoints_baseline_v6_foraging
```

Useful task flags:

- `--food-count`
- `--time-limit`
- `--pellet-reward`
- `--step-cost`
- `--sensor-radius`
- `--sensor-ring-edges`
- `--sensor-sectors`
- `--eval-report-episodes`

Checkpoint-time reports are written under the checkpoint root:

- `eval_reports.jsonl`
- `eval_reports.csv`
- `run_summary.json`

Best checkpoint selection now uses:

1. highest `mean_food_eaten`
2. tie-break by `food_per_100_steps`
3. tie-break by `mean_reward`

## Evaluate A Checkpoint

Visual rollout:

```powershell
python test_model.py --checkpoint-root .\rllib_checkpoints_baseline_v6_foraging
```

Headless batch scoring:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v6_foraging `
  --episodes 10 `
  --no-render `
  --summary-json .\rllib_checkpoints_baseline_v6_foraging\eval_latest.json `
  --summary-csv .\rllib_checkpoints_baseline_v6_foraging\eval_latest.csv
```

Optional toggles:

- `--hide-sensor-overlay`
- `--food-count`
- `--time-limit`
- `--sensor-radius`
- `--sensor-ring-edges`
- `--sensor-sectors`

## Visual Debugging

`debug_visual.py` runs scripted simulator-only scenarios:

- `sensor_overlay_demo`
- `dense_patch`
- `edge_sweep`
- `straight_swim`
- `left_turn`
- `right_turn`
- `random_forage`

Example:

```powershell
python debug_visual.py --scenario sensor_overlay_demo
python debug_visual.py --scenario dense_patch
```

The polar food sensor overlay is enabled by default in human render mode. The render now includes a fixed legend panel that explains:

- ring ranges
- the `12` sector resolution
- the forward-facing sector orientation
- the cyan intensity ramp for `0`, `1`, `2`, and `3+` pellets

Use `--hide-sensor-overlay` to disable both the wedges and the legend.

## Environment Validation

`debug_env.py` now validates the continuous-foraging contract.

Available probes:

- `joint_rest_decay`
- `propulsion_grid`
- `mirror_turn`
- `steering_authority`
- `drag_anisotropy`
- `dt_sensitivity`
- `food_field_contract`
- `polar_sensor_contract`
- `foraging_reward_contract`
- `baseline` aggregates all of them

Smoke run:

```powershell
python debug_env.py --probe-set baseline --seed 0 --no-plots
```

## Delayed Watcher

The watcher now defaults to the foraging run names:

- PID file: `baseline_v6_foraging.pid`
- log file: `baseline_v6_foraging.out.log`
- checkpoint root: `rllib_checkpoints_baseline_v6_foraging`

Start it with:

```powershell
python watch_training.py
```

What it writes under `<checkpoint_root>\auto_eval\`:

- scheduled check snapshots like `check_90m.json`
- `watcher.log`
- batch evaluation summaries for the best checkpoint and final checkpoint
- `evaluation_manifest.json`
- `visual_eval_command.txt`

If training is already finished, it evaluates immediately. If it is still running at both scheduled checks, it exits after writing the second snapshot.

## Notes

- The world border is visual only in this pass.
- The observation is intentionally local; global pellet coordinates are not exposed to the policy.
- Multi-fish behavior is still out of scope for this version.
