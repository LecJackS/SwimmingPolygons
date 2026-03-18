# V7 Shared Policy School

`07 Shared Policy School` is the first multi-fish stage.

V7 keeps the articulated 3-segment fish body from V6, but moves to a shared-world multi-agent task:

- `8` fish by default
- one shared pellet field
- one shared trainable policy for all `fish_i`
- one shared team reward
- no fish-fish collisions in this first milestone

V6 remains the frozen single-fish articulated foraging baseline. V7 is the active schooling branch.

## Environment Contract

- Env class: `SchoolingFishEnv`
- Action space per fish: `Box(low=-1, high=1, shape=(2,))`
  - `action[0]`: drive
  - `action[1]`: steer
- Observation space per fish: `82D`
  - `36` local food bins: `3` rings x `12` sectors
  - `36` local peer bins: `3` rings x `12` sectors
  - `10` proprioception features:
    - root forward velocity
    - root lateral velocity
    - root angular velocity
    - joint 0 angle
    - joint 1 angle
    - joint 0 velocity
    - joint 1 velocity
    - previous drive
    - previous steer
    - episode progress
- Episode objective:
  - forage until timeout
  - food respawns immediately after capture
- Reward:
  - every fish receives the same scalar each step
  - `team_pellets_eaten_this_step * pellet_reward - step_cost`

Default task parameters:

- `num_fish = 8`
- `food_count = 48`
- `food_capture_radius = 0.45`
- `time_limit = 600`
- `sensor_radius = 4.5`
- `sensor_ring_edges = [1.5, 3.0, 4.5]`
- `sensor_num_sectors = 12`

## Run Training

Canonical background launcher:

```powershell
cd "c:\Users\adminlcarreira\SwimmingPolygons"
.\.venv\Scripts\Activate.ps1
cd "07 Shared Policy School"

.\run_baseline.ps1
```

Direct training command:

```powershell
python agent.py `
  --device cuda `
  --train-iterations 100 `
  --checkpoint-every-iterations 5 `
  --num-env-runners 4 `
  --num-envs-per-runner 2 `
  --eval-report-episodes 10 `
  --checkpoint-root .\rllib_checkpoints_baseline_v7_school
```

Useful task flags:

- `--num-fish`
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

Best checkpoint selection uses:

1. highest `mean_team_food_eaten`
2. tie-break by `team_food_per_100_steps`
3. tie-break by `mean_team_reward`

## Evaluate A Checkpoint

Visual rollout:

```powershell
python test_model.py --checkpoint-root .\rllib_checkpoints_baseline_v7_school
```

Headless batch scoring:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v7_school `
  --episodes 10 `
  --no-render `
  --summary-json .\rllib_checkpoints_baseline_v7_school\eval_latest.json `
  --summary-csv .\rllib_checkpoints_baseline_v7_school\eval_latest.csv
```

Optional toggles:

- `--policy-mode trained|random`
- `--focus-agent-id`
- `--hide-sensor-overlay`
- `--num-fish`
- `--food-count`
- `--time-limit`

Random-policy baseline:

```powershell
python test_model.py `
  --policy-mode random `
  --episodes 50 `
  --no-render `
  --summary-json .\random_policy_baseline.json `
  --summary-csv .\random_policy_baseline.csv
```

## Visual Debugging

`debug_visual.py` runs scripted simulator-only scenarios:

- `sensor_overlay_demo`
- `dense_patch`
- `edge_sweep`
- `random_forage`

Example:

```powershell
python debug_visual.py --scenario sensor_overlay_demo
python debug_visual.py --scenario dense_patch --focus-agent-id fish_3
```

The render shows:

- all fish
- the shared pellet field
- sensor wedges for one focus fish only
- a fixed legend explaining:
  - cyan food bins
  - gold peer bins
  - ring ranges
  - sector orientation

## Environment Validation

`debug_env.py` validates the V7 shared-school contract.

Available probes:

- `food_field_contract`
- `spawn_separation`
- `peer_sensor_contract`
- `shared_reward_contract`
- `simultaneous_step_consistency`
- `baseline` aggregates all of them

Smoke run:

```powershell
python debug_env.py --probe-set baseline --seed 0 --no-plots
```

## Delayed Watcher

The watcher defaults to the V7 run names:

- PID file: `baseline_v7_school.pid`
- log file: `baseline_v7_school.out.log`
- checkpoint root: `rllib_checkpoints_baseline_v7_school`

Start it with:

```powershell
python watch_training.py --eval-device cpu --no-visual-launch
```

What it writes under `<checkpoint_root>\auto_eval\`:

- scheduled check snapshots like `check_90m.json`
- `watcher.log`
- batch evaluation summaries for the best checkpoint and final checkpoint
- `evaluation_manifest.json`
- `visual_eval_command.txt`

## Notes

- Fish do not collide in this first milestone.
- Fish interact indirectly through food depletion and local peer sensing.
- All fish map to one shared trainable policy ID: `shared_fish_policy`.
