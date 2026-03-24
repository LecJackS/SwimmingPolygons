# V8 Color Communication School

`08 Color Communication School` is the first explicit communication stage.

V8 keeps the articulated 3-segment fish body from V7, but changes the multi-agent task:

- `20` fish by default: `10` red, `10` blue
- `96` pellets by default: `48` red, `48` blue
- one shared trainable policy for all fish
- individual rewards, not shared team reward
- one local broadcast token per fish per step
- one-ring typed sector observation with food, fish, and message channels

V7 remains the frozen shared-team-reward baseline. V8 is now the active communication branch.

## Environment Contract

- Env class: `CommunicatingSchoolEnv`
- Action space per fish:
  - `Dict({"motion": Box(shape=(2,), low=-1, high=1), "message": Discrete(4)})`
  - `motion[0]`: drive
  - `motion[1]`: steer
  - `message`: local token `0..3`, where `0` is silence
- Observation space per fish: `48D`
  - `6` edible-food sectors
  - `6` non-edible-food sectors
  - `6` teammate sectors
  - `6` opponent sectors
  - `6` teammate-message sectors
  - `6` opponent-message sectors
  - `12` proprio / identity features:
    - self-color bit
    - root forward velocity
    - root lateral velocity
    - root angular velocity
    - joint 0 angle
    - joint 1 angle
    - joint 0 velocity
    - joint 1 velocity
    - previous drive
    - previous steer
    - previous message token
    - episode progress
- Episode objective:
  - forage until timeout
  - red fish only eat red pellets
  - blue fish only eat blue pellets
  - pellets respawn immediately after capture
- Reward:
  - each fish gets its own scalar each step
  - `food_eaten_this_step * pellet_reward - step_cost`

Default task parameters:

- `num_red_fish = 10`
- `num_blue_fish = 10`
- `num_red_pellets = 48`
- `num_blue_pellets = 48`
- `food_capture_radius = 0.45`
- `time_limit = 300`
- `sector_radius = 5.0`
- `sector_num = 6`

Food spawn contract:

- training reset and pellet respawn both use map-wide uniform random sampling
- red and blue pellets use the same sampler and are spatially mixed
- the sampler keeps the existing light constraints:
  - minimum pellet spacing
  - minimum distance from fish spawn positions
- structured layouts are reserved for a few deterministic env probes where exact geometry is required

## Run Training

Canonical background launcher:

```powershell
cd "c:\Users\adminlcarreira\SwimmingPolygons"
.\.venv\Scripts\Activate.ps1
cd "08 Color Communication School"

.\run_baseline.ps1
```

`run_baseline.ps1` now starts a dedicated child PowerShell that runs `python -u agent.py` in the foreground of that child shell. The PID file tracks that child shell, and the child shell appends directly to `baseline_v8_color_comm.out.log` and `baseline_v8_color_comm.err.log`, so closing the original launching terminal does not tear down Ray worker consoles.

Direct training command:

```powershell
python agent.py `
  --device cuda `
  --train-iterations 200 `
  --checkpoint-every-iterations 10 `
  --num-env-runners 8 `
  --num-envs-per-runner 2 `
  --light-eval-episodes 4 `
  --train-batch-size 8000 `
  --minibatch-size 1024 `
  --num-epochs 6 `
  --time-limit 300 `
  --checkpoint-root .\rllib_checkpoints_baseline_v8_color_comm
```

The canonical PPO sampler defaults are now:

- `count_steps_by = "agent_steps"`
- `rollout_fragment_length = 250`
- `sample_timeout_s = 180.0`
- `train_batch_size = 8000`
- `minibatch_size = 1024`
- `num_epochs = 6`

Useful task flags:

- `--num-red-fish`
- `--num-blue-fish`
- `--num-red-pellets`
- `--num-blue-pellets`
- `--time-limit`
- `--pellet-reward`
- `--step-cost`
- `--sector-radius`
- `--sector-num`
- `--light-eval-episodes`
- `--restore-from-checkpoint`
- `--gamma`
- `--gae-lambda`
- `--learning-rate`
- `--entropy-coeff`
- `--train-batch-size`
- `--minibatch-size`
- `--num-epochs`
- `--fcnet-hiddens`
- `--fcnet-activation`

Checkpoint-time reports are written under the checkpoint root:

- `eval_reports.jsonl`
- `eval_reports.csv`
- `run_summary.json`

Each checkpoint report includes:

- light-eval normal metrics only
- `eval_mode = "light"`
- `light_eval_episodes`

The expensive full normal+muted evaluation now runs after training for the selected best checkpoint and `checkpoint_final`, not inside the hot training loop.

## Targeted Campaign

`train_until_target.py` now runs the open-ended V8 `4x` campaign against the canonical random baseline.

It:

- reads `random_policy_baseline.json`
- computes `target_multiple * random_mean_pellets_per_fish`
- runs family/seed/phase training rounds under `rllib_checkpoints_target_v8_4x`
- evaluates a shortlisted checkpoint set after each phase
- runs the expensive normal+muted confirmation only when a candidate clears the target on the cheaper normal-only shortlist pass
- stops when a checkpoint confirms the target or the campaign retires every live combo
- persists phase status incrementally so interrupted phases can be resumed safely
- supports `--resume-existing` to recover an existing manifest/root without wiping prior work

Smoke validation:

```powershell
python train_until_target.py --smoke --force-clean
```

Full campaign:

```powershell
python train_until_target.py --force-clean
```

Resume an interrupted campaign:

```powershell
python train_until_target.py --resume-existing
```

Detached campaign launcher:

```powershell
.\run_target_campaign.ps1 -ForceClean
```

Detached watchdog launcher for unattended recovery:

```powershell
.\run_target_watchdog.ps1
```

Outputs:

- `target_training_manifest.json`
- phase roots under `rllib_checkpoints_target_v8_4x/<family>/seed<seed>/phase<nn>`
- watchdog artifacts:
  - `target_training_watchdog.pid`
  - `target_training_watchdog.out.log`
  - `target_training_watchdog.err.log`

The current default target is `4.0x` random, not `2.0x`.

## Evaluate A Checkpoint

Visual rollout:

```powershell
python test_model.py --checkpoint-root .\rllib_checkpoints_baseline_v8_color_comm
```

Render profiles:

- `--render-profile fast` is the default high-FPS viewer
- `--render-profile full` restores the richer overlay, legend, and token labels for diagnosis
- `--hide-sensor-overlay` only affects the `full` profile
- `--render-engine auto` is the default backend-aware renderer
- `--render-engine blit` forces the cached-blit path
- `--render-engine safe` forces a full redraw each frame with no background reuse

Headless batch scoring:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v8_color_comm `
  --episodes 2 `
  --no-render `
  --summary-json .\rllib_checkpoints_baseline_v8_color_comm\eval_latest.json `
  --summary-csv .\rllib_checkpoints_baseline_v8_color_comm\eval_latest.csv
```

Optional toggles:

- `--policy-mode trained|random`
- `--checkpoint-list-file`
- `--mute-mode normal|both`
- `--render-profile fast|full`
- `--render-engine auto|blit|safe`
- `--focus-agent-id`
- `--hide-sensor-overlay`
- `--mute-messages`
- `--num-red-fish`
- `--num-blue-fish`
- `--num-red-pellets`
- `--num-blue-pellets`
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
- `message_demo`
- `color_capture_demo`
- `random_forage`
- `blit_boundary_stress`

Example:

```powershell
python debug_visual.py --scenario sensor_overlay_demo
python debug_visual.py --scenario message_demo --focus-agent-id fish_0
python debug_visual.py --scenario sensor_overlay_demo --render-profile full
python debug_visual.py --scenario blit_boundary_stress --render-profile fast --render-engine auto
```

The V8 demos now start from the same random scattered food field used by training, seeded by `--seed`, so the visual starting state matches the learning setup instead of using fixed color clusters.

The render shows:

- red and blue fish
- red and blue pellets
- one focus-fish typed overlay
- the current message token above every fish
- a fixed legend explaining the channel colors and token notes

## Environment Validation

`debug_env.py` validates the V8 color-communication contract.

Available probes:

- `color_food_field_contract`
- `food_spawn_distribution_contract`
- `spawn_separation`
- `color_matched_consumption_contract`
- `communication_locality_contract`
- `observation_contract`
- `simultaneous_step_consistency`
- `baseline` aggregates all of them

Smoke run:

```powershell
python debug_env.py --probe-set baseline --seed 0 --no-plots
```

## Delayed Watcher

The watcher defaults to the V8 run names:

- PID file: `baseline_v8_color_comm.pid`
- log file: `baseline_v8_color_comm.out.log`
- checkpoint root: `rllib_checkpoints_baseline_v8_color_comm`

Start it with:

```powershell
python watch_training.py --eval-device cpu --no-visual-launch
```

What it writes under `<checkpoint_root>\auto_eval\`:

- scheduled check snapshots like `check_90m.json`
- `watcher.log`
- combined full batch summaries for the best checkpoint and final checkpoint
- `evaluation_manifest.json`
- `visual_eval_command.txt`

## Performance Benchmark

Use the local benchmark harness to measure env, inference, and render throughput:

```powershell
python benchmark_perf.py `
  --checkpoint-root .\rllib_checkpoints_smoke_v8_color_comm `
  --summary-json .\perf_benchmark.json `
  --summary-csv .\perf_benchmark.csv
```

The harness reports:

- env-only no-render speed
- random-policy fast/full render speed
- trained-policy no-render speed
- trained-policy fast/full render speed
- backend and render-engine information for each case

## Notes

- Same-color fish compete. Only the fish that eats a matching-color pellet gets reward.
- Both colors can hear nearby messages.
- Messages are local and delayed by one step in the observation path.
- Wrong-color fish do not consume or deny pellets in this first pass.
