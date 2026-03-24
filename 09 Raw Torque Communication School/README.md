# V9 Raw Torque Communication School

`09 Raw Torque Communication School` is the raw-control fork of V8.

V9 keeps the same full multi-agent forage-and-communication task shape:

- `10` red fish and `10` blue fish by default
- `48` red pellets and `48` blue pellets by default
- one shared policy for every fish
- per-step message token `0..3`

The locomotion interface is different:

- `motion[0]` and `motion[1]` are normalized joint torques
- there is no hardcoded oscillator
- there is no drive-based propulsion proxy
- there is no steer torque bias
- forward motion must emerge from articulated-body motion and drag

## Environment Contract

- Env class: `CommunicatingSchoolEnv`
- Action space per fish:
  - `Dict({"motion": Box(shape=(2,), low=-1, high=1), "message": Discrete(4)})`
  - `motion[0]`: joint-0 torque command
  - `motion[1]`: joint-1 torque command
  - `message`: local token `0..3`
- Observation space per fish: `111D`
  - `36` current exteroception features:
    - edible food
    - non-edible food
    - teammate
    - opponent
    - teammate messages
    - opponent messages
  - `3` context features:
    - team index
    - previous emitted message token
    - episode progress
  - `8 x 9` low-level history stack:
    - forward velocity
    - lateral velocity
    - angular velocity
    - joint angles `2`
    - joint velocities `2`
    - applied torques `2`

## Reward Modes

`forage` is the canonical task:

- same pellet reward structure as V8
- no locomotion shaping

`locomotion_debug` is validation-only:

- allows `0` blue fish and `0` pellets
- rewards forward swimming
- penalizes lateral slip, spin, torque effort, and joint-limit abuse

## Run Training

Canonical launcher:

```powershell
cd "c:\Users\adminlcarreira\SwimmingPolygons"
.\.venv\Scripts\Activate.ps1
cd "09 Raw Torque Communication School"

.\run_baseline.ps1
```

Direct training command:

```powershell
python agent.py `
  --device cuda `
  --train-iterations 200 `
  --checkpoint-every-iterations 20 `
  --num-env-runners 8 `
  --num-envs-per-runner 2 `
  --light-eval-episodes 2 `
  --train-batch-size 16000 `
  --minibatch-size 2048 `
  --num-epochs 6 `
  --rollout-fragment-length 300 `
  --gamma 0.97 `
  --gae-lambda 0.97 `
  --learning-rate 3e-4 `
  --fcnet-hiddens 512,512,256 `
  --checkpoint-root .\rllib_checkpoints_baseline_v9_raw_torque_comm
```

Useful V9-only flags:

- `--reward-mode forage|locomotion_debug`
- `--history-length`
- `--actuator-time-constant`
- `--restore-from-checkpoint`

## Evaluate A Checkpoint

Visual rollout:

```powershell
python test_model.py --checkpoint-root .\rllib_checkpoints_baseline_v9_raw_torque_comm
```

Headless batch scoring:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v9_raw_torque_comm `
  --episodes 5 `
  --no-render `
  --summary-json .\rllib_checkpoints_baseline_v9_raw_torque_comm\eval_latest.json `
  --summary-csv .\rllib_checkpoints_baseline_v9_raw_torque_comm\eval_latest.csv
```

Locomotion metrics are included in batch summaries:

- `mean_forward_velocity`
- `mean_lateral_velocity`
- `mean_abs_angular_velocity`
- `mean_abs_applied_torque`
- `mean_joint_limit_occupancy`
- `mean_joint_velocity_zero_crossings_per_fish`
- `mean_capture_distance`

## Validate The Env

Run the baseline V9 probe set:

```powershell
python debug_env.py --probe-set baseline --seed 0 --no-plots
```

Current V9 probes:

- `torque_rest_decay`
- `scripted_wave_propulsion`
- `mirror_torque_turn`
- `no_hidden_drive_assist`
- `history_contract`

## Debug Locomotion Smoke

This is the shortest useful PPO sanity check for torque-wave learning:

```powershell
python agent.py `
  --reward-mode locomotion_debug `
  --num-red-fish 1 `
  --num-blue-fish 0 `
  --num-red-pellets 0 `
  --num-blue-pellets 0 `
  --train-iterations 40 `
  --checkpoint-every-iterations 20 `
  --num-env-runners 4 `
  --num-envs-per-runner 1 `
  --train-batch-size 4800 `
  --minibatch-size 1200 `
  --num-epochs 6 `
  --learning-rate 0.001 `
  --entropy-coeff 0.03 `
  --checkpoint-root .\rllib_checkpoints_smoke_v9_locomotion_debug_long
```

## Notes

- V8 remains frozen.
- V9 removes the hardcoded locomotion prior, so learning is harder and slower.
- The debug locomotion mode is only a capability proof; the canonical task remains `forage`.
