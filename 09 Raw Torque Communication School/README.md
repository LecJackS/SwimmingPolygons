# V9 Muscle Activation Communication School

`09 Raw Torque Communication School` is the corrected V9 branch.

V9 keeps the same full multi-agent forage-and-communication task shape as V8:

- `10` red fish and `10` blue fish by default
- `48` red pellets and `48` blue pellets by default
- one shared policy for every fish
- per-step message token `0..3`

The locomotion interface is different:

- `motion[i]` is a normalized muscle-activation command for hinge `i`
- the number of hinges is `num_body_segments - 1`
- default morphology is now a longer `5`-segment fish
- activations are filtered before they become torques
- passive stiffness and damping resist parked curled postures
- there is no hardcoded oscillator
- there is no drive-based propulsion proxy
- there is no steer torque bias
- forward motion must emerge from body shape change interacting with drag

Existing pre-correction V9 checkpoints should be treated as invalid.

## Environment Contract

- Env class: `CommunicatingSchoolEnv`
- Default `num_body_segments`: `5`
- Action space per fish:
  - `Dict({"motion": Box(shape=(num_body_segments - 1,), low=-1, high=1), "message": Discrete(4)})`
  - one direct control value per hinge
  - `message`: local token `0..3`
- Observation space per fish is segment-count dependent
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
  - `8 x (3 + 3 * (num_body_segments - 1))` low-level history stack in `full_v9`:
    - forward velocity
    - lateral velocity
    - angular velocity
    - all joint angles
    - all joint velocities
    - all joint activations

Food sensing and food capture are mouth-anchored in V9.
Fish/fish sensing and communication stay body-centered.

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
  --rollout-fragment-length 500 `
  --gamma 0.97 `
  --gae-lambda 0.97 `
  --learning-rate 3e-4 `
  --fcnet-hiddens 512,512,256 `
  --activation-time-constant 0.12 `
  --joint-passive-stiffness 10.0 `
  --body-linear-drag 1.0 `
  --checkpoint-root .\rllib_checkpoints_baseline_v9_muscle_activation_comm
```

Useful V9-only flags:

- `--reward-mode forage|locomotion_debug`
- `--history-length`
- `--activation-time-constant`
- `--joint-passive-stiffness`
- `--body-linear-drag`
- `--restore-from-checkpoint`
- `--swim-assist-start-weight`
- `--swim-assist-min-iterations`
- `--swim-assist-disable-forward-velocity`
- `--swim-assist-disable-joint-limit-occupancy`
- `--swim-assist-disable-negative-forward-frac`
- `--swim-assist-disable-consecutive-evals`
- `--swim-assist-fade-evals`

## Swim Assist Curriculum

V9 `forage` can now use a temporary locomotion assist during training:

- the assist is blended into training reward from iteration `0`
- it is based on positive forward progress plus the existing anti-slip / anti-spin / anti-limit penalties
- light eval stays assist-off so checkpoint selection remains pure
- once the swimmers pass the motion gate consistently, the assist fades to `0` and stays off for the rest of the run

This is meant as a bootstrap for full-scale forage training, not a permanent task change.

## Evaluate A Checkpoint

Current status:

- the corrected V9 forage checkpoints are still weak visual demos
- use `fast` for live checkpoint inspection
- use `full` only for diagnostic capture or GIF export

Live checkpoint view:

```powershell
python test_model.py --checkpoint-root .\rllib_checkpoints_baseline_v9_muscle_activation_comm --render-profile fast
```

Diagnostic full-mode GIF capture:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v9_muscle_activation_comm `
  --render-profile full `
  --render-engine safe `
  --save-gif .\media\v9_checkpoint_full.gif `
  --gif-seconds 6 `
  --gif-fps 12
```

Headless batch scoring:

```powershell
python test_model.py `
  --checkpoint-root .\rllib_checkpoints_baseline_v9_muscle_activation_comm `
  --episodes 5 `
  --no-render `
  --summary-json .\rllib_checkpoints_baseline_v9_muscle_activation_comm\eval_latest.json `
  --summary-csv .\rllib_checkpoints_baseline_v9_muscle_activation_comm\eval_latest.csv
```

## Scripted Visual Baseline

Use the deterministic scripted-wave demo when you want to verify the renderer or simulator independently of a trained checkpoint:

```powershell
python debug_visual.py --scenario scripted_wave_demo --render-profile fast
```

Regenerate the canonical scripted-wave GIF:

Live `full` mode is intentionally not the supported path on Windows. If you request `--render-profile full` without `--save-gif`, the viewer warns and falls back to `fast`.

```powershell
python debug_visual.py `
  --scenario scripted_wave_demo `
  --render-profile full `
  --render-engine safe `
  --save-gif .\media\v9_scripted_wave_full.gif `
  --gif-seconds 6 `
  --gif-fps 12
```

Locomotion diagnostics in summaries now include:

- `mean_forward_velocity`
- `mean_lateral_velocity`
- `mean_abs_angular_velocity`
- `mean_abs_activation`
- `mean_abs_applied_torque`
- `mean_joint_limit_occupancy`
- `fraction_joint_limit_high_steps`
- `fraction_joints_quiet_steps`
- `fraction_negative_forward_velocity_steps`
- `mean_joint_velocity_zero_crossings_per_fish`
- `mean_activation_sign_changes_per_fish`
- `mean_capture_distance`

## Validate The Env

Run the corrected V9 probe set:

```powershell
python debug_env.py --probe-set baseline --seed 0 --no-plots
```

Current V9 probes:

- `activation_rest_decay`
- `constant_activation_no_cruise`
- `scripted_wave_propulsion`
- `wave_beats_static_activation`
- `mouth_capture_contract`
- `history_contract`

## Debug Locomotion Smoke

This is the shortest useful PPO sanity check for wave learning in the corrected simulator:

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
  --checkpoint-root .\rllib_checkpoints_smoke_v9_locomotion_debug_corrected
```

## Notes

- V8 remains frozen.
- V9 is now a muscle-activation swimmer, not the earlier raw-torque version.
- The debug locomotion mode is only a simulator check; the canonical task remains `forage`.

