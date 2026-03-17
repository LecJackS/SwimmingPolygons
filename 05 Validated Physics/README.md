# Version 5: Validated Physics

`V5` forks from the current `04 Accelerated Physics` codebase and keeps V4 frozen as the previous stable experimental baseline.

## Intent

This version is reserved for realism-first simulator validation and the next training iteration:

- Audit the simulator contract before more major training changes.
- Add visual debugging tools that are independent of policy quality.
- Continue RLlib training only after the simulator passes stronger validation.

## Current Status

V5 starts as a working copy of V4 and currently includes:

- Gymnasium-native `OctopusEnv` with decoupled translational and rotational dynamics.
- RLlib PPO training/evaluation scripts with staged curriculum and early-stop logic.
- Env-only validation harness in `debug_env.py`.
- Scripted visual debugging runner in `debug_visual.py`.

## Run V5

Train:

```bash
cd "05 Validated Physics"
python agent.py --train-iterations 20
```

Each checkpoint now emits deterministic fixed-distance evaluation reports under the checkpoint root:

- `eval_reports.jsonl`
- `eval_reports.csv`
- `run_summary.json`

The default report sweep is:

- `0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 10.0`

Evaluate latest checkpoint:

```bash
cd "05 Validated Physics"
python test_model.py --max-frames 1000
```

Evaluate at a fixed spawn distance:

```bash
cd "05 Validated Physics"
python test_model.py --max-frames 1000 --fixed-food-distance 3.0
```

Score a checkpoint deterministically over multiple episodes without rendering:

```bash
cd "05 Validated Physics"
python test_model.py --checkpoint-root ./rllib_checkpoints --episodes 10 --distances 5.7,10.0 --no-render --summary-json ./rllib_checkpoints/eval_latest.json --summary-csv ./rllib_checkpoints/eval_latest.csv
```

The batch evaluator uses the same deterministic scoring path as `agent.py`, so checkpoint review and checkpoint-time reports are directly comparable.

## Validate The Environment

Run the baseline validation harness:

```bash
cd "05 Validated Physics"
python debug_env.py --probe-set baseline --seed 0
```

Run the realism-first physics audit:

```bash
cd "05 Validated Physics"
python debug_env.py --probe-set physics_audit --seed 0
```

Run the control and observation contract audit:

```bash
cd "05 Validated Physics"
python debug_env.py --probe-set control_contract --seed 0
```

Artifacts are written under `media/debug_reports/<timestamp>/` and include `summary.json`, machine-readable tables, and optional plots.

## Visual Debugging

Render a scripted simulator scenario:

```bash
cd "05 Validated Physics"
python debug_visual.py --scenario toward_target --seed 0
```

Available scenarios:

- `neutral`
- `forward`
- `turn_left`
- `turn_right`
- `toward_target`
- `away_from_target`
- `orbit_failure`

## Immediate Goals

- Freeze the simulator contract only after `baseline`, `physics_audit`, and `control_contract` all pass.
- Keep RLlib, action space, and observation shape unchanged unless the new audits prove they are invalid.
- Use checkpoint-time evaluation reports and deterministic batch scoring to judge training quality before changing reward or curriculum again.
- Resume reward/curriculum iteration only after the simulator audit is green and the current reports show where training stalls.

## Longer Horizon

- Reach reliable single-fish distance-10 mastery.
- Add fish presets for body geometry and dynamics.
- Evolve toward a multi-fish interactive environment after single-fish behavior is trustworthy.
