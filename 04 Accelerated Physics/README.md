# Version 4: Accelerated Physics

`V4` is the next iteration after `03 2D Improved Physics`.

## Intent

This version is reserved for performance-focused work:

- Better training throughput and GPU utilization.
- Checkpointing and recovery workflow improvements.
- Controlled hyperparameter and environment optimization experiments.

## Current Status

Physics baseline implemented.

This version now includes:

- Gymnasium-native `OctopusEnv` with V3-like decoupled dynamics.
- Flat normalized observation vector with relative-only target encoding.
- RLlib PPO training/evaluation scripts with CLI options, checkpointing, and staged curriculum.

## Run V4 (RLlib)

Train:

```bash
cd "04 Accelerated Physics"
python agent.py --train-iterations 20
```

Train with early stop at distance 10:

```bash
cd "04 Accelerated Physics"
python agent.py --train-iterations 200 --checkpoint-every-iterations 5 --curriculum-enabled --curriculum-stages 0.7,1.0,1.4,2.0,2.8,4.0,5.7,8.0,10.0 --curriculum-success-rate 0.8 --curriculum-eval-episodes 20 --curriculum-consecutive-evals 2 --early-stop-enabled --early-stop-distance 10 --early-stop-success-rate 0.9 --early-stop-eval-episodes 20 --early-stop-consecutive-evals 3
```

Curriculum behavior:

- Training always uses fixed-distance stages shared globally across workers.
- Stage promotion requires fixed-distance eval success and is monotonic (no demotion).
- Final stop remains the strict early-stop gate at distance `10`.

Evaluate latest checkpoint:

```bash
cd "04 Accelerated Physics"
python test_model.py --max-frames 1000
```

Evaluate at a fixed spawn distance (no curriculum in eval):

```bash
cd "04 Accelerated Physics"
python test_model.py --max-frames 1000 --fixed-food-distance 3.0
```

Evaluate a specific checkpoint:

```bash
cd "04 Accelerated Physics"
python test_model.py --checkpoint-path ".\\rllib_checkpoints\\<checkpoint_dir>"
```

## Next Steps

- Tune reward shaping while keeping training stable.
- Benchmark CPU vs GPU throughput and document tradeoffs.
- Add a first fish-variant configuration set (size/mass/drag presets).
- Introduce multi-fish simulation gradually (single controlled fish first).

## Benchmark Profiles (Time-to-Threshold)

CPU profile:

```bash
cd "04 Accelerated Physics"
python agent.py --device cpu --train-iterations 100 --checkpoint-every-iterations 5 --curriculum-enabled --curriculum-stages 0.7,1.0,1.4,2.0,2.8,4.0,5.7,8.0,10.0 --curriculum-success-rate 0.8 --curriculum-eval-episodes 20 --curriculum-consecutive-evals 2 --curriculum-time-limit-base 100 --curriculum-time-limit-max 180 --early-stop-distance 10 --early-stop-success-rate 0.9 --early-stop-eval-episodes 20 --early-stop-consecutive-evals 3
```

CUDA profile:

```bash
cd "04 Accelerated Physics"
python agent.py --device cuda --train-iterations 100 --checkpoint-every-iterations 5 --curriculum-enabled --curriculum-stages 0.7,1.0,1.4,2.0,2.8,4.0,5.7,8.0,10.0 --curriculum-success-rate 0.8 --curriculum-eval-episodes 20 --curriculum-consecutive-evals 2 --curriculum-time-limit-base 100 --curriculum-time-limit-max 180 --early-stop-distance 10 --early-stop-success-rate 0.9 --early-stop-eval-episodes 20 --early-stop-consecutive-evals 3
```

Compare profiles by **time-to-final-stage-pass / time-to-early-stop**, not GPU utilization percentage.
