# Evaluation

## Shared Policy Contract

Evaluation must match training:

- three cameras in the recorded benchmark order
- DINOv3 visual stream
- strict causal action memory
- `EVAL_CONSOLIDATE_ALL=1`
- no legacy empty-recent behavior
- 50-step prediction/execution chunks on RMBench
- benchmark-specific action/proprio dimensions and token caps

Do not infer these values from a checkpoint filename. Use its
`config/history_config.yaml` and `config/input_provenance.json`.

## RMBench

The canonical protocol runs a successful one-trial `cover_blocks` smoke before
submitting all nine tasks. Full evaluation uses seeds 100000-100099 for each
task and reports the task-balanced mean.

RMBench DINO stores use:

```bash
export EVAL_CONSOLIDATE_ALL=1
export MEMFT_LEGACY_EMPTY_RECENT=0
export RMBENCH_POLICY_INPUT_COLOR_ORDER=bgr
export MEMFT_PREDICTION_HORIZON=50
export MEMFT_EXECUTION_HORIZON=50
export DINOV3_MODEL_PATH=/path/to/dinov3-vitb16
```

Use `benchmarks/rmbench/eval/run_local.py` for a local two-process run. The
script creates an immutable server config, starts the policy process, verifies
its checkpoint identity, and invokes the official evaluator through the
four-view wrapper.

```bash
python benchmarks/rmbench/eval/run_local.py \
  --rmbench-root /path/to/RMBench \
  --checkpoint /path/to/stage2/step-100000 \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task cover_blocks \
  --trials 1 \
  --policy-python "$PWD/.venv/bin/python" \
  --sim-python /path/to/rmbench/.venv/bin/python \
  --output outputs/rmbench/cover-smoke
```

## RoboCasa

The evaluator uses three synchronized views and replays the recorded scene XML
and initial state. Task groups and trial counts are defined in
`benchmarks/robocasa/eval/eval_standard_3view.py`.

Use the two-process launcher:

```bash
python benchmarks/robocasa/eval/run_local.py \
  --robocasa-root /path/to/patched-robocasa \
  --checkpoint /path/to/stage2/step-150000 \
  --dinov3-model "$DINOV3_MODEL_PATH" \
  --architecture full_attention \
  --task-group organize_condiments \
  --pilot-trials 1 \
  --new5-scene-root /path/to/recorded-scenes \
  --sim-python /path/to/patched-robocasa/.venv-mem/bin/python \
  --output outputs/robocasa/organize-condiments-smoke
```

The launcher creates an auditable `paper_metrics.json`. Sweeten Hot Chocolate
uses the paper's recovered success rule
`stable_placement_count == goal_count`, while retaining the stricter raw
evaluator result in the same file. Other RoboCasa tasks report raw success
unchanged.

See `benchmarks/robocasa/README.md` for required replay assets and custom tasks.

## Reporting

Retain `trials.jsonl`, per-task statistics, evaluator arguments, checkpoint
provenance, logs, and videos. A trial counts as successful only under the
benchmark's terminal success predicate. Failure-type analysis is a separate
manual annotation and must not change the primary success denominator.
