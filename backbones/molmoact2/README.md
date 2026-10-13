# IMPRINT for MolmoAct2

This directory contains the native PyTorch/LeRobot integration used to attach
the policy-agnostic IMPRINT memory encoder to the MolmoAct2 action expert.
It is based on the bundled MolmoAct2 source snapshot.

The repository root remains the source of truth for:

- RMBench stores and causal action construction
- Stage-1 memory pretraining
- DINOv3 and WAN feature contracts
- simulator installation and benchmark assets

This directory owns:

- Flax Stage-1 to PyTorch checkpoint conversion
- MolmoAct2 memory encoder and checkpoint loading
- memory conditioning inside the action expert
- compact RMBench Stage-2 data loading
- MolmoAct2 policy serving and closed-loop evaluation

## Environment

Use a separate environment from the root JAX/OpenPI environment:

```bash
cd backbones/molmoact2/experiments
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
python -m pip install -e "./lerobot"
```

Download the public MolmoAct2 base model before training:

```bash
hf download allenai/MolmoAct2 \
  --revision e432d85f6e039edca44afb93c262f3084ab72a9c \
  --local-dir artifacts/models/MolmoAct2
```

## Stage 1

Stage 1 is policy-agnostic. Use the repository-root training pipeline and the
RMBench full-attention profile:

```bash
export IMPRINT_STAGE1_STORE=/path/to/rmbench/stage1
export IMPRINT_OUTPUT_DIR=/path/to/new/stage1-output

bash scripts/reproduce/train_stage1.sh \
  configs/paper/rmbench/full_attention/experiment.env
```

The selected checkpoint is `stage1_step160000.npz`.

## Stage 2

Copy the portable profile and set its required paths:

```bash
cp configs/paper/rmbench/molmoact2/experiment.env \
  configs/paper/rmbench/molmoact2/local.env

export MOLMOACT2_STAGE1_CHECKPOINT=/path/to/stage1_step160000.npz
export MOLMOACT2_STAGE2_ROOT=/path/to/rmbench/stage2
export MOLMOACT2_STAGE1_FEATURE_ROOT=/path/to/rmbench/stage1/features
export MOLMOACT2_SAVE_FOLDER=/path/to/new/molmoact2-stage2-output
export MOLMOACT2_PYTHON=$PWD/backbones/molmoact2/experiments/.venv/bin/python

bash scripts/reproduce/train_molmoact2_stage2.sh \
  configs/paper/rmbench/molmoact2/experiment.env
```

The paper configuration:

- starts from `allenai/MolmoAct2`
- imports the 160k Stage-1 memory encoder
- finetunes the memory encoder, VLM language stack, connector, and action
  expert
- freezes the MolmoAct2 image encoder
- uses the native fixed-FiLM memory seam
- trains for 100,000 updates with a 50-step, 16-D action target
- uses 256 memory tokens with dimensions `48/768/16/16/2048`

## Evaluation

The closed-loop implementation is under
[`experiments/rmbench_eval`](experiments/rmbench_eval). Evaluation must use:

- online WAN and DINOv3 history features
- stored Pi0.5 language embeddings for the memory encoder
- RGB-to-BGR conversion only at policy input
- strictly causal executed actions
- `EVAL_CONSOLIDATE_ALL=1`
- 50-step prediction and execution chunks

The policy server entrypoint is
`experiments/scripts/serve_rmbench_molmoact2_tcp.py`. The evaluator package
contains the client patch, four-view recorder, runtime contract checks, and
smoke worker used by the experiment. Simulator dependencies must remain in a
separate RMBench environment.

The release evaluator downloads:

- `Amazon-FAR/imprint_models`, path `simulation/rmbench/molmoact2/`, for the
  model-only Stage-2 checkpoint
- `Amazon-FAR/imprint_datasets`, path
  `simulation/rmbench/molmoact2_eval_assets/`, for compact metadata and stored
  language embeddings
- pinned upstream DINOv3 and Qwen repositories for visual and tokenizer assets

Optimizer state, trainer state, simulator archives, and source tarballs are
not part of the public model release.

## Focused Tests

```bash
cd backbones/molmoact2/experiments
PYTHONPATH="$PWD:$PWD/lerobot/src" python -m pytest -q \
  tests/test_memory_bottleneck_contract.py \
  tests/test_memory_bottleneck_encoder.py \
  tests/test_memory_checkpoint.py \
  tests/test_action_expert_memory.py \
  tests/test_rmbench_stage2.py \
  tests/test_rmbench_eval_package.py
```

## Published Configuration

The released MolmoAct2 checkpoint uses:

| Setting | Value |
| --- | --- |
| Memory mode | `finetune` |
| Memory tokens | 256 x 1024 |
| Cross-attention blocks | 2 |
| Memory attention heads | 8 |
| Action horizon/dimension | 50 / 16 |
| Global/device batch | 4 / 1 |
| Trainable policy components | language stack, connector, action expert |
| Frozen policy component | image encoder |
| Training updates | 100,000 |

The upstream MolmoAct2 and vendored LeRobot licenses are retained alongside
their source.
