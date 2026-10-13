# Reproducibility

## Immutable Inputs

Every reported run is identified by:

1. Git commit.
2. Feature-store manifest and hash inventory.
3. Stage-1 checkpoint SHA256.
4. Stage-2 checkpoint metadata SHA256.
5. History configuration SHA256.
6. Evaluation arguments, seeds, camera order, and color order.

Do not reuse a mutable output directory. The training launchers refuse to write
over a nonempty checkpoint path unless OpenPI's explicit overwrite flag is
provided.

## Critical Invariants

- `action_causal[0]` is zero.
- `action_causal[1:] == action[:-1]`.
- A sampled visual variant uses its own frame IDs and its own causal actions.
- Camera order is stable from store construction through evaluation.
- DINOv3 recent and alignment streams use the same pooled `(3, 64, 768)`
  features.
- RMBench caps are 4096 latent, 768 recent, and 1244 action/proprio tokens.
- RoboCasa caps are 4096 latent, 768 recent, and 1776 action/proprio tokens.
- Stage 2 loads the complete Pi0.5 checkpoint plus the matching Stage-1 memory
  encoder.
- The Stage-1 decoder is not loaded unless an auxiliary Stage-2 profile
  explicitly requests it.

## Deterministic Retraining

The release launchers default to `IMPRINT_DETERMINISTIC_TRAINING=1`. This sets:

```text
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
XLA_FLAGS=--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true
```

Stage 2 also derives every worker's Python and NumPy RNG from Torch's seeded
worker stream. This makes offline history-variant selection stochastic within a
run but repeatable across runs.

This mode was added after the reported checkpoints were trained. The historical
RMBench recurrent run used nondeterministic B200/XLA kernels, and its legacy
Stage-2 loader created an unseeded NumPy generator for history variants.
Therefore, the published paper checkpoints are the canonical artifacts for
reproducing reported rollout numbers; a fresh deterministic training run
reproduces the documented recipe, not the exact historical parameter tensors.

In a controlled four-B200 audit, two historical-mode runs consumed an identical
500-batch sequence but differed by 0.50% relative L2 in the memory encoder and
1.04% in the decoder. Repeating the audit with the deterministic settings above
produced bit-identical losses and zero parameter drift. The deterministic
500-step runtime was within normal run-to-run timing noise of historical mode.

The audited historical RMBench recurrent environment used CUDA 12.6, cuDNN
9.5, Transformers 4.53.2, and Tokenizers 0.21.1. The portable release lock uses
CUDA 12.8, cuDNN 9.7, Transformers 4.56.2, and Tokenizers 0.22.2. Controlled
cross-runtime drift was no larger than same-runtime historical-mode drift, but
the exact runtime remains part of checkpoint provenance.

## Original Snapshots

The cleanup process did not modify the source directories from which the
release was assembled. Local archives include the canonical recurrent source,
RMBench evaluator, RoboCasa launchers/evaluator, and full-attention/recurrent
training source. Their checksums are stored beside the archives and are not
part of the Git repository because they contain historical infrastructure
paths.

## Verification

```bash
bash scripts/reproduce/check_release.sh
```

This compiles Python files, checks shell syntax, verifies release profiles, and
scans tracked files for credentials and machine-specific paths.
