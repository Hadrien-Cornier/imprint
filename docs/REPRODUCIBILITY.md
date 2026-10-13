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
