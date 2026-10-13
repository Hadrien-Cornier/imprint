# Customized RoboCasa Evaluation Environment

The paper uses a patched RoboCasa 1.0.1 environment rather than an unmodified
PyPI installation. The exact source archive is distributed under
`simulation/robocasa/environment/` in the public Hugging Face dataset
`Amazon-FAR/imprint_datasets`.

The archive SHA256 is:

```text
ea113f5e102a59d8c7b893dbc9f4dfe3e993cc5dd1a0ec5f4173d9012061bd91
```

It contains the migrated memory-task classes, legacy scene remapping, recorded
state replay fixes, tests, and the dependency lock copied into this directory.

```bash
hf download Amazon-FAR/imprint_datasets \
  simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  --repo-type dataset \
  --local-dir artifacts/downloads/imprint_datasets

bash scripts/reproduce/setup_robocasa_eval.sh \
  artifacts/downloads/imprint_datasets/simulation/robocasa/environment/robocasa-memory-setup.tar.gz \
  .external/robocasa-memory-setup
```

The simulator assets are not embedded in the source archive. The setup helper
downloads the six official RoboCasa bundles and the pinned legacy replay assets
before reporting success. Expect approximately 10 GB of asset downloads. Keep
this simulator environment separate from the IMPRINT policy environment.

Older preserved policy environments may require a DINOv3-compatible
Transformers overlay. In that case, pass
`--dinov3-runtime-overlay /path/to/python-package-overlay` to the launcher;
omit it when the policy environment already supports DINOv3.

The simulator bundle must retain
`scripts/replay/replay_memory_demos.py`. The launcher discovers that path
automatically and uses the canonical recorded-replay and causal-action
protocol; pass `--memory-replay-helper` only for a different bundle layout.

Run a smoke with the coordinated launcher:

```bash
python benchmarks/robocasa/eval/run_local.py \
  --robocasa-root .external/robocasa-memory-setup \
  --checkpoint /path/to/stage2/step-150000 \
  --dinov3-model /path/to/dinov3-vitb16 \
  --architecture full_attention \
  --task-group organize_condiments \
  --pilot-trials 1 \
  --new5-scene-root /path/to/recorded-scenes \
  --output outputs/robocasa/organize-condiments-smoke
```
