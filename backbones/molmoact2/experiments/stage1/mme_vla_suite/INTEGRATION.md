# MME-VLA (RoboMME) memory suite — integration into this openpi

The full **`mme_vla_suite`** (all 14 memory-augmented VLA variants from
[RoboMME/robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning))
is vendored into this openpi fork as a sibling package under `src/mme_vla_suite/`.

## What was integrated

- **All 14 variants** (a 3×3 grid + 2 symbolic), selected via YAML configs in
  `models/config/robomme/`:

  | representation | × integration | variants |
  |---|---|---|
  | perceptual (frame-sampling / token-dropping) | context / modulation / expert | 6 |
  | recurrent (TTT / RMT) | context / modulation / expert | 6 |
  | symbolic (simple / grounded subgoal) | (prompt) | 2 |

- Memory model: `models/integration/history_pi0.py` (`HistoryPi0`,
  `HistoryPi0Config` — subclass of openpi `Pi0Config`),
  `history_observation.py` (`HistAugObservation` — subclass of openpi
  `Observation`), `history_gemma.py` (memory cross-attn / expert).
- Representations: `models/representation/{percep_mem,recur_mem,ttt,rmt,mem_encoder}.py`.
- Policy/transforms: `policies/robomme_policy.py`; training: `training/config.py`
  (`get_config`, configs `pi05_baseline`, `mme_vla_suite`), `training/dataset.py`,
  `shared/mem_buffer.py`.

## openpi reconciliation (the only stock-openpi changes)

The suite targets the same mid-2025 pi0.5 openpi generation as this fork
(`pi0.py`, `siglip.py`, `array_typing.py`, `transforms.py` are byte-identical).
Two minimal compat patches were applied to `src/openpi/models/model.py`:

1. `preprocess_observation`: relaxed the strict image-key subset check and now
   iterates `observation.images.keys()` — so memory variants that add extra
   history/memory image keys preprocess without error.
2. `BaseModel.load`: filter `None`-valued leaves before
   `check_pytree_equality` — so optional memory params/fields don't trip
   parameter loading.

`HistoryPi0Config.get_freeze_filter` is self-contained (keeps SigLIP/`.*img.*`
and LoRA frozen), so **no** change to stock `pi0_config.py` was needed.

Extra dep added to the openpi venv: `hydra-core` (omegaconf/einops/transformers
4.53.2/opencv already present).

## Verify

From the openpi_rc root:
```bash
.venv-openpi/bin/python -m mme_vla_suite.verify_integration
# -> 14/14 variant configs OK against this openpi
```
This loads every variant YAML, builds a `HistoryPi0Config`, and constructs the
`HistAugObservation` inputs_spec (perceptual→`static_image_emb`,
recurrent→`recur_image_emb`, symbolic→`symbolic_tokenized_prompt`).

## Train a memory variant

```bash
# baseline pi0.5 (no memory)
.venv-openpi/bin/python scripts/train.py pi05_baseline --exp-name=...
# memory suite — pick a variant by pointing history_config at its yaml
.venv-openpi/bin/python scripts/train.py mme_vla_suite \
    --model.history_config=perceptual-framesamp-context.yaml --exp-name=...
```
(`get_config("mme_vla_suite").model.use_history=True`; the variant yaml sets
representation/integration/budget.)

## Known follow-ups (not done here)

- **Datasets**: the suite's `RoboMMEDataConfig` / `dataset_builder/` expect the
  RoboMME dataset + (for perceptual/recurrent) an offline SigLIP feature store
  built by `dataset_builder/build_robomme_dataset.py`. To train on **robocasa**
  data instead, point a `DataConfig` at a robocasa LeRobot dataset and build the
  memory feature store with the suite's builder (or use the lighter
  observation-history-as-extra-image-keys route described in the project memory).
- **GPU model build / training**: verified at import + config + inputs_spec
  level; a full JAX model instantiation + finetune is GPU/time-heavy and not run
  here.
- Vendored on git branch `robomme-memory-vla` in this openpi_rc checkout.
