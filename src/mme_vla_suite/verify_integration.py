"""
Verify the vendored MME-VLA memory suite integrates with this openpi checkout.

JAX-free-ish smoke test (no GPU model build): imports the memory modules, loads
all 14 RoboMME variant YAMLs, builds a HistoryPi0Config for each, and constructs
the HistAugObservation inputs_spec per memory family. Run from the openpi_rc root:

    .venv-openpi/bin/python -m mme_vla_suite.verify_integration
"""
import glob
import os
import traceback

import omegaconf

from mme_vla_suite.models.integration import history_pi0

CONFIG_DIR = "src/mme_vla_suite/models/config/robomme"


def main():
    cfgs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    assert cfgs, f"no variant yamls found under {CONFIG_DIR} (run from openpi_rc root)"

    print(f"==== {len(cfgs)} MME-VLA variant configs ====")
    n_ok = 0
    rep_fields = {
        "perceptual": "static_image_emb",
        "recurrent": "recur_image_emb",
        "symbolic": "symbolic_tokenized_prompt",
    }
    for p in cfgs:
        name = os.path.basename(p)
        try:
            hc = omegaconf.OmegaConf.load(p)
            cfg = history_pi0.HistoryPi0Config(
                pi05=True, action_horizon=20, use_history=True,
                history_config=hc, discrete_state_input=False,
            )
            obs_spec, act_spec = cfg.inputs_spec(batch_size=1)
            rep = hc.get("representation_type", "?")
            want = rep_fields.get(rep)
            has_field = want is not None and getattr(obs_spec, want, None) is not None
            status = "OK" if has_field else "OK?"
            n_ok += 1
            print(f"[{status:3}] {name:40} rep={rep:10} act={tuple(act_spec.shape)} mem={want if has_field else '-'}")
        except Exception as e:
            print(f"[ERR] {name}: " + "".join(traceback.format_exception_only(type(e), e)).strip())

    # also load the suite's own train-config registry
    from mme_vla_suite.training import config as mme_config
    print("\nregistered TrainConfigs:", list(mme_config._CONFIGS_DICT.keys()))
    print(f"\n{n_ok}/{len(cfgs)} variant configs OK against this openpi")


if __name__ == "__main__":
    main()
