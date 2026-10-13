"""Skeleton smoke test: the Stage-1 trainer imports and exposes main()."""
import importlib


def test_pretrain_bottleneck_importable():
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    assert hasattr(mod, "main") and callable(mod.main)
    assert hasattr(mod, "parse_args")


def test_pretrain_bottleneck_help():
    # argparse builds without touching the (unbuilt) store.
    mod = importlib.import_module("mme_vla_suite.scripts.pretrain_bottleneck")
    args = mod.parse_args(["--config", "x.yaml"])
    assert args.config == "x.yaml"
