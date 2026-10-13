"""CPU tests for the standalone DINOv3-derived store builder."""

import json
import os
from dataclasses import dataclass

import numpy as np


def test_jsonable_converts_dataclass_metadata():
    from mme_vla_suite.dataset_builder.build_dinov3_store import _jsonable

    @dataclass
    class _SizeDict:
        height: int
        width: int

    assert _jsonable({"size": _SizeDict(height=256, width=256)}) == {
        "size": {"height": 256, "width": 256}
    }


def test_dinov3_image_size_must_support_exact_local_pooling():
    from mme_vla_suite.dataset_builder.build_dinov3_store import validate_pool_geometry

    assert validate_pool_geometry(image_size=256, patch_size=16, pool_grid=(8, 8)) == (16, 16)
    try:
        validate_pool_geometry(image_size=224, patch_size=16, pool_grid=(8, 8))
    except ValueError as exc:
        assert "must divide" in str(exc)
    else:
        raise AssertionError("14x14 patches must not be accepted for an 8x8 pool grid")


class _FakeDinoExtractor:
    model_info = {
        "model_id": "fake/dinov3-vitb16",
        "revision": "test-revision",
        "feature_dim": 3,
        "tokens_per_view": 4,
        "pool_grid": [2, 2],
        "target_normalization": "per_token_layer_norm_at_loss",
    }

    def encode(self, frames):
        frames = np.asarray(frames)
        assert frames.ndim == 5
        t, v = frames.shape[:2]
        out = np.empty((t, v, 4, 3), np.float32)
        for ti in range(t):
            for vi in range(v):
                out[ti, vi] = ti * 10 + vi
        return out


def _build_source_store(root):
    source = root / "source"
    ep = source / "features" / "episode_0"
    ep.mkdir(parents=True)

    frames = np.zeros((3, 2, 8, 8, 3), np.uint8)
    np.save(ep / "frames.npy", frames)
    np.save(ep / "recent_siglip.npy", np.zeros((3, 2, 64, 2048), np.float16))
    np.save(ep / "align_prefix.npy", np.zeros((3, 128, 2048), np.float16))
    np.save(ep / "action_proprio.npy", {"action": np.zeros((3, 12), np.float32)}, allow_pickle=True)
    (ep / "chunk_index.json").write_text(json.dumps({"num_steps": 3, "num_views": 2}))

    meta = source / "meta"
    meta.mkdir()
    (meta / "samples.json").write_text("[[0, 0], [0, 1], [0, 2]]")
    return source


def test_pool_dinov3_patches_drops_prefix_tokens_and_preserves_view_grid():
    from mme_vla_suite.dataset_builder.build_dinov3_store import pool_dinov3_patches

    # One CLS + two registers + a 4x4 patch grid. The values encode the patch index.
    hidden = np.full((1, 19, 1), -999.0, np.float32)
    hidden[0, 3:, 0] = np.arange(16, dtype=np.float32)

    pooled = pool_dinov3_patches(
        hidden,
        num_register_tokens=2,
        patch_grid=(4, 4),
        pool_grid=(2, 2),
    )

    assert pooled.shape == (1, 4, 1)
    # 2x2 local means: [[0,1],[4,5]], [[2,3],[6,7]], ...
    assert np.allclose(pooled[..., 0], [[2.5, 4.5, 10.5, 12.5]])


def test_build_dinov3_store_replaces_legacy_visual_streams(tmp_path):
    from mme_vla_suite.dataset_builder.build_dinov3_store import build_derived_store

    source = _build_source_store(tmp_path)
    output = tmp_path / "dino"
    build_derived_store(
        source_store=source,
        output_store=output,
        extractor=_FakeDinoExtractor(),
        copy_mode="copy",
    )

    ep = output / "features" / "episode_0"
    dino = np.load(ep / "dinov3.npy")
    assert dino.shape == (3, 2, 4, 3)
    assert np.all(dino[2, 1] == 21.0)
    assert not (ep / "recent_siglip.npy").exists()
    assert not (ep / "align_prefix.npy").exists()
    assert np.array_equal(np.load(ep / "frames.npy"), np.load(source / "features" / "episode_0" / "frames.npy"))

    manifest = json.loads((output / "meta" / "dino_store_manifest.json").read_text())
    assert manifest["source_store"] == str(source.resolve())
    assert manifest["episodes"] == 1
    assert manifest["replacement_streams"]["recent"]["file"] == "dinov3.npy"
    assert manifest["replacement_streams"]["recent"]["shape"] == ["T", "V", 4, 3]
    assert manifest["replacement_streams"]["alignment"]["file"] == "dinov3.npy"
    assert manifest["replacement_streams"]["alignment"]["reshape"] == ["T", "V*P", "D"]
    assert manifest["extractor"] == _FakeDinoExtractor.model_info


def test_build_dinov3_store_rejects_nonempty_output(tmp_path):
    from mme_vla_suite.dataset_builder.build_dinov3_store import build_derived_store

    source = _build_source_store(tmp_path)
    output = tmp_path / "already_exists"
    output.mkdir()
    (output / "unrelated.txt").write_text("do not overwrite")

    try:
        build_derived_store(
            source_store=source,
            output_store=output,
            extractor=_FakeDinoExtractor(),
            copy_mode="copy",
        )
    except FileExistsError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("builder unexpectedly overwrote a nonempty output directory")


def test_finalize_existing_dinov3_store_writes_manifest_without_reencoding(tmp_path):
    from mme_vla_suite.dataset_builder.build_dinov3_store import (
        build_derived_store,
        finalize_existing_derived_store,
    )

    source = _build_source_store(tmp_path)
    output = tmp_path / "dino"
    build_derived_store(
        source_store=source,
        output_store=output,
        extractor=_FakeDinoExtractor(),
        copy_mode="copy",
    )
    (output / "meta" / "dino_store_manifest.json").unlink()
    before = np.load(output / "features" / "episode_0" / "dinov3.npy").copy()

    manifest = finalize_existing_derived_store(
        source_store=source,
        output_store=output,
        extractor=_FakeDinoExtractor(),
    )

    assert manifest["episodes"] == 1
    assert np.array_equal(before, np.load(output / "features" / "episode_0" / "dinov3.npy"))
    assert (output / "meta" / "dino_store_manifest.json").exists()
