import dataclasses

import flax.traverse_util as traverse_util
import numpy as np

import openpi.models.model as model
import openpi.shared.array_typing as at
from openpi.training.weight_loaders import WeightLoader
from openpi.training.weight_loaders import download

_PI0_REQUIRED_ROOTS = frozenset(
    {
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    }
)
_PI05_FORBIDDEN_ROOTS = frozenset({"time_mlp_in", "time_mlp_out"})
_MEMORY_PARAM_PARTS = frozenset(
    {
        "mem_encoder",
        "aux_decoder",
        "mem_attn",
        "mem_rms_norm",
        "mem_rms_norm_ffn",
        "q_einsum_mem",
        "kv_einsum_mem",
        "out_einsum_mem",
    }
)


def _has_path_part(path: tuple[object, ...], names: frozenset[str]) -> bool:
    return any(isinstance(part, str) and part in names for part in path)


def _is_memory_parameter(path: tuple[object, ...]) -> bool:
    return _has_path_part(path, _MEMORY_PARAM_PARTS)


def validate_native_pi0_parameter_trees(
    reference: dict,
    loaded: dict,
) -> None:
    """Reject a wrong base checkpoint or an incomplete native pi0 load."""
    flat_reference = traverse_util.flatten_dict(reference)
    flat_loaded = traverse_util.flatten_dict(loaded)

    reference_roots = {str(path[0]) for path in flat_reference if path}
    loaded_roots = {str(path[0]) for path in flat_loaded if path}

    missing_roots = sorted(_PI0_REQUIRED_ROOTS - loaded_roots)
    if missing_roots:
        raise ValueError(f"The base checkpoint is not a complete pi0 checkpoint; missing roots: {missing_roots}.")

    forbidden_roots = sorted(_PI05_FORBIDDEN_ROOTS & loaded_roots)
    if forbidden_roots:
        raise ValueError(f"A pi0.5 checkpoint was supplied to the native pi0 loader; found roots: {forbidden_roots}.")

    model_missing_roots = sorted(_PI0_REQUIRED_ROOTS - reference_roots)
    if model_missing_roots:
        raise ValueError(f"The target model is not native pi0; its parameter tree is missing: {model_missing_roots}.")

    missing_base = sorted(
        "/".join(map(str, path))
        for path in flat_reference
        if not _is_memory_parameter(path) and path not in flat_loaded
    )
    if missing_base:
        preview = ", ".join(missing_base[:12])
        raise KeyError(f"pi0_base is missing {len(missing_base)} non-memory model parameters: {preview}")

    shape_mismatches = []
    for path, reference_value in flat_reference.items():
        if _is_memory_parameter(path) or path not in flat_loaded:
            continue
        loaded_value = flat_loaded[path]
        if loaded_value.shape != reference_value.shape:
            shape_mismatches.append(f"{'/'.join(map(str, path))}: {loaded_value.shape} != {reference_value.shape}")
    if shape_mismatches:
        raise ValueError("pi0_base parameter shapes do not match the target model: " + "; ".join(shape_mismatches[:12]))


@dataclasses.dataclass(frozen=True)
class NativePi0Stage1BottleneckWeightLoader(WeightLoader):
    """Load pi0_base and overlay a VLA-independent Stage-1 memory encoder."""

    params_path: str
    stage1_ckpt: str
    strict: bool = True

    def load(self, params: at.Params) -> at.Params:
        loaded = model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        if self.strict:
            validate_native_pi0_parameter_trees(params, loaded)

        flat_reference = traverse_util.flatten_dict(params)
        flat_loaded = traverse_util.flatten_dict(loaded)
        merged = {}
        for path, reference_value in flat_reference.items():
            if path in flat_loaded:
                value = flat_loaded[path]
                merged[path] = value.astype(reference_value.dtype) if value.dtype != reference_value.dtype else value
            else:
                merged[path] = reference_value

        checkpoint_path = download.maybe_download(self.stage1_ckpt)
        stage1 = np.load(checkpoint_path, allow_pickle=True)

        def stage1_target(key: str) -> tuple[object, ...]:
            parts = key.split("/")
            parts[0] = "mem_encoder"
            return tuple(int(part) if part.isdigit() else part for part in parts)

        transferred = 0
        discarded = 0
        for key in stage1.files:
            if not key.startswith("bottleneck/"):
                discarded += 1
                continue
            target = stage1_target(key)
            if target not in merged:
                if self.strict:
                    raise KeyError(f"Stage-1 parameter {key!r} maps to absent model path {target!r}.")
                continue
            value = stage1[key]
            if value.shape != merged[target].shape:
                raise ValueError(f"Stage-1 shape mismatch for {key!r}: {value.shape} != {merged[target].shape}.")
            merged[target] = value.astype(merged[target].dtype) if value.dtype != merged[target].dtype else value
            transferred += 1

        memory_keys = [path for path in merged if path and path[0] == "mem_encoder"]
        if self.strict and transferred != len(memory_keys):
            raise KeyError(
                "Stage-1 transfer did not initialize the full memory encoder: "
                f"{transferred}/{len(memory_keys)} parameters."
            )

        print(
            "[NativePi0Stage1BottleneckWeightLoader] loaded strict pi0_base and "
            f"transferred {transferred} Stage-1 memory parameters "
            f"(discarded {discarded} decoder/non-bottleneck parameters)."
        )
        return traverse_util.unflatten_dict(merged)
