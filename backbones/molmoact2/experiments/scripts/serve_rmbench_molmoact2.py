"""Official RMBench policy-model entrypoint for online MolmoAct2 memory rollouts."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rmbench_molmoact2_adapter import (  # noqa: E402
    RMBenchMolmoAct2ServerModel,
    eval,
    reset_model,
)


def _load_memory_module():
    module_name = "_rmbench_molmoact2_memory"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    module_path = _SCRIPT_DIR.parent / "lerobot" / "src" / "lerobot" / "policies" / "molmoact2" / "rmbench_memory.py"
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RMBench memory module from {module_path}.")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_memory = _load_memory_module()
CAMERA_ORDER = _memory.CAMERA_ORDER
DINO_MODEL_ID = _memory.DINO_MODEL_ID
DINO_MODEL_REVISION = _memory.DINO_MODEL_REVISION
DinoV3RecentEncoder = _memory.DinoV3RecentEncoder
RMBenchLanguageEmbeddingResolver = _memory.RMBenchLanguageEmbeddingResolver
RMBenchMemoryLayout = _memory.RMBenchMemoryLayout
RMBenchOnlineMemoryConditioner = _memory.RMBenchOnlineMemoryConditioner
build_wan_frontend = _memory.build_wan_frontend


log = logging.getLogger(__name__)
_PREDICTION_HORIZON = 50
_WAN_FRONTEND_VARIANT = "Wan2.2-TI2V-5B"
_MEMORY_LAYOUT_FIELDS = (
    "temporal_factor",
    "downsample",
    "latent_cap",
    "max_recent_tokens",
    "max_step_tokens",
    "latent_dim",
    "recent_dim",
    "stream_dim",
    "num_views",
    "pool_h",
    "pool_w",
    "wan_h",
    "wan_w",
)


@dataclass(frozen=True)
class ServerConfig:
    checkpoint: str
    stage2_root: str | None = None
    feature_root: str | None = None
    dino_model_path: str | None = None
    memory_enabled: bool = True
    device: str = "cuda"
    norm_tag: str = "rmbench_stage2"
    prediction_horizon: int = _PREDICTION_HORIZON
    flow_seed: int | None = None
    stage1_source_root: str | None = None
    offline_feature_root: str | None = None

    def validate(self) -> None:
        if not isinstance(self.memory_enabled, bool):
            raise TypeError("memory_enabled must be a boolean")
        required_directories = {"checkpoint": self.checkpoint}
        if self.memory_enabled:
            required_directories.update(
                {
                    "stage2_root": self.stage2_root,
                    "feature_root": self.feature_root,
                    "dino_model_path": self.dino_model_path,
                }
            )
        for identifier, value in required_directories.items():
            if not isinstance(value, (str, Path)):
                raise TypeError(
                    f"{identifier} must identify a directory, got {value!r}."
                )
            path = Path(value).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(
                    f"{identifier} must identify an existing directory, got {value!r}."
                )
        if (
            isinstance(self.prediction_horizon, bool)
            or not isinstance(self.prediction_horizon, int)
            or self.prediction_horizon != _PREDICTION_HORIZON
        ):
            raise ValueError(
                f"prediction_horizon must be {_PREDICTION_HORIZON} for RMBench, "
                f"got {self.prediction_horizon!r}."
            )
        if self.flow_seed is not None and (
            isinstance(self.flow_seed, bool)
            or not isinstance(self.flow_seed, int)
            or self.flow_seed < 0
        ):
            raise ValueError(
                "flow_seed must be a non-negative integer or null, "
                f"got {self.flow_seed!r}."
            )
        if self.offline_feature_root is not None and str(self.offline_feature_root).strip():
            raise ValueError(
                "offline visual features are forbidden for RMBench online memory rollouts."
            )
        if self.memory_enabled and self.stage1_source_root is not None:
            stage1_path = Path(self.stage1_source_root).expanduser()
            if not stage1_path.is_dir():
                raise FileNotFoundError(
                    "stage1_source_root must identify an existing directory, "
                    f"got {self.stage1_source_root!r}."
                )

    def runtime_provenance(self) -> dict[str, object]:
        provenance: dict[str, object] = {
            "checkpoint": str(Path(self.checkpoint).expanduser()),
            "camera_order": list(CAMERA_ORDER),
            "prediction_horizon": self.prediction_horizon,
            "flow_seed": self.flow_seed,
            "memory_enabled": self.memory_enabled,
            "visual_history": (
                "online_wan_and_dinov3" if self.memory_enabled else "none"
            ),
            "offline_visual_features": False,
        }
        if not self.memory_enabled:
            provenance["language_source"] = "molmoact2_prompt"
            return provenance
        provenance.update(
            {
                "stage2_metadata_root": str(Path(self.stage2_root).expanduser()),
                "language_feature_root": str(Path(self.feature_root).expanduser()),
                "dino_model_id": DINO_MODEL_ID,
                "dino_model_revision": DINO_MODEL_REVISION,
                "dino_model_path": str(Path(self.dino_model_path).expanduser()),
                "language_source": "stored_pi05_embeddings",
                "wan_frontend_variant": _WAN_FRONTEND_VARIANT,
            }
        )
        if self.stage1_source_root is not None:
            source_root = Path(self.stage1_source_root).expanduser()
            provenance["stage1_source_root"] = str(source_root)
            provenance["wan_stage1_source_root"] = str(source_root)
            provenance["wan_resolved_source_path"] = str(source_root.resolve())
            source_version = _wan_source_version(source_root)
            if source_version is not None:
                provenance["wan_source_version"] = source_version
            module_file = (
                source_root
                / "mme_vla_suite"
                / "models"
                / "representation"
                / "wan_frontend.py"
            )
            if module_file.is_file():
                provenance["wan_frontend_module_file"] = str(module_file.resolve())
        return provenance


def _wan_source_version(source_root: Path) -> str | None:
    pyproject_path = source_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return None
    try:
        import tomllib

        with pyproject_path.open("rb") as file:
            project = tomllib.load(file).get("project")
    except (ImportError, OSError, ValueError):
        return None
    if not isinstance(project, Mapping):
        return None
    version = project.get("version")
    return version if isinstance(version, str) and version else None


def load_stage2_memory_layout(stage2_root: str | Path) -> RMBenchMemoryLayout:
    """Load the exact live-memory layout staged with the Stage-2 dataset."""
    layout_path = Path(stage2_root).expanduser() / "meta" / "memory_layout.json"
    if not layout_path.is_file():
        raise FileNotFoundError(
            "RMBench Stage-2 memory layout metadata is required at "
            f"{layout_path}."
        )
    try:
        with layout_path.open(encoding="utf-8") as file:
            raw_layout = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"RMBench Stage-2 memory layout metadata at {layout_path} is invalid JSON."
        ) from exc
    if not isinstance(raw_layout, Mapping):
        raise ValueError(
            f"RMBench Stage-2 memory layout metadata at {layout_path} must be a JSON object."
        )

    actual_fields = set(raw_layout)
    expected_fields = set(_MEMORY_LAYOUT_FIELDS)
    missing_fields = sorted(expected_fields - actual_fields)
    unknown_fields = sorted(actual_fields - expected_fields)
    if missing_fields or unknown_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields {missing_fields}")
        if unknown_fields:
            details.append(f"unknown fields {unknown_fields}")
        raise ValueError(
            "RMBench Stage-2 memory layout metadata at "
            f"{layout_path} must contain exactly {_MEMORY_LAYOUT_FIELDS}; "
            + "; ".join(details)
            + "."
        )
    try:
        return RMBenchMemoryLayout(
            **{field: raw_layout[field] for field in _MEMORY_LAYOUT_FIELDS}
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"RMBench Stage-2 memory layout metadata at {layout_path} is incoherent: {exc}"
        ) from exc


def _value_from_config(source: object, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if isinstance(source, Mapping):
            if name in source and source[name] is not None:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return None


def _memory_config_from_policy(policy: object) -> object:
    handles = getattr(policy, "_handles", None)
    model = getattr(handles, "model", None)
    checkpoint_config = getattr(model, "config", None)
    memory_config = getattr(checkpoint_config, "memory", None)
    if memory_config is None and isinstance(checkpoint_config, Mapping):
        memory_config = checkpoint_config.get("memory")
    if memory_config is None:
        raise ValueError(
            "MolmoAct2 checkpoint does not expose its memory configuration; "
            "cannot construct an RMBench online-memory layout."
        )
    return memory_config


def _validate_checkpoint_memory_layout(
    layout: RMBenchMemoryLayout,
    policy: object,
) -> None:
    """Require the runtime layout to agree with the checkpoint's memory widths."""
    memory_config = _memory_config_from_policy(policy)
    enabled = _value_from_config(memory_config, ("enabled",))
    if enabled is not True:
        raise ValueError(
            "MolmoAct2 checkpoint memory.enabled must be true for RMBench Stage-2 serving."
        )

    comparable_dimensions = {
        "cosmos_latent_dim": layout.latent_dim,
        "recent_dim": layout.recent_dim,
        "action_dim": layout.stream_dim,
        "proprio_dim": layout.stream_dim,
    }
    for field_name, expected_value in comparable_dimensions.items():
        value = _value_from_config(memory_config, (field_name,))
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "MolmoAct2 checkpoint memory configuration field "
                f"{field_name!r} must be an integer, got {value!r}."
            )
        if value != expected_value:
            raise ValueError(
                "RMBench Stage-2 memory layout disagrees with MolmoAct2 checkpoint "
                f"memory.{field_name}: metadata={expected_value}, checkpoint={value}."
            )

    language_dim = _value_from_config(memory_config, ("language_dim",))
    if language_dim is not None and language_dim != 2048:
        raise ValueError(
            "MolmoAct2 checkpoint memory language_dim must be 2048 for saved "
            f"Pi0.5 embeddings, got {language_dim!r}."
        )


def _prepend_stage1_source(stage1_source_root: str | None) -> None:
    if stage1_source_root is None:
        return
    stage1_path = str(Path(stage1_source_root).expanduser())
    if stage1_path in sys.path:
        sys.path.remove(stage1_path)
    sys.path.insert(0, stage1_path)


def build_server_model(config: ServerConfig) -> RMBenchMolmoAct2ServerModel:
    config.validate()
    if config.memory_enabled:
        _prepend_stage1_source(config.stage1_source_root)

    # Keep configuration-only imports free of torch and checkpoint dependencies.
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    policy_config = MolmoAct2Config(
        checkpoint_path=config.checkpoint,
        device=config.device,
        memory_enabled=config.memory_enabled,
        memory_mode="frozen",
        norm_tag=config.norm_tag,
    )
    policy = MolmoAct2Policy(policy_config)
    policy.eval()

    if not config.memory_enabled:
        return RMBenchMolmoAct2ServerModel(
            policy,
            memory_enabled=False,
            prediction_horizon=config.prediction_horizon,
            norm_tag=config.norm_tag,
            flow_seed=config.flow_seed,
        )

    layout = load_stage2_memory_layout(config.stage2_root)
    _validate_checkpoint_memory_layout(layout, policy)
    language_resolver = RMBenchLanguageEmbeddingResolver.build(
        Path(config.stage2_root),
        Path(config.feature_root),
    )
    dino_encoder = DinoV3RecentEncoder(config.dino_model_path, device=config.device)
    wan_frontend = build_wan_frontend(
        config.device,
        variant=_WAN_FRONTEND_VARIANT,
    )

    def conditioner_factory() -> RMBenchOnlineMemoryConditioner:
        return RMBenchOnlineMemoryConditioner(
            layout=layout,
            dino_encoder=dino_encoder,
            wan_frontend=wan_frontend,
            offline_feature_root=config.offline_feature_root,
        )

    return RMBenchMolmoAct2ServerModel(
        policy,
        memory_enabled=True,
        language_resolver=language_resolver,
        conditioner_factory=conditioner_factory,
        prediction_horizon=config.prediction_horizon,
        norm_tag=config.norm_tag,
        flow_seed=config.flow_seed,
    )


def _required_string(usr_args: Mapping[str, object], identifier: str) -> str:
    try:
        value = usr_args[identifier]
    except KeyError as exc:
        raise KeyError(f"RMBench server requires argument {identifier!r}.") from exc
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"RMBench server argument {identifier!r} must be a non-empty string, "
            f"got {value!r}."
        )
    return value


def get_model(usr_args: Mapping[str, object]) -> RMBenchMolmoAct2ServerModel:
    if not isinstance(usr_args, Mapping):
        raise TypeError(
            "RMBench server get_model expects a mapping of user arguments, got "
            f"{type(usr_args).__name__}."
        )
    memory_enabled = usr_args.get("memory_enabled", True)
    if not isinstance(memory_enabled, bool):
        raise TypeError(
            "RMBench server argument 'memory_enabled' must be a boolean, "
            f"got {memory_enabled!r}."
        )
    config = ServerConfig(
        checkpoint=_required_string(usr_args, "checkpoint"),
        stage2_root=(
            _required_string(usr_args, "stage2_root") if memory_enabled else None
        ),
        feature_root=(
            _required_string(usr_args, "feature_root") if memory_enabled else None
        ),
        dino_model_path=(
            _required_string(usr_args, "dino_model_path") if memory_enabled else None
        ),
        stage1_source_root=(
            _required_string(usr_args, "stage1_source_root")
            if memory_enabled
            else None
        ),
        memory_enabled=memory_enabled,
        device=str(usr_args.get("device", "cuda")),
        norm_tag=str(usr_args.get("norm_tag", "rmbench_stage2")),
        prediction_horizon=usr_args.get("prediction_horizon", _PREDICTION_HORIZON),
        flow_seed=usr_args.get("flow_seed"),
        offline_feature_root=usr_args.get("offline_feature_root"),
    )
    config.validate()
    log.info("RMBench MolmoAct2 runtime provenance: %s", json.dumps(config.runtime_provenance(), sort_keys=True))
    return build_server_model(config)
