# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "MolmoAct2Config",
    "PI0Config",
    "PI05Config",
    "PI0FastConfig",
    "SmolVLAConfig",
    "SARMConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "XVLAConfig",
    "WallXConfig",
]

_LAZY_EXPORTS = {
    "ACTConfig": (".act.configuration_act", "ACTConfig"),
    "DiffusionConfig": (".diffusion.configuration_diffusion", "DiffusionConfig"),
    "GrootConfig": (".groot.configuration_groot", "GrootConfig"),
    "MolmoAct2Config": (".molmoact2.configuration_molmoact2", "MolmoAct2Config"),
    "PI0Config": (".pi0.configuration_pi0", "PI0Config"),
    "PI0FastConfig": (".pi0_fast.configuration_pi0_fast", "PI0FastConfig"),
    "PI05Config": (".pi05.configuration_pi05", "PI05Config"),
    "SmolVLAConfig": (".smolvla.configuration_smolvla", "SmolVLAConfig"),
    "SmolVLANewLineProcessor": (".smolvla.processor_smolvla", "SmolVLANewLineProcessor"),
    "SARMConfig": (".sarm.configuration_sarm", "SARMConfig"),
    "TDMPCConfig": (".tdmpc.configuration_tdmpc", "TDMPCConfig"),
    "VQBeTConfig": (".vqbet.configuration_vqbet", "VQBeTConfig"),
    "WallXConfig": (".wall_x.configuration_wall_x", "WallXConfig"),
    "XVLAConfig": (".xvla.configuration_xvla", "XVLAConfig"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
