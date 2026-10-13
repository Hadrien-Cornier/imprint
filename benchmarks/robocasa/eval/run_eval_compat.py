"""Launch the DINO evaluator with legacy RoboCasa scene-index compatibility."""

from __future__ import annotations

import os
import runpy
import sys
from numbers import Integral
from pathlib import Path

import gymnasium as gym
from robocasa.models.scenes import scene_registry
from robocasa.models.scenes import scene_builder
from robocasa.models.fixtures.stove import Stove
from robocasa.models.fixtures.windows import WindowProc
from robocasa.utils.errors import PlacementError, SamplingError
from robosuite.models.base import MujocoModel
from robosuite.models.tasks import task as task_module

# Robosuite validates the physical EGL index at import time, while EGL sees
# only the remapped local CUDA device when each worker is pinned to one GPU.
if local_egl_device := os.environ.get("MME_EGL_LOCAL_DEVICE_ID"):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = local_egl_device


class LegacyFramedWindow(WindowProc):
    """Accept the retired scene-layout ``ofs`` field during archived replay."""

    def __init__(self, *args, ofs=None, **kwargs):
        del ofs
        super().__init__(*args, **kwargs)


# RoboCasa 1.0.1 retains the legacy framed_window fixture registry and the
# compatible procedural implementation, but its scene-builder alias was
# removed. Archived PanTransfer layouts still use the alias and ``ofs``.
scene_builder.FIXTURES.setdefault("framed_window", LegacyFramedWindow)


_get_knobs_state = Stove.get_knobs_state
_update_stove_state = Stove.update_state


def mask_missing_stove_entries(self, env):
    available_joints = set(env.sim.model.joint_names)
    available_sites = set(env.sim.model.site_names)
    missing_joints = {}
    missing_sites = {}
    for location, joint in self.knob_joints.items():
        expected_joint = f"{self.naming_prefix}knob_{location}_joint"
        if joint is not None and expected_joint not in available_joints:
            missing_joints[location] = joint
            self.knob_joints[location] = None
    for location, site in self.burner_sites.items():
        expected_site = f"{self.naming_prefix}burner_on_{location}"
        if site is not None and expected_site not in available_sites:
            missing_sites[location] = site
            self.burner_sites[location] = None
    return missing_joints, missing_sites


def restore_stove_entries(self, missing_joints, missing_sites):
    self.knob_joints.update(missing_joints)
    self.burner_sites.update(missing_sites)


def get_existing_knobs_state(self, env):
    """Ignore stale fixture knob entries absent from the recorded XML."""
    missing_joints, missing_sites = mask_missing_stove_entries(self, env)
    try:
        return _get_knobs_state(self, env)
    finally:
        restore_stove_entries(self, missing_joints, missing_sites)


def update_existing_stove_state(self, env):
    """Update only burner sites represented by the recorded XML."""
    missing_joints, missing_sites = mask_missing_stove_entries(self, env)
    try:
        return _update_stove_state(self, env)
    finally:
        restore_stove_entries(self, missing_joints, missing_sites)


Stove.get_knobs_state = get_existing_knobs_state
Stove.update_state = update_existing_stove_state


def _legacy_index(value, mapping):
    if isinstance(value, Integral) and int(value) in mapping:
        return mapping[int(value)]
    return value


_get_layout_path = scene_registry.get_layout_path
_get_style_path = scene_registry.get_style_path


def get_layout_path_with_legacy_indices(layout_id):
    return _get_layout_path(
        _legacy_index(layout_id, scene_registry.LEGACY_LAYOUT_BY_OLD_INDEX)
    )


def get_style_path_with_legacy_indices(style_id):
    return _get_style_path(
        _legacy_index(style_id, scene_registry.LEGACY_STYLE_BY_OLD_INDEX)
    )


if os.environ.get("MME_EVALUATOR_LEGACY_SCENE_INDICES", "1") == "1":
    scene_registry.get_layout_path = get_layout_path_with_legacy_indices
    scene_registry.get_style_path = get_style_path_with_legacy_indices

    # Several RoboCasa scene modules import these helpers by value during
    # package initialization, so update existing references as well.
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "get_layout_path", None) is _get_layout_path:
            module.get_layout_path = get_layout_path_with_legacy_indices
        if getattr(module, "get_style_path", None) is _get_style_path:
            module.get_style_path = get_style_path_with_legacy_indices


_gym_make = gym.make


def make_with_transient_scene_retry(*args, **kwargs):
    """Build a disposable RoboCasa env even when its sampled layout is invalid.

    The evaluator immediately resets each rollout to its recorded XML and
    state. This retry only chooses a constructible temporary scene before that
    reset, so it cannot alter the evaluated layout.
    """
    env_id = str(args[0]) if args else str(kwargs.get("id", ""))
    if not env_id.startswith("robocasa/"):
        return _gym_make(*args, **kwargs)

    base_seed = int(kwargs.get("seed", 0))
    last_error = None
    for offset in range(32):
        attempt_kwargs = dict(kwargs)
        attempt_kwargs["seed"] = base_seed + offset
        attempt_kwargs["render_gpu_device_id"] = 0
        try:
            return _gym_make(*args, **attempt_kwargs)
        except (
            PlacementError,
            SamplingError,
            ValueError,
            AttributeError,
        ) as error:
            last_error = error
            print(
                "Transient RoboCasa scene construction failed "
                f"for {env_id} with seed={base_seed + offset}: {error}",
                flush=True,
            )
    raise RuntimeError(
        f"Unable to construct a transient {env_id} scene after 32 seeds"
    ) from last_error


gym.make = make_with_transient_scene_retry


_get_ids = task_module.get_ids


def get_existing_ids(sim, elements, element_type="geom", inplace=False):
    """Ignore transient-scene names absent from the recorded XML."""
    values = elements if isinstance(elements, list) else [elements]
    ids = []
    for value in values:
        try:
            ids.extend(_get_ids(sim, [value], element_type=element_type))
        except ValueError:
            print(
                f"Skipping stale {element_type} mapping from disposable scene: "
                f"{value}",
                flush=True,
            )
    if inplace and isinstance(elements, list):
        elements[:] = ids
    return ids


task_module.get_ids = get_existing_ids

_generate_id_mappings = task_module.Task.generate_id_mappings


def generate_recorded_scene_id_mappings(self, sim):
    try:
        return _generate_id_mappings(self, sim)
    except ValueError as error:
        # Segmentation mappings are not consumed by these RGB policy rollouts.
        print(f"Skipping stale disposable-scene ID mapping: {error}", flush=True)
        self._instances_to_ids = {}
        self._geom_ids_to_instances = {}
        self._site_ids_to_instances = {}
        self._classes_to_ids = {}
        self._geom_ids_to_classes = {}
        self._site_ids_to_classes = {}


task_module.Task.generate_id_mappings = generate_recorded_scene_id_mappings


def set_recorded_scene_sites_visibility(self, sim, visible):
    """Toggle only visualization sites present in the recorded model."""
    for site_name in self.sites:
        try:
            site_id = sim.model.site_name2id(site_name)
        except ValueError:
            continue
        alpha = sim.model.site_rgba[site_id][3]
        if (visible and alpha < 0) or (not visible and alpha > 0):
            sim.model.site_rgba[site_id][3] = -alpha


MujocoModel.set_sites_visibility = set_recorded_scene_sites_visibility

evaluator_script = os.environ["MME_EVALUATOR_SCRIPT"]
if os.environ.get("MME_EVALUATOR_SKIP_DISPOSABLE_RESET") == "1":
    source = Path(evaluator_script).read_text()
    source = source.replace(
        "    raw_env.reset()\n    model_xml = raw_env.edit_model_xml(demo.model_file)",
        "    # The archived XML below constructs the actual evaluation scene.\n"
        "    model_xml = raw_env.edit_model_xml(demo.model_file)",
        1,
    )
    exec(
        compile(source, evaluator_script, "exec"),
        {"__name__": "__main__", "__file__": evaluator_script},
    )
else:
    runpy.run_path(evaluator_script, run_name="__main__")
