# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Defines Unitree Go1 quadruped constants."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go1"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)

FEET_ONLY_FLAT_TERRAIN_TORQUE_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain_torque.xml"
)
FEET_ONLY_ROUGH_TERRAIN_TORQUE_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain_torque.xml"
)
FULL_FLAT_TERRAIN_TORQUE_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_flat_terrain_torque.xml"
)
FULL_COLLISIONS_FLAT_TERRAIN_TORQUE_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain_torque.xml"
)

_TORQUE_XML_BY_POSITION_XML = {
    str(FEET_ONLY_FLAT_TERRAIN_XML): FEET_ONLY_FLAT_TERRAIN_TORQUE_XML,
    str(FEET_ONLY_ROUGH_TERRAIN_XML): FEET_ONLY_ROUGH_TERRAIN_TORQUE_XML,
    str(FULL_FLAT_TERRAIN_XML): FULL_FLAT_TERRAIN_TORQUE_XML,
    str(FULL_COLLISIONS_FLAT_TERRAIN_XML): (
        FULL_COLLISIONS_FLAT_TERRAIN_TORQUE_XML
    ),
}


def xml_for_control_mode(
    xml_path: str | epath.Path, control_mode: str
) -> epath.Path:
  """Returns the scene XML for a Go1 control mode."""
  if control_mode == "position":
    return epath.Path(xml_path)
  if control_mode != "torque":
    raise ValueError(
        f"Unsupported Go1 control mode: {control_mode!r}. "
        "Expected 'position' or 'torque'."
    )

  try:
    return _TORQUE_XML_BY_POSITION_XML[str(xml_path)]
  except KeyError as e:
    raise ValueError(
        f"Torque control is not supported for Go1 XML {xml_path!r}."
    ) from e


def task_to_xml(task_name: str, control_mode: str = "position") -> epath.Path:
  position_xml = {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]
  return xml_for_control_mode(position_xml, control_mode)


FEET_SITES = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "trunk"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
