"""Constants for the direct-torque Go1 environments."""

from etils import epath

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants

ROOT_PATH = mjx_env.ROOT_PATH / "direct_torque" / "locomotion" / "go1"

FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain_torque.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain_torque.xml"
)
FULL_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_flat_terrain_torque.xml"
)
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain_torque.xml"
)

_TORQUE_XML_BY_POSITION_XML = {
    go1_constants.FEET_ONLY_FLAT_TERRAIN_XML.name: FEET_ONLY_FLAT_TERRAIN_XML,
    go1_constants.FEET_ONLY_ROUGH_TERRAIN_XML.name: FEET_ONLY_ROUGH_TERRAIN_XML,
    go1_constants.FULL_FLAT_TERRAIN_XML.name: FULL_FLAT_TERRAIN_XML,
    go1_constants.FULL_COLLISIONS_FLAT_TERRAIN_XML.name: (
        FULL_COLLISIONS_FLAT_TERRAIN_XML
    ),
}


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


def torque_xml_path(xml_path: str | epath.Path) -> epath.Path:
  """Returns the direct-torque scene corresponding to a Go1 scene."""
  xml_path = epath.Path(xml_path)
  if xml_path.name.endswith("_torque.xml"):
    return xml_path
  try:
    return _TORQUE_XML_BY_POSITION_XML[xml_path.name]
  except KeyError as error:
    raise ValueError(
        f"No direct-torque Go1 scene exists for XML {xml_path!r}."
    ) from error
