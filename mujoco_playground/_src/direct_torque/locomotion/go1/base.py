"""Base class for direct-torque Go1 environments."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.direct_torque.locomotion.go1 import go1_constants as consts
from mujoco_playground._src.locomotion.go1 import base as go1_base


def get_assets() -> Dict[str, bytes]:
  """Loads direct-torque XMLs alongside the original Go1 assets."""
  assets = go1_base.get_assets()
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  return assets


class Go1TorqueEnv(go1_base.Go1Env):
  """Go1 base environment with motor actuators driven by direct torque."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(
        xml_path=consts.torque_xml_path(xml_path),
        config=config,
        config_overrides=config_overrides,
    )

  def _get_assets(self) -> Dict[str, bytes]:
    return get_assets()

  def _configure_model(self) -> None:
    """Leaves motor and joint parameters at their XML-defined values."""

  def _get_control(
      self, position_target: jax.Array, action: jax.Array
  ) -> jax.Array:
    del position_target
    scaled_action = action * self._config.action_scale
    lower_limits, upper_limits = self.mjx_model.actuator_forcerange.T
    return jp.where(
        scaled_action >= 0,
        scaled_action * upper_limits,
        scaled_action * (-lower_limits),
    )

  def _get_reset_control(self, position_target: jax.Array) -> jax.Array:
    del position_target
    return jp.zeros(self.mjx_model.nu)
