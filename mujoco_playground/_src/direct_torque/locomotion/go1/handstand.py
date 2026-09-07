"""Direct-torque handstand tasks for Go1."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.direct_torque.locomotion.go1 import base
from mujoco_playground._src.locomotion.go1 import handstand as go1_handstand


def default_config() -> config_dict.ConfigDict:
  config = go1_handstand.default_config()
  config.ctrl_dt = 0.004
  config.sim_dt = 0.004
  config.episode_length = 2500
  return config


class Handstand(go1_handstand.Handstand, base.Go1TorqueEnv):
  """Perform a handstand with direct torque control."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config=config, config_overrides=config_overrides)


class Footstand(go1_handstand.Footstand, base.Go1TorqueEnv):
  """Perform a footstand with direct torque control."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config=config, config_overrides=config_overrides)
