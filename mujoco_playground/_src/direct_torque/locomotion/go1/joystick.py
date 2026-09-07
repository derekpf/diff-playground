"""Direct-torque joystick tasks for Go1."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.direct_torque.locomotion.go1 import base
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick


def default_config() -> config_dict.ConfigDict:
  config = go1_joystick.default_config()
  config.ctrl_dt = 0.004
  config.sim_dt = 0.004
  config.episode_length = 5000
  config.reward_config.scales.action_rate = -0.0001
  return config


class Joystick(go1_joystick.Joystick, base.Go1TorqueEnv):
  """Track a joystick command with direct torque control."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        task=task,
        config=config,
        config_overrides=config_overrides,
    )
