"""Direct-torque fall recovery task for Go1."""

from typing import Any, Dict, Optional, Union

from ml_collections import config_dict

from mujoco_playground._src.direct_torque.locomotion.go1 import base
from mujoco_playground._src.locomotion.go1 import getup as go1_getup


def default_config() -> config_dict.ConfigDict:
  config = go1_getup.default_config()
  config.ctrl_dt = 0.004
  config.sim_dt = 0.004
  config.episode_length = 1500
  config.reward_config.scales.action_rate = -0.00001
  return config


class Getup(go1_getup.Getup, base.Go1TorqueEnv):
  """Recover from a fall with direct torque control."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config=config, config_overrides=config_overrides)

  def _post_init(self) -> None:
    super()._post_init()
    self._settle_steps = 0
