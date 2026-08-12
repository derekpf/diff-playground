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
"""Base classes for Go1."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "unitree_go1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class Go1Env(mjx_env.MjxEnv):
  """Base class for Go1 environments."""

  _POSITION_CONTROL = "position"
  _TORQUE_CONTROL = "torque"
  _POSITION_CTRL_DT = 0.02
  _TORQUE_CTRL_DT = 0.002
  _TORQUE_SIM_DT = 0.002

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    config_overrides = config_overrides or {}
    position_episode_length = config.episode_length
    super().__init__(
        config.copy_and_resolve_references(), config_overrides
    )

    self._control_mode = self._config.control_mode
    if self._control_mode not in (self._POSITION_CONTROL, self._TORQUE_CONTROL):
      raise ValueError(
          f"Unsupported Go1 control mode: {self._control_mode!r}. "
          "Expected 'position' or 'torque'."
      )

    # The position-mode defaults are retained in the task configs so existing
    # callers are unchanged. Torque mode has a smaller default control period,
    # while an explicit override always takes precedence.
    ctrl_dt = self._config.ctrl_dt
    sim_dt = self._config.sim_dt
    if self._control_mode == self._TORQUE_CONTROL:
      if "ctrl_dt" not in config_overrides and ctrl_dt == 0.02:
        ctrl_dt = self._TORQUE_CTRL_DT
      if "sim_dt" not in config_overrides and sim_dt == 0.004:
        sim_dt = self._TORQUE_SIM_DT
    self._ctrl_dt = ctrl_dt
    self._sim_dt = sim_dt
    if "episode_length" in config_overrides:
      self._episode_length = self._config.episode_length
    elif self._control_mode == self._TORQUE_CONTROL:
      self._episode_length = round(
          position_episode_length * self._POSITION_CTRL_DT / self.dt
      )
    else:
      self._episode_length = self._config.episode_length

    xml_path = consts.xml_for_control_mode(xml_path, self._control_mode)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mj_model.opt.ccd_iterations = 20

    # Keep physical joint damping in both modes. Position actuators get the
    # task's configured PD gains; torque motors retain their XML gain/bias.
    self._mj_model.dof_damping[6:] = self._config.Kd
    if self._control_mode == self._POSITION_CONTROL:
      self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
      self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp

    if self._mj_model.nu != 12:
      raise ValueError(
          "Go1 control modes require a model with exactly 12 actuators; "
          f"got {self._mj_model.nu} for {xml_path}."
      )
    force_range = np.asarray(self._mj_model.actuator_forcerange)
    if not np.all(np.isfinite(force_range)) or not np.allclose(
        force_range[:, 0], -force_range[:, 1]
    ):
      raise ValueError(
          f"Go1 model {xml_path} does not define symmetric actuator torque "
          "limits."
      )
    self._torque_limits = jp.array(force_range[:, 1])
    if self._control_mode == self._TORQUE_CONTROL and (
        not np.allclose(self._mj_model.actuator_gainprm[:, 0], 1.0)
        or not np.allclose(self._mj_model.actuator_biasprm, 0.0)
    ):
      raise ValueError(
          f"Go1 torque model {xml_path} must use unit-gain, zero-bias motors."
      )

    # Increase offscreen framebuffer size to render at higher resolutions.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path.as_posix()
    self._imu_site_id = self._mj_model.site("imu").id

    # Contact sensor ids.
    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in consts.FEET_GEOMS
    ]

  # Sensor readings.

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    return jp.vstack([
        mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])

  def _get_control(
      self, position_target: jax.Array, action: jax.Array
  ) -> jax.Array:
    """Converts a task position target and normalized action to actuator ctrl."""
    if self._control_mode == self._POSITION_CONTROL:
      return position_target
    return action * self._config.action_scale * self._torque_limits

  def _get_reset_control(self, position_target: jax.Array) -> jax.Array:
    """Returns the initial position target or zero torque for reset."""
    if self._control_mode == self._POSITION_CONTROL:
      return position_target
    return jp.zeros_like(position_target)

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def control_mode(self) -> str:
    return self._control_mode

  @property
  def episode_length(self) -> int:
    return self._episode_length

  @property
  def torque_limits(self) -> jax.Array:
    return self._torque_limits

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def n_substeps(self) -> int:
    return max(1, int(round(self.dt / self.sim_dt)))
