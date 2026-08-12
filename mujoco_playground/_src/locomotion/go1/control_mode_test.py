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
"""Tests for the shared Go1 position and torque control modes."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest

from mujoco_playground._src import locomotion
from mujoco_playground._src import wrapper


_GO1_ENVS = (
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
    "Go1Getup",
    "Go1Handstand",
    "Go1Footstand",
)


class ControlModeTest(absltest.TestCase):
  """Tests shared position and torque actuator behavior."""

  def test_torque_models_and_one_step(self) -> None:
    for env_name in _GO1_ENVS:
      env = locomotion.load(
          env_name,
          config_overrides={"impl": "jax", "control_mode": "torque"},
      )

      self.assertTrue(env.xml_path.endswith("_torque.xml"))
      self.assertEqual(env.action_size, 12)
      np.testing.assert_allclose(env.mj_model.actuator_gainprm[:, 0], 1.0)
      np.testing.assert_allclose(env.mj_model.actuator_biasprm, 0.0)
      np.testing.assert_allclose(
          env.mj_model.actuator_ctrlrange,
          env.mj_model.actuator_forcerange,
      )
      np.testing.assert_allclose(
          env.torque_limits,
          np.array([23.7, 23.7, 35.55] * 4),
      )

      state = jax.jit(env.reset)(jax.random.PRNGKey(0))
      np.testing.assert_allclose(state.data.ctrl, 0.0)
      state = jax.jit(env.step)(state, jp.full((12,), 0.25))
      for value in jax.tree_util.tree_leaves(state.obs):
        self.assertTrue(bool(jp.all(jp.isfinite(value))))
      self.assertTrue(bool(jp.all(jp.isfinite(state.data.qpos))))

  def test_torque_action_scaling(self) -> None:
    for env_name, action_scale in (
        ("Go1JoystickFlatTerrain", 0.5),
        ("Go1Getup", 0.5),
        ("Go1Handstand", 0.3),
    ):
      env = locomotion.load(
          env_name,
          config_overrides={"impl": "jax", "control_mode": "torque"},
      )
      action = jp.full((12,), 0.25)
      np.testing.assert_allclose(
          env._get_control(jp.zeros(12), action),
          action * action_scale * env.torque_limits,
      )

  def test_getup_torque_reset_skips_settling(self) -> None:
    env = locomotion.load(
        "Go1Getup",
        config_overrides={"impl": "jax", "control_mode": "torque"},
    )
    self.assertEqual(env._settle_steps, 0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    np.testing.assert_allclose(state.data.ctrl, 0.0)

  def test_torque_timing_defaults_and_overrides(self) -> None:
    env = locomotion.load(
        "Go1Getup",
        config_overrides={"impl": "jax", "control_mode": "torque"},
    )
    self.assertEqual(env.dt, 0.002)
    self.assertEqual(env.sim_dt, 0.002)
    self.assertEqual(env.n_substeps, 1)

    env = locomotion.load(
        "Go1Getup",
        config_overrides={
            "impl": "jax",
            "control_mode": "torque",
            "ctrl_dt": 0.01,
            "sim_dt": 0.002,
        },
    )
    self.assertEqual(env.dt, 0.01)
    self.assertEqual(env.sim_dt, 0.002)
    self.assertEqual(env.n_substeps, 5)

  def test_episode_lengths_match_position_duration(self) -> None:
    expected_lengths = {
        "Go1JoystickFlatTerrain": 1000,
        "Go1JoystickRoughTerrain": 1000,
        "Go1Getup": 300,
        "Go1Handstand": 500,
        "Go1Footstand": 500,
    }
    for env_name, position_length in expected_lengths.items():
      position_env = locomotion.load(
          env_name, config_overrides={"impl": "jax"}
      )
      torque_env = locomotion.load(
          env_name,
          config_overrides={"impl": "jax", "control_mode": "torque"},
      )

      self.assertEqual(position_env.episode_length, position_length)
      self.assertEqual(
          torque_env.episode_length,
          round(position_length * 0.02 / torque_env.dt),
      )

  def test_episode_length_override_is_exact(self) -> None:
    for control_mode in ("position", "torque"):
      env = locomotion.load(
          "Go1Getup",
          config_overrides={
              "impl": "jax",
              "control_mode": control_mode,
              "episode_length": 17,
          },
      )
      self.assertEqual(env.episode_length, 17)

  def test_torque_episode_length_uses_custom_control_period(self) -> None:
    env = locomotion.load(
        "Go1Getup",
        config_overrides={
            "impl": "jax",
            "control_mode": "torque",
            "ctrl_dt": 0.01,
            "sim_dt": 0.002,
        },
    )
    self.assertEqual(env.episode_length, 600)

  def test_torque_episode_length_does_not_mutate_config(self) -> None:
    config = locomotion.get_default_config("Go1Getup")
    env = locomotion.load(
        "Go1Getup",
        config=config,
        config_overrides={"impl": "jax", "control_mode": "torque"},
    )

    self.assertEqual(config.episode_length, 300)
    self.assertEqual(env.episode_length, 3000)

  def test_wrapper_defaults_to_effective_episode_length(self) -> None:
    env = locomotion.load(
        "Go1Getup",
        config_overrides={"impl": "jax", "control_mode": "torque"},
    )
    wrapped_env = wrapper.wrap_for_brax_training(env)

    self.assertEqual(wrapped_env.env.episode_length, env.episode_length)


if __name__ == "__main__":
  absltest.main()
