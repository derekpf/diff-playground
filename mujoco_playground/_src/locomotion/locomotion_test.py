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
"""Tests for the locomotion environments."""

from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest, parameterized

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.barkour import joystick as barkour_joystick
from mujoco_playground._src.locomotion.h1 import joystick as h1_joystick
from mujoco_playground._src.locomotion.op3 import joystick as op3_joystick


class TestSuite(parameterized.TestCase):
  """Tests for the locomotion environments."""

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in locomotion.ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = locomotion.load(env_name, config_overrides={"impl": "jax"})
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)
    obs_shape = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    self.assertEqual(obs_shape, env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())

  def test_go1_contact_reward_has_hard_forward_value(self) -> None:
    env = locomotion.load(
        "Go1JoystickFlatTerrain", config_overrides={"impl": "jax"}
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.info["command"] = jp.array([1.0, 0.0, 0.5])
    state.info["steps_until_next_cmd"] = jp.array(100000, dtype=jp.int32)
    action = jp.zeros(env.action_size)
    step = jax.jit(env.step)

    state = step(state, action)
    state = step(state, action)

    contact_values = np.asarray(
        jp.array([
            state.data.sensordata[env._mj_model.sensor_adr[sensorid]]
            for sensorid in env._feet_floor_found_sensor
        ])
    )
    self.assertTrue(np.any(contact_values == 0.0))

    contact = contact_values > 0.0
    feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr])
    vel_xy_norm_sq = np.sum(np.square(feet_vel[..., :2]), axis=-1)
    expected = (
        np.sum(vel_xy_norm_sq * contact)
        * float(env._config.reward_config.scales.feet_slip)
    )
    np.testing.assert_allclose(
        state.metrics["reward/feet_slip"], expected, rtol=1e-5, atol=1e-6
    )

  @parameterized.parameters(
      "BarkourJoystick",
      "BerkeleyHumanoidJoystickFlatTerrain",
      "SpotFlatTerrainJoystick",
  )
  def test_first_contact_reward_is_zero_on_the_initial_swing(
      self, env_name: str
  ) -> None:
    env = locomotion.load(env_name, config_overrides={"impl": "jax"})
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))

    # The reset air time is exactly zero, so the first-contact predicate is
    # false even when the foot sensor reports contact on the first step.
    np.testing.assert_allclose(
        state.metrics["reward/feet_air_time"], 0.0, atol=1e-6
    )

  @parameterized.named_parameters(
      ("h1", "H1JoystickGaitTracking", 0.0),
      ("spot", "SpotJoystickGaitTracking", 2.0),
  )
  def test_discrete_gait_gate_has_hard_forward_value(
      self, env_name: str, boundary: float
  ) -> None:
    env = locomotion.load(env_name, config_overrides={"impl": "jax"})
    global_linvel = jp.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(
        env._cost_lin_vel_z(global_linvel, boundary), 0.0
    )
    np.testing.assert_allclose(
        env._cost_lin_vel_z(global_linvel, boundary + 1.0), 1.0
    )

  @parameterized.parameters(
      barkour_joystick.Joystick._cost_termination,
      h1_joystick.Joystick._cost_termination,
      op3_joystick.Joystick._cost_termination,
  )
  def test_termination_cost_uses_boolean_softness(self, cost_fn) -> None:
    env = SimpleNamespace(bool_softness=0.0, reward_st_enable=False)
    cost = lambda step: cost_fn(env, jp.array(True), step)

    np.testing.assert_allclose(cost(jp.array(499.0)), 1.0)
    np.testing.assert_allclose(cost(jp.array(500.0)), 0.0)
    env.bool_softness = 0.1
    self.assertGreater(float(cost(jp.array(500.0))), 0.0)

    env.reward_st_enable = True
    np.testing.assert_allclose(cost(jp.array(500.0)), 0.0)
    self.assertLess(float(jax.grad(cost)(jp.array(500.0))), 0.0)


if __name__ == "__main__":
  absltest.main()
