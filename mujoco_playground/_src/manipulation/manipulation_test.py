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
"""Tests for the manipulation environments."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src import manipulation


def _load_env(env_name: str):
  config = manipulation.get_default_config(env_name)
  overrides = {"impl": "jax"} if "impl" in config else {}
  return manipulation.load(env_name, config_overrides=overrides)


class TestSuite(parameterized.TestCase):
  """Tests for the manipulation environments."""

  @parameterized.named_parameters(
      {"testcase_name": f"test_can_create_{env_name}", "env_name": env_name}
      for env_name in manipulation.ALL_ENVS
  )
  def test_can_create_all_environments(self, env_name: str) -> None:
    env = _load_env(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(42))
    state = jax.jit(env.step)(state, jp.zeros(env.action_size))
    self.assertIsNotNone(state)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)
    obs_shape = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
    self.assertEqual(obs_shape, env.observation_size)
    self.assertFalse(jp.isnan(state.data.qpos).any())

  @parameterized.named_parameters(
      ("aloha", "AlohaHandOver"),
      ("panda_pick", "PandaPickCube"),
      ("panda_open", "PandaOpenCabinet"),
      ("panda_cartesian", "PandaPickCubeCartesian"),
  )
  def test_zero_collision_rewards_match_nominal_forward(
      self, env_name: str
  ) -> None:
    env = _load_env(env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    if env_name == "AlohaHandOver":
      no_collision = 1.0 - env.hand_table_collision(state.data)
    elif env_name == "PandaPickCube":
      no_collision = env._get_reward(state.data, state.info)[
          "no_floor_collision"
      ]
    elif env_name == "PandaOpenCabinet":
      no_collision = env._get_rewards(state.data, state.info)[
          "no_barrier_collision"
      ]
    else:
      state = jax.jit(env.step)(state, jp.zeros(env.action_size))
      no_collision = state.metrics["reward/no_box_collision"]

    np.testing.assert_allclose(no_collision, 1.0, atol=1e-6)

  def test_robotiq_success_step_count_boundary_is_inclusive(self) -> None:
    env = _load_env("PandaRobotiqPushCube")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    body_id = env._obj_body
    xpos = state.data.xpos.at[body_id].set(
        state.data.mocap_pos[env._mocap_target].ravel()
    )
    xquat = state.data.xquat.at[body_id].set(
        state.data.mocap_quat[env._mocap_target].ravel()
    )
    state = state.replace(data=state.data.replace(xpos=xpos, xquat=xquat))
    state.info["success_step_count"] = jp.array(
        env._config.reward_config.success_step_count
    )

    success, sub_success = env._get_success_reward(state)
    np.testing.assert_allclose(success, sub_success, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
