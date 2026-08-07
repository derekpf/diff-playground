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
"""Tests for the project softjax adapter."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest
from ml_collections import config_dict

from mujoco_playground import MjxEnv
from mujoco_playground._src import softjax as sj
from mujoco_playground._src import wrapper


class _SoftnessEnv(MjxEnv):

  def __init__(self):
    super().__init__(config_dict.create(ctrl_dt=0.01, sim_dt=0.01))

  def reset(self, rng):
    del rng
    raise NotImplementedError

  def step(self, state, action):
    del state, action
    raise NotImplementedError

  @property
  def xml_path(self):
    raise NotImplementedError

  @property
  def action_size(self):
    return 0

  @property
  def mj_model(self):
    raise NotImplementedError

  @property
  def mjx_model(self):
    raise NotImplementedError

  def reward(self, value):
    return sj.relu(value, softness=self.reward_softness)


class SoftjaxTest(absltest.TestCase):

  def test_softness(self):
    self.assertNotIn("reward_softness", MjxEnv.__dict__)
    env = _SoftnessEnv()
    self.assertEqual(env.reward_softness, 0.01)

  def test_reward_softness_is_instance_scoped(self):
    first = _SoftnessEnv()
    second = _SoftnessEnv()
    first.reward_softness = 0.1

    self.assertEqual(first.reward_softness, 0.1)
    self.assertEqual(second.reward_softness, 0.01)
    np.testing.assert_allclose(
        jax.jit(first.reward)(0.0), 0.1 * np.log(2.0), rtol=1e-5
    )
    np.testing.assert_allclose(
        jax.jit(second.reward)(0.0), 0.01 * np.log(2.0), rtol=1e-5
    )

  def test_wrapper_delegates_reward_softness(self):
    env = _SoftnessEnv()
    wrapped = wrapper.Wrapper(env)
    wrapped.reward_softness = 0.1

    self.assertEqual(env.reward_softness, 0.1)
    np.testing.assert_allclose(
        wrapped.reward(0.0), 0.1 * np.log(2.0), rtol=1e-5
    )

  def test_standalone_wrappers_use_the_default_softness(self):
    np.testing.assert_allclose(sj.relu(0.0), 0.01 * np.log(2.0), rtol=1e-5)

  def test_explicit_softness_is_honored(self):
    np.testing.assert_allclose(
        sj.relu(0.0, softness=0.1), 0.1 * np.log(2.0), rtol=1e-5
    )
    np.testing.assert_allclose(sj.relu(0.0, 0.1), 0.1 * np.log(2.0), rtol=1e-5)

  def test_clip_at_lower_bound(self):
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0), 0.00693147, rtol=1e-5
    )
    np.testing.assert_allclose(
        sj.clip(jp.array(0.0), 0.0, 10000.0, softness=0.1),
        0.0693147,
        rtol=1e-5,
    )

  def test_abs(self):
    np.testing.assert_allclose(sj.abs(0.01), 0.00462117, rtol=1e-5)

  def test_comparisons(self):
    expected = 0.731059
    np.testing.assert_allclose(sj.greater(0.01, 0.0), expected, rtol=1e-5)
    np.testing.assert_allclose(sj.greater_equal(0.01, 0.0), expected, rtol=1e-5)
    np.testing.assert_allclose(sj.less(0.0, 0.01), expected, rtol=1e-5)
    np.testing.assert_allclose(sj.less_equal(0.0, 0.01), expected, rtol=1e-5)

  def test_st_comparisons_have_hard_forward_and_soft_gradients(self):
    comparisons = (
        (sj.greater_st, (0.0, 0.0, 1.0)),
        (sj.greater_equal_st, (0.0, 1.0, 1.0)),
        (sj.less_st, (1.0, 0.0, 0.0)),
        (sj.less_equal_st, (1.0, 1.0, 0.0)),
    )
    for comparison, expected in comparisons:
      with self.subTest(comparison=comparison.__name__):
        values = comparison(jp.array([-1.0, 0.0, 1.0]), 0.0)
        np.testing.assert_allclose(values, expected)
        _, gradient = jax.jvp(
            lambda x: comparison(x, 0.0),
            (jp.array(0.0),),
            (jp.array(1.0),),
        )
        self.assertNotEqual(float(gradient), 0.0)

  def test_st_comparisons_honor_explicit_softness(self):
    comparisons = (
        sj.greater_st,
        sj.greater_equal_st,
        sj.less_st,
        sj.less_equal_st,
    )
    for comparison in comparisons:
      with self.subTest(comparison=comparison.__name__):
        gradients = []
        for softness in (0.1, 1.0):
          _, gradient = jax.jvp(
              lambda x: comparison(x, 0.0, softness=softness),
              (jp.array(0.0),),
              (jp.array(1.0),),
          )
          gradients.append(float(gradient))
        self.assertNotEqual(gradients[0], gradients[1])

        positional_gradient = jax.jvp(
            lambda x: comparison(x, 0.0, 0.1),
            (jp.array(0.0),),
            (jp.array(1.0),),
        )[1]
        np.testing.assert_allclose(positional_gradient, gradients[0])

  def test_reductions(self):
    values = jp.array([0.0, 0.01])
    np.testing.assert_allclose(sj.max(values), 0.01, rtol=1e-5)
    np.testing.assert_allclose(sj.min(values), 0.0, atol=1e-5)
    np.testing.assert_allclose(
        sj.any(sj.greater(jp.zeros(4), 0.0)), 0.9375, rtol=1e-5
    )
    np.testing.assert_allclose(
        sj.any(sj.greater_st(jp.zeros(4), 0.0)), 0.0, atol=1e-5
    )

  def test_relu(self):
    np.testing.assert_allclose(sj.relu(0.0), 0.00693147, rtol=1e-5)

  def test_modes_are_forwarded(self):
    self.assertEqual(sj.clip(0.0, 0.0, 10000.0, mode="hard"), 0.0)
    self.assertEqual(sj.relu(-1.0, mode="hard"), 0.0)


if __name__ == "__main__":
  absltest.main()
