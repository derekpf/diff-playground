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
"""Tests for smooth reward primitives."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest, parameterized

from mujoco_playground._src import reward
from mujoco_playground._src import softjax as sj

_SIGMOIDS = (
    "gaussian",
    "hyperbolic",
    "long_tail",
    "reciprocal",
    "cosine",
    "linear",
    "quadratic",
    "tanh_squared",
)


class RewardTest(parameterized.TestCase):

  @parameterized.parameters(
      ((-0.2, 0.2), 0.0, (-0.2, 0.2)),
      ((-0.2, 0.2), 0.5, (-0.2, 0.2)),
      ((0.0, 0.0), 0.0, (0.0,)),
      ((0.0, 0.0), 0.5, (0.0,)),
  )
  def test_tolerance_is_one_at_inclusive_boundaries(
      self, bounds, margin, boundary_values
  ):
    for softness in (0.1, 0.05, 0.01, 0.005, 0.001):
      values = reward.tolerance(
          jp.array(boundary_values),
          bounds=bounds,
          margin=margin,
          softness=softness,
      )
      np.testing.assert_allclose(values, 1.0)

  @parameterized.parameters(0.0, 0.5)
  def test_tolerance_is_finite_and_differentiable(self, margin):
    def objective(x):
      return reward.tolerance(
          x, bounds=(-0.2, 0.2), margin=margin, sigmoid="gaussian"
      ).sum()

    values, gradients = jax.jit(jax.value_and_grad(objective))(
        jp.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    )
    self.assertTrue(np.all(np.isfinite(values)))
    self.assertTrue(np.all(np.isfinite(gradients)))
    self.assertGreater(np.linalg.norm(np.asarray(gradients)), 0.0)

  def test_tolerance_with_infinite_bounds_is_finite(self):
    values = reward.tolerance(
        jp.array([-1.0, 0.0, 1.0]), bounds=(2.5, float("inf")), margin=1.25
    )
    self.assertTrue(np.all(np.isfinite(values)))

  @parameterized.parameters(*_SIGMOIDS)
  def test_sigmoids_are_finite_and_differentiable(self, sigmoid):
    def objective(x):
      return reward.tolerance(
          x, bounds=(-0.2, 0.2), margin=0.5, sigmoid=sigmoid
      ).sum()

    values, gradients = jax.jit(jax.value_and_grad(objective))(
        jp.array([-1.5, -0.2, 0.0, 0.2, 1.5])
    )
    self.assertTrue(np.all(np.isfinite(values)))
    self.assertTrue(np.all(np.isfinite(gradients)))
    self.assertGreater(np.linalg.norm(np.asarray(gradients)), 0.0)

  def test_soft_primitives_are_finite_at_boundaries(self):
    def objective(x):
      vector = jp.stack([x, jp.zeros_like(x)])
      comparisons = sj.logical_and(
          sj.greater_equal(x, -0.2), sj.less_equal(x, 0.2)
      )
      gates = sj.logical_or(
          sj.any(sj.greater(vector, 0.0), axis=0),
          sj.all(sj.less(vector, 2.0), axis=0),
      )
      return (
          sj.abs(x)
          + sj.norm(vector, axis=0)
          + sj.div(x, sj.norm(vector, axis=0))
          + sj.clip(x, -0.5, 0.5)
          + sj.min(jp.stack([x, 0.25 * jp.ones_like(x)]), axis=0)
          + sj.max(jp.stack([x, -0.25 * jp.ones_like(x)]), axis=0)
          + sj.where(comparisons, x, -x)
          + gates
          + sj.arcsin(sj.clip(x, -1.0, 1.0))
          + sj.arccos(sj.clip(x, -1.0, 1.0))
      ).sum()

    values, gradients = jax.jit(jax.value_and_grad(objective))(
        jp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    )
    self.assertTrue(np.all(np.isfinite(values)))
    self.assertTrue(np.all(np.isfinite(gradients)))
    self.assertGreater(np.linalg.norm(np.asarray(gradients)), 0.0)


if __name__ == "__main__":
  absltest.main()
