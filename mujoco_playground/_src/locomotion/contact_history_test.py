"""Contact event histories in differentiable joystick environments."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import sensor
from mujoco_playground._src import locomotion
from mujoco_playground._src import mjx_env


class ContactHistoryTest(parameterized.TestCase):

  def _setup_env(self, name):
    self.env = locomotion.load(name)
    self.env.reward_softness = 0.05
    self.env.bool_softness = 0.1
    self.env.reward_st_enable = True
    self.env._mjx_model = self.env.mjx_model.replace(
        opt=self.env.mjx_model.opt.replace(
            sensor_softness=0.1, sensor_st_enable=True))
    self.state = jax.jit(self.env.reset)(jax.random.PRNGKey(0))

  def _state_with_gap(self, gap, sensor_softness=0.1):
    contacts = self.state.data._impl.contact
    contacts = contacts.replace(dist=contacts.includemargin + gap)
    data = self.state.data.replace(
        _impl=self.state.data._impl.replace(contact=contacts))
    model = self.env.mjx_model.replace(opt=self.env.mjx_model.opt.replace(
        sensor_softness=sensor_softness))
    data = sensor.sensor_acc(model, data)
    shape = self.state.info["feet_air_time"].shape
    info = {
        **self.state.info,
        "feet_air_time": jp.full(shape, 0.2),
        "last_contact": jp.zeros(shape, dtype=data.sensordata.dtype),
    }
    if "swing_peak" in self.state.info:
      info["swing_peak"] = jp.full(shape, 0.08)
    return self.state.replace(data=data, info=info,
                              metrics=dict(self.state.metrics))

  def _events(self, gap):
    result = self.env.step(self._state_with_gap(gap),
                           jp.zeros(self.env.action_size))
    values = [
        result.info["feet_air_time"].sum(),
        result.info["last_contact"].sum(),
    ]
    if "swing_peak" in self.state.info:
      values.append(result.info["swing_peak"].sum())
    return jp.array(values)

  @parameterized.parameters(
      "G1JoystickFlatTerrain",
      "SpotFlatTerrainJoystick",
      "T1JoystickFlatTerrain",
      "H1JoystickGaitTracking",
      "BerkeleyHumanoidJoystickFlatTerrain",
      "BarkourJoystick",
  )
  def test_nominal_forward_and_touchdown_history_gradients(self, name):
    self._setup_env(name)
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      for gap in (-0.001, 0.0, 0.001):
        actual = self.env.step(self._state_with_gap(jp.array(gap)),
                               jp.zeros(self.env.action_size))
        nominal = self.env.step(
            self._state_with_gap(jp.array(gap), sensor_softness=0.0),
            jp.zeros(self.env.action_size))
        for key in ("feet_air_time", "swing_peak", "last_contact"):
          if key not in self.state.info:
            continue
          np.testing.assert_array_equal(actual.info[key], nominal.info[key])

      for st in (False, True):
        self.env.reward_st_enable = st
        derivative = jax.jacrev(self._events)(jp.array(0.001))
        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertGreater(float(derivative[0]), 0.0)
        self.assertLess(float(derivative[1]), 0.0)
        if len(derivative) > 2:
          self.assertGreater(float(derivative[2]), 0.0)
    self.assertTrue(np.all(np.isfinite(derivative)))
    self.assertTrue(np.all(np.abs(derivative) > 1e-3))
    self.assertTrue(jp.issubdtype(actual.info["last_contact"].dtype,
                                  jp.floating))


if __name__ == "__main__":
  absltest.main()
