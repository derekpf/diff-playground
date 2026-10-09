"""Native contact sensor gradients in Go1 rewards and event histories."""

from unittest import mock
from absl.testing import absltest
import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import sensor
from mujoco_playground._src import locomotion, mjx_env


class ContactSurrogateTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.env = locomotion.load("Go1JoystickFlatTerrain")
    cls.env.reward_softness = cls.env.bool_softness = 0.1
    cls.env.reward_st_enable = True
    cls.env._mjx_model = cls.env.mjx_model.replace(opt=cls.env.mjx_model.opt.replace(
        sensor_softness=0.1, sensor_st_enable=True))
    cls.state = jax.jit(cls.env.reset)(jax.random.PRNGKey(0))

  def _state_with_gap(self, gap, softness=0.1, st=True):
    data = self.state.data
    contacts = data._impl.contact
    contacts = contacts.replace(dist=contacts.includemargin + gap)
    data = data.replace(_impl=data._impl.replace(contact=contacts))
    model = self.env.mjx_model.replace(opt=self.env.mjx_model.opt.replace(
        sensor_softness=softness, sensor_st_enable=st))
    data = sensor.sensor_acc(model, data)
    return self.state.replace(data=data, info={
        **self.state.info,
        "feet_air_time": jp.full(4, 0.2),
        "swing_peak": jp.full(4, 0.08),
        "last_contact": jp.zeros(4, dtype=data.sensordata.dtype),
        "command": jp.array([1.0, 0.0, 0.0]),
    }, metrics=dict(self.state.metrics))

  def _events(self, gap, softness=0.1, st=True):
    result = self.env.step(self._state_with_gap(gap, softness, st), jp.zeros(12))
    return jp.array([
        result.metrics["reward/feet_air_time"],
        result.metrics["reward/feet_height"],
        result.info["feet_air_time"].sum(),
        result.info["swing_peak"].sum(),
    ])

  def test_touchdown_rewards_and_resets_receive_native_sensor_gradients(self):
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      for st in (False, True):
        for gap in (-0.001, 0.001):
          fn = lambda g: self._events(g, st=st)
          derivative = jax.jit(jax.jacrev(fn))(jp.array(gap))
          self.assertTrue(np.all(np.isfinite(derivative)))
          self.assertTrue(np.all(np.abs(derivative) > 1e-6))
          print("native Go1 event gap gradients", st, gap, derivative)
      fn = lambda g: self._events(g, softness=0.)
      np.testing.assert_array_equal(jax.jit(jax.jacrev(fn))(jp.array(.001)),
                                    jp.zeros(4))

  def test_st_events_preserve_nominal_sensor_forward(self):
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      for gap in (-0.1, -0.001, 0.0, 0.001, 0.1):
        actual = self.env.step(self._state_with_gap(jp.array(gap)), jp.zeros(12))
        expected = self.env.step(self._state_with_gap(jp.array(gap), softness=0.),
                                 jp.zeros(12))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
          np.testing.assert_array_equal(a, b)
        keep = float(gap >= 0.0)
        np.testing.assert_array_equal(actual.info["feet_air_time"],
                                      jp.full(4, (0.2 + self.env.dt) * keep))
        np.testing.assert_array_equal(actual.info["swing_peak"],
                                      jp.full(4, 0.08 * keep))

  def test_event_history_honors_relaxed_reward_setting(self):
    previous = self.env.reward_st_enable
    self.env.reward_st_enable = False
    try:
      with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
        actual = self.env.step(self._state_with_gap(jp.array(.001)),
                               jp.zeros(12))
        self.assertGreater(float(actual.info["last_contact"].min()), 0.0)
        self.assertLess(float(actual.info["last_contact"].max()), 1.0)
        self.assertGreater(float(actual.info["feet_air_time"].min()), 0.0)
        self.assertLess(float(actual.info["feet_air_time"].max()),
                        .2 + self.env.dt)
        gradient = jax.jacrev(lambda gap: self._events(gap)[2:])(
            jp.array(.001))
      self.assertTrue(np.all(np.isfinite(gradient)))
      self.assertTrue(np.all(np.abs(gradient) > 1e-6))
    finally:
      self.env.reward_st_enable = previous


  def test_contact_history_reaches_next_reward_and_critic_observation(self):
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      def downstream(gap):
        first = self.env.step(self._state_with_gap(gap), jp.zeros(12))
        # Isolate history from the clock, peak and current sensor paths.
        info = jax.tree.map(jax.lax.stop_gradient, first.info)
        info["last_contact"] = first.info["last_contact"]
        following = first.replace(
            data=jax.tree.map(jax.lax.stop_gradient, first.data),
            info=info, metrics=dict(first.metrics))
        second = self.env.step(following, jp.zeros(12))
        return (first.info["last_contact"], second.obs["state"],
                second.obs["privileged_state"],
                second.metrics["reward/feet_air_time"])

      derivatives = jax.jit(jax.jacrev(downstream))(jp.array(.001))
      history, actor_obs, critic_obs, reward = derivatives
      self.assertTrue(np.all(np.abs(history) > 1e-3))
      np.testing.assert_array_equal(actor_obs, jp.zeros_like(actor_obs))
      self.assertGreater(float(jp.linalg.norm(critic_obs)), 1e-3)
      self.assertGreater(float(jp.abs(reward)), 1e-3)
      self.assertTrue(np.all(np.isfinite(critic_obs)))

  def test_reset_and_step_contact_types_match_in_scan(self):
    initial = self.state.replace(info=dict(self.state.info),
                                 metrics=dict(self.state.metrics))
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      def rollout(state):
        return jax.lax.scan(
            lambda s, _: (self.env.step(s, jp.zeros(12)), None),
            state, None, length=4)[0]
      result = jax.jit(rollout)(initial)
    self.assertEqual(result.info["last_contact"].dtype,
                     initial.info["last_contact"].dtype)
    self.assertTrue(jp.issubdtype(result.info["last_contact"].dtype, jp.floating))

  def test_below_peak_height_reaches_touchdown_height_reward(self):
    previous = self.env.reward_softness, self.env.reward_st_enable
    try:
      with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
        def reward(height, peak):
          state = self._state_with_gap(jp.array(-.001))
          state.info["swing_peak"] = jp.full(4, peak)
          data = state.data.replace(site_xpos=state.data.site_xpos.at[
              self.env._feet_site_id, -1].set(height))
          result = self.env.step(state.replace(data=data), jp.zeros(12))
          return result.metrics["reward/feet_height"]

        for softness in (.005, .05, .5):
          self.env.reward_softness = softness
          for st in (False, True):
            self.env.reward_st_enable = st
            for peak, height, direction in ((.08, .07, 1.), (.12, .11, -1.)):
              gradient = jax.grad(lambda z: reward(z, peak))(jp.full(4, height))
              # A foot below the stored peak receives a reward derivative
              # toward the target for both ST modes and several widths.
              self.assertTrue(np.all(np.isfinite(gradient)))
              self.assertTrue(np.all(direction * gradient > 0.))
              if st and softness == .05:
                np.testing.assert_allclose(
                    gradient, jp.full(4, direction * .1800664), rtol=1e-5)
    finally:
      self.env.reward_softness, self.env.reward_st_enable = previous

  def test_flight_histories_reach_following_touchdown_rewards(self):
    with mock.patch.object(mjx_env, "step", side_effect=lambda m, d, c, n: d):
      def following_rewards(gap):
        first = self.env.step(self._state_with_gap(gap), jp.zeros(12))
        info = jax.tree.map(jax.lax.stop_gradient, first.info)
        for key in ("feet_air_time", "swing_peak"):
          info[key] = first.info[key]
        touchdown = self._state_with_gap(jp.array(-.001)).replace(
            info=info, metrics=dict(first.metrics))
        second = self.env.step(touchdown, jp.zeros(12))
        return jp.array([second.metrics["reward/feet_air_time"],
                         second.metrics["reward/feet_height"]])

      derivative = jax.jacrev(following_rewards)(jp.array(.001))
    self.assertTrue(np.all(np.isfinite(derivative)))
    self.assertTrue(np.all(derivative > 1e-3))

if __name__ == "__main__":
  absltest.main()
