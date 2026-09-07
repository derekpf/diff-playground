"""Behavioral tests for position and direct-torque Go1 environments."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest

from mujoco_playground._src import locomotion, wrapper

_POSITION_ENVS = (
    "Go1JoystickFlatTerrain",
    "Go1JoystickRoughTerrain",
    "Go1Getup",
    "Go1Handstand",
    "Go1Footstand",
)

_TORQUE_ENVS = (
    "Go1JoystickFlatTerrainTorque",
    "Go1JoystickRoughTerrainTorque",
    "Go1GetupTorque",
    "Go1HandstandTorque",
    "Go1FootstandTorque",
)

_GO1_TORQUE_LIMITS = np.array(
    [[-23.7, 23.7], [-23.7, 23.7], [-35.55, 35.55]] * 4
)


class DirectTorqueTest(absltest.TestCase):
  """Tests the observable actuator behavior of Go1 environments."""

  def test_position_environments_use_position_targets(self) -> None:
    for env_name in _POSITION_ENVS:
      env = locomotion.load(env_name, config_overrides={"impl": "jax"})
      self.assertEqual(env.action_size, 12)

      target = jp.linspace(-0.5, 0.5, 12)
      action = jp.full((12,), 0.25)
      np.testing.assert_allclose(env._get_control(target, action), target)

      state = jax.jit(env.reset)(jax.random.PRNGKey(0))
      self.assertTrue(bool(jp.any(state.data.ctrl != 0.0)))
      state = jax.jit(env.step)(state, action)
      self.assertTrue(bool(jp.any(state.data.ctrl != 0.0)))

  def test_position_environments_honor_configured_pd(self) -> None:
    env = locomotion.load(
        "Go1Getup",
        config_overrides={
            "impl": "jax",
            "Kp": 17.0,
            "Kd": 0.25,
        },
    )

    np.testing.assert_allclose(env.mj_model.dof_damping[6:], 0.25)
    np.testing.assert_allclose(env.mj_model.actuator_gainprm[:, 0], 17.0)
    np.testing.assert_allclose(env.mj_model.actuator_biasprm[:, 1], -17.0)

  def test_direct_torque_models_and_one_step(self) -> None:
    for env_name in _TORQUE_ENVS:
      env = locomotion.load(env_name, config_overrides={"impl": "jax"})

      self.assertEqual(env.action_size, 12)
      np.testing.assert_allclose(env.mj_model.actuator_gainprm[:, 0], 1.0)
      np.testing.assert_allclose(env.mj_model.actuator_biasprm, 0.0)
      np.testing.assert_allclose(
          env.mj_model.actuator_ctrlrange,
          env.mj_model.actuator_forcerange,
      )
      np.testing.assert_allclose(
          env.mjx_model.actuator_forcerange,
          _GO1_TORQUE_LIMITS,
      )

      state = jax.jit(env.reset)(jax.random.PRNGKey(0))
      np.testing.assert_allclose(state.data.ctrl, 0.0)
      action = jp.tile(jp.array([0.25, -0.25, 0.25]), 4)
      state = jax.jit(env.step)(state, action)
      scaled_action = action * env._config.action_scale
      expected_ctrl = jp.where(
          scaled_action >= 0,
          scaled_action * env.mjx_model.actuator_forcerange[:, 1],
          scaled_action * (-env.mjx_model.actuator_forcerange[:, 0]),
      )
      np.testing.assert_allclose(state.data.ctrl, expected_ctrl)
      for value in jax.tree_util.tree_leaves(state.obs):
        self.assertTrue(bool(jp.all(jp.isfinite(value))))
      self.assertTrue(bool(jp.all(jp.isfinite(state.data.qpos))))

  def test_direct_torque_does_not_apply_position_gains(self) -> None:
    env = locomotion.load(
        "Go1GetupTorque",
        config_overrides={
            "impl": "jax",
            "Kp": 17.0,
            "Kd": 0.25,
        },
    )

    np.testing.assert_allclose(env.mj_model.dof_damping[6:], 0.0)
    np.testing.assert_allclose(env.mj_model.actuator_gainprm[:, 0], 1.0)
    np.testing.assert_allclose(env.mj_model.actuator_biasprm, 0.0)

  def test_direct_torque_scales_asymmetric_limits_without_clipping(self) -> None:
    env = locomotion.load(
        "Go1JoystickFlatTerrainTorque",
        config_overrides={"impl": "jax", "action_scale": 0.25},
    )
    asymmetric_limits = jp.tile(jp.array([[-2.0, 3.0]]), (12, 1))
    env._mjx_model = env.mjx_model.replace(
        actuator_forcerange=asymmetric_limits
    )
    action = jp.array([-10.0, -1.0, 0.0, 1.0, 10.0] * 2 + [0.0, 0.0])
    scaled_action = action * env._config.action_scale

    control = env._get_control(jp.zeros(12), action)
    expected_control = jp.where(
        scaled_action >= 0,
        scaled_action * asymmetric_limits[:, 1],
        scaled_action * (-asymmetric_limits[:, 0]),
    )
    np.testing.assert_allclose(control, expected_control)
    self.assertGreater(float(control[4]), float(asymmetric_limits[4, 1]))
    self.assertLess(float(control[0]), float(asymmetric_limits[0, 0]))

  def test_mujoco_applies_physical_torque_limits(self) -> None:
    env = locomotion.load(
        "Go1JoystickFlatTerrainTorque", config_overrides={"impl": "jax"}
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action = jp.full((12,), 3.0)
    state = jax.jit(env.step)(state, action)

    scaled_action = action * env._config.action_scale
    expected_control = scaled_action * env.mjx_model.actuator_forcerange[:, 1]
    np.testing.assert_allclose(state.data.ctrl, expected_control)
    self.assertTrue(
        bool(
            jp.all(
                state.data.actuator_force
                <= env.mjx_model.actuator_forcerange[:, 1] + 1e-6
            )
        )
    )
    self.assertTrue(
        bool(
            jp.all(
                state.data.actuator_force
                >= env.mjx_model.actuator_forcerange[:, 0] - 1e-6
            )
        )
    )

  def test_direct_torque_timing_and_physical_horizons(self) -> None:
    expected_lengths = {
        "Go1JoystickFlatTerrainTorque": 5000,
        "Go1JoystickRoughTerrainTorque": 5000,
        "Go1GetupTorque": 1500,
        "Go1HandstandTorque": 2500,
        "Go1FootstandTorque": 2500,
    }
    for env_name, expected_length in expected_lengths.items():
      env = locomotion.load(env_name, config_overrides={"impl": "jax"})
      self.assertEqual(env.dt, 0.004)
      self.assertEqual(env.sim_dt, 0.004)
      self.assertEqual(env.n_substeps, 1)
      self.assertEqual(env.episode_length, expected_length)

  def test_getup_direct_torque_skips_settling(self) -> None:
    env = locomotion.load(
        "Go1GetupTorque", config_overrides={"impl": "jax"}
    )
    self.assertEqual(env._settle_steps, 0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    np.testing.assert_allclose(state.data.ctrl, 0.0)

  def test_direct_torque_reward_action_rate_defaults(self) -> None:
    self.assertEqual(
        locomotion.get_default_config(
            "Go1JoystickFlatTerrainTorque"
        ).reward_config.scales.action_rate,
        -0.0001,
    )
    self.assertEqual(
        locomotion.get_default_config("Go1GetupTorque")
        .reward_config.scales.action_rate,
        -0.00001,
    )

  def test_episode_length_override_is_exact(self) -> None:
    env = locomotion.load(
        "Go1GetupTorque",
        config_overrides={"impl": "jax", "episode_length": 17},
    )
    self.assertEqual(env.episode_length, 17)

  def test_wrapper_uses_direct_torque_episode_length(self) -> None:
    env = locomotion.load(
        "Go1GetupTorque", config_overrides={"impl": "jax"}
    )
    wrapped_env = wrapper.wrap_for_brax_training(env)

    self.assertEqual(wrapped_env.env.episode_length, env.episode_length)


if __name__ == "__main__":
  absltest.main()
