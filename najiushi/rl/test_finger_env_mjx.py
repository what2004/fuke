"""Test pure JAX task equations against native MuJoCo; no MJX dependency.

These tests DO NOT execute the MJX physics engine or Playground wrappers.
Use check_mjx.py --task in the server training environment for that check.
"""

import unittest
from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from finger_env import FingerHeightEnv
from finger_env_mjx import FingerHeightMjxEnv, limit_command, noisy_tendon_length, observation, task_sample, tip_kinematics


class ToyData(NamedTuple):
    qpos: object
    qvel: object
    qacc: object
    ctrl: object
    time: object
    site_xpos: object
    cvel: object
    subtree_com: object
    ten_length: object
    ten_velocity: object
    sensordata: object

    def replace(self, **kwargs):
        return self._replace(**kwargs)


class ToyState(NamedTuple):
    data: object
    obs: object
    reward: object
    done: object
    metrics: dict
    info: dict

    def replace(self, **kwargs):
        return self._replace(**kwargs)


class ToyPhysics:
    """Only tests JAX control flow; deliberately NOT a MuJoCo replacement."""

    @staticmethod
    def forward(model, data):
        return data

    @staticmethod
    def step(model, data):
        return data.replace(time=data.time + .001, site_xpos=jp.array([[0., 0., .16]]))


def toy_env():
    env = object.__new__(FingerHeightMjxEnv)
    env.config = FingerHeightEnv().config
    env._mjx = ToyPhysics
    env._state_type = ToyState
    env._mjx_model = None
    env.tip = env.tip_body = env.tip_root = env.tendon = env.actuator = 0
    env.tendon_sensor_adr, env.tendon_velocity_sensor_adr = 0, 1
    env.low, env.high = .063, .115
    env._sim_dt, env._n_substeps, env.max_steps = .001, 20, 400
    env._initial_data = ToyData(jp.zeros(3), jp.zeros(3), jp.zeros(3), jp.array([.115]),
                                jp.float32(0), jp.array([[0., 0., .12]]), jp.zeros((1, 6)),
                                jp.zeros((1, 3)), jp.array([.114]), jp.zeros(1), jp.array([.114, 0.]))
    return env


class MjxTaskMathTests(unittest.TestCase):
    def test_sensor_path_and_noise_jit_vmap(self):
        env = toy_env()
        keys = jax.random.split(jax.random.PRNGKey(7), 16)
        batch = jax.jit(jax.vmap(env.reset))(keys)
        measured = np.asarray(batch.info['tendon_length_observed'])
        np.testing.assert_allclose(batch.info['tendon_length_sensor'], .114)
        np.testing.assert_allclose(batch.obs[:, 14]*.1, measured, atol=1e-8)
        self.assertTrue(np.all(np.abs(measured-.114) <= .005+1e-7))
        self.assertGreater(np.ptp(measured), .001)
        repeat = jax.jit(jax.vmap(env.reset))(keys)
        np.testing.assert_array_equal(batch.obs, repeat.obs)
        read = jax.jit(lambda key: noisy_tendon_length(key, .114, 0., .005))
        self.assertAlmostEqual(float(read(keys[0])), .114, places=6)
        # Give the sensor a different value to the engine's raw length. The
        # observation must use sensordata, not silently read ten_length.
        env.config = replace(env.config, tendon_noise_level=0.)
        env._initial_data = env._initial_data.replace(sensordata=jp.array([.120, .030]))
        state = jax.jit(env.reset)(keys[0])
        self.assertAlmostEqual(float(state.obs[14]), 1.2, places=6)
        self.assertAlmostEqual(float(state.obs[15]), .3, places=6)

    def test_state_control_flow_jit_vmap_and_autoreset(self):
        env = toy_env()
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        initial = reset(jax.random.PRNGKey(0))
        initial_info = dict(initial.info)

        def episode(s):
            def tick(s, _):
                s = env.step(s, jp.array([1.]))
                return s, s.info["is_success"]
            return jax.lax.scan(tick, s, None, length=400)

        final, success = jax.jit(episode)(initial)
        self.assertTrue(bool(final.done))
        self.assertTrue(bool(final.info["truncated"]))
        self.assertFalse(bool(final.info["terminated"]))
        self.assertFalse(bool(success[48]))  # 49 * 20 = 980 qualifying ticks
        self.assertTrue(bool(success[49]))   # exactly 1000 qualifying ticks
        self.assertEqual(int(final.info["task_steps"]), 400)
        self.assertEqual(int(initial.info["task_steps"]), 0)
        self.assertEqual(set(initial_info), set(final.info))
        frozen = step(final, jp.array([-1.]))
        np.testing.assert_array_equal(frozen.obs, final.obs)

        # Simulate cached Brax reset: data/obs replaced; task info retained.
        cached = final.replace(data=initial.data, obs=initial.obs, done=jp.float32(0))
        restarted = step(cached, jp.array([1.]))
        self.assertEqual(int(restarted.info["task_steps"]), 1)
        self.assertFalse(bool(restarted.info["success_ever"]))
        self.assertAlmostEqual(float(restarted.info["command_length"]), .1146, places=6)
        self.assertEqual(int(restarted.info["hold_steps"]), 20)

        # Brax EpisodeWrapper must see done=0 for timeouts, done=1 for failures.
        adapted = env.for_brax_training()
        penultimate = final.replace(done=jp.float32(0), info={
            **final.info, "task_steps": jp.int32(399), "truncated": jp.array(False)})
        timeout = jax.jit(adapted.step)(penultimate, jp.array([0.]))
        self.assertTrue(bool(timeout.info["truncated"]))
        self.assertEqual(float(timeout.done), 0)
        failure = jax.jit(adapted.step)(initial, jp.array([jp.nan]))
        self.assertEqual(float(failure.done), 1)

        batch = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(1), 3))
        batch = jax.jit(jax.vmap(env.step))(batch, jp.array([[-1.], [0.], [1.]]))
        self.assertEqual(batch.obs.shape, (3, 16))
        self.assertAlmostEqual(float(batch.info["command_length"][0]), .115, places=6)
        self.assertAlmostEqual(float(batch.info["command_length"][2]), .1146, places=6)

    def test_nonfinite_action_terminates_without_nan_observation(self):
        env = toy_env()
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        result = jax.jit(env.step)(state, jp.array([jp.nan]))
        self.assertTrue(bool(result.info["terminated"]))
        self.assertFalse(bool(result.info["truncated"]))
        self.assertTrue(np.isfinite(result.obs).all())
        self.assertEqual(float(result.reward), -env.config.failure_penalty)
        self.assertEqual(float(result.data.time), 0)

    def test_command_jit_and_batch(self):
        env = FingerHeightEnv()
        command = jax.jit(lambda a, u: limit_command(a, u, env.low, env.high,
                                                     env.config.max_command_speed, env.config.control_dt))
        actions = jp.array([-2., -1., 0., 1., 2.])
        result = jax.jit(jax.vmap(command))(actions, jp.full((5,), env.high))
        np.testing.assert_allclose(result, [env.high, env.high, env.high - .0004,
                                           env.high - .0004, env.high - .0004], atol=1e-8)
        for length in [env.low, .100, env.high]:
            np.testing.assert_allclose(command(env.action_for_length(length)[0], env.high),
                                       np.clip(length, env.high - .0004, env.high), atol=1e-8)

    def test_world_tip_velocity_against_mujoco(self):
        env = FingerHeightEnv()
        m, d = env.model, env.data
        body = int(m.site_bodyid[env.tip])
        root = int(m.body_rootid[body])
        read = jax.jit(tip_kinematics)
        rng = np.random.default_rng(5)
        v = np.zeros(6)
        for _ in range(30):
            d.qpos[:] = rng.uniform(np.deg2rad([-90, -80, -72]), 0)
            d.qvel[:] = rng.normal(size=3)
            mujoco.mj_forward(m, d)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, env.tip, v, 0)
            _, velocity = read(jp.array(d.site_xpos[env.tip]), jp.array(d.cvel[body]), jp.array(d.subtree_com[root]))
            np.testing.assert_allclose(velocity, v[3:], atol=5e-8, rtol=2e-6)

    def test_integer_hold_boundary_and_loss(self):
        env = FingerHeightEnv()
        c = env.config
        sample = jax.jit(lambda n, ever, first, t, z, speed: task_sample(
            jp.zeros(3), jp.array([0., 0., z]), jp.array([speed, 0., 0.]),
            n, ever, first, t, c, .001))
        result = sample(jp.int32(998), jp.array(False), jp.float32(-1), jp.float32(.999), .16, 0.)
        self.assertFalse(bool(result[2]))
        result = sample(result[0], result[3], result[4], jp.float32(1), .16, 0.)
        self.assertTrue(bool(result[2]))
        self.assertAlmostEqual(float(result[1]), 1)
        self.assertAlmostEqual(float(result[4]), 1)
        result = sample(result[0], result[3], result[4], jp.float32(1.001), .16, .02)
        self.assertFalse(bool(result[2]))
        self.assertEqual(int(result[0]), 0)
        self.assertTrue(bool(result[3]))
        result = sample(jp.int32(800), result[3], result[4], jp.float32(2), .18, 0.)
        self.assertEqual(int(result[0]), 0)

    def test_full_episode_task_parity_on_native_physics_samples(self):
        """Same physics samples, independently evaluated NumPy/JAX task logic."""
        env = FingerHeightEnv()
        env.config = replace(env.config, tendon_noise_level=0.)
        obs0, _ = env.reset(seed=0)
        replay = mujoco.MjData(env.model)
        mujoco.mj_resetData(env.model, replay)
        replay.ctrl[0] = env.high
        mujoco.mj_forward(env.model, replay)
        samples, expected, snapshots = [], [], []
        v = np.zeros(6)
        # Reach, leave, and reach again: test counter reset and success history.
        lengths = [.100] * 150 + [env.high] * 100 + [.100] * 150
        for length in lengths:
            obs, reward, terminated, truncated, info = env.step(env.action_for_length(length))
            self.assertFalse(terminated)
            expected.append((obs.copy(), reward, info.copy()))
            replay.ctrl[0] = info["command_length"]
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, replay)
                mujoco.mj_forward(env.model, replay)
                mujoco.mj_objectVelocity(env.model, replay, mujoco.mjtObj.mjOBJ_SITE, env.tip, v, 0)
                samples.append(np.r_[replay.qvel, replay.site_xpos[env.tip], v[3:], replay.time])
            snapshots.append((replay.qpos.copy(), replay.qvel.copy(),
                              replay.site_xpos[env.tip].copy(), v[3:].copy(),
                              replay.ten_length[env.tendon], replay.ten_velocity[env.tendon]))
        self.assertTrue(truncated)

        def evaluate(inputs):
            def tick(carry, x):
                count, ever, first = carry
                result = task_sample(x[:3], x[3:6], x[6:9], count, ever, first,
                                     x[9], env.config, env.model.opt.timestep)
                count, held, success, ever, first, terms = result
                return (count, ever, first), (held, success, ever, first, terms)
            return jax.lax.scan(tick, (jp.int32(0), jp.array(False), jp.float32(-1)), inputs)[1]

        held, success, ever, first, terms = jax.device_get(jax.jit(evaluate)(jp.asarray(samples, jp.float32)))
        block_terms = terms.reshape(env.max_steps, env.substeps, -1).sum(axis=1)
        observe = jax.jit(lambda q, qv, p, v, u, h, s, tl, tv: observation(
            q, qv, p, v, u, h, s, tl, tv, env.low, env.high, env.max_steps, env.config))
        for i, (snapshot, (obs, reward, info)) in enumerate(zip(snapshots, expected)):
            index = (i + 1) * env.substeps - 1
            q, qv, p, velocity, tl, tv = snapshot
            generated = observe(q, qv, p, velocity, info["command_length"], held[index],
                                jp.int32(i + 1), tl, tv)
            np.testing.assert_allclose(generated, obs, atol=3e-6, rtol=3e-6)
            # Command-change penalty is evaluated once per control step.
            expected_reward = float(block_terms[i].sum()) + info["reward_terms"]["command_change"]
            self.assertAlmostEqual(expected_reward, reward, places=6)
            self.assertEqual(bool(success[index]), info["is_success"])
            self.assertEqual(bool(ever[index]), info["success_ever"])
            target_first = -1 if info["first_success_time"] is None else info["first_success_time"]
            self.assertAlmostEqual(float(first[index]), target_first, places=5)
        self.assertTrue(bool(success[-1]))
        self.assertEqual(obs0.shape, (16,))


if __name__ == "__main__":
    unittest.main()
