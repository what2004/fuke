"""Random sampling, pulse timing and post-release recovery semantics."""
from dataclasses import replace
from pathlib import Path
import unittest

import jax
import jax.numpy as jp
import numpy as np

from finger_env import FingerHeightEnv, TaskConfig, PUSH_POINTS, PUSH_DIRECTIONS
from finger_env_mjx import FingerHeightMjxEnv, task_sample


CONFIG = Path(__file__).with_name('config_push.json')


class RandomPushTests(unittest.TestCase):
    def test_point_layout_and_direction_normalization(self):
        self.assertEqual(len(PUSH_POINTS), 10)
        self.assertEqual(len(PUSH_DIRECTIONS), 8)
        np.testing.assert_allclose(np.linalg.norm(PUSH_DIRECTIONS, axis=1), 1.)
        np.testing.assert_array_equal(PUSH_DIRECTIONS[:, 0], 0.)
        for body, pos in PUSH_POINTS.values():
            self.assertTrue(body.startswith('index_'))
            self.assertGreater(pos[1], 0.)

    def test_seeded_native_sampling_and_coverage(self):
        env = FingerHeightEnv(config_path=CONFIG)
        points, directions, enabled = set(), set(), set()
        try:
            obs, info = env.reset(seed=9)
            again, info2 = env.reset(seed=9)
            np.testing.assert_array_equal(obs, again)
            self.assertEqual(info, info2)
            for seed in range(128):
                _, info = env.reset(seed=seed)
                points.add(info['push_point'])
                directions.add(info['push_direction_index'])
                enabled.add(info['push_sampled'])
                self.assertGreaterEqual(np.linalg.norm(info['push_force_world_n']), 1.-1e-6)
                self.assertLessEqual(np.linalg.norm(info['push_force_world_n']), 4.+1e-6)
            self.assertEqual(len(points), 10)
            self.assertEqual(len(directions), 8)
            self.assertEqual(enabled, {True, False})
        finally:
            env.close()

    def test_jax_sampling_is_batched_reproducible_and_in_range(self):
        env = object.__new__(FingerHeightMjxEnv)
        env.config, env._sim_dt = TaskConfig.load(CONFIG), .001
        env.high = .115
        sample = jax.jit(jax.vmap(lambda key: env._initial_info(key, jp.float32)))
        keys = jax.random.split(jax.random.PRNGKey(6), 128)
        batch, repeat = sample(keys), sample(keys)
        for name in batch:
            np.testing.assert_array_equal(batch[name], repeat[name])
        self.assertEqual(len(np.unique(batch['push_point_index'])), 10)
        self.assertEqual(len(np.unique(batch['push_direction_index'])), 8)
        magnitudes = np.linalg.norm(batch['push_force'], axis=1)
        self.assertGreaterEqual(magnitudes.min(), 1.-1e-6)
        self.assertLessEqual(magnitudes.max(), 4.+1e-6)
        self.assertTrue(np.all(np.asarray(batch['push_start_tick']) >= 3000))
        self.assertTrue(np.all(np.asarray(batch['push_start_tick']) <= 5000))

    def test_native_exact_pulse_ticks_and_reset(self):
        env = FingerHeightEnv(config_path=CONFIG)
        try:
            env.reset(seed=0)
            env.push_sampled = True
            env.push_start_tick, env.push_duration_ticks = 20, 3
            env.push_force = np.array([0., 0., -1.])
            for _ in range(40):
                env._physics_step()
            self.assertEqual(env.push_applied_ticks, 3)
            np.testing.assert_array_equal(env.data.qfrc_applied, 0.)
            env.reset(seed=0)
            self.assertEqual(env.push_applied_ticks, 0)
            np.testing.assert_array_equal(env.data.qfrc_applied, 0.)
        finally:
            env.close()

    def test_height_only_and_recovery_gate(self):
        config = TaskConfig.load(CONFIG)
        args = (jp.zeros(3), jp.array([0., 0., .16]), jp.array([.1, 0., 0.]),
                jp.int32(999), jp.array(False), jp.float32(-1), jp.float32(6), config, .001)
        blocked = task_sample(*args, eligible=False)
        self.assertFalse(bool(blocked[2]))
        self.assertEqual(int(blocked[0]), 0)
        allowed = task_sample(*args, eligible=True)
        self.assertTrue(bool(allowed[2]))
        strict_args = (*args[:7], replace(config, require_low_speed=True), .001)
        self.assertFalse(bool(task_sample(*strict_args)[2]))


if __name__ == '__main__':
    unittest.main()
