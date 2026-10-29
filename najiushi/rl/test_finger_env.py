"""Task-level checks; no PPO, GPU or additional test packages required."""

import unittest
from dataclasses import replace

import numpy as np

from finger_env import FingerHeightEnv


class FingerTaskTests(unittest.TestCase):
    def setUp(self):
        self.env = FingerHeightEnv()
        self.env.reset(seed=1)

    def tearDown(self):
        self.env.close()

    def test_mapping_slew_and_reward_accounting(self):
        np.testing.assert_allclose(self.env.action_for_length(self.env.low), [1])
        np.testing.assert_allclose(self.env.action_for_length(self.env.high), [-1])
        previous = self.env.command
        obs, reward, terminated, truncated, info = self.env.step([1])
        self.assertEqual(obs.shape, (16,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertAlmostEqual(previous - info["command_length"], 0.0004)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))
        self.assertFalse(terminated or truncated)

    def test_success_continues_and_can_be_lost(self):
        action = self.env.action_for_length(0.100)
        achieved = None
        for _ in range(150):
            _, _, terminated, truncated, info = self.env.step(action)
            if info["is_success"] and achieved is None:
                achieved = info["time"]
                self.assertGreaterEqual(info["hold_seconds"] + 1e-9, 1)
            self.assertFalse(terminated or truncated)
        self.assertIsNotNone(achieved)
        self.assertTrue(info["is_success"])
        # Leaving the goal region clears CURRENT success, but not episode history.
        for _ in range(100):
            _, _, terminated, truncated, info = self.env.step([-1])
            self.assertFalse(terminated or truncated)
        self.assertFalse(info["is_success"])
        self.assertTrue(info["success_ever"])
        self.assertEqual(info["hold_seconds"], 0)

    def test_reset_is_fixed_and_clears_episode_state(self):
        initial, _ = self.env.reset()
        for _ in range(120):
            self.env.step(self.env.action_for_length(0.100))
        self.assertTrue(self.env.success_ever)
        obs, info = self.env.reset(seed=99)
        np.testing.assert_array_equal(np.delete(obs, 14), np.delete(initial, 14))
        self.assertFalse(info["success_ever"])
        self.assertIsNone(info["first_success_time"])
        self.assertEqual(self.env.steps, 0)
        self.assertEqual(self.env.data.time, 0)
        self.assertFalse(self.env.data.xfrc_applied.any())

    def test_named_tendon_sensors_noise_and_reproducibility(self):
        obs, info = self.env.reset(seed=7)
        again, _ = self.env.reset(seed=7)
        np.testing.assert_array_equal(obs, again)
        other, _ = self.env.reset(seed=8)
        self.assertNotEqual(float(obs[14]), float(other[14]))
        for _ in range(50):
            obs, _, _, _, info = self.env.step(self.env.action_for_length(.100))
            self.assertAlmostEqual(info['tendon_length_sensor'], self.env.data.ten_length[self.env.tendon])
            self.assertAlmostEqual(float(obs[14])*.1, info['tendon_length_observed'], places=7)
            self.assertLessEqual(abs(info['tendon_length_observed']-info['tendon_length_sensor']), .005)
            self.assertAlmostEqual(float(obs[15])*.1, self.env.data.ten_velocity[self.env.tendon], places=6)
        # Observation noise does not change actuator force, physical motion or reward.
        clean = FingerHeightEnv()
        clean.config = replace(clean.config, tendon_noise_level=0.)
        clean.reset(seed=7)
        self.env.reset(seed=7)
        try:
            for _ in range(50):
                noisy_obs, reward, _, _, _ = self.env.step([0])
                clean_obs, clean_reward, _, _, clean_info = clean.step([0])
                np.testing.assert_array_equal(self.env.data.qpos, clean.data.qpos)
                np.testing.assert_array_equal(self.env.data.actuator_force, clean.data.actuator_force)
                self.assertEqual(reward, clean_reward)
                self.assertEqual(clean_info['tendon_length_observed'], clean_info['tendon_length_sensor'])
                np.testing.assert_array_equal(np.delete(noisy_obs, 14), np.delete(clean_obs, 14))
        finally:
            clean.close()

    def test_time_limit_requires_reset(self):
        for index in range(self.env.max_steps):
            _, _, terminated, truncated, _ = self.env.step([-1])
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == self.env.max_steps - 1)
        with self.assertRaises(RuntimeError):
            self.env.step([-1])
        self.env.reset()
        self.env.step([-1])

    def test_rejects_invalid_actions(self):
        for action in ([np.nan], [np.inf], [0, 1], 0):
            with self.assertRaises(ValueError):
                self.env.step(action)


if __name__ == "__main__":
    unittest.main()
