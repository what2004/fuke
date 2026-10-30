"""Check pulse timing, force removal and height-only recovery semantics."""
import unittest
from types import SimpleNamespace

import numpy as np
import mujoco

from check_policy_push import FingertipPulse, height_metrics, set_replay_duration
from finger_env import FingerHeightEnv


class PolicyPushTests(unittest.TestCase):
    def test_segment_points_follow_bodies_and_force_direction(self):
        env = FingerHeightEnv()
        try:
            env.reset(seed=0)
            env.data.qpos[:] = np.deg2rad([0, -40, -36])
            env._update_kinematics()
            for point in ['tip', 'distal', 'middle', 'proximal']:
                pulse = FingertipPulse(env, 0., .1, [0, 0, -1], True, point)
                expected = (env.data.site_xpos[env.tip] if point == 'tip'
                            else env.data.xipos[pulse.body])
                np.testing.assert_allclose(pulse.point(), expected)
                down, up, sideways = (np.zeros(env.model.nv) for _ in range(3))
                for force, dest in [([0, 0, -1], down), ([0, 0, 1], up), ([1, 0, 0], sideways)]:
                    mujoco.mj_applyFT(env.model, env.data, np.asarray(force, dtype=float),
                                     np.zeros(3), pulse.point(), pulse.body, dest)
                self.assertGreater(np.linalg.norm(down), 0)
                np.testing.assert_allclose(up, -down)
                np.testing.assert_allclose(sideways, 0., atol=1e-12)
                # A force on a parent segment cannot drive its child joints directly.
                if point == 'proximal':
                    np.testing.assert_allclose(down[1:], 0.)
                if point == 'middle':
                    self.assertAlmostEqual(down[2], 0.)
        finally:
            env.close()

    def test_extended_replay_and_long_pulse(self):
        env = FingerHeightEnv()
        try:
            set_replay_duration(env, 12.)
            self.assertEqual(env.max_steps, 600)
            self.assertEqual(env._policy_training_max_steps, 400)
            pulse = FingertipPulse(env, 4., 2., [0, 0, -.5], True)
            self.assertEqual(pulse.start_tick, 4000)
            self.assertEqual(pulse.duration_ticks, 2000)
            env.reset(seed=0)
            for _ in range(600):
                _, _, terminated, truncated, _ = env.step(env.action_for_length(.1))
                self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertAlmostEqual(env.data.time, 12.)
        finally:
            env.close()

    def test_fingertip_pulse_exact_ticks_and_removal(self):
        env = FingerHeightEnv()
        try:
            env.reset(seed=0)
            pulse = FingertipPulse(env, .02, .003, [0, 0, -.1], True)
            for _ in range(40):
                pulse.step()
            self.assertEqual(pulse.applied_steps, 3)
            self.assertEqual([i for i, r in enumerate(pulse.rows) if r['push_active']], [20, 21, 22])
            self.assertTrue(np.all(env.data.qfrc_applied == 0))
            impulse = sum(r['force_z_n']*.001 for r in pulse.rows)
            self.assertAlmostEqual(impulse, -.0003)
            pulse.close()
        finally:
            env.close()

    def test_recovery_ignores_speed_but_requires_continuous_height_hold(self):
        config = SimpleNamespace(height_tolerance=.005, hold_seconds=.003)
        reference = [dict(time=i*.001, height=.16, height_error=0., tip_speed=1.) for i in range(1, 11)]
        rows = [dict(r) for r in reference]
        rows[4].update(height=.17, height_error=.01)
        metrics = height_metrics(rows, reference, config, .001, .002)
        self.assertTrue(metrics['left_height_band_after_push'])
        self.assertTrue(metrics['final_height_in_band'])
        self.assertAlmostEqual(metrics['height_hold_completed_after_release_s'], .005)
        self.assertAlmostEqual(metrics['max_height_deviation_from_no_push_m'], .01)

    def test_invalid_push_rejected(self):
        env = FingerHeightEnv()
        try:
            for start, duration, force in [(.0005, .1, [0, 0, -.1]),
                                            (4., 0., [0, 0, -.1]),
                                            (7.9, .2, [0, 0, -.1])]:
                with self.assertRaises(ValueError):
                    FingertipPulse(env, start, duration, force, True)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
