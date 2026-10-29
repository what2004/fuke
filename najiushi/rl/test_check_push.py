"""Verify fingertip force projection, physical-step timing and repeatability."""
import unittest
import mujoco
import numpy as np
from check_push import PushConfig, PushProbeEnv, recovery_metrics, rollout


class PushTests(unittest.TestCase):
    def test_force_timing_within_control_step_and_cleanup(self):
        push = PushConfig(.043, .007, (0., 0., -.1), .100)
        env = PushProbeEnv(push)
        try:
            env.reset()
            for _ in range(4):
                env.step(env.action_for_length(.100))
            active = [r for r in env.rows if r['push_active']]
            self.assertEqual(len(active), 7)
            self.assertAlmostEqual(active[0]['time'], .044)
            self.assertAlmostEqual(active[-1]['time'], .050)
            np.testing.assert_allclose(env.impulse, [0, 0, -.0007], atol=1e-12)
            self.assertFalse(env.data.qfrc_applied.any())
            env.reset()
            self.assertEqual(env.applied_steps, 0)
            self.assertFalse(env.data.qfrc_applied.any())
        finally:
            env.close()

    def test_force_at_tip_matches_site_jacobian(self):
        push = PushConfig(0., .001, (0., .03, -.1), .100)
        env = PushProbeEnv(push)
        try:
            env.reset()
            env.data.qpos[:] = np.deg2rad([-20, -30, -27])
            env._update_kinematics()
            jac = np.zeros((3, env.model.nv))
            mujoco.mj_jacSite(env.model, env.data, jac, None, env.tip)
            expected = jac.T @ np.array(push.force_world_n)
            env._physics_step()
            np.testing.assert_allclose(env.data.qfrc_applied, expected, atol=1e-12)
        finally:
            env.close()
        self.assertFalse(env.data.qfrc_applied.any())

    def test_repeatable_pulse_recovers_and_matches_unforced_prefix(self):
        push = PushConfig(4., .1, (0., 0., -.1), .100)
        baseline, task, _, _ = rollout(push, False)
        rows, _, count, impulse = rollout(push, True)
        again, _, _, _ = rollout(push, True)
        np.testing.assert_array_equal([r['height'] for r in rows], [r['height'] for r in again])
        np.testing.assert_array_equal([r['height'] for r in rows[:4000]], [r['height'] for r in baseline[:4000]])
        self.assertEqual(count, 100)
        np.testing.assert_allclose(impulse, [0, 0, -.01], atol=1e-12)
        metrics = recovery_metrics(rows, baseline, task, push)
        self.assertTrue(metrics['lost_height_or_speed_qualification'])
        self.assertTrue(metrics['final_qualifies'])
        self.assertGreater(metrics['max_height_deviation_from_no_push_m'], .00005)
        self.assertLess(metrics['full_hold_reacquired_after_release_s'], 2.)
        self.assertGreaterEqual(metrics['full_hold_reacquired_after_release_s'], 1.)

    def test_reject_invalid_schedule(self):
        for start, duration, force in [(-1, .1, (0, 0, 1)), (4, 0, (0, 0, 1)),
                                        (7.95, .1, (0, 0, 1)), (4.0005, .1, (0, 0, 1)),
                                        (4, .1, (0, np.nan, 1))]:
            with self.assertRaises(ValueError):
                PushConfig(start, duration, force, .1).steps(.001, 8.)


if __name__ == '__main__':
    unittest.main()
