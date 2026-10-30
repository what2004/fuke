"""Test a saved PPO policy with one fingertip push and a matched no-push run.

Height-only recovery is reported separately from the original training success.
The policy keeps controlling the tendon throughout; no training occurs here.
"""
import argparse
from contextlib import nullcontext
import csv
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from play_ppo import load_policy, load_run, replay_viewer
from finger_env import PUSH_POINTS


def set_replay_duration(env, seconds):
    ticks = seconds / env.config.control_dt
    if (not np.isfinite(seconds) or seconds <= 0
            or not np.isclose(ticks, round(ticks), rtol=0, atol=1e-7)):
        raise ValueError('Episode duration must be positive and align with control steps')
    env._policy_training_max_steps = env.max_steps
    env.config = replace(env.config, episode_seconds=seconds)
    env.max_steps = round(ticks)


PUSH_BODIES = {'proximal': 'index_proximal', 'middle': 'index_middle', 'distal': 'index_distal'}


class FingertipPulse:
    def __init__(self, env, start, duration, force, enabled, point='tip'):
        self.env, self.enabled = env, enabled
        self.point_name = point
        self.force = np.asarray(force, dtype=float)
        dt = env.model.opt.timestep
        ticks = np.asarray([start, duration]) / dt
        if (self.force.shape != (3,) or not np.isfinite(self.force).all()
                or not np.isfinite(ticks).all() or start < 0 or duration <= 0
                or start + duration >= env.config.episode_seconds
                or not np.allclose(ticks, np.round(ticks), atol=1e-7, rtol=0)):
            raise ValueError('Push must be finite, align with physics ticks, and end before episode end')
        self.start_tick, self.duration_ticks = map(int, np.round(ticks))
        if point == 'tip':
            self.body = int(env.model.site_bodyid[env.tip])
        elif point in PUSH_BODIES or point in PUSH_POINTS:
            body_name = PUSH_BODIES[point] if point in PUSH_BODIES else PUSH_POINTS[point][0]
            self.body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self.body < 0:
                raise ValueError(f'Missing push body: {body_name}')
        else:
            raise ValueError(f'Unknown push point: {point}')
        self.original_step = env._physics_step
        self.rows, self.applied_steps = [], 0

    def point(self):
        # For the uniform cylinders, the inertial center is the cylinder center.
        if self.point_name != 'tip' and self.point_name in PUSH_POINTS:
            return (self.env.data.xpos[self.body] + self.env.data.xmat[self.body].reshape(3, 3)
                    @ np.asarray(PUSH_POINTS[self.point_name][1]))
        return (self.env.data.site_xpos[self.env.tip] if self.point_name == 'tip'
                else self.env.data.xipos[self.body]).copy()

    def step(self):
        e = self.env
        tick = round(e.data.time / e.model.opt.timestep)
        active = self.enabled and self.start_tick <= tick < self.start_tick + self.duration_ticks
        force = self.force if active else np.zeros(3)
        e.data.qfrc_applied[:] = 0
        e.data.xfrc_applied[:] = 0
        position = self.point()
        if active:
            mujoco.mj_applyFT(e.model, e.data, force, np.zeros(3),
                              position, self.body, e.data.qfrc_applied)
            self.applied_steps += 1
        self.original_step()
        self.rows.append(dict(time=float(e.data.time), height=float(e.tip_position[2]),
                              height_error=e.height_error, tip_speed=e.tip_speed,
                              command_length=e.command, tendon_length=float(e.data.ten_length[e.tendon]),
                              force_x_n=float(force[0]), force_y_n=float(force[1]), force_z_n=float(force[2]),
                              push_point=self.point_name,
                              point_x_m=float(position[0]), point_y_m=float(position[1]), point_z_m=float(position[2]),
                              push_active=int(active)))

    def close(self):
        self.env._physics_step = self.original_step
        self.env.data.qfrc_applied[:] = 0
        self.env.data.xfrc_applied[:] = 0


def height_metrics(rows, reference, config, start, duration):
    """Use 1 ms samples; a complete one-second height hold defines recovery."""
    end = start + duration
    affected = [(r, b) for r, b in zip(rows, reference) if r['time'] > start + 1e-8]
    dt = rows[1]['time'] - rows[0]['time']
    held, first, recovered = 0., None, None
    for row in rows:
        if row['time'] <= end + 1e-8:
            continue
        if abs(row['height_error']) <= config.height_tolerance:
            if first is None:
                first = row['time'] - end
            held += dt
            if recovered is None and held + 1e-8 >= config.hold_seconds:
                recovered = row['time'] - end
        else:
            held = 0.
    return dict(max_height_error_after_push_m=max(abs(r['height_error']) for r, _ in affected),
                max_height_deviation_from_no_push_m=max(abs(r['height']-b['height']) for r, b in affected),
                first_height_in_band_after_release_s=first,
                height_hold_completed_after_release_s=recovered,
                left_height_band_after_push=any(abs(r['height_error']) > config.height_tolerance for r, _ in affected),
                final_height_in_band=bool(abs(rows[-1]['height_error']) <= config.height_tolerance))


def draw_push(window, env, force, active, start, duration, pulse):
    """A fixed-length arrow indicates direction, not displacement or force scale."""
    with window.lock():
        scene = window.user_scn
        scene.ngeom = 0
        if active and np.linalg.norm(force) > 0:
            geom = scene.geoms[0]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.array([1., .1, .05, 1.], dtype=np.float32))
            tip = pulse.point()
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .002,
                                tip, tip + np.asarray(force)/np.linalg.norm(force)*.04)
            scene.ngeom = 1
    status = 'PUSH ON' if active else ('Waiting for push' if env.data.time < start else 'Push finished')
    window.set_texts((int(mujoco.mjtFontScale.mjFONTSCALE_150),
                      int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                      f't = {env.data.time:.2f} s | {status}\n'
                      f'World force: {list(force)} N | pulse {start:g}-{start+duration:g} s\n'
                      f'Push point: {pulse.point_name}\n'
                      f'Tip height: {env.tip_position[2]*100:.3f} cm\n'
                      'Red arrow: force direction (visual scale only)', ''))


def rollout(env, policy, seed, start, duration, force, enabled, render, slowdown=1., keep_open=False, point='tip'):
    import jax
    import jax.numpy as jp
    obs, _ = env.reset(seed=seed)
    pulse = FingertipPulse(env, start, duration, force, enabled, point)
    env._physics_step = pulse.step
    context = replay_viewer(env.model, env.data) if render else nullcontext(None)
    key = jax.random.PRNGKey(seed)
    controls = []
    next_report = .5
    try:
        with context as window:
            for step in range(env.max_steps):
                wall = time.monotonic()
                if window and not window.is_running():
                    raise RuntimeError('Viewer closed before experiment completed')
                key, action_key = jax.random.split(key)
                # Keep the recorded 8 s countdown before extending beyond it.
                # The policy sees zero remaining training time after that point.
                training_steps = getattr(env, '_policy_training_max_steps', env.max_steps)
                obs[13] = max(0., 1 - env.steps / training_steps)
                action = np.asarray(policy(jp.asarray(obs), action_key)[0])
                with window.lock() if window else nullcontext():
                    obs, _, terminated, truncated, info = env.step(action)
                controls.append(dict(time=info['time'], action=float(action[0]),
                                     command_length=info['command_length']))
                if terminated:
                    raise RuntimeError('Simulation failed; do not interpret partial traces as recovery')
                if window:
                    draw_push(window, env, force, bool(pulse.rows[-1]['push_active']), start, duration, pulse)
                    window.sync()
                    time.sleep(max(0, env.config.control_dt*slowdown-(time.monotonic()-wall)))
                if step == 0 or info['time'] >= next_report - 1e-8:
                    print(f"{'push' if enabled else 'baseline'} t={info['time']:.2f}s "
                          f"height={info['height']*100:.3f}cm", flush=True)
                    if info['time'] >= next_report - 1e-8:
                        next_report += .5
                if truncated:
                    break
            if window and keep_open:
                print('Playback finished. Close the window to save results and exit.', flush=True)
                while window.is_running():
                    window.sync()
                    time.sleep(.05)
    finally:
        pulse.close()
    return pulse, controls


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--push-start', type=float, default=4.)
    parser.add_argument('--push-duration', type=float, default=2.)
    parser.add_argument('--episode-seconds', type=float, default=12., help='local replay duration; saved training config unchanged')
    parser.add_argument('--force', type=float, nargs=3, default=[0., 0., -.1], metavar=('FX', 'FY', 'FZ'))
    parser.add_argument('--point', choices=list(PUSH_POINTS)+list(PUSH_BODIES), default='tip',
                        help='tip site or the center of a finger segment; point follows that segment')
    parser.add_argument('--render', action='store_true', help='show the pushed episode after baseline')
    parser.add_argument('--slowdown', type=float, default=1., help='wall-clock playback factor, e.g. 5; physics unchanged')
    parser.add_argument('--keep-open', action='store_true', help='freeze final state until the viewer is closed')
    args = parser.parse_args()
    if not np.isfinite(args.slowdown) or args.slowdown < 1:
        parser.error('--slowdown must be finite and at least 1')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    import jax.numpy as jp
    env, checkpoint = load_run(args.run_dir, args.checkpoint)
    try:
        # This paired experiment supplies its own explicit pulse. A snapshot
        # trained with randomized pushes must not add a second hidden pulse.
        if getattr(env.config, 'push_enabled', False):
            env.config = replace(env.config, push_enabled=False)
        set_replay_duration(env, args.episode_seconds)
        # Validate before loading the policy or starting the baseline.
        FingertipPulse(env, args.push_start, args.push_duration, args.force, True, args.point)
        policy = jax.jit(load_policy(checkpoint))
        policy(jp.zeros(16, jp.float32), jax.random.PRNGKey(args.seed))[0].block_until_ready()
        baseline, baseline_actions = rollout(env, policy, args.seed, args.push_start,
                                             args.push_duration, args.force, False, False, point=args.point)
        pushed, pushed_actions = rollout(env, policy, args.seed, args.push_start,
                                         args.push_duration, args.force, True, args.render,
                                         args.slowdown, args.keep_open, args.point)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
        out = args.run_dir.resolve() / f'policy-push-{stamp}'
        out.mkdir()
        for name, rows in [('baseline', baseline.rows), ('push', pushed.rows),
                           ('baseline_actions', baseline_actions), ('push_actions', pushed_actions)]:
            save_csv(out / f'{name}.csv', rows)
        metrics = height_metrics(pushed.rows, baseline.rows, env.config, args.push_start, args.push_duration)
        summary = dict(controller='saved PPO policy, closed loop; no training', checkpoint=str(checkpoint),
                       evaluation='height only; speed still recorded; original training task unchanged',
                       target_height_m=env.config.target_height, height_tolerance_m=env.config.height_tolerance,
                       height_hold_seconds=env.config.hold_seconds, seed=args.seed,
                       episode_seconds=env.config.episode_seconds,
                       policy_time_observation='original training countdown, clamped to zero beyond training duration',
                       push_start_s=args.push_start, push_duration_s=args.push_duration, force_world_n=args.force,
                       push_point=args.point, force_frame='world',
                       applied_physics_steps=pushed.applied_steps,
                       impulse_world_n_s=(np.asarray(args.force)*pushed.applied_steps*env.model.opt.timestep).tolist(),
                       **metrics)
        (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        for label, probe, actions in [('no push', baseline, baseline_actions), ('push', pushed, pushed_actions)]:
            t = [r['time'] for r in probe.rows]
            axes[0].plot(t, [r['height']*100 for r in probe.rows], label=label)
            axes[1].plot(t, [r['command_length']*1000 for r in probe.rows], label=label)
            axes[2].plot(t, [r['tip_speed'] for r in probe.rows], label=label)
            axes[3].plot([r['time'] for r in actions], [r['action'] for r in actions], label=label)
        axes[0].axhspan((env.config.target_height-env.config.height_tolerance)*100,
                       (env.config.target_height+env.config.height_tolerance)*100, color='green', alpha=.12)
        axes[0].axhline(env.config.target_height*100, color='green', ls=':')
        for ax, label in zip(axes, ['Tip height (cm)', 'Target tendon length (mm)', 'Tip speed (m/s)', 'Policy action']):
            ax.axvspan(args.push_start, args.push_start+args.push_duration, color='red', alpha=.15)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
            ax.legend()
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'PPO push test: {args.point}, world force {args.force} N; height-only evaluation')
        fig.tight_layout()
        fig.savefig(out/'recovery.png', dpi=150)
        plt.close(fig)
        print(json.dumps(summary, indent=2), flush=True)
        print(f'Results: {out}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
