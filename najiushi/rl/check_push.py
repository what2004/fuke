"""Native MuJoCo: one world-frame force at the fingertip, plus a no-push control.

No PPO or MJX changes. Save physical-step traces, recovery metrics and plots.
"""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from finger_env import FingerHeightEnv


@dataclass(frozen=True)
class PushConfig:
    start_seconds: float
    duration_seconds: float
    force_world_n: tuple
    command_length_m: float

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        return cls(**{**values, "force_world_n": tuple(values["force_world_n"])})

    def steps(self, dt, episode_seconds):
        force = np.asarray(self.force_world_n, dtype=float)
        numbers = [self.start_seconds, self.duration_seconds, self.command_length_m]
        if force.shape != (3,) or not np.isfinite(force).all() or not np.isfinite(numbers).all():
            raise ValueError("Push settings must contain finite values and a 3D force")
        if self.start_seconds < 0 or self.duration_seconds <= 0:
            raise ValueError("Push start must be nonnegative and duration must be positive")
        end = self.start_seconds + self.duration_seconds
        if end >= episode_seconds:
            raise ValueError("Push must end before the episode ends")
        ticks = np.array([self.start_seconds, self.duration_seconds]) / dt
        if not np.allclose(ticks, np.round(ticks), rtol=0, atol=1e-7):
            raise ValueError("Push timing must align with the physical timestep")
        return int(round(ticks[0])), int(round(ticks[1]))


class PushProbeEnv(FingerHeightEnv):
    """Apply force at index_tip, not at the distal body's center of mass."""

    def __init__(self, push, enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.push = push
        self.enabled = enabled
        self.start_tick, self.duration_ticks = push.steps(self.model.opt.timestep, self.config.episode_seconds)
        self.action_for_length(push.command_length_m)
        self.tip_body = int(self.model.site_bodyid[self.tip])
        self.spring_ids = [self._id(mujoco.mjtObj.mjOBJ_TENDON, name)
                           for name in ("index_pip_return", "index_mcp_return")]
        self.rows = []
        self.applied_steps = 0
        self.impulse = np.zeros(3)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.rows = []
        self.applied_steps = 0
        self.impulse = np.zeros(3)
        return result

    def _physics_step(self):
        tick = round(self.data.time / self.model.opt.timestep)
        active = self.enabled and self.start_tick <= tick < self.start_tick + self.duration_ticks
        force = np.asarray(self.push.force_world_n) if active else np.zeros(3)
        # This experiment owns its applied-force buffers; clear at every substep.
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        if active:
            mujoco.mj_applyFT(self.model, self.data, force, np.zeros(3),
                              self.data.site_xpos[self.tip], self.tip_body, self.data.qfrc_applied)
            self.applied_steps += 1
            self.impulse += force * self.model.opt.timestep
        super()._physics_step()
        row = dict(time=float(self.data.time), height=self.tip_position[2],
                   height_error=self.height_error, tip_speed=self.tip_speed,
                   command_length=self.command, tendon_length=self.data.ten_length[self.tendon],
                   active_pull_n=-self.data.actuator_force[self.actuator], push_active=int(active))
        row.update(zip(("force_x_n", "force_y_n", "force_z_n"), force))
        row.update(zip(("mcp_deg", "pip_deg", "dip_deg"), np.rad2deg(self.data.qpos)))
        for name, index in zip(("pip_spring_elastic_n", "mcp_spring_elastic_n"), self.spring_ids):
            extension = max(0., float(self.data.ten_length[index] - self.model.tendon_lengthspring[index, 1]))
            row[name] = float(self.model.tendon_stiffness[index]) * extension
        self.rows.append(row)

    def close(self):
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        super().close()


def recovery_metrics(rows, reference, task, push):
    """Recovery is measured from pulse end, using 1 ms samples, not plot rounding."""
    end = push.start_seconds + push.duration_seconds
    eps = 1e-8
    post = [r for r in rows if r["time"] >= end - eps]
    affected = [(r, b) for r, b in zip(rows, reference) if r["time"] > push.start_seconds + eps]
    held, first_entry, recovered, stable_entry = 0., None, None, None
    dt = rows[1]["time"] - rows[0]["time"]
    for r in post:
        # Sample exactly at pulse end describes a still-forced step, not recovery time.
        if r["time"] <= end + eps:
            continue
        valid = abs(r["height_error"]) <= task.height_tolerance and r["tip_speed"] <= task.speed_limit
        if valid:
            if first_entry is None:
                first_entry = r["time"] - end
            held += dt
            if recovered is None and held + eps >= task.hold_seconds:
                recovered = r["time"] - end
                stable_entry = r["time"] - end - held
        else:
            held = 0.
    before = [r for r in rows if r["time"] <= push.start_seconds + eps][-1]
    return {
        "max_target_height_error_after_push_m": max(abs(r["height_error"]) for r, _ in affected),
        "max_height_deviation_from_no_push_m": max(abs(r["height"] - b["height"]) for r, b in affected),
        "max_tip_speed_after_push_m_s": max(r["tip_speed"] for r, _ in affected),
        "first_qualifying_sample_after_release_s": first_entry,
        "full_hold_reacquired_after_release_s": recovered,
        "sustained_recovery_started_after_release_s": stable_entry,
        "final_height_m": rows[-1]["height"],
        "pre_push_height_m": before["height"],
        "final_qualifies": bool(abs(rows[-1]["height_error"]) <= task.height_tolerance
                                and rows[-1]["tip_speed"] <= task.speed_limit),
        "lost_height_or_speed_qualification": any(abs(r["height_error"]) > task.height_tolerance
                                                  or r["tip_speed"] > task.speed_limit for r, _ in affected),
    }


def rollout(push, enabled, config=None, render=False):
    env = PushProbeEnv(push, enabled=enabled, config_path=config)
    env.reset(seed=0)
    action = env.action_for_length(push.command_length_m)
    context = nullcontext(None)
    if render:
        from mujoco import viewer
        context = viewer.launch_passive(env.model, env.data, show_right_ui=False)
    try:
        with context as window:
            for _ in range(env.max_steps):
                start = time.monotonic()
                if window and not window.is_running():
                    raise RuntimeError("Viewer closed before the experiment finished")
                with window.lock() if window else nullcontext():
                    _, _, terminated, _, _ = env.step(action)
                if terminated:
                    raise RuntimeError("Simulation failed; no recovery claim can be made")
                if window:
                    window.sync()
                    time.sleep(max(0, env.config.control_dt - (time.monotonic() - start)))
        return env.rows, env.config, env.applied_steps, env.impulse.copy()
    finally:
        env.close()


def plot_recovery(rows, reference, task, push, directory):
    import os
    # Keep the cache beside this experiment when the user's config is read-only.
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["time"] for r in rows])
    def values(key, source=rows):
        return np.array([r[key] for r in source])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout="constrained")
    ax = axes[0, 0]
    ax.axhspan((task.target_height-task.height_tolerance)*100, (task.target_height+task.height_tolerance)*100,
               color="green", alpha=.10, label="Height tolerance")
    ax.axhline(task.target_height*100, color="green", ls=":", label="Target")
    ax.plot(t, values("height", reference)*100, "--", color="0.45", label="No push")
    ax.plot(t, values("height")*100, label="Push baseline")
    ax.set_ylabel("Tip height (cm)")
    ax = axes[0, 1]
    ax.plot(t, values("tip_speed"), label="Tip speed")
    ax.plot(t, values("tip_speed", reference), "--", color="0.45", label="No push")
    ax.axhline(task.speed_limit, color="green", ls=":", label="Speed limit")
    ax.set_ylabel("3D tip speed (m/s)")
    for key, label in [("mcp_deg", "MCP"), ("pip_deg", "PIP"), ("dip_deg", "DIP")]:
        axes[1, 0].plot(t, values(key), label=label)
    axes[1, 0].set_ylabel("Joint angle (degrees)")
    for key, label in [("command_length", "Length command"), ("tendon_length", "Actual tendon")]:
        axes[1, 1].plot(t, values(key)*1000, label=label)
    axes[1, 1].set_ylabel("Active tendon length (mm)")
    for key, label in [("active_pull_n", "Active pull"), ("pip_spring_elastic_n", "PIP elastic component"),
                       ("mcp_spring_elastic_n", "MCP elastic component")]:
        axes[2, 0].plot(t, values(key), label=label)
    axes[2, 0].set_ylabel("Tendon force (N)")
    for key, label in [("force_x_n", "World Fx"), ("force_y_n", "World Fy"), ("force_z_n", "World Fz")]:
        axes[2, 1].step(t, values(key), where="pre", label=label)
    axes[2, 1].set_ylabel("Force at fingertip (N)")
    for ax in axes.flat:
        ax.axvspan(push.start_seconds, push.start_seconds+push.duration_seconds, color="orange", alpha=.2)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlim(0, t[-1])
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    fig.suptitle("Fixed tendon command: one fingertip push and recovery\n"
                 f"Target {task.target_height*100:.1f} cm | command {push.command_length_m:.3f} m | "
                 f"world force {push.force_world_n} N at {push.start_seconds:g} s for {push.duration_seconds:g} s",
                 fontsize=12)
    fig.savefig(directory / "recovery.png", dpi=160)
    fig.savefig(directory / "recovery.svg")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0, 0].plot(t, (values("height")-values("height", reference))*1000)
    axes[0, 0].set_ylabel("Height change vs no push (mm)")
    axes[0, 1].plot(t, values("tip_speed"), label="Tip speed")
    axes[0, 1].axhline(task.speed_limit, color="green", ls=":", label="Speed limit")
    axes[0, 1].set_ylabel("3D tip speed (m/s)")
    for key, label in [("mcp_deg", "MCP"), ("pip_deg", "PIP"), ("dip_deg", "DIP")]:
        axes[1, 0].plot(t, values(key)-values(key, reference), label=label)
    axes[1, 0].set_ylabel("Joint change vs no push (degrees)")
    for key, label in [("active_pull_n", "Active pull"), ("pip_spring_elastic_n", "PIP elastic"),
                       ("mcp_spring_elastic_n", "MCP elastic")]:
        axes[1, 1].plot(t, values(key)-values(key, reference), label=label)
    axes[1, 1].set_ylabel("Force change vs no push (N)")
    left, right = max(0, push.start_seconds-.15), min(t[-1], push.start_seconds+push.duration_seconds+1.5)
    selected = (t >= left) & (t <= right)
    for ax in axes.flat:
        ax.axvspan(push.start_seconds, push.start_seconds+push.duration_seconds, color="orange", alpha=.2)
        ax.set_xlim(left, right)
        for line in ax.lines:
            if len(line.get_ydata()) == len(t):
                line.set_data(t[selected], np.asarray(line.get_ydata())[selected])
        ax.relim();ax.autoscale_view(scalex=False)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Simulation time (s)")
    fig.suptitle("Push/release detail | orange = force active | same fixed command in both trials\n"
                 "Spring traces show elastic components only (damping excluded)", fontsize=12)
    fig.savefig(directory / "recovery_detail.png", dpi=160)
    fig.savefig(directory / "recovery_detail.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push-config", type=Path, default=Path(__file__).with_name("push_config.json"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    push = PushConfig.load(args.push_config)
    baseline, task, _, _ = rollout(push, False, args.config)
    rows, _, steps, impulse = rollout(push, True, args.config, args.render)
    dt = rows[1]["time"] - rows[0]["time"]
    metrics = recovery_metrics(rows, baseline, task, push)
    metrics.update(applied_physics_steps=steps, applied_duration_s=steps*dt,
                   impulse_world_n_s=impulse.tolist(), controller="fixed tendon command; no PPO")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
    directory = args.output_dir or Path(__file__).with_name("runs") / f"push-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    for filename, data in [("push.csv", rows), ("no_push.csv", baseline)]:
        with (directory / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    snapshot = {"push": vars(push), "task": vars(task), "mujoco": mujoco.__version__}
    model_path = Path(__file__).resolve().parents[1] / "descrption/hand.xml"
    (directory / "hand.xml").write_text(model_path.read_text())
    for filename, obj in [("metrics.json", metrics), ("config.json", snapshot)]:
        (directory / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    plot_recovery(rows, baseline, task, push, directory)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Results: {directory.resolve()}")


if __name__ == "__main__":
    main()
