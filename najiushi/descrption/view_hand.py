"""Open the hand viewer and report fingertip height in world coordinates.

Space: pause/resume simulation. R: reset to the home keyframe.
The green index_tip site marks the center of the distal cylinder's end face.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np


class HeightHoldMonitor:
    """Measure uninterrupted success in simulation seconds, including pauses."""

    def __init__(self, target=0.16, tolerance=0.005, speed_limit=0.01, hold_time=1.0):
        self.target = target
        self.tolerance = tolerance
        self.speed_limit = speed_limit
        self.hold_time = hold_time
        self.reset()

    def reset(self):
        self.since = None
        self.last_time = None
        self.held = 0.0
        self.success = False

    def update(self, sim_time, height, speed):
        if self.last_time is not None and sim_time < self.last_time:
            self.reset()
        self.last_time = sim_time
        if abs(height - self.target) <= self.tolerance and speed <= self.speed_limit:
            if self.since is None:
                self.since = sim_time
            self.held = sim_time - self.since
        else:
            self.since = None
            self.held = 0.0
        self.success = self.since is not None and self.held + 1e-9 >= self.hold_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=Path(__file__).with_name("hand.xml"))
    parser.add_argument("--print-hz", type=float, default=5, help="terminal updates per second")
    parser.add_argument("--duration", type=float, help="exit after this many wall-clock seconds")
    parser.add_argument("--headless", action="store_true", help="test telemetry without opening a window")
    parser.add_argument("--target-height", type=float, default=0.16, help="target world Z, meters")
    parser.add_argument("--height-tol", type=float, default=0.005, help="height tolerance, meters")
    parser.add_argument("--speed-limit", type=float, default=0.01, help="maximum 3D fingertip speed, m/s")
    parser.add_argument("--hold-time", type=float, default=1, help="required continuous simulation seconds")
    parser.add_argument("--ctrl", type=float, help="optional initial tendon target length, meters")
    args = parser.parse_args()
    if args.print_hz <= 0 or (args.duration is not None and args.duration <= 0):
        parser.error("print-hz and duration must be positive")
    if args.headless and args.duration is None:
        args.duration = 5
    if not np.isfinite([args.target_height, args.height_tol, args.speed_limit, args.hold_time]).all():
        parser.error("target parameters must be finite")
    if min(args.height_tol, args.speed_limit, args.hold_time) <= 0:
        parser.error("height-tol, speed-limit and hold-time must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    data = mujoco.MjData(model)
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "index_tip")
    if tip < 0:
        raise ValueError("The XML must define the index_tip site at the fingertip")
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if args.ctrl is not None and not model.actuator_ctrlrange[0, 0] <= args.ctrl <= model.actuator_ctrlrange[0, 1]:
        parser.error(f"ctrl must be within {model.actuator_ctrlrange[0].tolist()}")
    monitor = HeightHoldMonitor(args.target_height, args.height_tol, args.speed_limit, args.hold_time)
    velocity = np.zeros(6)

    def reset():
        if home >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home)
        else:
            mujoco.mj_resetData(model, data)
        if args.ctrl is not None:
            data.ctrl[0] = args.ctrl
        mujoco.mj_forward(model, data)
        monitor.reset()

    def observe():
        mujoco.mj_forward(model, data)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tip, velocity, 0)
        xyz = data.site_xpos[tip].copy()
        speed = float(np.linalg.norm(velocity[3:]))
        monitor.update(data.time, xyz[2], speed)
        return xyz, speed

    reset()
    state = {"paused": False, "reset": False}

    def key_callback(key):
        if key == 32:
            state["paused"] = not state["paused"]
        elif key in (82, 114):
            state["reset"] = True

    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(
            model, data, key_callback=key_callback, show_right_ui=True
        )

    print("指尖定义：绿色点 index_tip，远节圆柱端面中心。高度为世界坐标 Z。", flush=True)
    print("右侧 Control 面板可调肌腱长度；空格暂停/继续，R 复位，Ctrl+C 退出。", flush=True)
    print(
        f"判定：目标 {args.target_height * 100:.2f} ± {args.height_tol * 100:.2f} cm，"
        f"指尖三维速度 ≤ {args.speed_limit * 100:.2f} cm/s，连续保持 {args.hold_time:.2f} 秒。",
        flush=True,
    )
    start = time.monotonic()
    next_print = start
    steps_per_frame = max(1, round(0.02 / model.opt.timestep))
    frame_time = steps_per_frame * model.opt.timestep
    try:
        with context as viewer:
            while viewer is None or viewer.is_running():
                tick = time.monotonic()
                if args.duration is not None and tick - start >= args.duration:
                    break
                with viewer.lock() if viewer is not None else nullcontext():
                    if state["reset"]:
                        reset()
                        state["reset"] = False
                    if not state["paused"]:
                        for _ in range(steps_per_frame):
                            mujoco.mj_step(model, data)
                            # Check every physics step so brief violations reset the timer.
                            observe()
                    xyz, speed = observe()
                    sim_time = data.time
                if viewer is not None:
                    viewer.sync()
                if tick >= next_print:
                    print(
                        f"t={sim_time:7.3f}s  指尖高度 Z={xyz[2]:.6f} m "
                        f"({xyz[2] * 100:.3f} cm)  "
                        f"误差={(xyz[2] - args.target_height) * 100:+.3f} cm  "
                        f"速度={speed * 100:.3f} cm/s  "
                        f"保持={min(monitor.held, args.hold_time):.2f}/{args.hold_time:.2f}s  "
                        + ("[达标]" if monitor.success else "[未达标]")
                        + ("  [暂停]" if state["paused"] else ""),
                        flush=True,
                    )
                    next_print = tick + 1 / args.print_hz
                time.sleep(max(0, frame_time - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
