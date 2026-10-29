"""Run a fixed tendon command through the local task, with optional live viewer.

No learning is performed. The CSV records rewards, command slew, success and
holding time; the final checks also verify reset and time-limit behavior.
"""

import argparse
from contextlib import nullcontext
import csv
from pathlib import Path
import time

import numpy as np

from finger_env import FingerHeightEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--length", type=float, default=0.100)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("runs") / "acceptance.csv")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    env = FingerHeightEnv(config_path=args.config)
    action = env.action_for_length(args.length)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    initial_obs, initial_info = env.reset(seed=0)
    assert initial_obs.shape == (16,) and initial_obs.dtype == np.float32
    assert np.array_equal(env.data.qpos, np.zeros(3)) and np.array_equal(env.data.qvel, np.zeros(3))
    assert initial_info["command_length"] == env.high and initial_info["hold_seconds"] == 0
    context = nullcontext(None)
    if args.render:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(env.model, env.data, show_right_ui=False)
    total_returns = []
    fieldnames = ["episode", "time", "height", "height_error", "tip_speed", "command_length",
                  "tendon_length", "tendon_length_sensor", "tendon_length_observed",
                  "hold_seconds", "is_success", "success_ever", "reward",
                  "tracking", "tip_speed_penalty", "joint_speed_penalty", "command_change", "holding", "success_bonus"]
    print(f"目标 {env.config.target_height * 100:.1f} cm；请求肌腱长度 {args.length:.3f} m；自动限速，不进行 PPO 训练。", flush=True)
    try:
        with args.csv.open("w", newline="") as output, context as viewer:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for episode in range(args.episodes):
                with viewer.lock() if viewer else nullcontext():
                    obs, info = env.reset(seed=episode)
                    # Only the len_if observation has Aero-style random noise.
                    assert np.array_equal(np.delete(obs, 14), np.delete(initial_obs, 14)), "Fixed physical observations differ"
                assert info["first_success_time"] is None and not info["success_ever"]
                last_command = info["command_length"]
                episode_return = 0.0
                first_success = None
                next_print = 0.0
                for step in range(env.max_steps):
                    tick = time.monotonic()
                    if viewer and not viewer.is_running():
                        print("Viewer closed; acceptance interrupted.")
                        return
                    with viewer.lock() if viewer else nullcontext():
                        obs, reward, terminated, truncated, info = env.step(action)
                    if viewer:
                        viewer.sync()
                    assert not terminated, "Simulation failed"
                    assert np.isfinite(obs).all() and np.isfinite(reward)
                    assert abs(info["command_length"] - last_command) <= env.config.max_command_speed * env.config.control_dt + 1e-12
                    assert np.isclose(reward, sum(info["reward_terms"].values()))
                    last_command = info["command_length"]
                    episode_return += reward
                    if info["is_success"] and first_success is None:
                        first_success = info["time"]
                    assert truncated == (step == env.max_steps - 1), "Episode ended early"
                    terms = info["reward_terms"]
                    row = {key: info[key] for key in fieldnames if key in info}
                    row.update(episode=episode + 1, reward=reward, tracking=terms["tracking"],
                               tip_speed_penalty=terms["tip_speed"], joint_speed_penalty=terms["joint_speed"],
                               command_change=terms["command_change"], holding=terms["holding"], success_bonus=terms["success"])
                    writer.writerow(row)
                    if info["time"] + 1e-9 >= next_print:
                        print(f"回合 {episode+1} t={info['time']:5.2f}s "
                              f"高度={info['height']*100:6.3f}cm 速度={info['tip_speed']*100:6.3f}cm/s "
                              f"指令={info['command_length']:.4f}m 保持={info['hold_seconds']:.2f}s "
                              f"奖励={reward:+.4f} {'达标' if info['is_success'] else '未达标'}", flush=True)
                        next_print += 0.5
                    if viewer:
                        time.sleep(max(0, env.config.control_dt - (time.monotonic() - tick)))
                assert first_success is not None, "Command did not achieve a full one-second hold"
                assert env.max_steps * env.config.control_dt - first_success >= 1, "Need at least one second of observation after success"
                assert info["is_success"], "Finger did not maintain success through the episode end"
                total_returns.append(episode_return)
                print(f"回合 {episode+1} 完成：首次达标 {first_success:.2f}s，总奖励 {episode_return:.4f}。", flush=True)
    finally:
        env.close()

    # A released finger must not be marked successful, and should earn less reward.
    env.reset(seed=0)
    release_return = 0.0
    for _ in range(env.max_steps):
        _, reward, terminated, _, info = env.step(env.action_for_length(env.high))
        assert not terminated and not info["success_ever"]
        release_return += reward
    assert min(total_returns) > release_return, "Reward does not favor reaching the target over release"
    try:
        env.step(action)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stepping after timeout should require reset")
    env.close()
    print(f"验收通过：初态、动作限速、达标后继续运行、超时重置及奖励对照均正常。释放对照总奖励={release_return:.4f}", flush=True)
    print(f"CSV: {args.csv.resolve()}", flush=True)


if __name__ == "__main__":
    main()
