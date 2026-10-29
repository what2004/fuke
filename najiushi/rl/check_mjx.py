"""Check this hand with the Python/MJX environment used for training.

No packages are installed and hand.xml is never modified. Run --native-only
for the ordinary MuJoCo reference; the default additionally requires MJX-JAX.
Use --task to check the full height task, including native/MJX parity.
"""

import argparse
import importlib.metadata
import inspect
from pathlib import Path
import sys


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def check_task(args):
    import jax
    import jax.numpy as jp
    import numpy as np

    from finger_env import FingerHeightEnv
    from finger_env_mjx import FingerHeightMjxEnv

    env = FingerHeightMjxEnv(config_path=args.config, xml_path=args.xml)
    native = FingerHeightEnv(config_path=args.config, xml_path=args.xml)
    action = env.action_for_length(.100)
    print("Task backend: MJX-JAX; devices:", jax.devices(), flush=True)
    print("Compiling reset and a complete task episode; initial compilation may take several minutes ...", flush=True)

    def episode(state):
        def tick(s, _):
            s = env.step(s, action)
            trace = (s.obs, s.reward, s.done, s.metrics["height"], s.metrics["tip_speed"],
                     s.info["hold_seconds"], s.info["is_success"], s.info["command_length"])
            return s, trace
        return jax.lax.scan(tick, state, None, length=env.max_steps)

    reset = jax.jit(env.reset)
    run = jax.jit(episode)
    initial = reset(jax.random.PRNGKey(0))
    initial_obs, _ = native.reset(seed=0)
    # NumPy and JAX use independent RNG algorithms; compare physical entries,
    # and verify the noisy tendon sample against its own sensor instead.
    np.testing.assert_allclose(np.delete(initial.obs, 14), np.delete(initial_obs, 14), atol=1e-5)
    noise_bound = env.config.tendon_noise_level * env.config.tendon_length_noise_m
    assert abs(float(initial.obs[14])*.1 - float(initial.info["tendon_length_sensor"])) <= noise_bound + 1e-7
    final, trace = run(initial)
    obs, rewards, done, heights, speeds, held, success, command = jax.device_get(trace)
    assert np.isfinite(obs).all() and np.isfinite(rewards).all()
    assert not done[:-1].any() and done[-1] == 1, "Task must finish only at the configured time limit"
    assert bool(final.info["truncated"]) and not bool(final.info["terminated"])
    assert bool(success[-1]) and np.isclose(held[-1], env.config.hold_seconds)
    max_delta = env.config.max_command_speed * env.dt
    assert np.max(np.abs(np.diff(np.r_[env.high, command]))) <= max_delta + 1e-7
    assert int(initial.info["task_steps"]) == 0, "step must not mutate the original state"

    reference_height, reference_rewards = [], []
    for _ in range(native.max_steps):
        _, reward, terminated, _, info = native.step(native.action_for_length(.100))
        assert not terminated
        reference_height.append(info["height"])
        reference_rewards.append(reward)
    assert info["is_success"], "Native baseline did not succeed"
    height_difference = abs(float(heights[-1]) - reference_height[-1])
    joint_difference = np.rad2deg(np.max(np.abs(np.asarray(final.data.qpos) - native.data.qpos)))
    print(f"Final height: MJX={heights[-1]*100:.4f} cm; native={reference_height[-1]*100:.4f} cm", flush=True)
    print(f"Largest trajectory height difference: {np.max(np.abs(heights-reference_height))*100:.4f} cm", flush=True)
    print(f"Largest final joint difference: {joint_difference:.4f} deg", flush=True)
    print(f"Total reward: MJX={rewards.sum():.5f}; native={sum(reference_rewards):.5f}", flush=True)
    assert height_difference <= .005, "Final native/MJX height difference exceeds 5 mm"
    assert joint_difference <= 5, "Final native/MJX joint difference exceeds 5 degrees"
    np.testing.assert_allclose(rewards.sum(), sum(reference_rewards), rtol=.1, atol=.1,
                               err_msg="Task rewards differ substantially; inspect backend trajectories")
    repeat, _ = run(reset(jax.random.PRNGKey(1)))
    np.testing.assert_allclose(np.delete(repeat.obs, 14), np.delete(final.obs, 14), atol=1e-5)
    frozen = jax.jit(env.step)(final, action)
    np.testing.assert_array_equal(frozen.obs, final.obs)

    print(f"Compiling {args.batch_size} task environments with vmap ...", flush=True)
    batch = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(2), args.batch_size))
    batch = jax.jit(jax.vmap(env.step))(
        batch, jp.linspace(-1, 1, args.batch_size, dtype=batch.obs.dtype)[:, None])
    assert batch.obs.shape == (args.batch_size, 16) and np.isfinite(batch.obs).all()
    assert not np.isclose(float(batch.info["command_length"][0]), float(batch.info["command_length"][-1]))
    print("MJX TASK PASS: reset, slew limits, observation, reward, hold timer, timeout, native parity and vmap.", flush=True)
    print("No PPO training was performed. These are finite smoke-test tolerances, not proof of identical backends.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path(__file__).resolve().parents[1] / "descrption/hand.xml")
    parser.add_argument("--steps", type=int, default=4000, help="physics steps per control phase")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--task", action="store_true", help="check the MJX height task for a complete episode; --steps is unused")
    parser.add_argument("--config", type=Path, help="task config.json override for --task")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        parser.error("steps must be positive; batch-size must be at least 2")
    if args.task and args.native_only:
        parser.error("--task requires MJX; do not combine it with --native-only")

    print("Python:", sys.executable, flush=True)
    for package in ("mujoco", "mujoco-mjx", "jax", "jaxlib"):
        print(f"{package}: {version(package)}", flush=True)
    if args.task:
        check_task(args)
        return

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    assert model.nq == 3 and model.nu == 1, "Expected the three-joint, one-actuator hand"
    initial = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, initial, 0)
    mujoco.mj_forward(model, initial)
    low, high = model.actuator_ctrlrange[0]
    controls = np.array([high, (low + high) / 2, low, high])
    names = ["release", "partial flexion", "full flexion", "release again"]

    def check(q, velocity, lengths, label):
        assert np.isfinite(q).all(), f"{label}: non-finite angles"
        assert np.isfinite(velocity).all(), f"{label}: non-finite velocities"
        assert np.isfinite(lengths).all(), f"{label}: non-finite tendon lengths"
        assert np.max(np.abs(q)) < np.pi, f"{label}: divergent joint angles"
        error = np.max(np.abs(q[..., 2] - 0.9 * q[..., 1]))
        assert error < np.deg2rad(1), f"{label}: DIP/PIP coupling error exceeds 1 degree"

    native = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, native, 0)
    reference = []
    for name, control in zip(names, controls):
        native.ctrl[0] = control
        states = []
        for _ in range(args.steps):
            mujoco.mj_step(model, native)
            states.append(np.r_[native.qpos, native.qvel, native.ten_length])
        states = np.array(states)
        check(states[:, :3], states[:, 3:6], states[:, 6:], "MuJoCo " + name)
        assert not np.any(native.warning.number), "MuJoCo issued simulation warnings"
        reference.append(native.qpos.copy())
        print(f"MuJoCo {name}: {np.rad2deg(native.qpos).round(3)} deg", flush=True)
    if args.native_only:
        print("NATIVE PASS. MJX was not checked.", flush=True)
        return

    if version("mujoco-mjx") == "not installed":
        raise RuntimeError("This Python has no mujoco-mjx. Run this script with the actual Aero training Python.")
    import jax
    import jax.numpy as jp
    from mujoco import mjx

    print("JAX devices:", jax.devices(), flush=True)
    print("Converting the unchanged model to MJX-JAX ...", flush=True)
    kwargs = {"impl": "jax"} if "impl" in inspect.signature(mjx.put_model).parameters else {}
    mx = mjx.put_model(model, **kwargs)
    data_kwargs = {"impl": "jax"} if "impl" in inspect.signature(mjx.put_data).parameters else {}
    initial_mjx = mjx.put_data(model, initial, **data_kwargs)
    dx = initial_mjx
    print("Model conversion PASS", flush=True)

    def rollout(data, control):
        def step(carry, _):
            carry = mjx.step(mx, carry.replace(ctrl=jp.reshape(control, (1,))))
            return carry, (carry.qpos, carry.qvel, carry.ten_length)
        return jax.lax.scan(step, data, None, length=args.steps)

    simulate = jax.jit(rollout)
    endpoints = []
    print("Compiling and stepping; first compilation may take a while ...", flush=True)
    for name, control in zip(names, controls):
        dx, trace = simulate(dx, jp.asarray(control, dtype=dx.qpos.dtype))
        q, velocity, lengths = jax.device_get(trace)
        check(q, velocity, lengths, "MJX " + name)
        endpoints.append(q[-1])
        print(f"MJX {name}: {np.rad2deg(q[-1]).round(3)} deg", flush=True)
    endpoints = np.array(endpoints)
    assert np.linalg.norm(endpoints[2] - endpoints[0]) > 0.1, "MJX actuator did not visibly bend the finger"
    delta = np.rad2deg(np.max(np.abs(endpoints - np.array(reference))))
    print(f"Native/MJX largest endpoint difference: {delta:.4f} deg", flush=True)
    if delta > 5:
        raise RuntimeError("Native/MJX endpoint difference exceeds 5 degrees; investigate before training.")

    print(f"Compiling vmap for {args.batch_size} independent hands ...", flush=True)
    batched = jax.jit(jax.vmap(lambda u: rollout(initial_mjx, u)))
    # All copies start at home but receive different target lengths.
    _, trace = batched(jp.linspace(low, high, args.batch_size, dtype=dx.qpos.dtype))
    q, velocity, lengths = jax.device_get(trace)
    assert q.shape == (args.batch_size, args.steps, model.nq)
    check(q, velocity, lengths, "batched MJX")
    assert np.linalg.norm(q[0, -1] - q[-1, -1]) > 0.1, "Batched actions did not produce different poses"
    print("MJX-JAX CHECK PASS: conversion, JIT stepping, tendon actuation, coupling and vmap.", flush=True)
    print("This is a model compatibility check, not a PPO training or external-contact validation.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CHECK FAILED [{type(exc).__name__}]: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
