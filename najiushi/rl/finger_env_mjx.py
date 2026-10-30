"""Functional MJX-JAX version of the native finger-height task.

Uses Playground's State interface: reset(rng) -> State, step(state, action)
-> State. No registration, training loop, random pushes or physics overrides.
The pure JAX task functions can be tested without MJX/Playground installed;
constructing FingerHeightMjxEnv requires both packages in the training Python.
"""

import inspect
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

try:
    from .finger_env import TaskConfig, PUSH_POINTS, PUSH_DIRECTIONS
except ImportError:
    from finger_env import TaskConfig, PUSH_POINTS, PUSH_DIRECTIONS


REWARD_NAMES = ("tracking", "tip_speed", "joint_speed", "command_change", "holding", "success", "failure")


def limit_command(action, command, low, high, rate, dt):
    """+1 pulls, -1 releases; limit the applied command at control frequency."""
    requested = low + (1 - jp.clip(action, -1, 1)) * 0.5 * (high - low)
    delta = rate * dt
    return jp.clip(command + jp.clip(requested - command, -delta, delta), low, high)


def noisy_tendon_length(key, sensor_length, level, scale):
    """Aero observation model: add U[-1, 1] * level * scale in meters."""
    return sensor_length + jax.random.uniform(key, shape=(), minval=-1., maxval=1.) * level * scale


def tip_kinematics(site_position, body_cvel, root_com):
    """World site velocity, shifting MuJoCo's COM-centered spatial velocity."""
    velocity = body_cvel[3:] + jp.cross(body_cvel[:3], site_position - root_com)
    return site_position, velocity


def task_sample(qvel, position, velocity, hold_steps, ever, first_time, sim_time, config, sim_dt, eligible=True):
    """One physics-step update, evaluated AFTER forward at the new qpos."""
    error = position[2] - config.target_height
    speed = jp.linalg.norm(velocity)
    qualifies = (jp.abs(error) <= config.height_tolerance) & eligible
    if config.require_low_speed:
        qualifies &= speed <= config.speed_limit
    required = int(np.ceil(config.hold_seconds / sim_dt - 1e-9))
    # Integer physics ticks prevent float32 drift at the one-second boundary.
    hold_steps = jp.where(qualifies, jp.minimum(hold_steps + 1, required), 0)
    held = jp.minimum(hold_steps * sim_dt, config.hold_seconds)
    success = hold_steps >= required
    first_time = jp.where(success & ~ever, sim_time, first_time)
    ever = ever | success
    terms = jp.stack((
        config.tracking_weight * jp.exp(-jp.square(error / config.height_reward_scale)),
        -config.tip_speed_weight * jp.minimum(jp.square(speed / config.tip_speed_scale), 100),
        -config.joint_speed_weight * jp.minimum(jp.mean(jp.square(qvel / config.joint_speed_scale)), 100),
        jp.zeros_like(error),
        config.holding_weight * qualifies,
        config.success_weight * success,
        jp.zeros_like(error),
    )) * sim_dt
    return hold_steps, held, success, ever, first_time, terms


def observation(qpos, qvel, position, velocity, command, held, steps, tendon_length,
                tendon_velocity, low, high, max_steps, config):
    """Same ordering and scales as FingerHeightEnv._observation (16 floats)."""
    tail = jp.stack((
        (position[2] - config.target_height) / 0.05,
        1 - 2 * (command - low) / (high - low),
        jp.minimum(held / config.hold_seconds, 1),
        jp.maximum(0, 1 - steps / max_steps),
        tendon_length / 0.1,
        tendon_velocity / 0.1,
    ))
    return jp.concatenate((qpos / jp.pi, qvel / config.joint_speed_scale,
                           jp.reshape(position[2] / 0.25, (1,)),
                           velocity / config.tip_speed_scale, tail)).astype(jp.float32)


class FingerHeightMjxEnv:
    """Playground-compatible duck-typed environment with immutable JAX state.

Model/constants live on self; ALL changing episode variables live in State.
Use jax.jit(env.reset/step), and jax.vmap for independent environments.
Call reset after done. A step on a completed state is an identity operation.
No Gymnasium/SB3 environment is introduced.
"""

    def __init__(self, config_path=None, xml_path=None):
        try:
            from mujoco import mjx
            from mujoco_playground._src.mjx_env import State
        except ImportError as exc:
            raise ImportError("Use the Aero training Python with mujoco-mjx and mujoco_playground installed") from exc
        self._mjx = mjx
        self._state_type = State
        self.config = TaskConfig.load(config_path)
        self._xml_path = str(Path(xml_path).resolve() if xml_path else
                             Path(__file__).resolve().parents[1] / "descrption/hand.xml")
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        m = self._mj_model
        if m.nq != 3 or m.nu != 1:
            raise ValueError("Expected the three-joint, one-actuator hand")
        self.tip = self._id(mujoco.mjtObj.mjOBJ_SITE, "index_tip")
        self.tendon = self._id(mujoco.mjtObj.mjOBJ_TENDON, "index_flexor")
        self.actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "index_tendon_length")
        self.tendon_sensor_adr = self._sensor_address("len_if")
        self.tendon_velocity_sensor_adr = self._sensor_address("vel_if")
        self.tip_body = int(m.site_bodyid[self.tip])
        self.tip_root = int(m.body_rootid[self.tip_body])
        self.low, self.high = map(float, m.actuator_ctrlrange[self.actuator])
        self._sim_dt = float(m.opt.timestep)
        self._n_substeps = round(self.config.control_dt / self._sim_dt)
        self.max_steps = round(self.config.episode_seconds / self.config.control_dt)
        if self.config.push_enabled:
            self._push_bodies = jp.array([self._id(mujoco.mjtObj.mjOBJ_BODY, body)
                                          for body, _ in PUSH_POINTS.values()])
            self._push_positions = jp.array([pos for _, pos in PUSH_POINTS.values()])
            for value in (self.config.push_start_min_s, self.config.push_start_max_s,
                          self.config.push_duration_min_s, self.config.push_duration_max_s):
                if not np.isclose(value/self.sim_dt, round(value/self.sim_dt), atol=1e-7, rtol=0):
                    raise ValueError('Random push time limits must align with physics ticks')
        if self._n_substeps < 1 or not np.isclose(self._n_substeps * self._sim_dt, self.dt):
            raise ValueError("control_dt must be an integer multiple of the XML timestep")
        if self.max_steps < 1 or not np.isclose(self.max_steps * self.dt, self.config.episode_seconds):
            raise ValueError("episode_seconds must be an integer multiple of control_dt")
        # Fail on unsupported physics rather than silently changing the XML.
        kwargs = {"impl": "jax"} if "impl" in inspect.signature(mjx.put_model).parameters else {}
        self._mjx_model = mjx.put_model(m, **kwargs)
        native = mujoco.MjData(m)
        mujoco.mj_resetData(m, native)
        native.qpos[:] = 0
        native.qvel[:] = 0
        native.ctrl[self.actuator] = self.high
        mujoco.mj_forward(m, native)
        kwargs = {"impl": "jax"} if "impl" in inspect.signature(mjx.put_data).parameters else {}
        self._initial_data = mjx.put_data(m, native, **kwargs)

    def _id(self, kind, name):
        index = mujoco.mj_name2id(self._mj_model, kind, name)
        if index < 0:
            raise ValueError(f"Missing model element: {name}")
        return index

    def _sensor_address(self, name):
        sensor = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        if self.mj_model.sensor_dim[sensor] != 1 or self.mj_model.sensor_objid[sensor] != self.tendon:
            raise ValueError(f"Expected scalar active-tendon sensor: {name}")
        return int(self.mj_model.sensor_adr[sensor])

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def action_size(self):
        return 1

    @property
    def observation_size(self):
        return 16

    @property
    def dt(self):
        return self.config.control_dt

    @property
    def sim_dt(self):
        return self._sim_dt

    @property
    def n_substeps(self):
        return self._n_substeps

    @property
    def episode_length(self):
        return self.max_steps

    @property
    def unwrapped(self):
        return self

    @property
    def model_assets(self):
        return {}

    def render(self, trajectory, **kwargs):
        from mujoco_playground._src.mjx_env import render_array
        return render_array(self.mj_model, trajectory, **kwargs)

    def for_brax_training(self):
        """Let Brax EpisodeWrapper own timeouts so value bootstrapping is correct."""
        return _BraxTimeoutAdapter(self)

    def action_for_length(self, length):
        """Pure JAX mapping; caller must provide a finite length in ctrlrange."""
        return jp.reshape(1 - 2 * (jp.asarray(length) - self.low) / (self.high - self.low), (1,))

    def _tip(self, data):
        return tip_kinematics(data.site_xpos[self.tip], data.cvel[self.tip_body],
                              data.subtree_com[self.tip_root])

    def _obs(self, data, info):
        position, velocity = self._tip(data)
        info["rng"], noise_key = jax.random.split(info["rng"])
        sensor_length = data.sensordata[self.tendon_sensor_adr]
        measured_length = noisy_tendon_length(noise_key, sensor_length,
                                              self.config.tendon_noise_level, self.config.tendon_length_noise_m)
        info.update(tendon_length_sensor=sensor_length, tendon_length_observed=measured_length)
        return observation(data.qpos, data.qvel, position, velocity,
                           info["command_length"], info["hold_seconds"], info["task_steps"],
                           measured_length, data.sensordata[self.tendon_velocity_sensor_adr],
                           self.low, self.high, self.max_steps, self.config)

    def _initial_info(self, rng, dtype):
        zero = jp.zeros((), dtype)
        info = {
            "rng": rng, "task_steps": jp.zeros((), jp.int32),
            "command_length": jp.asarray(self.high, dtype),
            "hold_steps": jp.zeros((), jp.int32), "hold_seconds": zero,
            "is_success": jp.asarray(False), "success_ever": jp.asarray(False),
            "first_success_time": jp.asarray(-1, dtype),
            "terminated": jp.asarray(False), "truncated": jp.asarray(False),
            "tendon_length_sensor": zero, "tendon_length_observed": zero,
            "push_sampled": jp.asarray(False), "push_point_index": jp.int32(0),
            "push_direction_index": jp.int32(0), "push_force": jp.zeros(3, dtype),
            "push_start_tick": jp.int32(0), "push_duration_ticks": jp.int32(0),
            "push_applied_ticks": jp.int32(0),
        }
        if self.config.push_enabled:
            c, dt = self.config, self.sim_dt
            keys = jax.random.split(rng, 7)
            direction = jax.random.randint(keys[2], (), 0, len(PUSH_DIRECTIONS))
            force = jp.asarray(PUSH_DIRECTIONS, dtype)[direction] * jax.random.uniform(
                keys[3], (), minval=c.push_force_min_n, maxval=c.push_force_max_n)
            info.update(rng=keys[0], push_point_index=jax.random.randint(keys[1], (), 0, len(PUSH_POINTS)),
                        push_direction_index=direction, push_force=force,
                        push_sampled=jax.random.uniform(keys[4]) < c.push_probability,
                        push_start_tick=jax.random.randint(keys[5], (), round(c.push_start_min_s/dt), round(c.push_start_max_s/dt)+1),
                        push_duration_ticks=jax.random.randint(keys[6], (), round(c.push_duration_min_s/dt), round(c.push_duration_max_s/dt)+1))
        return info

    def _apply_push(self, data, info):
        tick = jp.rint(data.time / self.sim_dt).astype(jp.int32)
        active = info['push_sampled'] & (tick >= info['push_start_tick']) & (tick < info['push_start_tick']+info['push_duration_ticks'])
        index = info['push_point_index']
        body = self._push_bodies[index]
        point = data.xpos[body] + data.xmat[body] @ self._push_positions[index]
        force = jp.where(active, info['push_force'], jp.zeros(3, data.qpos.dtype))
        qforce = self._mjx.apply_ft(self.mjx_model, data, force, jp.zeros(3, data.qpos.dtype), point, body)
        return data.replace(qfrc_applied=qforce), active.astype(jp.int32)

    def _metrics(self, data, info, terms, previous=None):
        position, velocity = self._tip(data)
        metrics = dict(previous) if previous is not None else {}
        metrics.update({f"reward/{name}": terms[i] for i, name in enumerate(REWARD_NAMES)})
        metrics.update(height=position[2], height_error=position[2] - self.config.target_height,
                       tip_speed=jp.linalg.norm(velocity), hold_seconds=info["hold_seconds"],
                       is_success=info["is_success"].astype(jp.float32),
                       success_ever=info["success_ever"].astype(jp.float32),
                       command_length=info["command_length"], tendon_length=data.ten_length[self.tendon],
                       tendon_length_sensor=info["tendon_length_sensor"],
                       tendon_length_observed=info["tendon_length_observed"])
        if self.config.push_enabled:
            finished = info['truncated'] | info['terminated']
            metrics.update(push_sampled=info['push_sampled'].astype(jp.float32),
                           push_applied_ticks=info['push_applied_ticks'].astype(jp.float32),
                           push_episode_count=(finished & info['push_sampled']).astype(jp.float32),
                           push_episode_recovered=(finished & info['push_sampled'] & info['success_ever']).astype(jp.float32))
        return metrics

    def reset(self, rng):
        data = self._mjx.forward(self.mjx_model, self._initial_data)
        info = self._initial_info(rng, data.qpos.dtype)
        terms = jp.zeros(len(REWARD_NAMES), data.qpos.dtype)
        zero = jp.zeros((), data.qpos.dtype)
        return self._state_type(data=data, obs=self._obs(data, info), reward=zero,
                                done=zero, metrics=self._metrics(data, info, terms), info=info)

    def step(self, state, action):
        action = jp.asarray(action, dtype=state.data.qpos.dtype)
        if action.shape != (1,):
            raise ValueError("Action must have shape (1,); batch with jax.vmap")
        return jax.lax.cond(state.done.astype(bool), lambda _: state,
                            lambda _: self._step_active(state, action), operand=None)

    def _step_active(self, state, action):
        c = self.config
        dtype = state.data.qpos.dtype
        info = dict(state.info)  # preserve wrapper bookkeeping, never mutate input
        # Cached Brax autoreset may replace data/obs but retain old task info.
        fresh = (state.data.time == 0) & (info["task_steps"] > 0)
        initial = self._initial_info(info["rng"], dtype)
        for key in initial:
            if key != "rng":
                info[key] = jp.where(fresh, initial[key], info[key])
        previous = info["command_length"]
        action_valid = jp.all(jp.isfinite(action))
        command = limit_command(jp.nan_to_num(action[0]), previous, self.low, self.high,
                                c.max_command_speed, self.dt)
        command = jp.where(action_valid, command, previous)
        data = state.data.replace(ctrl=state.data.ctrl.at[self.actuator].set(command))
        # Keep all scan carry shapes/dtypes invariant, including failures.
        carry = (data, info["hold_steps"], info["hold_seconds"], info["is_success"],
                 info["success_ever"], info["first_success_time"],
                 jp.zeros(len(REWARD_NAMES), dtype), info['push_applied_ticks'], ~action_valid)

        def physics_tick(carry, _):
            def advance(values):
                data, count, held, success, ever, first, terms, applied_ticks, failed = values
                increment = jp.int32(0)
                if c.push_enabled:
                    data, increment = self._apply_push(data, info)
                candidate = self._mjx.step(self.mjx_model, data)
                candidate = self._mjx.forward(self.mjx_model, candidate)
                pos, vel = self._tip(candidate)
                valid = jp.all(jp.isfinite(jp.concatenate((candidate.qpos, candidate.qvel,
                                                          candidate.qacc, pos, vel, candidate.ten_length))))

                def accept(_):
                    eligible = jp.asarray(True)
                    if c.push_enabled:
                        eligible = ~info['push_sampled'] | (jp.rint(candidate.time/self.sim_dt).astype(jp.int32) > info['push_start_tick']+info['push_duration_ticks'])
                    count1, held1, success1, ever1, first1, sample = task_sample(
                        candidate.qvel, pos, vel, count, ever, first, candidate.time, c, self.sim_dt, eligible)
                    return candidate, count1, held1, success1, ever1, first1, terms + sample, applied_ticks+increment, failed

                def reject(_):
                    # Preserve last finite data and stop remaining substeps.
                    return data, count, held, success, ever, first, terms, applied_ticks, jp.asarray(True)

                return jax.lax.cond(valid, accept, reject, operand=None)

            result = jax.lax.cond(carry[-1], lambda values: values, advance, carry)
            return result, None

        result, _ = jax.lax.scan(physics_tick, carry, None, length=self.n_substeps)
        data, count, held, success, ever, first, terms, applied_ticks, failed = result
        terms = terms.at[3].set(-c.command_change_weight * jp.square(
            (command - previous) / (c.max_command_speed * self.dt)) * self.dt)
        terms = terms.at[6].set(jp.where(failed, -c.failure_penalty, 0))
        steps = info["task_steps"] + 1
        truncated = (steps >= self.max_steps) & ~failed
        info.update(task_steps=steps, command_length=command, hold_steps=count,
                    push_applied_ticks=applied_ticks,
                    hold_seconds=held, is_success=success, success_ever=ever,
                    first_success_time=first, terminated=failed, truncated=truncated)
        return state.replace(data=data, obs=self._obs(data, info), reward=jp.sum(terms),
                             done=(failed | truncated).astype(dtype), info=info,
                             metrics=self._metrics(data, info, terms, state.metrics))


class _BraxTimeoutAdapter:
    """Report true failures as done; EpisodeWrapper supplies time-limit done.

Otherwise a timeout already marked done by the task could be mistaken for a
true terminal state by Brax's truncation calculation.
"""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, rng):
        return self.env.reset(rng)

    def step(self, state, action):
        result = self.env.step(state, action)
        return result.replace(done=result.info["terminated"].astype(result.done.dtype))
