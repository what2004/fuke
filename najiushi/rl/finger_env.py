"""Native MuJoCo task prototype, independent of Gymnasium/SB3.

This mutable NumPy environment is for local debugging. It is NOT an MJX/JAX
environment: the task equations still need a functional JAX implementation
and backend parity checks before using the Aero/Brax trainer.

Observation (float32, 16 values; all scales explicit):
  q/π (3), qvel/10 (3), tip_z/0.25 (1), tip_xyz_velocity/0.1 (3),
  height_error/0.05 (1), applied_command mapped to [-1,1] (1),
  hold progress (1), remaining episode fraction (1),
  measured active tendon length/0.1 (1), tendon velocity sensor/0.1 (1).
Length sensing follows Aero: len_if + uniform observation noise (default ±5 mm).
Values such as normalized velocity are not clipped: they can exceed one.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np


# Body-local positions, in meters. Three points per cylinder plus the fingertip.
PUSH_POINTS = {
    **{f'{name}_{percent}': (body, (0., length*percent/100, 0.))
       for name, body, length in [('proximal', 'index_proximal', .045),
                                  ('middle', 'index_middle', .028),
                                  ('distal', 'index_distal', .022)]
       for percent in (25, 50, 75)},
    'tip': ('index_distal', (0., .022, 0.)),
}
# World YZ plane: the model has hinges about X and no lateral bending DOF.
PUSH_DIRECTIONS = np.array([[0., np.cos(a), np.sin(a)]
                            for a in np.arange(8)*np.pi/4])


@dataclass(frozen=True)
class TaskConfig:
    target_height: float = 0.16
    height_tolerance: float = 0.005
    speed_limit: float = 0.01
    hold_seconds: float = 1.0
    control_dt: float = 0.02
    episode_seconds: float = 8.0
    max_command_speed: float = 0.02
    height_reward_scale: float = 0.01
    tip_speed_scale: float = 0.1
    joint_speed_scale: float = 10.0
    tracking_weight: float = 1.0
    tip_speed_weight: float = 0.02
    joint_speed_weight: float = 0.01
    command_change_weight: float = 0.01
    holding_weight: float = 0.2
    success_weight: float = 0.5
    failure_penalty: float = 5.0
    tendon_noise_level: float = 1.0
    tendon_length_noise_m: float = 0.005
    require_low_speed: bool = True
    push_enabled: bool = False
    push_probability: float = .8
    push_force_min_n: float = 1.
    push_force_max_n: float = 4.
    push_start_min_s: float = 3.
    push_start_max_s: float = 5.
    push_duration_min_s: float = .5
    push_duration_max_s: float = 2.

    @classmethod
    def load(cls, path=None):
        path = Path(path) if path else Path(__file__).with_name("config.json")
        config = cls(**json.loads(path.read_text()))
        values = vars(config)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("All task settings must be finite")
        positive = ("height_tolerance", "speed_limit", "hold_seconds", "control_dt",
                    "episode_seconds", "max_command_speed", "height_reward_scale",
                    "tip_speed_scale", "joint_speed_scale")
        if any(values[name] <= 0 for name in positive):
            raise ValueError("Time, tolerance, rate and normalization scales must be positive")
        if any(value < 0 for name, value in values.items() if name.endswith(("weight", "penalty"))):
            raise ValueError("Reward weights and failure penalty must be nonnegative")
        if config.tendon_noise_level < 0 or config.tendon_length_noise_m < 0:
            raise ValueError("Tendon observation noise must be nonnegative")
        if type(config.require_low_speed) is not bool or type(config.push_enabled) is not bool:
            raise ValueError('require_low_speed and push_enabled must be booleans')
        if not 0 <= config.push_probability <= 1:
            raise ValueError('push_probability must be in [0, 1]')
        for low, high in [('push_force_min_n', 'push_force_max_n'),
                          ('push_start_min_s', 'push_start_max_s'),
                          ('push_duration_min_s', 'push_duration_max_s')]:
            if not 0 <= values[low] <= values[high]:
                raise ValueError(f'Invalid push range: {low}, {high}')
        if config.push_duration_min_s <= 0:
            raise ValueError('Push duration must be positive')
        if config.push_enabled and (config.push_start_max_s + config.push_duration_max_s
                                    + config.hold_seconds >= config.episode_seconds):
            raise ValueError('Leave enough time after the latest push for recovery and holding')
        return config


class FingerHeightEnv:
    """reset() -> (observation, info); step() -> (obs, reward, terminated, truncated, info)."""

    def __init__(self, config_path=None, xml_path=None):
        self.config = TaskConfig.load(config_path)
        xml_path = Path(xml_path) if xml_path else Path(__file__).resolve().parents[1] / "descrption/hand.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 3 or self.model.nu != 1:
            raise ValueError("Expected three finger joints and one active tendon actuator")
        self.tip = self._id(mujoco.mjtObj.mjOBJ_SITE, "index_tip")
        self.tendon = self._id(mujoco.mjtObj.mjOBJ_TENDON, "index_flexor")
        self.actuator = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "index_tendon_length")
        self.tendon_sensor_adr = self._sensor_address("len_if")
        self.tendon_velocity_sensor_adr = self._sensor_address("vel_if")
        self.low, self.high = self.model.actuator_ctrlrange[self.actuator].copy()
        self.substeps = round(self.config.control_dt / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * self.model.opt.timestep, self.config.control_dt):
            raise ValueError("control_dt must be an integer multiple of the XML timestep")
        self.max_steps = round(self.config.episode_seconds / self.config.control_dt)
        if self.max_steps < 1 or not np.isclose(self.max_steps * self.config.control_dt, self.config.episode_seconds):
            raise ValueError("episode_seconds must be an integer multiple of control_dt")
        self._velocity = np.zeros(6)
        self._ready = False
        if self.config.push_enabled:
            dt = self.model.opt.timestep
            for value in (self.config.push_start_min_s, self.config.push_start_max_s,
                          self.config.push_duration_min_s, self.config.push_duration_max_s):
                if not np.isclose(value/dt, round(value/dt), rtol=0, atol=1e-7):
                    raise ValueError('Random push time limits must align with physics ticks')

    def _id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"Missing model element: {name}")
        return index

    def action_for_length(self, length):
        if not np.isfinite(length) or not self.low <= length <= self.high:
            raise ValueError(f"Target length must be within [{self.low}, {self.high}] m")
        return np.array([1 - 2 * (length - self.low) / (self.high - self.low)], dtype=np.float32)

    def _sensor_address(self, name):
        sensor = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        if self.model.sensor_dim[sensor] != 1 or self.model.sensor_objid[sensor] != self.tendon:
            raise ValueError(f"Expected scalar active-tendon sensor: {name}")
        return int(self.model.sensor_adr[sensor])

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("This first version uses a fixed initial state; reset options are not supported")
        self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.command = float(self.high)
        self.data.ctrl[self.actuator] = self.command
        self.steps = 0
        self.held = 0.0
        self.success = False
        self.success_ever = False
        self.first_success_time = None
        self._ready = True
        self.push_point_index = self.push_direction_index = 0
        self.push_force = np.zeros(3)
        self.push_start_tick = self.push_duration_ticks = self.push_applied_ticks = 0
        self.push_sampled = False
        if self.config.push_enabled:
            c, rng, dt = self.config, self.np_random, self.model.opt.timestep
            self.push_point_index = int(rng.integers(len(PUSH_POINTS)))
            self.push_direction_index = int(rng.integers(len(PUSH_DIRECTIONS)))
            self.push_sampled = bool(rng.random() < c.push_probability)
            self.push_force = PUSH_DIRECTIONS[self.push_direction_index] * rng.uniform(c.push_force_min_n, c.push_force_max_n)
            self.push_start_tick = int(rng.integers(round(c.push_start_min_s/dt), round(c.push_start_max_s/dt)+1))
            self.push_duration_ticks = int(rng.integers(round(c.push_duration_min_s/dt), round(c.push_duration_max_s/dt)+1))
        self._update_kinematics()
        return self._observation(), self._info({})

    def _update_kinematics(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                self.tip, self._velocity, 0)
        self.tip_position = self.data.site_xpos[self.tip].copy()
        self.tip_velocity = self._velocity[3:].copy()
        self.tip_speed = float(np.linalg.norm(self.tip_velocity))
        self.height_error = float(self.tip_position[2] - self.config.target_height)

    def _observation(self):
        c = self.config
        self.tendon_length_sensor = float(self.data.sensordata[self.tendon_sensor_adr])
        self.observed_tendon_length = self.tendon_length_sensor + self.np_random.uniform(-1, 1) * c.tendon_noise_level * c.tendon_length_noise_m
        return np.concatenate((
            self.data.qpos / np.pi,
            self.data.qvel / c.joint_speed_scale,
            [self.tip_position[2] / 0.25],
            self.tip_velocity / c.tip_speed_scale,
            [self.height_error / 0.05, self.action_for_length(self.command)[0],
             min(self.held / c.hold_seconds, 1), max(0, 1 - self.steps / self.max_steps),
             self.observed_tendon_length / 0.1,
             self.data.sensordata[self.tendon_velocity_sensor_adr] / 0.1],
        )).astype(np.float32)

    def _info(self, terms):
        return {
            "time": float(self.data.time), "height": float(self.tip_position[2]),
            "height_error": self.height_error, "tip_speed": self.tip_speed,
            "command_length": self.command,
            "tendon_length": float(self.data.ten_length[self.tendon]),
            "tendon_length_sensor": self.tendon_length_sensor,
            "tendon_length_observed": self.observed_tendon_length,
            "hold_seconds": self.held, "is_success": self.success,
            "success_ever": self.success_ever, "first_success_time": self.first_success_time,
            "reward_terms": terms,
            "push_sampled": self.push_sampled,
            "push_point": list(PUSH_POINTS)[self.push_point_index],
            "push_direction_index": self.push_direction_index,
            "push_force_world_n": self.push_force.tolist(),
            "push_start_tick": self.push_start_tick,
            "push_duration_ticks": self.push_duration_ticks,
            "push_applied_ticks": self.push_applied_ticks,
        }

    def step(self, action):
        if not self._ready:
            raise RuntimeError("Call reset before stepping or after an episode ends")
        action = np.asarray(action, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Action must be one finite number in an array with shape (1,)")
        c = self.config
        requested = self.low + (1 - float(np.clip(action[0], -1, 1))) / 2 * (self.high - self.low)
        previous = self.command
        max_change = c.max_command_speed * c.control_dt
        self.command = float(np.clip(previous + np.clip(requested - previous, -max_change, max_change), self.low, self.high))
        self.data.ctrl[self.actuator] = self.command
        terms = dict.fromkeys(("tracking", "tip_speed", "joint_speed", "command_change", "holding", "success"), 0.0)
        terminated = False
        for _ in range(self.substeps):
            self._physics_step()
            valid = np.isfinite(np.r_[self.data.qpos, self.data.qvel, self.tip_position, self.tip_velocity]).all()
            if not valid or np.any(self.data.warning.number):
                terminated = True
                break
            eligible = not self.push_sampled or round(self.data.time/self.model.opt.timestep) > self.push_start_tick+self.push_duration_ticks
            qualifies = (eligible and abs(self.height_error) <= c.height_tolerance
                         and (not c.require_low_speed or self.tip_speed <= c.speed_limit))
            self.held = min(self.held + self.model.opt.timestep, c.hold_seconds) if qualifies else 0.0
            self.success = self.held + 1e-9 >= c.hold_seconds
            if self.success and not self.success_ever:
                self.first_success_time = float(self.data.time)
            self.success_ever |= self.success
            # Integrate reward rates in simulation time, independent of control frequency.
            dt = self.model.opt.timestep
            terms["tracking"] += c.tracking_weight * np.exp(-(self.height_error / c.height_reward_scale) ** 2) * dt
            terms["tip_speed"] -= c.tip_speed_weight * min((self.tip_speed / c.tip_speed_scale) ** 2, 100) * dt
            terms["joint_speed"] -= c.joint_speed_weight * min(float(np.mean((self.data.qvel / c.joint_speed_scale) ** 2)), 100) * dt
            terms["holding"] += c.holding_weight * float(qualifies) * dt
            terms["success"] += c.success_weight * float(self.success) * dt
        terms["command_change"] = -c.command_change_weight * ((self.command - previous) / max_change) ** 2 * c.control_dt
        if terminated:
            terms["failure"] = -c.failure_penalty
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        self._ready = not (terminated or truncated)
        obs = self._observation()
        if terminated:
            obs = np.nan_to_num(obs, nan=0, posinf=0, neginf=0)
        return obs, float(sum(terms.values())), terminated, truncated, self._info(terms)

    def _physics_step(self):
        """Advance one physical step; local experiments can override this hook."""
        if self.config.push_enabled:
            self.data.qfrc_applied[:] = 0
            tick = round(self.data.time/self.model.opt.timestep)
            active = self.push_sampled and self.push_start_tick <= tick < self.push_start_tick+self.push_duration_ticks
            if active:
                body_name, local = list(PUSH_POINTS.values())[self.push_point_index]
                body = self._id(mujoco.mjtObj.mjOBJ_BODY, body_name)
                point = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ np.asarray(local)
                mujoco.mj_applyFT(self.model, self.data, self.push_force, np.zeros(3), point, body, self.data.qfrc_applied)
                self.push_applied_ticks += 1
        mujoco.mj_step(self.model, self.data)
        self._update_kinematics()

    def close(self):
        self._ready = False
