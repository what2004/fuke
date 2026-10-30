# Server preparation verification

Date: 2026-10-29 (Asia/Shanghai).

An isolated local `.venv-server-check` was created from `requirements-core.lock`.
It is not included in the upload archive and does not modify Microduck's Python.

Verified with real installed libraries:

- Python 3.12.14; JAX/JAXlib 0.8.2; MuJoCo/MJX 3.10.0;
  Brax 0.14.2; bundled Playground 0.2.0.
- `server/preflight.py --allow-cpu`: passed, including model conversion,
  named tendon sensors, real MJX reset/JIT step, and Brax/wrapper API checks.
- `check_mjx.py --task --batch-size 4`: passed on CPU. Final height was
  approximately 0.158433 m in both engines. Largest trajectory height
  difference was 0.0024 cm; largest final joint difference was 0.0001 degree.
  Total rewards were 11.79592 (MJX) and 11.79590 (native).
- All 24 local unit tests passed in the new isolated dependency environment.
- Shell scripts pass `bash -n` and server Python scripts compile.

A real CPU Brax PPO smoke run reached its initial policy evaluation (reward
approximately 0.743). The subsequent training phase did not complete in a
reasonable local CPU run time, so this local run was stopped. Checkpoint saving
and PPO completion are NOT verified by that CPU run; use the V100 smoke run
for those acceptance steps. No remote/server training process was stopped.

Replay preparation subsequently verified a real Brax checkpoint save/load
round trip with temporary **untrained** network weights, saved observation
normalization, and one complete native MuJoCo replay episode. This check was
run outside the local execution sandbox after Orbax filesystem writes stalled
inside it. This validates the replay plumbing, not the user's trained policy.

Not verified locally: V100 CUDA kernels, GPU memory usage, GPU training speed,
long-run learning convergence, or sim-to-real transfer. The server must run
`preflight`, `check`, and `smoke` before the full training run.

## 随机扰动训练接口（2026-10-29）

- 新增 10 个 body-local 施力点与 8 个世界 YZ 平面单位方向；每回合采样一个矩形力脉冲。
- `config_push.json`：12 秒、80% 有扰动、1～4 N、第 3～5 秒开始、持续 0.5～2 秒；
  先采用撤力后高度保持成功条件。另提供 0.25～1 N 的小力配置。
- 本地 39 项单元测试通过，覆盖采样复现、点方向覆盖、精确施力步数、清除外力、
  高度判定与撤力门控，以及 warm-start 配置验证。
- 真实 MJX-JAX 验证：128 次 jit/vmap reset 覆盖全部点与方向，幅值符合 1～4 N；
  80 种原生 MuJoCo/MJX 广义力投影比较通过；真实 jit step 在同一指定扰动下与原生
  qpos/高度比较通过，20 个施力步准确，随后外力归零。
- `train_ppo.py --task-config .../config_push.json --restore-checkpoint .../000001032192 --dry-run`
  通过。warm start 恢复策略和归一化器，重建 critic 与 optimizer。
- 未在本次本地验证中运行新的 GPU PPO 学习，不代表随机扰动恢复成功率已达标。
