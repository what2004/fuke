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

Real Brax PPO smoke-run result will be recorded below after completion.

Not verified locally: V100 CUDA kernels, GPU memory usage, GPU training speed,
long-run learning convergence, or sim-to-real transfer. The server must run
`preflight`, `check`, and `smoke` before the full training run.
