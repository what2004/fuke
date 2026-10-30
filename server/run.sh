#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
stage="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  preflight|check|smoke|train|test) ;;
  *) echo "Usage: bash server/run.sh {preflight|check|smoke|train|test} [extra arguments]"; exit 2 ;;
esac
server_python="$project_root/.venv-server/bin/python"
if [[ ! -x "$server_python" ]]; then
  echo "Missing .venv-server. Run: bash server/setup_v100.sh" >&2; exit 1
fi
# CUDA wheel libraries are inside the venv rather than the system linker path.
nvidia_lib_root="$("$server_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/nvidia"
if [[ -d "$nvidia_lib_root" ]]; then
  nvidia_libraries="$(find "$nvidia_lib_root" -maxdepth 2 -type d -name lib | paste -sd:)"
  if [[ -n "$nvidia_libraries" ]]; then
    export LD_LIBRARY_PATH="$nvidia_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export MPLCONFIGDIR="$project_root/server/cache/matplotlib"
export JAX_COMPILATION_CACHE_DIR="$project_root/server/cache/jax"
mkdir -p server/logs "$MPLCONFIGDIR" "$JAX_COMPILATION_CACHE_DIR"
log="server/logs/${stage}-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
echo "Stage: $stage | log: $project_root/$log"
case "$stage" in
  preflight) command=("$server_python" server/preflight.py "$@") ;;
  check) command=("$server_python" najiushi/rl/check_mjx.py --task "$@") ;;
  smoke) command=("$server_python" najiushi/rl/train_ppo.py --smoke "$@") ;;
  train) command=("$server_python" najiushi/rl/train_ppo.py "$@") ;;
  test) command=("$server_python" -m unittest discover -s najiushi/rl -p 'test_*.py' "$@") ;;
esac
"${command[@]}" 2>&1 | tee "$log"
