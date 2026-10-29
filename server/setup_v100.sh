#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
server_python="${SERVER_PYTHON:-python3.12}"
"$server_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
# Only create/install into this project's dedicated environment.
if [[ ! -x .venv-server/bin/python ]]; then
  "$server_python" -m venv .venv-server
fi
.venv-server/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing .venv-server must use Python 3.12"'
.venv-server/bin/python -m pip install --upgrade pip
.venv-server/bin/python -m pip install -r server/requirements-v100.lock
.venv-server/bin/python -m pip check
.venv-server/bin/python -m pip freeze > server/installed-server.txt
echo "Setup complete. Next: bash server/run.sh preflight"
