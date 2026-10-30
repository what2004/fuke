#!/usr/bin/env bash
# Run only after synchronizing the latest source to the server.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
checkpoint="$project_root/najiushi/rl/runs/finger-height-20261029-193528-0f1992/checkpoints/000001032192"
task_config="$project_root/najiushi/rl/config_push_warmup.json"
if [[ ! -f "$checkpoint/ppo_network_config.json" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  echo "Restore the complete original training run before starting." >&2
  exit 1
fi
arguments=(--task-config "$task_config" --restore-checkpoint "$checkpoint")
echo '1/4: GPU, dependencies and MJX preflight'
bash server/run.sh preflight
echo '2/4: Validate the small-force task and existing checkpoint'
bash server/run.sh train "${arguments[@]}" --dry-run
echo '3/4: Small PPO smoke run, starting from the existing checkpoint'
bash server/run.sh smoke "${arguments[@]}"
echo '4/4: Formal small-force training, starting from the original checkpoint'
bash server/run.sh train "${arguments[@]}"
