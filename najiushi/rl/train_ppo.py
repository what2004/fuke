"""Train the single-finger height task with Aero's Playground/Brax PPO stack.

--dry-run validates settings without importing MJX/Brax or creating output.
--smoke runs a small real training job to check the SERVER pipeline; it is not
expected to learn a good policy. This script never installs/upgrades packages.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
PPO_KEYS = {
    "num_timesteps", "num_envs", "num_eval_envs", "num_evals", "unroll_length",
    "batch_size", "num_minibatches", "num_updates_per_batch", "num_resets_per_eval",
    "learning_rate", "entropy_cost", "discounting", "gae_lambda", "reward_scaling",
    "clipping_epsilon", "max_grad_norm", "normalize_observations", "deterministic_eval",
    "max_devices_per_host",
}


def versions():
    result = {}
    for name in ("mujoco", "mujoco-mjx", "jax", "jaxlib", "brax", "playground", "flax", "orbax-checkpoint"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "ppo_config.json")
    parser.add_argument("--task-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--xml", type=Path, default=ROOT.parent / "descrption/hand.xml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name", help="new subdirectory name; existing directories are never reused")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-eval-envs", type=int)
    return parser.parse_args(argv)


def build_plan(args):
    try:
        from .finger_env import TaskConfig
    except ImportError:
        from finger_env import TaskConfig
    raw = json.loads(args.config.read_text())
    if set(raw) != {"seed", "network", "ppo", "smoke_overrides"}:
        raise ValueError("PPO config must contain seed, network, ppo and smoke_overrides")
    if set(raw["ppo"]) != PPO_KEYS or not set(raw["smoke_overrides"]) <= PPO_KEYS:
        raise ValueError("Unknown or missing PPO settings; check ppo_config.json")
    ppo = dict(raw["ppo"])
    if args.smoke:
        ppo.update(raw["smoke_overrides"])
    for key in ("num_timesteps", "num_envs", "num_eval_envs"):
        override = getattr(args, key)
        if override is not None:
            ppo[key] = override
    integers = ("num_timesteps", "num_envs", "num_eval_envs", "num_evals", "unroll_length",
                "batch_size", "num_minibatches", "num_updates_per_batch", "num_resets_per_eval", "max_devices_per_host")
    for name in integers:
        if type(ppo[name]) is not int or ppo[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if ppo["num_evals"] < 2:
        raise ValueError("num_evals must be at least 2 for initial and post-training evaluations")
    if ppo["max_devices_per_host"] != 1:
        raise ValueError("This first entry point uses one device; keep max_devices_per_host=1")
    if ppo["batch_size"] * ppo["num_minibatches"] % ppo["num_envs"]:
        raise ValueError("Brax requires batch_size * num_minibatches divisible by num_envs")
    for name in ("normalize_observations", "deterministic_eval"):
        if type(ppo[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean")
    for name in PPO_KEYS - set(integers) - {"normalize_observations", "deterministic_eval"}:
        value = ppo[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    for name in ("learning_rate", "reward_scaling", "max_grad_norm"):
        if ppo[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < ppo["discounting"] <= 1 or not 0 <= ppo["gae_lambda"] <= 1 or not 0 < ppo["clipping_epsilon"] < 1:
        raise ValueError("Invalid discounting, gae_lambda or clipping_epsilon")
    network = raw["network"]
    if set(network) != {"policy_hidden_layer_sizes", "value_hidden_layer_sizes"}:
        raise ValueError("Unexpected network settings")
    for name, sizes in network.items():
        if not isinstance(sizes, list) or not sizes or any(type(n) is not int or n < 1 for n in sizes):
            raise ValueError(f"{name} must be a nonempty list of positive integers")
    seed = raw["seed"] if args.seed is None else args.seed
    if type(seed) is not int or seed < 0 or seed >= 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if args.run_name and (args.run_name in (".", "..") or Path(args.run_name).name != args.run_name):
        raise ValueError("run-name must be a single directory name")
    task = TaskConfig.load(args.task_config)
    xml = ET.parse(args.xml).getroot()
    sim_dt = float(xml.find("option").get("timestep"))
    substeps = round(task.control_dt / sim_dt)
    episode_length = round(task.episode_seconds / task.control_dt)
    if substeps < 1 or not math.isclose(substeps * sim_dt, task.control_dt):
        raise ValueError("Task control_dt must be a multiple of the XML timestep")
    if episode_length < 1 or not math.isclose(episode_length * task.control_dt, task.episode_seconds):
        raise ValueError("episode_seconds must be a multiple of control_dt")
    return {"seed": seed, "smoke": args.smoke, "ppo": ppo, "network": network,
            "task": asdict(task), "episode_length": episode_length, "action_repeat": 1,
            "sim_dt": sim_dt, "substeps": substeps,
            "xml_path": str(args.xml.resolve()), "task_config_path": str(args.task_config.resolve()),
            "output_dir": str(args.output_dir.resolve()), "run_name": args.run_name}


def load_backend():
    # Set process defaults before importing JAX; respect user overrides.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import jax
        from brax.training.agents.ppo import train as ppo
        from brax.training.agents.ppo import networks
        from mujoco_playground import wrapper
        try:
            from .finger_env_mjx import FingerHeightMjxEnv
        except ImportError:
            from finger_env_mjx import FingerHeightMjxEnv
    except ImportError as exc:
        raise RuntimeError("Use the existing Aero training Python with JAX, mujoco-mjx, Playground and Brax installed") from exc
    return SimpleNamespace(jax=jax, train=ppo.train, networks=networks,
                           wrapper=wrapper, env_type=FingerHeightMjxEnv)


def check_signature(function, kwargs):
    signature = inspect.signature(function)
    accepts_extra = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    unknown = set(kwargs) - set(signature.parameters)
    if unknown and not accepts_extra:
        raise RuntimeError(f"Installed Brax API does not accept: {sorted(unknown)}; no training started")


def json_metrics(value):
    """Convert scalar/array device metrics to portable, strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_metrics(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return json_metrics(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_metrics(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_training(plan):
    backend = load_backend()
    if backend.jax.process_count() != 1:
        raise RuntimeError("This entry point supports one host, not distributed multi-host training")
    network_factory = functools.partial(backend.networks.make_ppo_networks,
                                        **{k: tuple(v) for k, v in plan["network"].items()})
    # Environment factory is direct: no modifications to Aero's registry needed.
    env = backend.env_type(config_path=plan["task_config_path"], xml_path=plan["xml_path"])
    eval_env = backend.env_type(config_path=plan["task_config_path"], xml_path=plan["xml_path"])
    if env.episode_length != plan["episode_length"]:
        raise RuntimeError("Episode length differs between the plan and environment")
    kwargs = {**plan["ppo"], "environment": env.for_brax_training(),
              "eval_env": eval_env.for_brax_training(), "episode_length": env.episode_length,
              "action_repeat": 1, "seed": plan["seed"], "network_factory": network_factory,
              "wrap_env_fn": functools.partial(backend.wrapper.wrap_for_brax_training, full_reset=True),
              "save_checkpoint_path": "", "progress_fn": None}
    check_signature(backend.train, kwargs)
    check_signature(backend.wrapper.wrap_for_brax_training,
                    {"episode_length": env.episode_length, "action_repeat": 1, "full_reset": True})
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    name = plan["run_name"] or f"finger-height-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(plan["output_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    # Store the exact task, model, settings and code used, for later local replay.
    write_json(run_dir / "task_config.json", plan["task"])
    write_json(run_dir / "ppo_config.json", plan)
    shutil.copy2(plan["xml_path"], run_dir / "hand.xml")
    code_dir = run_dir / "source"
    code_dir.mkdir()
    for filename in ("train_ppo.py", "finger_env_mjx.py", "finger_env.py"):
        shutil.copy2(ROOT / filename, code_dir / filename)
    metadata = {"status": "running", "started_at": now.isoformat(), "versions": versions(),
                "python": sys.executable, "devices": [str(d) for d in backend.jax.local_devices()],
                "observation_size": env.observation_size, "action_size": env.action_size,
                "model_sha256": hashlib.sha256(Path(plan["xml_path"]).read_bytes()).hexdigest(),
                "checkpoint_contents": "Brax policy/value/observation-normalizer parameters; not a full optimizer-state resume"}
    write_json(run_dir / "metadata.json", metadata)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    start = time.monotonic()

    def progress(num_steps, metrics):
        record = {"num_steps": int(num_steps), "elapsed_seconds": time.monotonic() - start,
                  "metrics": json_metrics(metrics)}
        with (run_dir / "progress.jsonl").open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        score = record["metrics"].get("eval/episode_reward", "n/a")
        print(f"steps={int(num_steps):,}  eval_reward={score}  elapsed={record['elapsed_seconds']:.1f}s", flush=True)

    kwargs.update(save_checkpoint_path=str(checkpoints), progress_fn=progress)
    print(f"Run directory: {run_dir}\nCompiling and training with Brax PPO ...", flush=True)
    try:
        _, _, final_metrics = backend.train(**kwargs)
        saved = sorted((p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
        if not saved:
            raise RuntimeError("Trainer returned without numeric checkpoint directories; inspect the installed Brax checkpoint API")
        latest = saved[-1]
        metadata.update(status="completed", elapsed_seconds=time.monotonic() - start,
                        latest_checkpoint=str(latest.relative_to(run_dir)),
                        actual_checkpoint_step=int(latest.name), final_metrics=json_metrics(final_metrics))
        write_json(run_dir / "metadata.json", metadata)
        (run_dir / "latest_checkpoint.txt").write_text(str(latest.relative_to(run_dir)) + "\n")
        print(f"Training finished. Checkpoint: {latest}", flush=True)
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - start)
        write_json(run_dir / "metadata.json", metadata)
        raise
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps({"plan": plan, "versions": versions()}, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN PASS: configuration validated. No training, MJX compilation or output directory was created.")
        return
    run_training(plan)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted; completed checkpoints remain in the run directory.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"TRAIN FAILED [{type(exc).__name__}]: {exc}", file=sys.stderr)
        sys.exit(1)
