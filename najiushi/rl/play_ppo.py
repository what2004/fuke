"""Replay a downloaded Brax PPO run with CPU inference and native MuJoCo.

Uses that run's model, task configuration, native environment source snapshot,
network configuration and saved observation normalizer. No training occurs.
"""
import argparse
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from zoneinfo import ZoneInfo


@contextmanager
def replay_viewer(model, data):
    """Wait for MuJoCo's render thread before Python tears down GLFW."""
    from mujoco import viewer
    existing_threads = set(threading.enumerate())
    window = viewer.launch_passive(model, data, show_right_ui=False)
    # launch_passive starts _launch_internal as a daemon on Linux. Handle.close
    # requests exit but does not join it, allowing atexit to terminate GLFW early.
    render_threads = [thread for thread in threading.enumerate()
                      if thread not in existing_threads
                      and thread.name.endswith('(_launch_internal)')]
    try:
        yield window
    finally:
        window.close()
        for thread in render_threads:
            thread.join()


def load_run(run_dir, checkpoint_name=None):
    run_dir = Path(run_dir).resolve()
    for filename in ('hand.xml', 'task_config.json', 'metadata.json', 'source/finger_env.py'):
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(f'Missing {filename}: download the complete training run directory')
    metadata = json.loads((run_dir / 'metadata.json').read_text())
    digest = hashlib.sha256((run_dir / 'hand.xml').read_bytes()).hexdigest()
    if digest != metadata.get('model_sha256'):
        raise ValueError('Downloaded hand.xml differs from the model recorded during training')
    if checkpoint_name is None:
        pointer = run_dir / 'latest_checkpoint.txt'
        if not pointer.is_file():
            raise FileNotFoundError('Missing latest_checkpoint.txt; specify --checkpoint directory-name')
        checkpoint = (run_dir / pointer.read_text().strip()).resolve()
    else:
        checkpoint = (run_dir / 'checkpoints' / checkpoint_name).resolve()
    if not checkpoint.is_relative_to(run_dir / 'checkpoints') or not checkpoint.is_dir():
        raise ValueError('Checkpoint must be a directory inside this run/checkpoints')
    if not (checkpoint / 'ppo_network_config.json').is_file():
        raise FileNotFoundError(f'Missing PPO network configuration in {checkpoint}')
    # Load the exact native environment snapshot shipped with the run.
    module_name = 'finger_replay_' + hashlib.sha256(str(run_dir).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, run_dir / 'source/finger_env.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    env = module.FingerHeightEnv(config_path=run_dir / 'task_config.json', xml_path=run_dir / 'hand.xml')
    return env, checkpoint


def normalize_observation_size(size):
    """Accept a flat size or the serialized ShapeDtypeStruct used by Brax."""
    if isinstance(size, dict) and size.get('shape') == [16]:
        return 16
    if type(size) is int and size == 16:
        return 16
    raise ValueError(f'Expected a flat 16-observation finger policy, got {size!r}')


def load_policy(checkpoint):
    """Restore recorded network defaults, including nullable initializer fields.

    Brax 0.14.2's generic load_config indexes its initializer registry even
    for JSON null defaults. Preserve these as None when reconstructing.
    """
    from brax.training import checkpoint as common_checkpoint
    from brax.training import networks as common_networks
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from ml_collections import config_dict
    config = json.loads((checkpoint / 'ppo_network_config.json').read_text())
    config['observation_size'] = normalize_observation_size(config['observation_size'])
    if config['action_size'] != 1:
        raise ValueError('This replay expects a 16-observation, one-action finger policy')
    kwargs = config['network_factory_kwargs']
    for key, value in list(kwargs.items()):
        if key == 'activation' and value is not None:
            kwargs[key] = common_networks.ACTIVATION[value]
        elif key.endswith('_init_fn') and value is not None:
            kwargs[key] = common_networks.KERNEL_INITIALIZER[value]
    network = common_checkpoint.get_network(config_dict.ConfigDict(config), ppo_networks.make_ppo_networks)
    params = ppo_checkpoint.load(str(checkpoint))
    return ppo_networks.make_inference_fn(network)(params, deterministic=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', help='optional directory name under checkpoints, e.g. 000001032192')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', action='store_true', help='live window at real-time speed')
    args = parser.parse_args(argv)
    if args.episodes < 1 or not 0 <= args.seed < 2**32-args.episodes:
        parser.error('episodes must be positive and seed must be a valid uint32 range')
    # CPU policy inference is sufficient; no CUDA libraries or GPU are needed.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import jax
    import jax.numpy as jp
    import numpy as np
    env, checkpoint = load_run(args.run_dir, args.checkpoint)
    policy = jax.jit(load_policy(checkpoint))
    policy(jp.zeros(16, jp.float32), jax.random.PRNGKey(args.seed))[0].block_until_ready()
    print(f'Loaded: {checkpoint}\nCPU policy inference + native MuJoCo; no training.', flush=True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
    output_dir = args.run_dir.resolve() / f'replay-{stamp}'
    output_dir.mkdir()
    env.reset(seed=args.seed)
    context = nullcontext(None)
    if args.render:
        context = replay_viewer(env.model, env.data)
    fields = ['episode', 'time', 'action', 'command_length', 'tendon_length', 'height', 'tip_speed',
              'hold_seconds', 'is_success', 'success_ever', 'reward', 'mcp_deg', 'pip_deg', 'dip_deg']
    results = []
    try:
        with context as window, (output_dir / 'trajectory.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields);writer.writeheader()
            for episode in range(args.episodes):
                with window.lock() if window else nullcontext():
                    obs, info = env.reset(seed=args.seed+episode)
                key = jax.random.PRNGKey(args.seed+episode)
                total_reward, next_report, terminated = 0., 0., False
                for _ in range(env.max_steps):
                    started = time.monotonic()
                    if window and not window.is_running():
                        raise RuntimeError('Viewer closed; replay interrupted')
                    key, action_key = jax.random.split(key)
                    action = np.asarray(policy(jp.asarray(obs), action_key)[0])
                    with window.lock() if window else nullcontext():
                        obs, reward, terminated, truncated, info = env.step(action)
                        angles = np.rad2deg(env.data.qpos).copy()
                    if window:
                        window.sync()
                    total_reward += reward
                    row = {name: info[name] for name in fields if name in info}
                    row.update(episode=episode+1, action=float(action[0]), reward=reward,
                               mcp_deg=angles[0], pip_deg=angles[1], dip_deg=angles[2])
                    writer.writerow(row)
                    if info['time'] >= next_report-1e-9:
                        print(f"episode={episode+1} t={info['time']:.2f}s "
                              f"height={info['height']*100:.3f}cm speed={info['tip_speed']:.4f}m/s "
                              f"hold={info['hold_seconds']:.2f}s success={info['is_success']}", flush=True)
                        next_report += 1.
                    if terminated or truncated:
                        break
                    if window:
                        time.sleep(max(0, env.config.control_dt-(time.monotonic()-started)))
                result = dict(episode=episode+1, seed=args.seed+episode, total_reward=total_reward,
                              final_height_m=info['height'], final_tip_speed_m_s=info['tip_speed'],
                              final_hold_seconds=info['hold_seconds'], final_success=info['is_success'],
                              success_ever=info['success_ever'], first_success_time=info['first_success_time'],
                              simulation_failed=terminated)
                results.append(result)
                print('Episode result:', json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        env.close()
    summary = dict(checkpoint=str(checkpoint), backend='native MuJoCo; CPU policy',
                   noise='training configuration; NumPy RNG differs from MJX RNG', episodes=results,
                   reached_and_held_rate=sum(r['success_ever'] for r in results)/len(results),
                   final_success_rate=sum(r['final_success'] for r in results)/len(results))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(f'Replay saved to: {output_dir}', flush=True)


if __name__ == '__main__':
    main()
