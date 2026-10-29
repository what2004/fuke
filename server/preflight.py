"""Check the server's actual device, pinned runtime and training APIs before PPO."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "najiushi/rl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true", help="local verification only")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12 for this locked environment")
    print("Python:", sys.executable, flush=True)
    print("Platform:", platform.platform(), flush=True)
    expected = {"jax": "0.8.2", "jaxlib": "0.8.2", "mujoco": "3.10.0",
                "mujoco-mjx": "3.10.0", "brax": "0.14.2", "playground": "0.2.0"}
    if not args.allow_cpu:
        expected.update({"jax-cuda12-plugin": "0.8.2", "jax-cuda12-pjrt": "0.8.2",
                         "nvidia-cudnn-cu12": "9.8.0.87", "nvidia-nccl-cu12": "2.27.7"})
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], check=True, capture_output=True, text=True)
        print(result.stdout, flush=True)
        for line in result.stdout.splitlines():
            driver = tuple(int(n) for n in line.split(',')[1].strip().split('.'))
            if driver < (525, 60, 13):
                raise RuntimeError("NVIDIA driver must support the CUDA 12 runtime (>=525.60.13)")
    for name, wanted in expected.items():
        found = metadata.version(name)
        print(f"{name}: {found}", flush=True)
        if found != wanted:
            raise RuntimeError(f"{name}: expected {wanted}, found {found}; use the locked setup")
    import jax
    import jax.numpy as jp
    devices = jax.devices()
    print("JAX devices:", devices, flush=True)
    if not args.allow_cpu and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("No JAX GPU device; refusing to silently train on CPU")
    product = jax.jit(lambda x: x @ x)(jp.ones((32, 32)))
    product.block_until_ready()
    # Exercise the convolution runtime as well: a GPU listed by JAX alone
    # does not prove that the installed cuDNN supports V100.
    conv = jax.jit(lambda x, w: jax.lax.conv_general_dilated(
        x, w, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")))
    conv(jp.ones((1, 8, 8, 1)), jp.ones((3, 3, 1, 1))).block_until_ready()
    from train_ppo import build_plan, parse_args, load_backend, check_signature
    plan = build_plan(parse_args(["--smoke"]))
    backend = load_backend()
    env = backend.env_type()
    check_signature(backend.train, {**plan['ppo'], 'environment': env.for_brax_training(),
                                   'eval_env': env.for_brax_training(), 'episode_length': env.episode_length,
                                   'action_repeat': 1, 'seed': 0, 'network_factory': None,
                                   'wrap_env_fn': None, 'save_checkpoint_path': '', 'progress_fn': None})
    check_signature(backend.wrapper.wrap_for_brax_training,
                    {'episode_length': env.episode_length, 'action_repeat': 1, 'full_reset': True})
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = jax.jit(env.step)(state, env.action_for_length(.100))
    state.obs.block_until_ready()
    assert state.obs.shape == (16,) and bool(jp.all(jp.isfinite(state.obs)))
    print("PREFLIGHT PASS: runtime, device kernels, model, sensors, JIT step and PPO APIs.", flush=True)
    print("Next: full task parity check, then a small real PPO smoke run.", flush=True)


if __name__ == '__main__':
    main()
