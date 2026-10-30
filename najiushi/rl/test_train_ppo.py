"""Check launcher plumbing without installing/running Brax or MJX."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import train_ppo


class TrainingEntryTests(unittest.TestCase):
    def plan(self, *args):
        return train_ppo.build_plan(train_ppo.parse_args(list(args)))

    def test_default_and_smoke_config(self):
        default = self.plan()
        self.assertEqual(default["episode_length"], 400)
        self.assertEqual(default["substeps"], 20)
        self.assertEqual(default["ppo"]["num_envs"], 64)
        self.assertEqual(default["task"]["target_height"], .16)
        smoke = self.plan("--smoke", "--num-timesteps", "2048", "--num-envs", "2")
        self.assertEqual(smoke["ppo"]["num_envs"], 2)
        self.assertEqual(smoke["ppo"]["num_timesteps"], 2048)
        self.assertEqual(smoke["ppo"]["num_updates_per_batch"], 1)

    def test_random_push_task_plan_and_warm_start_validation(self):
        task = Path(__file__).with_name('config_push.json')
        plan = self.plan('--task-config', str(task))
        self.assertEqual(plan['episode_length'], 600)
        self.assertTrue(plan['task']['push_enabled'])
        self.assertFalse(plan['task']['require_low_speed'])
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            config = dict(observation_size={'shape': [16], 'dtype': 'float32'}, action_size=1,
                          normalize_observations=True, network_factory_kwargs=plan['network'])
            (checkpoint/'ppo_network_config.json').write_text(json.dumps(config))
            warm = self.plan('--task-config', str(task), '--restore-checkpoint', str(checkpoint))
            self.assertEqual(warm['restore_checkpoint'], str(checkpoint.resolve()))
            config['action_size'] = 2
            (checkpoint/'ppo_network_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'action/normalization'):
                self.plan('--restore-checkpoint', str(checkpoint))

    def test_invalid_config_fails_before_training(self):
        for args in [("--num-envs", "63"), ("--seed", "-1"),
                     ("--num-timesteps", "0"), ("--run-name", "../unsafe")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(*args)

    def test_dry_run_has_no_backend_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "must-not-exist"
            with patch.object(train_ppo, "load_backend") as load, redirect_stdout(io.StringIO()):
                train_ppo.main(["--smoke", "--dry-run", "--output-dir", str(destination)])
                load.assert_not_called()
            self.assertFalse(destination.exists())

    def test_api_mismatch_is_explicit(self):
        def old_train(environment):
            pass
        with self.assertRaisesRegex(RuntimeError, "does not accept"):
            train_ppo.check_signature(old_train, {"environment": None, "save_checkpoint_path": "/tmp/x"})

    def fake_backend(self, calls, fail=False):
        class FakeEnv:
            episode_length = 400
            observation_size = 16
            action_size = 1

            def __init__(self, **kwargs):
                calls.setdefault("envs", []).append(kwargs)

            def for_brax_training(self):
                return self

        def wrap(env, episode_length, action_repeat, full_reset=False, randomization_fn=None):
            calls["wrapper"] = (episode_length, action_repeat, full_reset)
            return env

        def train(**kwargs):
            calls["kwargs"] = kwargs
            kwargs["wrap_env_fn"](kwargs["environment"], episode_length=kwargs["episode_length"],
                                  action_repeat=kwargs["action_repeat"], randomization_fn=None)
            kwargs["progress_fn"](0, {"eval/episode_reward": np.float32(0)})
            if fail:
                raise RuntimeError("simulated backend failure")
            folder = Path(kwargs["save_checkpoint_path"]) / "000000001024"
            folder.mkdir()
            (folder / "ppo_network_config.json").write_text("{}")
            kwargs["progress_fn"](1024, {"eval/episode_reward": np.float32(2), "test/array": np.array([1, 2])})
            return None, None, {"eval/episode_reward": np.float32(2)}

        return SimpleNamespace(
            env_type=FakeEnv, train=train,
            jax=SimpleNamespace(process_count=lambda: 1, local_devices=lambda: ["test device"]),
            wrapper=SimpleNamespace(wrap_for_brax_training=wrap),
            networks=SimpleNamespace(make_ppo_networks=lambda **kwargs: None))

    def test_mock_training_logs_snapshots_and_checkpoint_pointer(self):
        calls = {}
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan("--smoke", "--output-dir", tmp, "--run-name", "test-run")
            with patch.object(train_ppo, "load_backend", return_value=self.fake_backend(calls)), redirect_stdout(io.StringIO()):
                run = train_ppo.run_training(plan)
                with self.assertRaises(FileExistsError):
                    train_ppo.run_training(plan)
            self.assertEqual(calls["wrapper"], (400, 1, True))
            self.assertEqual(calls["kwargs"]["seed"], 1)
            self.assertEqual(calls["kwargs"]["network_factory"].keywords["policy_hidden_layer_sizes"], (64, 64))
            self.assertEqual(len(calls["envs"]), 4)  # separate train/eval envs on each call
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["actual_checkpoint_step"], 1024)
            self.assertEqual(metadata["latest_checkpoint"], "checkpoints/000000001024")
            records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()]
            self.assertEqual([r["num_steps"] for r in records], [0, 1024])
            self.assertEqual(records[-1]["metrics"]["test/array"], [1, 2])
            self.assertEqual((run / "hand.xml").read_bytes(), Path(plan["xml_path"]).read_bytes())
            self.assertTrue((run / "source/finger_env_mjx.py").is_file())
            self.assertEqual(json.loads((run / "task_config.json").read_text())["target_height"], .16)

    def test_mock_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan("--smoke", "--output-dir", tmp, "--run-name", "failed-run")
            with patch.object(train_ppo, "load_backend", return_value=self.fake_backend({}, fail=True)), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated backend failure"):
                    train_ppo.run_training(plan)
            run = Path(tmp) / "failed-run"
            self.assertEqual(json.loads((run / "metadata.json").read_text())["status"], "failed")
            self.assertFalse((run / "latest_checkpoint.txt").exists())

    def test_nonfinite_metric_is_valid_json(self):
        self.assertEqual(train_ppo.json_metrics({"x": np.nan, "y": np.array([1., np.inf])}),
                         {"x": None, "y": [1., None]})


if __name__ == "__main__":
    unittest.main()
