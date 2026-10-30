"""Check complete-run loading, model integrity and checkpoint selection."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from play_ppo import load_run, normalize_observation_size


class ReplayTests(unittest.TestCase):
    def test_accepts_brax_serialized_observation_shape(self):
        self.assertEqual(normalize_observation_size({
            'shape': [16], 'dtype': 'unserializable object: float32'}), 16)
        self.assertEqual(normalize_observation_size(16), 16)
        for size in (15, {'shape': [15]}, {'shape': [1, 16]}, {'state': 16}):
            with self.assertRaises(ValueError):
                normalize_observation_size(size)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        root = Path(__file__).resolve().parent
        (self.run/'source').mkdir()
        shutil.copy2(root.parent/'descrption/hand.xml', self.run/'hand.xml')
        shutil.copy2(root/'config.json', self.run/'task_config.json')
        shutil.copy2(root/'finger_env.py', self.run/'source/finger_env.py')
        self.checkpoint = self.run/'checkpoints/000001032192'
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint/'ppo_network_config.json').write_text('{}')
        (self.run/'latest_checkpoint.txt').write_text('checkpoints/000001032192\n')
        (self.run/'metadata.json').write_text(json.dumps({
            'model_sha256': hashlib.sha256((self.run/'hand.xml').read_bytes()).hexdigest()}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_snapshot_and_relocated_checkpoint(self):
        env, checkpoint = load_run(self.run)
        try:
            self.assertEqual(checkpoint, self.checkpoint)
            self.assertTrue(type(env).__module__.startswith('finger_replay_'))
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (16,))
        finally:
            env.close()

    def test_rejects_changed_model_and_incomplete_download(self):
        (self.run/'hand.xml').write_text((self.run/'hand.xml').read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            load_run(self.run)
        (self.run/'task_config.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'complete'):
            load_run(self.run)

    def test_rejects_checkpoint_outside_run(self):
        (self.run/'latest_checkpoint.txt').write_text('../outside\n')
        with self.assertRaisesRegex(ValueError, 'inside'):
            load_run(self.run)


if __name__ == '__main__':
    unittest.main()
