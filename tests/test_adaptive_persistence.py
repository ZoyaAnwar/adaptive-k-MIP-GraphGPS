"""Checkpoint restoration, baseline compatibility, and output isolation."""
import ast
import io
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_mass_calibration as fixtures


torch = fixtures.torch
identity = fixtures.identity


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class PersistenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(sys.modules, {'graphgps.layer.sparse_attention_layer': fixtures.attention}):
            cls.checkpoint = fixtures.load('checkpoint_attention', 'graphgps/adaptive_k/checkpoint_attention.py')

    def layer(self, k=50):
        return self.checkpoint.MassSparseAttention(4, 2, 2, 2, k, 'none')

    def test_serialized_checkpoint_restores_budget_and_outputs(self):
        torch.manual_seed(4)
        original = torch.nn.Sequential(self.layer(2), self.layer(3)).eval()
        stream = io.BytesIO()
        torch.save({'model_state': original.state_dict()}, stream)
        stream.seek(0)
        restored = torch.nn.Sequential(self.layer(), self.layer()).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True)['model_state'])
        self.assertEqual([m.k for m in restored], [2, 3])
        x = torch.randn(1, 7, 4)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), original(x))

    def test_baseline_state_dict_format_and_values_are_unchanged(self):
        baseline = fixtures.attention.SparseAttention(4, 2, 2, 2, 7, 'none')
        self.assertFalse(any(key.endswith('_extra_state') for key in baseline.state_dict()))
        self.assertEqual(self.layer(1).get_extra_state()['k'], 1)
        self.assertEqual(self.layer(50).get_extra_state()['k'], 50)
        with self.assertRaises(ValueError):
            self.layer().set_extra_state({'version': 1, 'k': 0})
        with self.assertRaises(RuntimeError):
            self.layer().load_state_dict(baseline.state_dict())

    def test_dataset_fingerprint_is_stable_and_order_sensitive(self):
        data = fixtures.Dataset()
        first = identity.dataset_fingerprint(data)
        self.assertEqual(first, identity.dataset_fingerprint(data))
        with patch.object(data, 'indices', return_value=[200, 100]):
            self.assertNotEqual(first, identity.dataset_fingerprint(data))
        data.graphs.reverse()
        self.assertNotEqual(first, identity.dataset_fingerprint(data))
        data.graphs.reverse()
        data.graphs[0].x[0, 0] += 1
        self.assertNotEqual(first, identity.dataset_fingerprint(data))


class OutputIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Execute the actual small main.py helpers without importing GraphGym.
        tree = ast.parse((fixtures.ROOT / 'main.py').read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('custom_set_out_dir', 'custom_set_run_dir')]
        self.namespace = {'os': os, 'makedirs_rm_exist': self.clean_dir}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), self.namespace)
        self.module_patch = patch.dict(sys.modules, {'graphgps.adaptive_k.run_identity': identity})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def clean_dir(self, path):
        Path(path).resolve().relative_to(Path(self.temp.name).resolve())
        if Path(path).exists():
            shutil.rmtree(path)
        Path(path).mkdir(parents=True)

    def cfg(self, schedule, resume=False):
        return SimpleNamespace(out_dir=self.temp.name, gt=SimpleNamespace(
            sparse=SimpleNamespace(k_schedule=schedule)),
            train=SimpleNamespace(auto_resume=resume))

    def test_adaptive_and_baseline_output_directories_are_distinct(self):
        for baseline in ('constant', 'linear', 'exponential'):
            normal = self.cfg(baseline)
            adaptive = self.cfg('adaptive_mass')
            for cfg in (normal, adaptive):
                self.namespace['custom_set_out_dir'](cfg, 'linear.yaml', '')
            self.assertEqual(normal.out_dir, os.path.join(self.temp.name, 'linear'))
            self.assertNotEqual(normal.out_dir, adaptive.out_dir)

    def test_root_guard_precedes_config_dump(self):
        cfg = self.cfg('adaptive_mass')
        root = Path(self.temp.name) / 'adaptive_mass' / 'linear'
        root.mkdir(parents=True)
        config = root / 'config.yaml'
        config.write_text('baseline config', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'non-adaptive_mass'):
            self.namespace['custom_set_out_dir'](cfg, 'linear.yaml', '')
        self.assertEqual(config.read_text(), 'baseline config')

    def test_rejects_baseline_directory_before_deletion_or_resume(self):
        cfg = self.cfg('adaptive_mass')
        self.namespace['custom_set_out_dir'](cfg, 'linear.yaml', '')
        directory = Path(cfg.out_dir) / '0'
        directory.mkdir(parents=True)
        sentinel = directory / 'baseline_checkpoint'
        sentinel.write_text('preserve', encoding='utf-8')
        for resume in (False, True):
            cfg.train.auto_resume = resume
            with self.assertRaisesRegex(RuntimeError, 'non-adaptive_mass'):
                self.namespace['custom_set_run_dir'](cfg, 0)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_adaptive_resume_keeps_existing_artifacts(self):
        cfg = self.cfg('adaptive_mass')
        self.namespace['custom_set_out_dir'](cfg, 'linear.yaml', 'trial')
        self.namespace['custom_set_run_dir'](cfg, 0)
        marker = Path(cfg.run_dir) / 'adaptive_mass_run.json'
        self.assertTrue(marker.exists())
        artifact = Path(cfg.run_dir) / 'calibration.npz'
        artifact.write_bytes(b'preserve')
        cfg.train.auto_resume = True
        self.namespace['custom_set_run_dir'](cfg, 0)
        self.assertEqual(artifact.read_bytes(), b'preserve')


class ColabRunnerTest(unittest.TestCase):
    def test_all_methods_construct_isolated_commands_for_both_datasets(self):
        bash = shutil.which('bash') if os.name != 'nt' else r'C:\Program Files\Git\bin\bash.exe'
        if not bash or not Path(bash).is_file():
            self.skipTest('Bash unavailable')
        subprocess.run([bash, '--noprofile', '--norc', '-n', 'colab/run_ablation.sh'],
                       cwd=fixtures.ROOT, check=True, capture_output=True)
        wrapper = 'python() { printf "CALL\\n" >> "$CAPTURE"; printf "%s\\n" "$@" >> "$CAPTURE"; }; export -f python; bash colab/run_ablation.sh'
        for dataset in ('PascalVOC-SP', 'Peptides-func'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                capture = Path(temp) / 'commands.txt'
                env = dict(os.environ, CAPTURE=capture.as_posix(), OUT_ROOT=Path(temp).as_posix(),
                           DATASET=dataset, MODE='full', SEEDS='0',
                           RUNS='fixed,linear,exponential,adaptive_mass')
                subprocess.run([bash, '--noprofile', '--norc', '-c', wrapper],
                               cwd=fixtures.ROOT, env=env, check=True, capture_output=True)
                calls = [block.splitlines() for block in capture.read_text().split('CALL\n')[1:]]
                self.assertEqual(len(calls), 4)
                roots = [call[call.index('out_dir') + 1] for call in calls]
                self.assertEqual(len(set(roots)), 4)
                for index, method in enumerate(('fixed', 'linear', 'exponential', 'adaptive_mass')):
                    self.assertTrue(roots[index].endswith('/' + method))
                    if method == 'adaptive_mass':
                        self.assertEqual(calls[index][calls[index].index('gt.sparse.k_schedule') + 1], 'adaptive_mass')
                    else:
                        self.assertNotIn('gt.sparse.k_schedule', calls[index])


if __name__ == '__main__':
    unittest.main()
