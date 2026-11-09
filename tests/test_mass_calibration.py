"""Strategy B arithmetic and controller tests; GPU validation remains a separate pilot."""
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = load('mass_statistics', 'graphgps/adaptive_k/mass_statistics.py')
identity = load('run_identity', 'graphgps/adaptive_k/run_identity.py')
try:
    import torch
except ImportError:
    torch = None


class StatisticsTest(unittest.TestCase):
    def test_dkw_and_calibration_boundaries(self):
        self.assertEqual(stats.calibration_epochs(16, 5), [5, 10, 15])
        self.assertEqual(stats.calibration_epochs(5, 5), [])
        epsilon, level, rank = stats.adjusted_quantile(8500, 20)
        self.assertAlmostEqual(epsilon, 0.0198296, places=6)
        self.assertEqual(rank, int(np.ceil(8500 * level)))
        self.assertGreater(stats.adjusted_quantile(8500, 49)[0], epsilon)
        with self.assertRaises(ValueError):
            stats.adjusted_quantile(1, 20)
        with self.assertRaises(ValueError):
            stats.calibration_epochs(100, 0)

    def test_no_floor_extremes_and_small_graph(self):
        p = np.zeros((8500, 50))
        p[:, 0] = 1
        self.assertEqual(stats.select_budget(p, 20)[0], 1)
        p[:] = 1 / 50
        self.assertEqual(stats.select_budget(p, 20)[0], 45)
        self.assertEqual(stats.select_budget(p, 20, tau=1)[0], 50)
        p[:] = 0
        p[:, :3] = 1 / 3
        self.assertEqual(stats.select_budget(p, 20)[0], 3)
        self.assertEqual(stats.select_budget(p, 20, tau=1)[0], 3)

    def test_confidence_adjusted_order_statistic(self):
        p = np.zeros((8500, 50))
        p[:, 0] = 1
        p[:300] = 1 / 50
        k, required, mass, summary = stats.select_budget(p, 20)
        self.assertEqual(k, 45)  # Empirical 95th percentile alone would choose 1.
        self.assertEqual(int((required == 45).sum()), 300)
        self.assertGreaterEqual(summary['coverage_lower_bound'], 0.95)
        self.assertTrue((mass >= 0.90 - 1e-12).all())

    def test_distribution_comparisons(self):
        p = np.zeros((100, 50))
        p[:, 0] = 1
        r = np.ones(100, dtype=int)
        equal = stats.compare_calibrations(p, r, p, r)
        self.assertTrue(all(value == 0 for value in equal.values()))
        other = np.full((100, 50), 1 / 50)
        changed = stats.compare_calibrations(other, np.full(100, 45), p, r)
        self.assertGreater(changed['mean_profile_js_divergence'], 0)
        self.assertEqual(changed['required_budget_cdf_distance'], 1)
        self.assertEqual(changed['required_budget_wasserstein'], 44)

    def test_probability_validation(self):
        for p in (np.zeros((10, 50)), np.full((10, 49), 1 / 49),
                  np.full((10, 50), np.nan)):
            with self.assertRaises(ValueError):
                stats.select_budget(p, 20)


if torch is not None:
    # Tiny dense oracle only inside tests. Production extraction uses LazyTensor.
    class SymbolicOracle:
        def __init__(self, value):
            self.value = value
        @property
        def shape(self):
            return self.value.shape
        def __mul__(self, other):
            return SymbolicOracle(self.value * other.value)
        def sum(self, axis):
            return SymbolicOracle(self.value.sum(axis))
        def __truediv__(self, scale):
            return SymbolicOracle(self.value / scale)
        def __neg__(self):
            return SymbolicOracle(-self.value)
        def argKmin(self, k, dim):
            return self.value.topk(k, dim=dim, largest=False).indices

    attention = load('sparse_attention', 'graphgps/layer/sparse_attention_layer.py')
    schedule_import = ModuleType('graphgps.adaptive_k.train_schedule')
    def apply(model, k):
        for module in model.modules():
            if isinstance(module, attention.SparseAttention):
                module.k = k
    schedule_import.apply_k_to_model = apply
    batch_import = ModuleType('torch_geometric.data')
    class Graph:
        def __init__(self, x):
            self.x, self.num_nodes = x, len(x)
        def clone(self):
            return Graph(self.x.clone())
        def to(self, device):
            return self  # CPU fixture for controller tests.
    class Batch:
        @staticmethod
        def from_data_list(graphs):
            assert len(graphs) == 1
            return graphs[0]
    batch_import.Batch = Batch
    with patch.dict(sys.modules, {
        'torch_geometric.data': batch_import,
        'graphgps.layer.sparse_attention_layer': attention,
        'graphgps.adaptive_k.train_schedule': schedule_import,
        'graphgps.adaptive_k.mass_statistics': stats,
        'graphgps.adaptive_k.run_identity': identity,
    }):
        calibration = load('mass_calibration', 'graphgps/adaptive_k/mass_calibration.py')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                attention.SparseAttention(4, 2, 2, 2, 7, 'none') for _ in range(2)])
            for layer in self.layers:
                layer.use_keops = True
            self.fail = False
        def parameters(self, *args, **kwargs):
            return iter([SimpleNamespace(device=torch.device('cuda:0'), is_cuda=False)])
        def forward(self, batch):
            x = batch.x.unsqueeze(0)
            for layer in self.layers:
                x = layer(x)
            if self.fail:
                raise RuntimeError('reference failure')
            return x

    class Dataset:
        transform = None
        def __init__(self):
            self.graphs = [Graph(torch.randn(n, 4)) for n in (3, 7)]
        def __len__(self):
            return len(self.graphs)
        def __getitem__(self, index):
            return self.graphs[index]
        def indices(self):
            return [100, 200]


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = SimpleNamespace(random_attention=False, mass_calibration=SimpleNamespace(
            interval=5, samples=8500, tau=0.90, q=0.95, delta=0.05, query_chunk_size=64))
        self.model, self.dataset = Model(), Dataset()
        patches = [patch.object(calibration, 'LazyTensor', SymbolicOracle),
                   patch.object(attention, 'LazyTensor', SymbolicOracle),
                   patch.object(torch.cuda, 'is_available', return_value=True),
                   patch.object(torch.cuda, 'device_count', return_value=0),
                   patch.object(torch.cuda, 'synchronize'),
                   patch.object(torch.cuda, 'reset_peak_memory_stats'),
                   patch.object(torch.cuda, 'max_memory_allocated', return_value=1234),
                   patch.object(torch.cuda, 'max_memory_reserved', return_value=4096),
                   patch.object(torch.cuda, 'mem_get_info', return_value=(10**9, 10**9))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def controller(self, start=0):
        return calibration.MassCalibration(self.model, self.cfg, self.dataset,
                                           self.temp.name, 42, 16, start)

    def test_sampling_archive_comparison_and_resume(self):
        controller = self.controller()
        self.assertEqual(controller.k, 50)
        self.assertIsNone(controller.step(0))
        first = controller.step(5)
        path = Path(self.temp.name) / 'mass_calibration/epoch_000005.npz'
        with np.load(path, allow_pickle=False) as saved:
            self.assertEqual(saved['probabilities'].shape, (8500, 50))
            self.assertEqual(len(saved['graph']), 8500)
            self.assertTrue(np.allclose(saved['probabilities'].sum(axis=1), 1))
            self.assertTrue((saved['query'] < np.where(saved['graph'] == 0, 3, 7)).all())
            self.assertTrue((saved['source_graph'] == np.where(saved['graph'] == 0, 100, 200)).all())
            self.assertTrue(0.47 < (saved['graph'] == 0).mean() < 0.53)
            self.assertTrue(0.47 < (saved['layer'] == 0).mean() < 0.53)
            self.assertTrue(0.47 < (saved['head'] == 0).mean() < 0.53)
            self.assertLess(len(np.unique(np.stack([saved[n] for n in
                            ('graph', 'query', 'layer', 'head')], axis=1), axis=0)), 8500)
            old = saved['probabilities'].copy()
        self.assertTrue(1 <= first['k'] <= 50)
        self.assertTrue(all(layer.k == first['k'] for layer in self.model.layers))
        controller.step(5)
        with np.load(path) as saved:
            np.testing.assert_array_equal(old, saved['probabilities'])
        restored = self.controller(start=7)
        self.assertEqual(restored.k, first['k'])
        self.assertIsNone(restored.step(7))
        next_stats = restored.step(10)
        self.assertEqual(next_stats['previous_calibration_epoch'], 5)
        self.assertIn('mean_profile_js_divergence', next_stats)
        self.assertIn('required_budget_cdf_distance', next_stats)
        self.assertTrue(path.with_suffix('.json').exists())
        # No monotonic cap or dependence on the previously deployed budget.
        # Controlled observations isolate the controller from layerwise feature collapse.
        def concentrated(module, x, queries, heads, chunk):
            p = np.zeros((len(queries), 50), dtype=np.float32)
            p[:, 0] = 1
            return p
        def uniform(module, x, queries, heads, chunk):
            p = np.zeros((len(queries), 50), dtype=np.float32)
            p[:, :x.shape[1]] = 1 / x.shape[1]
            return p
        with patch.object(calibration, 'top_probabilities', side_effect=concentrated):
            low = restored.step(15)['k']
        self.assertEqual(low, 1)
        self.assertLess(low, next_stats['k'])
        with patch.object(calibration, 'top_probabilities', side_effect=uniform):
            high = restored.step(15)['k']
        self.assertEqual(high, 7)
        self.assertGreater(high, low)

    def test_restore_model_rng_and_hooks_after_failure(self):
        controller = self.controller()
        self.model.train()
        self.model.layers[0].eval()
        modes = [m.training for m in self.model.modules()]
        python_state, np_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, 'reference failure'):
            controller.step(5)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertTrue(all(m.k == 50 for m in self.model.layers))
        self.assertTrue(all(not m._forward_pre_hooks for m in self.model.layers))
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_invalid_backend_random_attention_and_missing_resume(self):
        self.cfg.random_attention = True
        with self.assertRaisesRegex(ValueError, 'random_attention'):
            self.controller()
        self.cfg.random_attention = False
        self.model.layers[0].use_keops = False
        with self.assertRaisesRegex(RuntimeError, 'LazyTensor'):
            self.controller()
        self.model.layers[0].use_keops = True
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.controller(start=7)

    def test_shared_epoch_profiler_owns_calibration_reset(self):
        controller = self.controller()
        class SharedProfiler:
            def calibration_peak(self):
                return dict(peak_allocated_mib=3.0, peak_reserved_mib=7.0)
        with patch.object(torch.cuda, 'reset_peak_memory_stats') as reset:
            result = controller.step(5, SharedProfiler())
        reset.assert_not_called()
        self.assertEqual(result['peak_allocated_mib'], 3.0)
        self.assertEqual(result['peak_reserved_mib'], 7.0)
        self.assertEqual(result['peak_gpu_memory_bytes'], 3 * 1024**2)

    def test_serialized_checkpoint_training_resume(self):
        with patch.dict(sys.modules, {'graphgps.layer.sparse_attention_layer': attention}):
            checkpoint = load('checkpoint_attention', 'graphgps/adaptive_k/checkpoint_attention.py')
        def model():
            value = Model()
            value.layers = torch.nn.ModuleList([
                checkpoint.MassSparseAttention(4, 2, 2, 2, 50, 'none') for _ in range(2)])
            for layer in value.layers:
                layer.use_keops = True
            return value
        self.model = model()
        controller = self.controller()
        selected = controller.step(5)['k']
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        self.model = model()
        stream.seek(0)
        self.model.load_state_dict(torch.load(stream, weights_only=True))
        self.assertTrue(all(layer.k == selected for layer in self.model.layers))
        restored = self.controller(start=7)
        self.assertEqual(restored.k, selected)
        self.assertIsNone(restored.step(7))
        self.assertEqual(restored.step(10)['previous_calibration_epoch'], 5)

    def test_missing_required_archive_is_rejected(self):
        controller = self.controller()
        controller.step(5)
        (Path(self.temp.name) / 'mass_calibration/epoch_000005.npz').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            self.controller(start=7)

    def test_resume_rejects_data_reordering_and_changed_content(self):
        controller = self.controller()
        controller.step(5)
        original = self.dataset.graphs[:]
        self.dataset.graphs.reverse()
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.controller(start=7)
        self.dataset.graphs = original
        self.dataset.graphs[0].x[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.controller(start=7)

    def test_early_resume_validates_data_and_checkpoint_budget(self):
        controller = self.controller()
        self.assertEqual(self.controller(start=2).k, 50)
        self.dataset.graphs[0].x[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'dataset identity'):
            self.controller(start=2)
        self.dataset.graphs[0].x[0, 0] -= 1
        # Use the original manifest after any float round-trip variation.
        self.controller()
        controller.step(5)
        self.model.layers[0].k = 50
        with self.assertRaisesRegex(ValueError, 'checkpoint budget'):
            self.controller(start=7)

    def test_sorted_extraction_and_small_graph(self):
        module = self.model.layers[0]
        x = self.dataset[0].x.unsqueeze(0)
        queries, heads = np.array([0, 1, 2, 0]), np.array([0, 1, 0, 1])
        result = calibration.top_probabilities(module, x, queries, heads, 1)
        q = module.split_heads(module.MQs(x), 2)[0]
        keys = module.split_heads(module.MKs(x), 2)[0]
        for row, (query, head) in enumerate(zip(queries, heads)):
            expected = ((q[head, query] * keys[head]).sum(-1) / 2**0.5).sort(descending=True).values.softmax(-1)
            np.testing.assert_allclose(result[row, :3], expected.detach().numpy(), atol=1e-7)
            self.assertTrue((result[row, 3:] == 0).all())


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and attention._HAS_KEOPS, 'requires CUDA and PyKeOps')
class CudaKeOpsTest(unittest.TestCase):
    def test_real_symbolic_extraction_matches_tiny_reference(self):
        module = attention.SparseAttention(4, 2, 2, 2, 50, 'none').cuda()
        x = torch.randn(1, 7, 4, device='cuda')
        queries, heads = np.array([0, 1, 6]), np.array([0, 1, 0])
        result = calibration.top_probabilities(module, x, queries, heads, 2)
        with torch.no_grad():
            q = module.split_heads(module.MQs(x), 2)[0]
            keys = module.split_heads(module.MKs(x), 2)[0]
            for row, (query, head) in enumerate(zip(queries, heads)):
                logits = (q[head, query] * keys[head]).sum(-1) / 2**0.5
                expected = logits.sort(descending=True).values.softmax(-1).cpu().numpy()
                np.testing.assert_allclose(result[row, :7], expected, atol=1e-6)
                self.assertTrue((result[row, 7:] == 0).all())


if __name__ == '__main__':
    unittest.main()
