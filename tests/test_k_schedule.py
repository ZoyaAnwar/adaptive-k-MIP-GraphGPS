"""Exercise the real scheduler with lightweight stand-ins for model dependencies."""
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class Attention:
    def __init__(self):
        self.k = -1


torch = ModuleType('torch')
nn = ModuleType('torch.nn')
nn.Module = object
torch.nn = nn
attention = ModuleType('graphgps.layer.sparse_attention_layer')
attention.SparseAttention = Attention
spec = importlib.util.spec_from_file_location(
    'train_schedule', Path(__file__).resolve().parents[1]
    / 'graphgps' / 'adaptive_k' / 'train_schedule.py',
)
scheduler = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'torch': torch, 'torch.nn': nn,
                              'graphgps.layer.sparse_attention_layer': attention}):
    spec.loader.exec_module(scheduler)


def config(schedule='exponential', **overrides):
    values = dict(k=15, k_schedule=schedule, k_start=None, k_end=None,
                  k_decay=0.5, k_min=1)
    values.update(overrides)
    return SimpleNamespace(**values)


class KScheduleTest(unittest.TestCase):
    def values(self, cfg, epochs=101):
        return [scheduler.k_for_epoch(cfg, e, epochs) for e in range(epochs)]

    def test_exponential_defaults_and_monotonicity(self):
        for epochs in (2, 3, 100, 250):
            with self.subTest(epochs=epochs):
                values = self.values(config(), epochs)
                self.assertEqual((values[0], values[-1]), (50, 10))
                self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(self.values(config())[50], 22)
        self.assertGreater(self.values(config())[25], 10)

    def test_explicit_endpoints_override_defaults(self):
        values = self.values(config(k_start=80, k_end=20))
        self.assertEqual((values[0], values[50], values[-1]), (80, 40, 20))
        self.assertEqual(self.values(config(k_start=100))[0], 100)
        self.assertEqual(self.values(config(k_end=5))[-1], 5)
        self.assertEqual(self.values(config(k_start=45))[-1], 10)

    def test_legacy_decay_is_ignored(self):
        self.assertEqual(self.values(config(k_decay=0.01)),
                         self.values(config(k_decay=0.99)))

    def test_adaptive_mass_initial_budget_ignores_baseline_floor(self):
        cfg = config('adaptive_mass', k=3, k_start=10, k_end=2, k_min=100)
        self.assertEqual(scheduler.k_for_epoch(cfg, 0, 100), 50)
        cfg.random_attention = True
        with self.assertRaisesRegex(ValueError, 'random_attention'):
            scheduler.k_for_epoch(cfg, 0, 100)

    def test_constant_ignores_adaptive_settings(self):
        cfg = config('constant', k=7, k_start=-10, k_end=90, k_min=20)
        self.assertEqual(self.values(cfg), [7] * 101)

    def test_linear_defaults_remain_fixed_k(self):
        self.assertEqual(self.values(config('linear')), [15] * 101)

    def test_linear_matches_previous_behavior(self):
        for start, end, floor in ((100, 10, 1), (50, 10, 10),
                                  (10, 50, 1), (20, 20, 1), (8, 2, 5)):
            for epochs in (1, 2, 100):
                with self.subTest(start=start, end=end, epochs=epochs):
                    expected = [max(floor, int(round(
                        start + (end - start) * (e / max(epochs - 1, 1)))))
                        for e in range(epochs)]
                    self.assertEqual(self.values(config('linear', k_start=start,
                                     k_end=end, k_min=floor), epochs), expected)

    def test_epoch_bounds_and_single_epoch(self):
        cfg = config()
        self.assertEqual(scheduler.k_for_epoch(cfg, 0, 1), 50)
        self.assertEqual(scheduler.k_for_epoch(cfg, -1, 100), 50)
        self.assertEqual(scheduler.k_for_epoch(cfg, 100, 100), 10)
        self.assertEqual(scheduler.k_for_epoch(cfg, 0, 0), 15)

    def test_floor_and_invalid_endpoints(self):
        self.assertEqual(self.values(config(k_min=12))[-1], 12)
        for start, end in ((0, 10), (-1, 10), (50, 0), (50, -1)):
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, 'positive endpoints'):
                    scheduler.k_for_epoch(config(k_start=start, k_end=end), 0, 100)

    def test_epoch_update_applies_same_k_only_to_attention(self):
        layers = [Attention(), SimpleNamespace(k=99), Attention()]
        model = SimpleNamespace(modules=lambda: iter(layers))
        k = scheduler.set_epoch_k(model, config(), 50, 101)
        self.assertEqual(k, 22)
        self.assertEqual([layer.k for layer in layers], [22, 99, 22])


class NewPascalVOCScheduleTest(unittest.TestCase):
    def test_mirrored_exponential_matches_reflected_exponential(self):
        actual = [scheduler.k_for_epoch(config('mirrored_exponential'), e, 200) for e in range(200)]
        reversed_reference = [60 - scheduler.k_for_epoch(config(), e, 200) for e in reversed(range(200))]
        self.assertEqual(actual, reversed_reference)
        self.assertEqual([actual[e] for e in (0, 50, 100, 149, 175, 199)], [50, 45, 38, 27, 19, 10])
        self.assertTrue(all(a >= b for a, b in zip(actual, actual[1:])))

    def test_cosine_turn_and_intermediate_values(self):
        actual = [scheduler.k_for_epoch(config('decrease_increase'), e, 200) for e in range(200)]
        self.assertEqual([actual[e] for e in (0, 35, 70, 105, 140, 160, 170, 199)],
                         [50, 44, 30, 16, 10, 11, 13, 15])
        # Equivalent squared-trigonometric form provides a full trajectory oracle.
        reference = [round(10 + 40 * math.cos(math.pi * e / 280)**2) if e <= 140
                     else round(10 + 5 * math.sin(math.pi * (e - 140) / 118)**2)
                     for e in range(200)]
        self.assertEqual(actual, reference)
        self.assertTrue(all(a >= b for a, b in zip(actual[:140], actual[1:141])))
        self.assertTrue(all(a <= b for a, b in zip(actual[140:], actual[141:])))

    def test_every_layer_and_independent_runs(self):
        for method in ('mirrored_exponential', 'decrease_increase'):
            runs = []
            for _ in range(2):
                cfg = config(method)
                layers = [Attention() for _ in range(8)]
                model = SimpleNamespace(modules=lambda: iter(layers))
                observed = []
                for epoch in range(200):
                    k = scheduler.set_epoch_k(model, cfg, epoch, 200)
                    self.assertIs(type(k), int)
                    self.assertEqual([layer.k for layer in layers], [k] * 8)
                    observed.append(k)
                runs.append(observed)
            self.assertEqual(runs[0], runs[1])

    def test_new_study_requires_the_defined_epoch_range(self):
        for method in ('mirrored_exponential', 'decrease_increase'):
            for epoch, epochs in ((0, 1), (0, 250), (-1, 200), (200, 200)):
                with self.assertRaises(ValueError):
                    scheduler.k_for_epoch(config(method), epoch, epochs)


if __name__ == '__main__':
    unittest.main()
