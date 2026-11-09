"""CUDA boundary/logging tests using deterministic counters; no GPU required."""
import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import test_mass_calibration as fixtures


memory = fixtures.load('memory_profiling', 'graphgps/memory_profiling.py')


class FakeCuda:
    def __init__(self):
        self.events = []
        self.allocated = self.reserved = 0
    def is_available(self):
        return True
    def synchronize(self, device):
        self.events.append('sync')
    def reset_peak_memory_stats(self, device):
        self.events.append('reset')
        self.allocated = self.reserved = 0
    def work(self, allocated, reserved):
        self.events.append('work')
        self.allocated = max(self.allocated, allocated * memory.MIB)
        self.reserved = max(self.reserved, reserved * memory.MIB)
    def max_memory_allocated(self, device):
        self.events.append('allocated')
        return self.allocated
    def max_memory_reserved(self, device):
        self.events.append('reserved')
        return self.reserved


class MemoryProfilerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cuda = FakeCuda()
        self.metadata = dict(schedule='constant', dataset='fixture', seed=0)
    def profiler(self, start=0, device='cuda:0'):
        return memory.GPUMemoryProfiler(device, self.temp.name, self.metadata,
                                        start, self.cuda)

    def test_epoch_peaks_units_reset_and_run_high_water(self):
        profiler = self.profiler()
        profiler.begin_epoch(0)
        self.assertEqual(self.cuda.events, ['sync', 'reset'])
        self.cuda.work(4, 12)
        first = profiler.end_epoch(0, 50, ['val', 'test'])
        self.assertEqual(first['peak_allocated_mib'], 4)
        self.assertEqual(first['peak_reserved_mib'], 12)
        profiler.begin_epoch(1)
        self.cuda.work(2, 6)
        second = profiler.end_epoch(1, 40, [])
        self.assertEqual(second['peak_allocated_mib'], 2)
        summary = profiler.finish()
        self.assertEqual(summary['run_peak_allocated_mib'], 4)
        self.assertEqual(summary['run_peak_reserved_mib'], 12)
        self.assertEqual(self.cuda.events.count('reset'), 2)
        self.assertEqual(self.cuda.events.count('sync'), 4)
        rows = [json.loads(line) for line in profiler.epoch_path.read_text().splitlines()]
        self.assertEqual([row['k'] for row in rows], [50, 40])
        self.assertEqual(rows[0]['metadata'], self.metadata)
        self.assertEqual(rows[0]['units'], 'MiB')
        self.assertFalse(rows[1]['calibration_performed'])
        self.assertTrue(summary['complete_epoch_history'])
        self.assertTrue(summary['finished'])

    def test_calibration_does_not_reset_or_hide_later_training_peak(self):
        profiler = self.profiler()
        profiler.begin_epoch(5)
        self.cuda.work(3, 7)
        self.cuda.synchronize(profiler.device)
        self.assertEqual(profiler.calibration_peak()['peak_allocated_mib'], 3)
        self.assertEqual(self.cuda.events.count('reset'), 1)
        self.cuda.work(11, 19)
        row = profiler.end_epoch(5, 8, ['val', 'test'])
        self.assertEqual(row['peak_allocated_mib'], 11)
        self.assertEqual(row['peak_reserved_mib'], 19)
        self.assertEqual(row['calibration_peak_allocated_mib'], 3)
        self.assertEqual(row['calibration_peak_reserved_mib'], 7)

    def test_cpu_does_not_call_cuda_and_reports_null(self):
        profiler = self.profiler(device='cpu')
        profiler.begin_epoch(0)
        row = profiler.end_epoch(0, 10, ['val', 'test'])
        summary = profiler.finish()
        self.assertEqual(self.cuda.events, [])
        self.assertFalse(row['cuda_enabled'])
        self.assertIsNone(row['peak_allocated_mib'])
        self.assertIsNone(row['peak_reserved_mib'])
        self.assertIsNone(summary['run_peak_allocated_mib'])

    def test_resume_replays_epochs_without_losing_run_peak(self):
        profiler = self.profiler()
        for epoch, peak in enumerate((3, 12)):
            profiler.begin_epoch(epoch)
            self.cuda.work(peak, peak + 1)
            profiler.end_epoch(epoch, 50, [])
        profiler.finish()
        # Replay epoch 1 from an older checkpoint; retain the physically observed peak.
        resumed = self.profiler(start=1)
        resumed.begin_epoch(1)
        self.cuda.work(4, 5)
        resumed.end_epoch(1, 50, [])
        summary = resumed.finish()
        rows = [json.loads(line) for line in resumed.epoch_path.read_text().splitlines()]
        self.assertEqual([row['epoch'] for row in rows], [0, 1])
        self.assertEqual(rows[-1]['peak_allocated_mib'], 4)
        self.assertEqual(summary['run_peak_allocated_mib'], 12)
        self.assertEqual(summary['run_peak_reserved_mib'], 13)

    def test_late_instrumentation_and_truncated_append_are_labelled(self):
        profiler = self.profiler(start=3)
        profiler.begin_epoch(3)
        self.cuda.work(4, 5)
        profiler.end_epoch(3, 50, [])
        with profiler.epoch_path.open('a') as stream:
            stream.write('{"epoch":')
        restored = self.profiler(start=4)
        summary = restored.finish()
        self.assertFalse(summary['complete_epoch_history'])
        self.assertEqual(summary['recorded_epochs'], [3])

    def test_boundary_mismatches_are_rejected(self):
        profiler = self.profiler()
        with self.assertRaises(RuntimeError):
            profiler.calibration_peak()
        profiler.begin_epoch(0)
        with self.assertRaises(RuntimeError):
            profiler.begin_epoch(1)
        with self.assertRaises(RuntimeError):
            profiler.end_epoch(1, 50, [])
        with self.assertRaises(RuntimeError):
            profiler.finish()


class TrainingBoundaryTest(unittest.TestCase):
    def test_all_schedules_cover_calibration_training_and_both_evaluations(self):
        tree = ast.parse((fixtures.ROOT / 'graphgps/train/custom_train.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'custom_train')
        function.decorator_list = []
        for schedule in ('constant', 'linear', 'exponential', 'adaptive_mass'):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as directory:
                cuda = FakeCuda()
                metadata = {}
                cfg = SimpleNamespace(
                    gt=SimpleNamespace(sparse=SimpleNamespace(k_schedule=schedule)),
                    train=SimpleNamespace(auto_resume=False, sampler='full_batch',
                                          ckpt_best=False, enable_ckpt=False, ckpt_clean=False),
                    dataset=SimpleNamespace(task='graph', name='fixture', format='fixture', split_index=0),
                    optim=SimpleNamespace(max_epoch=6, scheduler='constant', batch_accumulation=1),
                    wandb=SimpleNamespace(use=False), metric_best='auto', run_dir=directory, seed=0)
                model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device='cuda:0')]))
                def train(*args):
                    cuda.events.append('train')
                    cuda.work(8, 12)
                def evaluate(*args, **kwargs):
                    cuda.events.append(kwargs['split'])
                    cuda.work(10 if kwargs['split'] == 'val' else 11,
                              16 if kwargs['split'] == 'val' else 18)
                class FakeCalibration:
                    k = 50
                    def __init__(self, *args):
                        pass
                    def step(self, epoch, profiler):
                        if epoch == 5:
                            cuda.events.append('calibration')
                            cuda.work(3, 6)
                            profiler.calibration_peak()
                        return None
                calibration_import = ModuleType('graphgps.adaptive_k.mass_calibration')
                calibration_import.MassCalibration = FakeCalibration
                namespace = dict(cfg=cfg, load_ckpt=lambda *a: 0, logging=logging, time=time,
                                 np=np, torch=fixtures.torch, train_epoch=train, eval_epoch=evaluate,
                                 _restore_perf_history=lambda *a: [[], [], []],
                                 is_eval_epoch=lambda e: e % 2 == 0, set_epoch_k=lambda *a: 50,
                                 GPUMemoryProfiler=lambda device, path, meta, start: memory.GPUMemoryProfiler(
                                     device, path, meta, start, cuda))
                exec(compile(ast.Module(body=[function], type_ignores=[]), 'custom_train.py', 'exec'), namespace)
                class Logger:
                    def write_epoch(self, epoch):
                        return dict(epoch=epoch, loss=1.0)
                    def close(self):
                        pass
                with patch.dict(sys.modules, {'graphgps.adaptive_k.mass_calibration': calibration_import}):
                    namespace['custom_train']([Logger(), Logger(), Logger()],
                                             [SimpleNamespace(dataset=[]) for _ in range(3)], model,
                                             None, SimpleNamespace(step=lambda: None))
                rows = [json.loads(line) for line in (Path(directory) / 'memory/epochs.jsonl').read_text().splitlines()]
                self.assertEqual(cuda.events.count('reset'), 6)
                for row in rows:
                    self.assertEqual(row['peak_allocated_mib'], 11 if row['epoch'] % 2 == 0 else 8)
                    self.assertEqual(row['evaluated_splits'], ['val', 'test'] if row['epoch'] % 2 == 0 else [])
                if schedule == 'adaptive_mass':
                    self.assertEqual(rows[5]['calibration_peak_allocated_mib'], 3)
                else:
                    self.assertFalse(any(row['calibration_performed'] for row in rows))
                summary = json.loads((Path(directory) / 'memory/run.json').read_text())
                self.assertEqual(summary['run_peak_allocated_mib'], 11)
                self.assertEqual(summary['run_peak_reserved_mib'], 18)


if __name__ == '__main__':
    unittest.main()
