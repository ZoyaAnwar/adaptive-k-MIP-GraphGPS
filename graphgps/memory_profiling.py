"""Comparable CUDA allocator peaks for complete training-loop epoch windows."""
import json
import logging
from pathlib import Path

import torch


MIB = 1024 ** 2


class GPUMemoryProfiler:
    def __init__(self, device, run_dir, metadata, start_epoch=0, cuda=None):
        self.device = torch.device(device)
        self.cuda = torch.cuda if cuda is None else cuda
        self.enabled = self.device.type == 'cuda' and self.cuda.is_available()
        self.metadata = dict(metadata)
        self.directory = Path(run_dir) / 'memory'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.epoch_path = self.directory / 'epochs.jsonl'
        self.run_path = self.directory / 'run.json'
        self.run_allocated = self.run_reserved = None
        self.rows = []
        self.epoch = None
        self.calibration = None
        self.finished = False
        if start_epoch > 0:
            if self.epoch_path.exists():
                text = self.epoch_path.read_text(encoding='utf-8')
                lines = text.splitlines()
                for index, line in enumerate(lines):
                    if line.strip():
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            if index == len(lines) - 1 and not text.endswith('\n'):
                                break  # An interrupted final append is not a complete epoch.
                            raise
                        if row['metadata'] != self.metadata:
                            raise ValueError('GPU memory history belongs to another experiment')
                        # Drop replayed epoch records, but preserve the observed run high-water mark.
                        self._accumulate(row)
                        if row['epoch'] < start_epoch:
                            self.rows.append(row)
            if self.run_path.exists():
                saved = json.loads(self.run_path.read_text(encoding='utf-8'))
                if saved['metadata'] != self.metadata:
                    raise ValueError('GPU memory summary belongs to another experiment')
                self._accumulate(saved, 'run_peak_')
        self.epoch_path.write_text(''.join(json.dumps(row) + '\n' for row in self.rows), encoding='utf-8')
        self._write_summary()

    def _accumulate(self, row, prefix='peak_'):
        for field, attr in (('allocated_mib', 'run_allocated'), ('reserved_mib', 'run_reserved')):
            value = row.get(prefix + field)
            previous = getattr(self, attr)
            if value is not None:
                setattr(self, attr, value if previous is None else max(previous, value))

    def _snapshot(self):
        if not self.enabled:
            return dict(peak_allocated_mib=None, peak_reserved_mib=None)
        return dict(peak_allocated_mib=self.cuda.max_memory_allocated(self.device) / MIB,
                    peak_reserved_mib=self.cuda.max_memory_reserved(self.device) / MIB)

    def begin_epoch(self, epoch):
        if self.epoch is not None:
            raise RuntimeError('Previous GPU memory epoch was not closed')
        if self.enabled:
            self.cuda.synchronize(self.device)
            self.cuda.reset_peak_memory_stats(self.device)
        self.epoch, self.calibration = epoch, None

    def calibration_peak(self):
        """Called after synchronized calibration, before any training work; never resets."""
        if self.epoch is None:
            raise RuntimeError('Calibration must be inside a profiled epoch')
        self.calibration = self._snapshot()
        return dict(self.calibration)

    def end_epoch(self, epoch, k, evaluated_splits):
        if self.epoch != epoch:
            raise RuntimeError('GPU memory epoch boundary mismatch')
        if self.enabled:
            self.cuda.synchronize(self.device)
        row = dict(metadata=self.metadata, epoch=epoch, k=int(k), units='MiB',
                   cuda_enabled=self.enabled, device=str(self.device),
                   scope='calibration_train_evaluation_and_epoch_bookkeeping',
                   evaluated_splits=list(evaluated_splits),
                   calibration_performed=self.calibration is not None,
                   calibration_peak_allocated_mib=None,
                   calibration_peak_reserved_mib=None, **self._snapshot())
        if self.calibration is not None:
            row.update({'calibration_' + key: value for key, value in self.calibration.items()})
        self._accumulate(row)
        self.rows.append(row)
        with self.epoch_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        self.epoch = None
        self._write_summary()
        logging.info('GPU memory epoch: %s', json.dumps(row))
        return row

    def _write_summary(self):
        epochs = sorted({row['epoch'] for row in self.rows})
        summary = dict(metadata=self.metadata, units='MiB',
                       scope='maximum_across_observed_epoch_windows_including_replays',
                       run_peak_allocated_mib=self.run_allocated,
                       run_peak_reserved_mib=self.run_reserved,
                       recorded_epochs=epochs,
                       complete_epoch_history=epochs == list(range(epochs[-1] + 1)) if epochs else False,
                       finished=self.finished)
        temporary = self.run_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        temporary.replace(self.run_path)
        return summary

    def finish(self):
        if self.epoch is not None:
            raise RuntimeError('Cannot finish GPU memory profiling inside an epoch')
        self.finished = True
        summary = self._write_summary()
        logging.info('GPU memory run: %s', json.dumps(summary))
        return summary
