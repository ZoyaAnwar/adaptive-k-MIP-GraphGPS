"""Opt-in study timings and latest recovery, independent of best checkpoints."""
import json
import importlib.metadata
import platform
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class StudyRuntime:
    def __init__(self, directory, recovery_every=1):
        self.directory = Path(directory) / 'study'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.recovery = self.directory / 'latest.ckpt'
        self.period = recovery_every
        self.started = time.perf_counter()
        self.session_id = uuid.uuid4().hex
        self.session_path = self.directory / 'sessions' / (self.session_id + '.json')
        self.session = dict(id=self.session_id, status='running',
                            started_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.session_path, self.session)
        self.timings = []
        self.staged = None

    def validate_dataset(self, dataset):
        from graphgps.adaptive_k.run_identity import dataset_fingerprint
        fingerprint=dataset_fingerprint(dataset)
        path=self.directory/'dataset_identity.json'
        if path.exists() and json.loads(path.read_text())['sha256'] != fingerprint:
            raise ValueError('Study training dataset identity or ordering changed')
        atomic_json(path,dict(sha256=fingerprint))

    def restore(self, model, optimizer, scheduler):
        if not self.recovery.exists():
            return None
        saved = torch.load(self.recovery, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        return saved['epoch'] + 1

    def prepare(self, start_epoch):
        path = self.directory / 'epochs.jsonl'
        if path.exists():
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines) - 1:
                        break
                    raise
                if row['epoch'] < start_epoch:
                    self.timings.append(row)
        path.write_text(''.join(json.dumps(row) + '\n' for row in self.timings), encoding='utf-8')
        self.session['start_epoch'] = start_epoch
        atomic_json(self.session_path, self.session)

    def begin_epoch(self, epoch):
        self.epoch_started = time.perf_counter()
        self.row = dict(epoch=epoch, session_id=self.session_id, calibration_wall_seconds=0.0,
                        train_seconds=0.0, val_seconds=0.0, test_seconds=0.0)

    def phase(self, name, seconds):
        self.row[name + '_seconds'] = seconds
        if name == 'calibration_wall' and seconds > 0:
            # Preserve observed calls across checkpoint replay, independently of canonical epochs.
            with (self.directory / 'calibration_calls.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=self.row['epoch'], session_id=self.session_id,
                                             calibration_wall_seconds=seconds)) + '\n')

    def stage_recovery(self, epoch, model, optimizer, scheduler):
        if (epoch + 1) % self.period == 0:
            temporary = self.recovery.with_suffix('.ckpt.tmp')
            torch.save(dict(epoch=epoch, model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), temporary)
            self.staged = temporary
    def end_epoch(self, epoch, model, optimizer, scheduler, best_epoch):
        if (epoch + 1) % self.period == 0 and self.staged is None:
            self.stage_recovery(epoch, model, optimizer, scheduler)
        self.row.update(total_epoch_wall_seconds=time.perf_counter() - self.epoch_started,
                        best_epoch=int(best_epoch))
        self.timings.append(self.row)
        with (self.directory / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(self.row) + '\n')
        atomic_json(self.directory / 'heartbeat.json', dict(
            session_id=self.session_id, last_completed_epoch=epoch,
            session_elapsed_seconds=time.perf_counter() - self.started))
        if self.staged is not None:
            self.staged.replace(self.recovery)
            self.staged = None

    def finish(self, succeeded, error=None):
        self.session.update(status='completed' if succeeded else 'failed', error=error,
                            elapsed_training_seconds=time.perf_counter() - self.started,
                            finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.session_path, self.session)


def run_study_training(function, cfg, *args):
    observer = StudyRuntime(cfg.run_dir, cfg.study.recovery_every)
    (observer.directory / 'effective_config.yaml').write_text(cfg.dump(), encoding='utf-8')
    versions = {}
    for name in ('torch', 'torch-geometric', 'pykeops', 'numpy', 'torchmetrics', 'rdkit'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic_json(observer.directory / 'environment.json', dict(
        python=platform.python_version(), packages=versions, cuda_build=torch.version.cuda,
        parameter_count=cfg.params, hidden_dimension=cfg.gt.dim_hidden))
    try:
        result = function(*args, study_observer=observer)
        from study.checkpoints import audit
        rows = [json.loads(line) for line in (Path(cfg.run_dir)/'val/stats.json').read_text().splitlines() if line.strip()]
        best = max(sorted(rows, key=lambda row: row['epoch']), key=lambda row: row[cfg.metric_best])['epoch']
        audit(cfg.run_dir, best, cfg.optim.max_epoch - 1, args[2])
    except BaseException as error:
        observer.finish(False, repr(error))
        raise
    observer.finish(True)
    return result
