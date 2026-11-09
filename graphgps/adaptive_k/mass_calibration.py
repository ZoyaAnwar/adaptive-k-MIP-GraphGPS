"""Memory-bounded, graph-uniform Strategy B calibration using symbolic KeOps."""
import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from graphgps.adaptive_k.mass_statistics import (
    adjusted_quantile, calibration_epochs, compare_calibrations, select_budget,
)
from graphgps.adaptive_k.train_schedule import apply_k_to_model
from graphgps.adaptive_k.run_identity import dataset_fingerprint
from graphgps.layer.sparse_attention_layer import SparseAttention, LazyTensor


@contextmanager
def frozen_reference(model, modules, seed):
    """Restore mode, k, gradients/checkpoint-compatible state and all RNGs on errors."""
    modes = [(module, module.training) for module in model.modules()]
    budgets = [module.k for module in modules]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count()))
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            model.eval()
            for module in modules:
                module.k = 50
            with torch.no_grad():
                yield
    finally:
        for module, mode in modes:
            module.training = mode
        for module, k in zip(modules, budgets):
            module.k = k
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.no_grad()
def top_probabilities(module, x, queries, heads, chunk_size):
    """Extract only sampled rows; the query-key matrix remains symbolic."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError('Calibration expects one graph per reference pass')
    q = module.split_heads(module.MQs(x), module.kq_dim)[0]
    keys = module.split_heads(module.MKs(x), module.kq_dim)[0]
    n_keys = keys.shape[-2]
    width = min(50, n_keys)
    result = np.zeros((len(queries), 50), dtype=np.float32)
    for head in np.unique(heads):
        rows = np.flatnonzero(heads == head)
        for start in range(0, len(rows), chunk_size):
            selected = rows[start:start + chunk_size]
            query_ids = torch.as_tensor(queries[selected], device=x.device)
            query = q[int(head), query_ids].contiguous()
            key = keys[int(head)].contiguous()
            scores = (LazyTensor(query[:, None, :]) *
                      LazyTensor(key[None, :, :])).sum(-1) / module.kq_dim ** 0.5
            indices = (-scores).argKmin(width, dim=1).long()
            logits = (query[:, None, :] * key[indices]).sum(-1).float() / module.kq_dim ** 0.5
            logits = logits.sort(dim=-1, descending=True).values
            result[selected, :width] = logits.softmax(dim=-1).cpu().numpy()
    return result


class MassCalibration:
    def __init__(self, model, sparse_cfg, dataset, run_dir, seed, max_epoch,
                 start_epoch=0):
        self.model, self.dataset = model, dataset
        self.options = sparse_cfg.mass_calibration
        self.modules = [m for m in model.modules() if isinstance(m, SparseAttention)]
        if sparse_cfg.random_attention or any(m.random_attention for m in self.modules):
            raise ValueError('adaptive_mass rejects random_attention=True')
        if not self.modules or any(not m.use_keops for m in self.modules):
            raise RuntimeError('adaptive_mass requires CUDA PyKeOps LazyTensor on every attention layer')
        if LazyTensor is None or not torch.cuda.is_available():
            raise RuntimeError('adaptive_mass requires CUDA PyKeOps LazyTensor')
        if not len(dataset) or self.options.query_chunk_size < 1:
            raise ValueError('Calibration needs a nonempty training set and positive chunk size')
        # Random runtime transforms would make tuple observations non-deterministic.
        if getattr(dataset, 'transform', None) is not None:
            raise ValueError('adaptive_mass requires a deterministic training dataset without runtime transforms')
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda':
            raise RuntimeError('adaptive_mass requires the model on CUDA')
        self.seed = int(seed)
        self.epochs = calibration_epochs(max_epoch, self.options.interval)
        self.signature = dict(seed=self.seed, max_epoch=max_epoch,
                              interval=self.options.interval, samples=self.options.samples,
                              k_max=50, tau=self.options.tau, q=self.options.q,
                              delta=self.options.delta, train_graphs=len(dataset),
                              heads=[m.num_heads for m in self.modules],
                              dataset_sha256=dataset_fingerprint(dataset))
        if self.epochs:
            adjusted_quantile(self.options.samples, len(self.epochs),
                              self.options.q, self.options.delta)
        self.directory = Path(run_dir) / 'mass_calibration'
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = self.directory / 'run_identity.json'
        if start_epoch > 0:
            if not identity.is_file() or json.loads(identity.read_text(encoding='utf-8')) != self.signature:
                raise ValueError('Cannot resume adaptive_mass: dataset identity/order or run settings differ')
        else:
            identity.write_text(json.dumps(self.signature, indent=2), encoding='utf-8')
        self.k, self.previous = 50, None
        previous_epochs = [e for e in self.epochs if e < start_epoch]
        if previous_epochs:
            path = self.directory / f'epoch_{previous_epochs[-1]:06d}.npz'
            if not path.exists():
                raise RuntimeError(f'Cannot resume adaptive_mass: missing {path}')
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved['signature'])) != self.signature:
                    raise ValueError('Calibration resume configuration differs from saved run')
                self.k = int(saved['k'])
                self.previous = {name: saved[name].copy() for name in
                                 ('probabilities', 'required', 'epoch', 'k')}
        if start_epoch > 0 and any(m.k != self.k for m in self.modules):
            raise ValueError('Adaptive checkpoint budget differs from calibration archive')
        apply_k_to_model(self.model, self.k)

    def step(self, epoch, memory_profiler=None):
        if epoch not in self.epochs:
            apply_k_to_model(self.model, self.k)
            logging.info('adaptive_mass epoch=%s requested_k=%s', epoch, self.k)
            return None
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        if memory_profiler is None:
            # Standalone calibration owns its window; training owns the full epoch.
            torch.cuda.reset_peak_memory_stats(self.device)
        m = self.options.samples
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 9187]))
        graphs = rng.integers(len(self.dataset), size=m)
        layers = rng.integers(len(self.modules), size=m)
        heads = np.array([rng.integers(self.modules[layer].num_heads) for layer in layers])
        queries = np.empty(m, dtype=np.int64)
        eligible = np.empty(m, dtype=np.int64)
        probabilities = np.zeros((m, 50), dtype=np.float32)
        with frozen_reference(self.model, self.modules, self.seed + epoch):
            for graph in np.unique(graphs):
                # One graph avoids padding and keeps the reference graph-local;
                # the existing deployed batching/padding behavior is unmodified.
                graph_seed = int(np.random.SeedSequence([self.seed, epoch, int(graph)]).generate_state(1)[0])
                random.seed(graph_seed)
                np.random.seed(graph_seed)
                torch.manual_seed(graph_seed)
                data = self.dataset[int(graph)].clone()
                n = data.num_nodes
                if not n:
                    raise ValueError('adaptive_mass cannot sample an empty graph')
                rows = np.flatnonzero(graphs == graph)
                queries[rows] = rng.integers(n, size=len(rows))
                eligible[rows] = min(50, n)
                handles = []
                try:
                    for layer, module in enumerate(self.modules):
                        selected = rows[layers[rows] == layer]
                        if not len(selected):
                            continue
                        def capture(module, args, selected=selected):
                            probabilities[selected] = top_probabilities(
                                module, args[0], queries[selected], heads[selected],
                                self.options.query_chunk_size)
                        handles.append(module.register_forward_pre_hook(capture))
                    batch = Batch.from_data_list([data]).to(self.device)
                    batch.split = 'train'
                    self.model(batch)
                finally:
                    for handle in handles:
                        handle.remove()
                del batch, data
        k, required, mass, summary = select_budget(
            probabilities, len(self.epochs), self.options.tau,
            self.options.q, self.options.delta)
        summary.update(epoch=epoch, previous_k=self.k)
        summary['layer_head_statistics'] = [
            dict(layer=layer, head=head, count=int(mask.sum()),
                 coverage=float((required[mask] <= k).mean()) if mask.any() else None,
                 mass_mean=float(mass[mask].mean()) if mask.any() else None)
            for layer, module in enumerate(self.modules)
            for head in range(module.num_heads)
            for mask in [(layers == layer) & (heads == head)]]
        if self.previous is not None:
            summary.update(compare_calibrations(
                probabilities, required, self.previous['probabilities'], self.previous['required']))
            summary.update(previous_calibration_epoch=int(self.previous['epoch']),
                           k_delta=k - int(self.previous['k']))
        torch.cuda.synchronize(self.device)
        summary['reference_seconds'] = time.perf_counter() - started
        if memory_profiler is not None:
            peaks = memory_profiler.calibration_peak()
        else:
            peaks = dict(peak_allocated_mib=torch.cuda.max_memory_allocated(self.device) / 1024**2,
                         peak_reserved_mib=torch.cuda.max_memory_reserved(self.device) / 1024**2)
        summary.update(peaks)
        # Preserve the existing allocated-byte diagnostic for downstream readers.
        summary['peak_gpu_memory_bytes'] = int(peaks['peak_allocated_mib'] * 1024**2)
        # Graph IDs are training-subset positions; source indices are also recorded.
        source = np.asarray(self.dataset.indices()) if hasattr(self.dataset, 'indices') else np.arange(len(self.dataset))
        target = self.directory / f'epoch_{epoch:06d}.npz'
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, probabilities=probabilities, required=required,
                                graph=graphs, source_graph=source[graphs], query=queries,
                                layer=layers, head=heads, eligible_keys=eligible,
                                epoch=epoch, k=k, signature=json.dumps(self.signature),
                                summary=json.dumps(summary))
        temporary.replace(target)
        summary['calibration_seconds'] = time.perf_counter() - started
        stats = target.with_suffix('.json')
        temporary = stats.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        temporary.replace(stats)
        self.k = k
        self.previous = dict(probabilities=probabilities, required=required, epoch=epoch, k=k)
        apply_k_to_model(self.model, k)
        logging.info('adaptive_mass calibration %s', json.dumps(summary))
        return summary
