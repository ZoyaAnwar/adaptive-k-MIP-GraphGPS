"""Resolve and apply a global SparseAttention k for the current epoch."""

from __future__ import annotations

import logging

import torch.nn as nn

from graphgps.layer.sparse_attention_layer import SparseAttention


def k_for_epoch(sparse_cfg, epoch: int, max_epoch: int) -> int:
    """Compute the single k used by all layers at ``epoch``.

    Schedules (``gt.sparse.k_schedule``):
      - constant: always ``gt.sparse.k``
      - linear: interpolate ``k_start`` -> ``k_end`` over epochs
      - exponential: geometric interpolation from ``k_start`` to ``k_end``
        (defaults 50 to 10), subject to the adaptive ``k_min`` floor
    """
    schedule = getattr(sparse_cfg, 'k_schedule', 'constant') or 'constant'
    k_default = int(sparse_cfg.k)
    if schedule == 'constant':
        return k_default
    if schedule == 'adaptive_mass':
        if getattr(sparse_cfg, 'random_attention', False):
            raise ValueError('adaptive_mass rejects random_attention=True')
        return 50  # Initial/reference budget; MassCalibration manages later epochs.
    default_start = 50 if schedule == 'exponential' else k_default
    default_end = 10 if schedule == 'exponential' else k_default
    k_start = sparse_cfg.k_start if sparse_cfg.k_start is not None else default_start
    k_end = sparse_cfg.k_end if sparse_cfg.k_end is not None else default_end
    k_start, k_end = int(k_start), int(k_end)
    k_min = max(1, int(getattr(sparse_cfg, 'k_min', 1)))

    if max_epoch <= 0:
        return max(k_min, k_default)
    epoch = int(min(max(epoch, 0), max(max_epoch - 1, 0)))

    if schedule == 'linear':
        if max_epoch == 1:
            return max(k_min, k_start)
        t = epoch / (max_epoch - 1)
        return max(k_min, int(round(k_start + (k_end - k_start) * t)))
    if schedule == 'exponential':
        if k_start <= 0 or k_end <= 0:
            raise ValueError('Exponential k schedule requires positive endpoints')
        t = epoch / max(max_epoch - 1, 1)
        return max(k_min, int(round(k_start * (k_end / k_start) ** t)))
    raise ValueError(
        f"Unknown gt.sparse.k_schedule={schedule!r}; "
        f"expected one of: constant, linear, exponential, adaptive_mass"
    )


def apply_k_to_model(model: nn.Module, k: int) -> int:
    """Set ``SparseAttention.k`` on every module (same k for all layers)."""
    k = int(k)
    n = 0
    for module in model.modules():
        if isinstance(module, SparseAttention):
            module.k = k
            n += 1
    return n


def set_epoch_k(model: nn.Module, sparse_cfg, epoch: int, max_epoch: int) -> int:
    """Resolve epoch k and apply it to all SparseAttention modules."""
    k = k_for_epoch(sparse_cfg, epoch, max_epoch)
    n = apply_k_to_model(model, k)
    if n:
        progress = min(max(epoch, 0), max(max_epoch - 1, 0)) / max(max_epoch - 1, 1)
        logging.info(
            'epoch=%s/%s progress=%.3f requested_k=%s on %s layer(s) '
            '(schedule=%s)',
            epoch, max(max_epoch - 1, 0), progress, k, n,
            getattr(sparse_cfg, 'k_schedule', 'constant'),
        )
    return k
