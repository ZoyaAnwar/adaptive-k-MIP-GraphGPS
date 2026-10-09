"""Epoch-wise adaptive-k curriculum for k-MIP SparseAttention.

Matches the k-MIP paper Limitations suggestion: start with a large k and
gradually decrease it as training progresses. Within an epoch, every GPS
layer uses the same k.
"""

from graphgps.adaptive_k.train_schedule import (
    apply_k_to_model,
    k_for_epoch,
    set_epoch_k,
)

__all__ = ['apply_k_to_model', 'k_for_epoch', 'set_epoch_k']
