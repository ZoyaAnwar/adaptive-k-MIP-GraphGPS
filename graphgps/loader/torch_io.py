"""torch.load helpers compatible with PyTorch >= 2.6 (weights_only default)."""
from __future__ import annotations

import torch


def torch_load(*args, **kwargs):
    """Load a trusted local checkpoint / processed dataset.

    PyTorch 2.6+ defaults weights_only=True, which breaks PyG Data pickles.
    Dataset and ckpt files in this repo are local/trusted, so we default to False.
    """
    kwargs.setdefault("weights_only", False)
    try:
        return torch.load(*args, **kwargs)
    except TypeError:
        # Older torch without weights_only kwarg
        kwargs.pop("weights_only", None)
        return torch.load(*args, **kwargs)
