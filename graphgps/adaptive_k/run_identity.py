"""Output isolation and streaming identity checks for adaptive_mass runs."""
import hashlib
import json
from pathlib import Path


def guard_run_directory(directory):
    directory = Path(directory)
    marker = directory / 'adaptive_mass_run.json'
    if directory.exists() and any(directory.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != {'strategy': 'adaptive_mass', 'version': 1}:
            raise RuntimeError(f'Refusing to overwrite or resume a non-adaptive_mass run: {directory}')


def mark_run_directory(directory):
    marker = Path(directory) / 'adaptive_mass_run.json'
    marker.write_text(json.dumps({'strategy': 'adaptive_mass', 'version': 1}), encoding='utf-8')


def dataset_fingerprint(dataset):
    """Hash source indices, graph order and all graph attributes, one graph at a time."""
    import numpy as np
    import torch

    digest = hashlib.sha256()
    def update(value):
        digest.update(type(value).__name__.encode() + b':')
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object arrays cannot define a dataset identity')
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (tuple, list, range)):
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            digest.update(repr(value).encode())
        else:
            raise ValueError(f'Unsupported dataset identity attribute: {type(value).__name__}')
        digest.update(b';')
    update(len(dataset))
    update(list(dataset.indices()) if hasattr(dataset, 'indices') else list(range(len(dataset))))
    for index in range(len(dataset)):
        graph = dataset[index]
        update(index)
        update(graph.num_nodes)
        update(graph.to_dict() if hasattr(graph, 'to_dict') else vars(graph))
    return digest.hexdigest()
