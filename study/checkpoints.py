"""CPU-only audit of locally produced checkpoints; never load untrusted pickle files."""
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(directory, best_epoch, final_epoch, model=None):
    import torch
    from study.experiment import atomic_json
    directory = Path(directory)
    rows = []
    for relative, epoch, trusted_rng in [(f'ckpt/{best_epoch}.ckpt', best_epoch, False),
                                         ('study/latest.ckpt', final_epoch, True)]:
        path = directory / relative
        saved = torch.load(path, map_location='cpu', weights_only=not trusted_rng)
        if saved.get('epoch') != epoch or not all(key in saved for key in
                ('model_state', 'optimizer_state', 'scheduler_state')):
            raise ValueError('Checkpoint schema/epoch mismatch: ' + relative)
        state = saved['model_state']
        if not isinstance(state, dict) or not state:
            raise ValueError('Empty model state: ' + relative)
        if any(not torch.isfinite(value).all().item() for value in state.values() if torch.is_tensor(value)):
            raise ValueError('Nonfinite checkpoint model state: ' + relative)
        if model is not None:
            expected = model.state_dict()
            if set(state) != set(expected) or any(
                    torch.is_tensor(value) and value.shape != expected[key].shape for key, value in state.items()):
                raise ValueError('Checkpoint architecture mismatch: ' + relative)
        budgets=[]
        for key,value in state.items():
            if key.endswith('_extra_state') and isinstance(value,dict) and 'k' in value:
                if value.get('version')!=1 or type(value['k']) is not int or not 1<=value['k']<=50:
                    raise ValueError('Invalid adaptive checkpoint k: '+relative)
                budgets.append(value['k'])
        if len(set(budgets))>1:
            raise ValueError('Adaptive checkpoint budgets differ across heads/layers')
        rows.append(dict(path=relative, epoch=epoch, sha256=digest(path), bytes=path.stat().st_size,
                         adaptive_k=budgets[0] if budgets else None))
    result = dict(schema_version=1, cpu_load_verified=True, checkpoints=rows)
    atomic_json(directory / 'study/checkpoint_audit.json', result)
    return result


def verify(directory, best_epoch, final_epoch):
    """Report-side validation needs no PyTorch; audit is produced by training/audit CLI."""
    directory = Path(directory)
    path = directory / 'study/checkpoint_audit.json'
    value = json.loads(path.read_text())
    expected = {f'ckpt/{best_epoch}.ckpt': best_epoch, 'study/latest.ckpt': final_epoch}
    rows = value.get('checkpoints', [])
    if not value.get('cpu_load_verified') or len(rows) != 2 or {r['path']: r['epoch'] for r in rows} != expected:
        raise ValueError('Checkpoint audit does not match validation-selected/final epochs')
    for row in rows:
        target = directory / row['path']
        if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
            raise ValueError('Checkpoint changed/corrupt since CPU audit: ' + row['path'])
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--best-epoch', type=int, required=True)
    parser.add_argument('--final-epoch', type=int, default=199)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.best_epoch, args.final_epoch), indent=2))
