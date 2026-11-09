"""Restore attention execution state after loading an existing checkpoint."""
from numbers import Integral

from graphgps.adaptive_k.train_schedule import apply_k_to_model, k_for_epoch
from graphgps.layer.sparse_attention_layer import SparseAttention


def restore_checkpoint_budget(model, checkpoint, sparse_cfg, training_epochs):
    """Use the saved training config, never an evaluation-only epoch override.

    Baseline checkpoints retain their original format. Adaptive budgets must
    already have been loaded from their serialized per-module extra state.
    Does not alter weights, RNG, model modes or training-resume epoch handling.
    """
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, SparseAttention)]
    if not layers:
        return None  # Other GPS attention backends have no k to restore.
    schedule = getattr(sparse_cfg, 'k_schedule', 'constant') or 'constant'
    if schedule == 'adaptive_mass':
        state = checkpoint.get('model_state', {})
        budgets = []
        for name, module in layers:
            key = name + '._extra_state' if name else '_extra_state'
            saved = state.get(key)
            if (not isinstance(saved, dict) or saved.get('version') != 1 or
                    type(saved.get('k')) is not int or not 1 <= saved['k'] <= 50):
                raise ValueError('Missing/invalid adaptive checkpoint budget: ' + key)
            if module.k != saved['k']:
                raise ValueError('Adaptive budget was not restored by model.load_state_dict: ' + key)
            budgets.append(saved['k'])
        if len(set(budgets)) != 1:
            raise ValueError('Adaptive checkpoint must have one global attention budget')
        return budgets[0]  # Never call the initial/reference adaptive schedule here.
    if schedule == 'constant':
        k = k_for_epoch(sparse_cfg, 0, 1)
    else:
        epoch = checkpoint.get('epoch')
        if (isinstance(training_epochs, bool) or not isinstance(training_epochs, Integral)
                or training_epochs < 1 or isinstance(epoch, bool)
                or not isinstance(epoch, Integral) or not 0 <= epoch < training_epochs):
            raise ValueError('Decay restoration requires the actual checkpoint epoch and original training duration')
        k = k_for_epoch(sparse_cfg, int(epoch), int(training_epochs))
    apply_k_to_model(model, k)
    return k
