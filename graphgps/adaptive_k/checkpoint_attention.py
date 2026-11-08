"""Opt-in checkpoint state for adaptive budgets; baseline attention is unchanged."""
from graphgps.layer.sparse_attention_layer import SparseAttention


class MassSparseAttention(SparseAttention):
    def get_extra_state(self):
        return {'version': 1, 'k': int(self.k)}

    def set_extra_state(self, state):
        if (not isinstance(state, dict) or state.get('version') != 1 or
                type(state.get('k')) is not int or not 1 <= state['k'] <= 50):
            raise ValueError('Invalid adaptive_mass checkpoint budget')
        self.k = state['k']
