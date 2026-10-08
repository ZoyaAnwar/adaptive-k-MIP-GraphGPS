import copy
import logging

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder

from graphgps.layer.gps_layer import GPSLayer


def _layer_ks(sparse_cfg, num_layers):
    """Resolve per-layer k values from gt.sparse.* schedule config."""
    schedule = getattr(sparse_cfg, 'k_schedule', 'constant') or 'constant'
    k_default = int(sparse_cfg.k)
    k_start = sparse_cfg.k_start if sparse_cfg.k_start is not None else k_default
    k_end = sparse_cfg.k_end if sparse_cfg.k_end is not None else k_default
    k_start, k_end = int(k_start), int(k_end)
    k_min = max(1, int(getattr(sparse_cfg, 'k_min', 1)))
    decay = float(getattr(sparse_cfg, 'k_decay', 0.5))

    if num_layers <= 0:
        return []
    if schedule == 'constant' or num_layers == 1:
        return [k_default] * num_layers
    if schedule == 'linear':
        if num_layers == 1:
            return [k_start]
        return [
            max(k_min, int(round(k_start + (k_end - k_start) * i / (num_layers - 1))))
            for i in range(num_layers)
        ]
    if schedule == 'exponential':
        return [
            max(k_min, int(round(k_start * (decay ** i))))
            for i in range(num_layers)
        ]
    raise ValueError(
        f"Unknown gt.sparse.k_schedule={schedule!r}; "
        f"expected one of: constant, linear, exponential"
    )


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-set edge dim for PNA.
            cfg.gnn.dim_edge = 16 if 'PNA' in cfg.gt.layer_type else cfg.gnn.dim_inner
            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gnn.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layer_ks = _layer_ks(cfg.gt.sparse, cfg.gt.layers)
        logging.info(f"SparseAttention k schedule ({cfg.gt.sparse.k_schedule}): {layer_ks}")
        layers = []
        for layer_idx, k_layer in enumerate(layer_ks):
            sparse_cfg = copy.deepcopy(cfg.gt.sparse)
            sparse_cfg.k = int(k_layer)
            layers.append(GPSLayer(
                dim_h=cfg.gt.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                sparse_cfg=sparse_cfg,
                log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
