import logging

import torch
import os
import os.path as osp
import numpy as np
from torch_geometric.utils import degree
from torch_geometric.utils import remove_self_loops
# from torch_geometric.utils import scatter
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import json_to_dict_list
from torch_geometric.graphgym.loader import get_loader, load_dataset
from torch_scatter import scatter

from yacs.config import CfgNode


def negate_edge_index(edge_index, batch=None):
    """Negate batched sparse adjacency matrices given by edge indices.

    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.

    Implementation inspired by `torch_geometric.utils.to_dense_adj`

    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.

    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        # Remove existing edges from the full N x N adjacency matrix
        flattened_size = n * n
        adj = adj.view([flattened_size])
        _idx1 = idx1[idx0 == i]
        _idx2 = idx2[idx0 == i]
        idx = _idx1 * n + _idx2
        zero = torch.zeros(_idx1.numel(), dtype=torch.short,
                           device=edge_index.device)
        scatter(zero, idx, dim=0, out=adj, reduce='mul')

        # Convert to edge index format
        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_negative = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_negative


def flatten_dict(metrics):
    """Flatten a list of train/val/test metrics into one dict to send to wandb.

    Args:
        metrics: List of Dicts with metrics

    Returns:
        A flat dictionary with names prefixed with "train/" , "val/" , "test/"
    """
    prefixes = ['train', 'val', 'test']
    result = {}
    for i in range(len(metrics)):
        # Take the latest metrics.
        stats = metrics[i][-1]
        result.update({f"{prefixes[i]}/{k}": v for k, v in stats.items()})
    return result


def cfg_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary.

    Yacs doesn't have a default function to convert the cfg object to plain
    python dict. The following function was taken from
    https://github.com/rbgirshick/yacs/issues/19
    """
    _VALID_TYPES = {tuple, list, str, int, float, bool}

    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            logging.warning(f"Key {'.'.join(key_list)} with "
                            f"value {type(cfg_node)} is not "
                            f"a valid type; valid types: {_VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_to_dict(v, key_list + [k])
        return cfg_dict


def make_wandb_name(cfg):
    # Format dataset name.
    dataset_name = cfg.dataset.format
    if dataset_name.startswith('OGB'):
        dataset_name = dataset_name[3:]
    if dataset_name.startswith('PyG-'):
        dataset_name = dataset_name[4:]
    if dataset_name in ['GNNBenchmarkDataset', 'TUDataset']:
        # Shorten some verbose dataset naming schemes.
        dataset_name = ""
    if cfg.dataset.name != 'none':
        dataset_name += "-" if dataset_name != "" else ""
        if cfg.dataset.name == 'LocalDegreeProfile':
            dataset_name += 'LDP'
        else:
            dataset_name += cfg.dataset.name
    # Format model name.
    model_name = cfg.model.type
    if cfg.model.type in ['gnn', 'custom_gnn']:
        model_name += f".{cfg.gnn.layer_type}"
    elif cfg.model.type == 'GPSModel':
        model_name = f"GPS.{cfg.gt.layer_type}.{cfg.gt.layers}l.{cfg.gt.n_heads}h"
        if cfg.gt.layer_type.split('+')[1] == 'SparseAttention':
            model_name += f".{cfg.gt.sparse.kq_dim}kq"
    elif cfg.model.type == 'MultiModel':
        model_name = f"Multi.{cfg.gt.layer_type}"
    model_name += f".{cfg.name_tag}" if cfg.name_tag else ""
    # Compose wandb run name.
    name = f"{dataset_name}.{model_name}.r{cfg.run_id}"
    return name

def batch_to_edge_idxs(batch):
    """
    Converts a batch to a list of batch.edge_index for each graph in the batch
    """
    edge_idxs = []
    for graph_id in range(batch.num_graphs):
        graph_mask = (batch.batch == graph_id)
        graph_edge_mask = graph_mask[batch.edge_index[0]] & graph_mask[batch.edge_index[1]]
        graph_edge_index = batch.edge_index[:, graph_edge_mask]
        graph_min_node = graph_mask.nonzero().min()
        graph_edge_index -= graph_min_node
        edge_idxs.append(graph_edge_index)

    return edge_idxs

def batch_to_labels_list(batch):
    """
    Converts a batch to a list of batch.y for each graph in the batch
    """
    labels_list = []
    for graph_id in range(batch.num_graphs):
        graph_mask = (batch.batch == graph_id)
        graph_labels = batch.y[graph_mask]
        labels_list.append(graph_labels)

    return labels_list

def report_epoch_times(dir):
    """
    Reads the epoch times in each split of the 'agg' subdir of the given `dir`
    And writes the averages and standard deviations into each dir/agg/*split*/time.txt
    """

    agg_dir = osp.join(dir, "agg")
    if not osp.exists(agg_dir):
        raise Exception(f"{agg_dir} does not exist")
    if not osp.isdir(agg_dir):
        raise Exception(f"{agg_dir} is not a directory")


    for split in os.listdir(agg_dir):
        # get the stats_list for this split
        dir_split = osp.join(agg_dir, split)
        fname_stats = osp.join(dir_split, "stats.json")
        stats_list = json_to_dict_list(fname_stats)

        # compute the overall mean and variance of the epoch times for this split
        epoch_time_means = np.array([x['time_epoch'] for x in stats_list])
        epoch_time_stds = np.array([x['time_epoch_std'] for x in stats_list])
        overall_epoch_time_mean = np.mean(epoch_time_means)
        overall_epoch_time_variance = np.mean(epoch_time_stds**2 + (epoch_time_means - overall_epoch_time_mean)**2)
        overall_epoch_time_std = np.sqrt(overall_epoch_time_variance)

        # save the overall mean and std of the epoch times for this split in a new file 'time.txt'
        with open(osp.join(dir_split, "time.txt"), "w") as f:
            f.write(f"{overall_epoch_time_mean} ± {overall_epoch_time_std}")

def report_epoch_times2(dir):
    """
    Reads the epoch times from the subdir of each split of each run in the given `dir`
    And writes the averages and standard deviations into each dir/agg/*split*/time.txt
    """

    agg_dir = osp.join(dir, "agg")
    splits = os.listdir(agg_dir)
    runs = [subdir for subdir in os.listdir(dir) if subdir != "agg"]

    for split in splits:

        epoch_time_means = []

        # get the mean for each split
        for run in runs:
            run_split_dir = osp.join(dir, run, split)
            fname_stats = osp.join(run_split_dir, "stats.json")
            stats_list = json_to_dict_list(fname_stats)

            epoch_time_mean = np.mean([x['time_epoch'] for x in stats_list])
            epoch_time_means.append(epoch_time_mean)

        overall_epoch_time_mean = np.mean(epoch_time_means)
        overall_epoch_time_std = np.std(epoch_time_means)

        # save the overall mean and std of the epoch times for this split in a new file 'time2.txt'
        with open(osp.join(agg_dir, split, "time2.txt"), "w") as f:
            f.write(f"{overall_epoch_time_mean} ± {overall_epoch_time_std}")


# had to override this because the original graphgym function did not infer the dimensions correctly for on disk datasets
def my_set_dataset_info(dataset):
    r"""
    Set global dataset information

    Args:
        dataset: PyG dataset object

    """

    is_on_disk_dataset = hasattr(dataset, 'is_on_disk_dataset') and dataset.is_on_disk_dataset
    # get sample to infer dim_in and dim_out from
    if is_on_disk_dataset:
        sample = dataset[0]
    else:
        sample = dataset.data

    # get dim_in and dim_out
    try:
        cfg.share.dim_in = sample.x.shape[1]
    except Exception:
        cfg.share.dim_in = 1
    try:
        if cfg.dataset.task_type == 'classification':
            if is_on_disk_dataset:
                cfg.share.dim_out = dataset.get_num_classes()
            else:
                cfg.share.dim_out = torch.unique(sample.y).shape[0]
        else:
            cfg.share.dim_out = sample.y.shape[1]
    except Exception:
        cfg.share.dim_out = 1

    # count number of dataset splits
    # PyG >=2.5: Data.keys is a method; older versions exposed it as a property.
    data_keys = dataset.data.keys() if callable(dataset.data.keys) else dataset.data.keys
    cfg.share.num_splits = 1
    for key in data_keys:
        if 'val' in key:
            cfg.share.num_splits += 1
            break
    for key in data_keys:
        if 'test' in key:
            cfg.share.num_splits += 1
            break

# had to override this because the original graphgym function did not infer the dimensions correctly for on disk datasets
def my_create_dataset():
    r"""
    Create dataset object

    Returns: PyG dataset object

    """
    dataset = load_dataset()
    my_set_dataset_info(dataset)

    return dataset


# had to override this because the original graphgym function did not infer the dimensions correctly for on disk datasets
def my_create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = my_create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=cfg.train.shuffle)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]

    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))

    return loaders



def dirichlet_energy(x, edge_index, batch=None):
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])

        x = x / torch.sqrt(deg + 1.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0

        if batch is not None:
            energy = scatter(energy, batch[dst], dim_size=x.shape[0], reduce='mean')
        else:
            energy = energy.mean()

        energy *= 0.5

    return float(energy.mean().detach().cpu())


def mean_average_distance(x, edge_index, batch):
    with torch.no_grad():
        src, dst = edge_index
        distance = 1 - torch.cosine_similarity(x[src], x[dst], dim=1)
        distance = scatter(distance, dst, dim_size=x.shape[0], reduce='sum')
        distance = scatter(distance, batch, dim_size=x.shape[0], reduce='mean')
    return float(distance.mean().detach().cpu())


def mean_norm(x):
    with torch.no_grad():
        norm = torch.norm(x, dim=1, p=2).mean()
    return float(norm.detach().cpu())

