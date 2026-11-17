"""Evaluate a trained k-MIP GPS checkpoint at one or more FIXED k values.

Example:
    python eval_fixed_k.py \
        --run-dir "results/local/PascalVOC-SP/full/linear/GPS+kmip-linear-50-to-10-linear" \
        --ks 2 5 10 15 20 30 50 --seeds 0 1 2 3 --splits val test

--run-dir must contain config.yaml and one folder per seed (0/, 1/, ...),
each with ckpt/<epoch>.ckpt. The script loads the weights once per seed,
sets SparseAttention.k on every layer, and runs val/test for each k.
It also reports the "trained" k (what the schedule used at the checkpoint
epoch) so you have a baseline to compare against.
Results are written to <out-dir>/fixed_k_results.csv and summary.csv.
"""
import argparse
import csv
import logging
import os
import os.path as osp
import sys
from collections import defaultdict

import numpy as np
import torch

import graphgps  # noqa: F401  (registers custom modules)
from graphgps.adaptive_k import apply_k_to_model, k_for_epoch
from graphgps.finetuning import get_final_pretrained_ckpt, set_new_cfg_allowed
from graphgps.loader.torch_io import torch_load
from graphgps.logger import create_logger
from graphgps.train.custom_train import eval_epoch
from graphgps.utils import my_create_loader
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.comp_budget import params_count

SPLIT_IDX = {'train': 0, 'val': 1, 'test': 2}
METRICS = ['f1', 'accuracy', 'mIoU', 'auc', 'loss']


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[2, 15])
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='default: every numeric subfolder of --run-dir')
    p.add_argument('--splits', nargs='+', default=['val', 'test'],
                   choices=list(SPLIT_IDX))
    p.add_argument('--dataset-dir', default='./datasets')
    p.add_argument('--device', default='auto', help='auto | cpu | cuda:0')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--no-trained-k', action='store_true',
                   help='skip the baseline run at the schedule k of the ckpt epoch')
    p.add_argument('--dry-run', action='store_true',
                   help='build model + load ckpts only, no dataset (sanity check)')
    return p.parse_args()


def load_run_cfg(run_dir, args):
    set_cfg(cfg)
    set_new_cfg_allowed(cfg, True)
    cfg.merge_from_file(osp.join(run_dir, 'config.yaml'))
    # Use the exact architecture from training; don't re-search hidden dim.
    cfg.model.auto_satisfy_param_budget = False
    cfg.train.mode = 'inference-only'
    cfg.train.auto_resume = False
    cfg.wandb.use = False
    cfg.pretrained.dir = ''
    cfg.dataset.dir = args.dataset_dir
    cfg.num_workers = 0
    if args.device == 'auto':
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        cfg.device = args.device
    cfg.accelerator = 'cuda' if cfg.device.startswith('cuda') else 'cpu'
    torch.set_num_threads(max(1, int(cfg.num_threads)))


def read_ckpt(run_dir, seed):
    ckpt_file = get_final_pretrained_ckpt(osp.join(run_dir, str(seed), 'ckpt'))
    ckpt = torch_load(ckpt_file, map_location='cpu')
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    epoch = int(osp.basename(ckpt_file).split('.')[0])
    return ckpt_file, state, epoch


def set_hidden_dim_from_state(state):
    # config.yaml stores the pre-search dim; training auto-scaled it to fit the
    # param budget, so read the real dim straight from the weights.
    dim = int(state['model.layers.0.norm2.weight'].shape[0])
    cfg.gnn.dim_inner = dim
    cfg.gt.dim_hidden = dim
    if cfg.gt.layer_type.endswith('+SparseAttention'):
        cfg.gt.sparse.val_dim = dim // cfg.gt.n_heads
    return dim


def main():
    args = parse()
    run_dir = osp.abspath(args.run_dir)
    out_dir = osp.abspath(args.out_dir or osp.join(run_dir, 'fixed_k_eval'))
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(osp.join(out_dir, 'eval.log'))])

    load_run_cfg(run_dir, args)
    seeds = args.seeds or sorted(int(d) for d in os.listdir(run_dir)
                                 if d.isdigit() and osp.isdir(osp.join(run_dir, d, 'ckpt')))
    logging.info(f'run_dir={run_dir}\nseeds={seeds} ks={args.ks} '
                 f'splits={args.splits} device={cfg.device}')

    if args.dry_run:
        cfg.share.dim_in, cfg.share.dim_out, cfg.share.num_splits = 14, 21, 3
        loaders = None
    else:
        loaders = my_create_loader()  # sets cfg.share.dim_in / dim_out

    rows = []
    for seed in seeds:
        seed_everything(seed)
        ckpt_file, state, epoch = read_ckpt(run_dir, seed)
        dim = set_hidden_dim_from_state(state)
        model = create_model()
        model.load_state_dict(state, strict=True)
        model.eval()
        cfg.params = params_count(model)
        trained_k = k_for_epoch(cfg.gt.sparse, epoch, cfg.optim.max_epoch)
        logging.info(f'\n[seed {seed}] loaded {ckpt_file} (epoch {epoch}, '
                     f'hidden {dim}, params {cfg.params}); schedule k at that epoch = {trained_k}')

        ks = [(k, 'fixed') for k in args.ks]
        if not args.no_trained_k and trained_k not in args.ks:
            ks.append((trained_k, 'trained_k'))

        for k, tag in ks:
            n = apply_k_to_model(model, k)
            if args.dry_run:
                logging.info(f'  k={k:>3} applied to {n} SparseAttention layers (dry run)')
                continue
            cfg.run_dir = osp.join(out_dir, f'seed{seed}', f'k{k}')
            os.makedirs(cfg.run_dir, exist_ok=True)
            loggers = create_logger()
            for split in args.splits:
                i = SPLIT_IDX[split]
                eval_epoch(loggers[i], loaders[i], model, split=split)
                stats = loggers[i].write_epoch(epoch)
                row = {'seed': seed, 'ckpt_epoch': epoch, 'k': k, 'kind': tag,
                       'split': split, **{m: stats.get(m) for m in METRICS}}
                rows.append(row)
                logging.info(f'  k={k:>3} ({tag:9}) {split:5} f1={row["f1"]:.4f} '
                             f'acc={row["accuracy"]:.4f} loss={row["loss"]:.4f}')
            for lg in loggers:
                lg.close()

    if not rows:
        logging.info('Dry run OK.')
        return

    fields = ['seed', 'ckpt_epoch', 'k', 'kind', 'split'] + METRICS
    with open(osp.join(out_dir, 'fixed_k_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # mean +- std across seeds (trained_k rows grouped together as k='trained')
    groups = defaultdict(list)
    for r in rows:
        key = ('trained' if r['kind'] == 'trained_k' else r['k'], r['split'])
        groups[key].append(r)
    summary = []
    logging.info('\n==== Summary (mean ± std over seeds) ====')
    for (k, split), rs in sorted(groups.items(), key=lambda x: (str(x[0][1]), str(x[0][0]).zfill(4))):
        s = {'k': k, 'split': split, 'n_seeds': len(rs)}
        for m in METRICS:
            v = np.array([r[m] for r in rs], dtype=float)
            s[f'{m}_mean'], s[f'{m}_std'] = round(v.mean(), 5), round(v.std(), 5)
        summary.append(s)
        logging.info(f'  {split:5} k={str(k):>7}  f1={s["f1_mean"]:.4f} ± {s["f1_std"]:.4f}  '
                     f'acc={s["accuracy_mean"]:.4f}  (n={len(rs)})')
    with open(osp.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    logging.info(f'\nSaved: {out_dir}/fixed_k_results.csv and summary.csv')


if __name__ == '__main__':
    main()
