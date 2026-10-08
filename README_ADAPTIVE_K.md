# Adaptive *k* for k-MIP Graph Transformers (this project)

> Upstream paper README is unchanged: see [`README.md`](README.md).  
> This file documents **our** extension, ports, and how to run Peptides on Mac / free Colab.

This repo starts from the official code for **k-MIP attention + GraphGPS**, then adds a **minimal adaptive-*k*** change: each GPS layer can use a different top-*k* width (constant / linear / exponential schedules).

| | Upstream paper | This project |
|---|---|---|
| Goal | Scale full attention via **fixed** *k*-MIP; theory for GraphGPS | Study whether **layer-wise adaptive *k*** helps |
| Core model | GPS + SparseAttention (k-MIP) | **Same**, plus per-layer *k* schedule |
| Main data focus here | LRGB, CityNetworks, point clouds | **Peptides-func** (small LRGB graphs; PE off) |
| Hardware target | A100 + CUDA KeOps | Mac M2 (CPU/naive) + **free Colab T4** |

Upstream: [arXiv](https://arxiv.org/pdf/2604.03815) · [OpenReview](https://openreview.net/forum?id=4Y5kxbH2fI) · [GitHub](https://github.com/JonasDeSchouwer/k-MIP-Attention-and-the-Expressive-Power-of-GraphGPS)

---

## 1. What the paper does (kept as baseline)

The paper introduces **k-Maximum Inner Product (k-MIP)** self-attention:

1. Compute query–key scores (symbolically via KeOps when available).
2. Keep only the **top-*k* keys per query**; mask the rest to \(-\infty\).
3. Softmax + weighted sum of values (sparse).

They embed this in **GraphGPS** (MPNN + global attention + MLP), prove:

- GraphGPS expressivity is upper-bounded by **S-SEG-WL** (encodings matter).
- k-MIP can **approximate full attention** arbitrarily well (so sparsification need not kill expressivity).

Empirically they use a **single fixed *k*** for all layers (configs often use `gt.sparse.k: 5` or `15`; sweeps also try `5` and `45`).

**Everything in that pipeline that we did not touch is still here:** GraphGPS stack, SparseAttention math, configs for LRGB / CityNetworks / point clouds, `run/experiments.py`, `profiling-experiment/`, etc.

---

## 2. What *we* changed (the actual contribution + plumbing)

### 2.1 Minimal research change — adaptive *k*

**Idea:** instead of one `k` for all layers, set \(k_\ell\) per layer \(\ell\).

| Schedule | Formula | Config |
|---|---|---|
| `constant` | \(k_\ell = k\) (paper default) | `gt.sparse.k_schedule: constant` |
| `linear` | interpolate `k_start` → `k_end` | `k_schedule: linear` |
| `exponential` | \(\max(k_{\min}, \mathrm{round}(k_{\mathrm{start}} \cdot \mathrm{decay}^{\ell}))\) | `k_schedule: exponential` |

**Where to find it**

| File | What changed |
|---|---|
| [`graphgps/config/gt_config.py`](graphgps/config/gt_config.py) | New keys: `k_schedule`, `k_start`, `k_end`, `k_decay`, `k_min` |
| [`graphgps/network/gps_model.py`](graphgps/network/gps_model.py) | `_layer_ks(...)` + per-layer `deepcopy(sparse_cfg)` with different `k` |
| [`graphgps/layer/gps_layer.py`](graphgps/layer/gps_layer.py) | **Unchanged** — still reads `sparse_cfg.k` |
| [`graphgps/layer/sparse_attention_layer.py`](graphgps/layer/sparse_attention_layer.py) | Still does top-*k*; only gets a different `k` per layer |

**Example (8 layers, linear 45→5):** roughly `[45, 39, 34, 28, 22, 16, 11, 5]`.  
**Example (4 layers, linear 15→3):** `[15, 11, 7, 3]` (verified in a local smoke run).

This is intentionally a **small surgical change**: no new attention mechanism, no theory rewrite, no KeOps rewrite.

### 2.2 Engineering so it runs on Mac + free Colab

These are **not** paper contributions; they are portability fixes.

| File | Why |
|---|---|
| [`graphgps/layer/sparse_attention_layer.py`](graphgps/layer/sparse_attention_layer.py) | Optional KeOps; **naive dense top-*k*** if no CUDA/KeOps (Mac / Colab without KeOps). Clamps `k ≤ N`. |
| [`graphgps/transform/dist_transforms.py`](graphgps/transform/dist_transforms.py) | `cupy` optional (CUDA-only helpers). |
| [`graphgps/config/defaults_config.py`](graphgps/config/defaults_config.py) | Restore `cfg.train.mode`, `cfg.device` removed in newer GraphGym. |
| [`main.py`](main.py) | Drop obsolete `set_agg_dir` import; sync `cfg.device` with `accelerator`. |
| [`graphgps/utils.py`](graphgps/utils.py) | PyG `Data.keys()` API fix. |
| [`graphgps/loader/master_loader.py`](graphgps/loader/master_loader.py) | Logging f-string fix. |
| [`graphgps/loader/torch_io.py`](graphgps/loader/torch_io.py) | **New** — `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6 (Colab). |
| [`graphgps/loader/dataset/peptides_functional.py`](graphgps/loader/dataset/peptides_functional.py) | Uses `torch_load` (Colab crash fix). |
| [`graphgps/loader/dataset/peptides_structural.py`](graphgps/loader/dataset/peptides_structural.py) | Same. |
| [`graphgps/finetuning.py`](graphgps/finetuning.py) | Checkpoint load via `torch_load`. |

### 2.3 New configs / Colab tooling (added files)

| Path | Role |
|---|---|
| `configs/LRGB/Peptides-func/colab/GPS+kmip-fixed.yaml` | Fixed *k*=15 baseline (PE off, batch 64, resume) |
| `configs/LRGB/Peptides-func/colab/GPS+kmip-linear.yaml` | Linear 45→5 |
| `configs/LRGB/Peptides-func/colab/GPS+kmip-exponential.yaml` | Exp decay from 45 |
| `configs/LRGB/Peptides-func/defaults/GPS+kmip-linear-k.yaml` | Local Mac smoke config |
| `colab/Adaptive_k_MIP_Colab.ipynb` | Free-tier Colab notebook |
| `colab/setup_env.sh` | Colab pip setup |
| `colab/run_ablation.sh` | Run fixed / linear / exponential |
| `colab/pack_for_colab.sh` | Zip project for upload |
| `colab/README.md` | Short Colab how-to |
| `requirements-mac.txt` | Apple Silicon lean env |

---

## 3. What is *not* changed (still the paper’s code)

- Attention formula (top-*k* MIP + softmax over *k* keys).
- GraphGPS layer structure (GatedGCN local + global attention).
- Theory (SEG-WL / approximation proofs) — not re-implemented; cite the paper.
- Official experiment CLI (`run/experiments.py`) and most YAML configs.
- Profiling / efficiency study under `profiling-experiment/`.
- Large-graph experiments (CityNetworks, ShapeNet, S3DIS) — **code present, not our current scope**.

We also keep **positional encodings disabled** on Peptides (`posenc_LapPE.enable: False`, `posenc_RWSE.enable: False`). That matches several of their kmip Peptides defaults and is what makes free Colab / 8 GB Mac preprocess possible. Turning PE on is a separate (RAM-heavy) experiment.

---

## 4. Scope vs the paper (honest status)

| Paper pillar | Status in this project |
|---|---|
| Fixed k-MIP method | **Kept** as `k_schedule: constant` |
| Adaptive *k* | **Added** (linear + exponential) |
| Theory (SEG-WL, approx.) | **Cite upstream**; not extended yet |
| Peptides / LRGB numbers | **In progress** (Colab ablation) |
| CityNetworks / 500k nodes / KeOps A100 | **Out of scope** for now (needs big GPU + KeOps) |
| Full paper hyperparameter budgets | Colab configs use paper-like 8 layers / ~500k params; epochs: quick 20 / standard 100 / full 250 |

**For a methods-style writeup on a small dataset, “done” looks like:** fixed vs linear vs exponential on Peptides-func (same seeds, full-ish epochs), plus AP + wall time / memory notes. Not a full reproduction of the ICLR workshop paper.

---

## 5. How to run

### Local (Mac M2) — smoke / debug

```bash
# micromamba env `kmipattn` (see requirements-mac.txt)
micromamba activate kmipattn
cd "/Users/zoyaanwar/Documents/kMLR570 Project"

python main.py \
  --cfg configs/LRGB/Peptides-func/defaults/GPS+kmip-linear-k.yaml \
  wandb.use False accelerator cpu optim.max_epoch 1
```

Uses **naive** top-*k* (no KeOps). Fine for Peptides (~150 nodes/graph).

### Free Colab T4 — real ablation

1. Pack: `bash colab/pack_for_colab.sh` → upload zip (do **not** clone upstream GitHub; it lacks adaptive-*k*).
2. Open `colab/Adaptive_k_MIP_Colab.ipynb`, enable **T4 GPU**, mount Drive.
3. `MODE=quick` (20 ep) → then `MODE=standard` (100) with resume → optional `MODE=full` (250).

```bash
# on Colab, from /content/kmip
MODE=standard RUNS=fixed,linear,exponential \
  OUT_ROOT=/content/drive/MyDrive/kmip_adaptive_k/results \
  bash colab/run_ablation.sh
```

Results land under Drive: `kmip_adaptive_k/results/{fixed,linear,exponential}/`.

---

## 6. Config knobs (adaptive *k*)

```yaml
gt:
  sparse:
    k: 15                 # used when k_schedule: constant
    k_schedule: linear    # constant | linear | exponential
    k_start: 45           # linear / exponential
    k_end: 5              # linear only
    k_decay: 0.7          # exponential only
    k_min: 5              # floor
```

Override on the CLI without editing YAML:

```bash
python main.py --cfg configs/.../GPS+kmip-fixed.yaml \
  gt.sparse.k_schedule exponential \
  gt.sparse.k_start 32 gt.sparse.k_decay 0.5 gt.sparse.k_min 2
```

At model build time the log should show something like:

```text
SparseAttention k schedule (linear): [45, 39, 34, 28, 22, 16, 11, 5]
```

---

## 7. Mental model of the codebase

```
main.py                    # entry (GraphGym)
graphgps/
  network/gps_model.py     # ★ builds layers + assigns k_ℓ
  layer/gps_layer.py       # MPNN + SparseAttention
  layer/sparse_attention_layer.py  # ★ top-k (KeOps or naive)
  config/gt_config.py      # ★ schedule hyperparameters
  loader/...               # Peptides etc. (+ torch_io Colab fix)
configs/LRGB/Peptides-func/
  defaults/                # original + local linear smoke
  colab/                   # ★ free-tier ablation YAMLs
colab/                     # ★ notebook + scripts
run/experiments.py         # upstream paper CLI (unchanged)
profiling-experiment/      # upstream efficiency study
```

★ = touch points for this project.

---

## 8. Upstream citation

If you use k-MIP / GraphGPS results or theory, cite the bibtex in [`README.md`](README.md).

Original license: MIT (see [`LICENSE`](LICENSE)).
