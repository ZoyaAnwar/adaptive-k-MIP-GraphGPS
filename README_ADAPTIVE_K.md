# Adaptive *k* for k-MIP Graph Transformers (this project)

> Upstream paper README is unchanged: see [`README.md`](README.md).  
> This file documents **our** extension, ports, and how to run Peptides on Mac / free Colab.

This repo starts from the official code for **k-MIP attention + GraphGPS**, then adds an **epoch-wise adaptive-*k*** curriculum (paper Limitations suggestion): start with a large *k* and decrease it as training progresses. **Within an epoch, every GPS layer uses the same *k*.**

| | Upstream paper | This project |
|---|---|---|
| Goal | Scale full attention via **fixed** *k*-MIP; theory for GraphGPS | Study **training curriculum** on *k* (large → small over epochs) |
| Core model | GPS + SparseAttention (k-MIP) | **Same**, plus epoch schedule updating `SparseAttention.k` |
| Main data focus here | LRGB, CityNetworks, point clouds | **Peptides-func** (small LRGB graphs; PE off) |
| Hardware target | A100 + CUDA KeOps | Mac M2 (CPU/naive) + **free Colab T4** |

Upstream: [arXiv](https://arxiv.org/pdf/2604.03815) · [OpenReview](https://openreview.net/forum?id=4Y5kxbH2fI) · [GitHub](https://github.com/JonasDeSchouwer/k-MIP-Attention-and-the-Expressive-Power-of-GraphGPS)

---

## 1. What the paper does (kept as baseline)

The paper introduces **k-Maximum Inner Product (k-MIP)** self-attention with a **single fixed *k*** for all layers and all epochs. In Limitations they suggest:

> start with a large *k* and gradually decrease it as training progresses

to mitigate supervision starvation. **That curriculum is what we implement.**

---

## 2. What *we* changed

### 2.1 Research change — epoch-wise adaptive *k*

| Schedule | Formula | Config |
|---|---|---|
| `constant` | \(k(e)=k\) (paper default) | `gt.sparse.k_schedule: constant` |
| `linear` | interpolate `k_start` → `k_end` over epochs | `k_schedule: linear` |
| `exponential` | \(\max(k_{\min}, \mathrm{round}(k_{\mathrm{start}} \cdot \mathrm{decay}^{e}))\) | `k_schedule: exponential` |

Within epoch \(e\), **all layers** share \(k(e)\).

| File | Role |
|---|---|
| [`graphgps/adaptive_k/train_schedule.py`](graphgps/adaptive_k/train_schedule.py) | `k_for_epoch`, `apply_k_to_model`, `set_epoch_k` |
| [`graphgps/train/custom_train.py`](graphgps/train/custom_train.py) | Calls `set_epoch_k` at the start of each epoch |
| [`graphgps/network/gps_model.py`](graphgps/network/gps_model.py) | Builds layers with a shared init *k*; train loop updates it |
| [`graphgps/config/gt_config.py`](graphgps/config/gt_config.py) | Schedule hyperparameters |

Example (100 epochs, linear 45→5): epoch 0 → \(k=45\), epoch 99 → \(k=5\), every layer identical within that epoch.

### 2.2 Engineering (Mac + free Colab)

| File | Why |
|---|---|
| [`graphgps/layer/sparse_attention_layer.py`](graphgps/layer/sparse_attention_layer.py) | Optional KeOps; naive dense top-*k* fallback; clamps `k ≤ N` |
| [`graphgps/loader/torch_io.py`](graphgps/loader/torch_io.py) | `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6 |
| Other GraphGym / PyG API fixes | See git history |

### 2.3 Colab tooling

| Path | Role |
|---|---|
| `configs/LRGB/Peptides-func/colab/GPS+kmip-*.yaml` | fixed / linear / exponential epoch curricula |
| `colab/Adaptive_k_MIP_Colab.ipynb` | Free-tier notebook (**clones this GitHub repo**) |
| `colab/setup_env.sh` / `run_ablation.sh` | Env + ablation runner |

---

## 3. How to run

### Local (Mac M2) — smoke / debug

```bash
micromamba activate kmipattn
cd "/Users/zoyaanwar/Documents/kMLR570 Project"

python main.py \
  --cfg configs/LRGB/Peptides-func/defaults/GPS+kmip-linear-k.yaml \
  wandb.use False accelerator cpu optim.max_epoch 2
```

### Free Colab T4 — real ablation

**Clone this repo** (has epoch-*k* + `colab/`). Do **not** clone the upstream paper repo.

1. Open [`colab/Adaptive_k_MIP_Colab.ipynb`](colab/Adaptive_k_MIP_Colab.ipynb) in Colab.
2. Runtime → **T4 GPU**. Mount Drive.
3. The notebook clones:

```text
https://github.com/ZoyaAnwar/adaptive-k-MIP-GraphGPS.git
```

into `/content/kmip`, then runs setup → smoke → ablation.

4. Epoch ladder: `MODE=quick` (20) → `MODE=standard` (100, resume) → optional `MODE=full` (250).

```bash
# on Colab, from /content/kmip
MODE=standard RUNS=fixed,linear,exponential \
  OUT_ROOT=/content/drive/MyDrive/kmip_adaptive_k/results \
  bash colab/run_ablation.sh
```

Results: `kmip_adaptive_k/results/{fixed,linear,exponential}/`.

To refresh after a push: re-run the clone cell, or `!git -C /content/kmip pull`.

---

## 4. Config knobs

```yaml
gt:
  sparse:
    k: 15                 # used when k_schedule: constant
    k_schedule: linear    # constant | linear | exponential  (over EPOCHS)
    k_start: 45           # linear / exponential
    k_end: 5              # linear only
    k_decay: 0.7          # exponential only
    k_min: 5              # floor
```

Log each epoch:

```text
Epoch 0/100: SparseAttention k=45 on 8 layer(s) (schedule=linear)
```

---

## 5. Mental model

```
main.py
graphgps/
  adaptive_k/train_schedule.py   # ★ epoch → k
  train/custom_train.py          # ★ set_epoch_k each epoch
  network/gps_model.py           # GPS layers (shared k)
  layer/sparse_attention_layer.py
  config/gt_config.py
configs/LRGB/Peptides-func/colab/
colab/                           # notebook clones GitHub
```

★ = this project's contribution touch points.

---

## 6. Upstream citation

Cite the bibtex in [`README.md`](README.md). Original license: MIT (see [`LICENSE`](LICENSE)).
