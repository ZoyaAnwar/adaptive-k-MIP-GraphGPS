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

Research defaults to linear 50 to 10, with k_min=10 and the same requested k in all layers. Constant mode ignores adaptive settings.

Current research datasets: PascalVOC-SP and Peptides-func. Full runs preserve paper-derived per-dataset training settings: PascalVOC-SP uses batch 50 / 200 epochs / hidden dimension 68; Peptides-func uses batch 200 / 250 epochs / hidden dimension 76. Both use 8 GPS layers and 4 heads; all other paper settings remain unchanged.

### 2.2 Engineering (Mac + free Colab)

| File | Why |
|---|---|
| [`graphgps/layer/sparse_attention_layer.py`](graphgps/layer/sparse_attention_layer.py) | Optional KeOps; naive dense top-*k* fallback; clamps `k ≤ N` |
| [`graphgps/loader/torch_io.py`](graphgps/loader/torch_io.py) | `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6 |
| Other GraphGym / PyG API fixes | See git history |

### 2.3 Colab tooling

| Path | Role |
|---|---|
| `configs/LRGB/{PascalVOC,Peptides-func}/defaults/GPS+kmip-linear-50-to-10.yaml` | Research: paper configs plus four linear schedule settings |
| `configs/LRGB/Peptides-func/colab/GPS+kmip-exponential.yaml` | Existing exploratory exponential config |
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

### Full research - paper-derived per-dataset training settings

**Clone this repo** (has epoch-*k* + `colab/`). Do **not** clone the upstream paper repo.

1. Open [`colab/Adaptive_k_MIP_Colab.ipynb`](colab/Adaptive_k_MIP_Colab.ipynb) in Colab.
2. Runtime → **T4 GPU**. Mount Drive.
3. The notebook clones:

```text
https://github.com/ZoyaAnwar/adaptive-k-MIP-GraphGPS.git
```

into `/content/kmip`, then runs setup → smoke → ablation.

4. Choose `DATASET=PascalVOC-SP` or `DATASET=Peptides-func`. PascalVOC-SP maps to the existing `configs/LRGB/PascalVOC/` directory.
5. Start research with `MODE=full RUNS=linear`. Full mode does not override max_epoch or batch size. Resume with the same dataset, mode, config, seed and OUT_ROOT; keep max_epoch unchanged.
6. `MODE=quick` (20 epochs) and `MODE=standard` (100 epochs) are independent short/debug and exploratory runs, respectively. They retain the selected config's batch size/model settings and never become earlier stages of a full run. The existing small `GPS+kmip-linear-k.yaml` remains a debug config.

```bash
# on Colab, from /content/kmip
DATASET=Peptides-func MODE=full RUNS=linear \
  OUT_ROOT=/content/drive/MyDrive/kmip_adaptive_k/results \
  bash colab/run_ablation.sh
```

Results: `kmip_adaptive_k/results/<DATASET>/<MODE>/<RUN>/`.

Fixed remains available using the unchanged selected paper config. The existing Peptides-func exponential config remains available unchanged as an exploratory run. Neither is launched by default. Other repository dataset support remains untouched and outside this research workflow.

To refresh after a push: re-run the clone cell, or `!git -C /content/kmip pull`.

---

## 4. Config knobs

```yaml
gt:
  sparse:
    k: 15                 # used when k_schedule: constant
    k_schedule: linear    # constant | linear | exponential  (over EPOCHS)
    k_start: 50           # linear research start
    k_end: 10             # linear only
    k_decay: 0.7          # exponential only
    k_min: 10             # adaptive floor; ignored by constant mode
```

Log each epoch:

```text
epoch=0/249 progress=0.000 requested_k=50 on 8 layer(s) (schedule=linear)
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
configs/LRGB/{PascalVOC,Peptides-func}/defaults/
colab/                           # notebook clones GitHub
```

★ = this project's contribution touch points.

---

## 6. Upstream citation

Cite the bibtex in [`README.md`](README.md). Original license: MIT (see [`LICENSE`](LICENSE)).
