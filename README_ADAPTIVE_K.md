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
| `exponential` | \(\mathrm{round}(k_{\mathrm{start}}(k_{\mathrm{end}}/k_{\mathrm{start}})^{e/(E-1)})\), subject to `k_min` | `k_schedule: exponential` |

Within epoch \(e\), **all layers** share \(k(e)\).

| File | Role |
|---|---|
| [`graphgps/adaptive_k/train_schedule.py`](graphgps/adaptive_k/train_schedule.py) | `k_for_epoch`, `apply_k_to_model`, `set_epoch_k` |
| [`graphgps/train/custom_train.py`](graphgps/train/custom_train.py) | Calls `set_epoch_k` at the start of each epoch |
| [`graphgps/network/gps_model.py`](graphgps/network/gps_model.py) | Builds layers with a shared init *k*; train loop updates it |
| [`graphgps/config/gt_config.py`](graphgps/config/gt_config.py) | Schedule hyperparameters |

Research defaults to linear 50 to 10, with k_min=10 and the same requested k in all layers. Constant mode ignores adaptive settings.

Exponential mode defaults to **50 to 10** across the full `optim.max_epoch` duration, reaching 10 at the final epoch. Explicit `k_start` and `k_end` take precedence; both must be positive. The existing `k_min` floor still applies. With one epoch, the schedule uses `k_start`. The legacy `k_decay` key is accepted but ignored. Linear mode retains its existing defaults (`gt.sparse.k` for omitted endpoints).

Select exponential decay through YAML (`gt.sparse.k_schedule: exponential`) or CLI overrides:

```bash
python main.py --cfg configs/LRGB/Peptides-func/defaults/GPS+kmip-linear-50-to-10.yaml gt.sparse.k_schedule exponential
```

Run the lightweight checks with `python -m unittest discover -s tests -p "test_k_schedule.py"`.

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

Fixed remains available using the unchanged selected paper config. The existing Peptides-func exponential config remains an exploratory run; its explicit start of 45 takes precedence and now decays geometrically to the default endpoint of 10. Neither is launched by default. Other repository dataset support remains untouched and outside this research workflow.

To refresh after a push: re-run the clone cell, or `!git -C /content/kmip pull`.

---

## 4. Config knobs

```yaml
gt:
  sparse:
    k: 15                 # used when k_schedule: constant
    k_schedule: linear    # constant | linear | exponential  (over EPOCHS)
    k_start: 50           # explicit start for linear or exponential
    k_end: 10             # explicit endpoint for linear or exponential
    k_decay: 0.7          # legacy key, accepted but ignored
    k_min: 10             # adaptive floor; ignored by constant mode
```

Log each epoch:

```text
epoch=0/249 progress=0.000 requested_k=50 on 8 layer(s) (schedule=linear)
```

---

## Strategy B attention mass calibration

Select the additional `adaptive_mass` schedule with the existing CLI mechanism:

```bash
python main.py --cfg configs/LRGB/Peptides-func/defaults/GPS+kmip-linear-50-to-10.yaml gt.sparse.k_schedule adaptive_mass

# Colab runner: isolated adaptive run using the same dataset/model settings.
DATASET=PascalVOC-SP RUNS=adaptive_mass SEEDS=0,1,2,3 bash colab/run_ablation.sh
```

Adaptive CLI runs automatically use `<out_dir>/adaptive_mass/<config-name[-tag]>/`
with per-seed directories below that. The runner additionally separates its
`<DATASET>/<MODE>/adaptive_mass` root. Directory identity markers are checked
before writing config files, clearing outputs, or resuming: a non-adaptive
experiment is rejected rather than reused. Baseline paths/defaults are unchanged.

Defaults are a reference and initial k of 50, 8,500 independent observations,
90% retained top-50 mass, 95% population coverage and 95% overall confidence.
`gt.sparse.mass_calibration.interval` defaults to 5 and is configurable.
Other settings under `gt.sparse.mass_calibration` are `samples` (8500), `tau`
(0.90), `q` (0.95), `delta` (0.05), and `query_chunk_size` (64). Baseline
`k`, `k_start`, `k_end` and **`k_min` do not constrain this strategy**; its
selected k may independently increase or decrease anywhere from 1 to 50.

The first five epochs use k=50. Calibration runs before zero-based epochs
5, 10, ... strictly below `optim.max_epoch`. Thus the planned count is
`C = floor((E - 1) / T)`; there is no unused final calibration. Each event
uses fresh seeded draws with replacement: uniform training graph, uniform
node within that graph, uniform SparseAttention layer, then uniform head.
For the standard architecture this samples eight layers and four heads,
without enforcing fixed per-layer/head quotas. Small graphs use only their
available keys, up to 50. The empirical quantile level is
`q + sqrt(log(2*C/delta)/(2*m))`, using order statistic `ceil(m*level)`.
For 250 epochs and T=5, C=49 and the adjustment is approximately 0.02112;
m stays 8500. Invalid settings whose adjusted level exceeds 1 are rejected.

CUDA and PyKeOps `LazyTensor` are mandatory for this option. Calibration
runs one graph at a time in frozen eval/no-gradient mode, always using k=50
in every attention module. Only sampled query/head rows are extracted in
chunks using symbolic top-k; no dense query-key score matrix is created.
All model modes, previous k values, RNG states and hooks are restored even
on failure. The existing attention code and batching/padding behavior are
unchanged; single-graph reference passes contain no padding. Consequently,
the mass guarantee describes the graph-local evaluation-mode reference,
not padded/stochastic training attention or coverage between checkpoints.
`random_attention=True`, runtime dataset transforms, and non-full-batch or
non-graph training datasets are rejected. One graph must still fit GPU
memory; query chunking does not eliminate the cost of its reference forward.

Each event logs selected k, DKW parameters, retained-mass statistics,
empirical coverage, a required-budget histogram, layer/head sample counts
and coverage, calibration runtime, and peak allocated GPU memory. Epoch
runtime includes calibration and archive I/O. Scalar diagnostics also go
to W&B when enabled. Outputs under `<run_dir>/mass_calibration/` are:

- `epoch_XXXXXX.npz`: sorted top-50 probabilities (8500 x 50, zero-padded
  for short graphs), required budgets, graph and source graph IDs, query,
  layer, head, eligible key counts, epoch, selected k, and run signature.
  Graph IDs index the training subset; query/layer/head IDs are zero-based.
- `epoch_XXXXXX.json`: full diagnostics including total calibration time.

Consecutive events compare the mean sorted probability profiles using
Jensen-Shannon divergence and the required-budget distributions using CDF
and Wasserstein distances. These compare **independent samples**, not paired
observations or temporal guarantees. Keep the archives with checkpoints:
training resume restores the last calibration strictly before the resumed
epoch, validates its settings, and replays later boundaries deterministically.
Missing required archives cause an explicit error. An opt-in `MassSparseAttention`
subclass includes its selected k in the normal model state dictionary; standard
checkpoint loading automatically restores k for both training and standalone
evaluation. Baseline attention retains its existing state-dictionary format and
computations. New adaptive checkpoints require this budget metadata; older
budget-less adaptive state dictionaries cannot silently default to k=50.

Training initialization computes a streaming SHA-256 fingerprint of all graph
attributes plus source indices and their order. A run manifest is saved even
before the first calibration. Resume checks this identity and run settings,
then verifies that checkpoint budgets match the corresponding archive, before
metric history is rewritten. Keep the run identity and calibration files with
training checkpoints. The fingerprint adds one CPU pass over the training set
at startup; graph count alone is insufficient. Earlier local prototype runs
without the manifest must be restarted rather than automatically resumed.

Run `python -m unittest discover -s tests -p "test_*.py"` for baseline and
Strategy B checks. Controller tests use real CPU tensors and a tiny symbolic
backend stand-in. CUDA/KeOps and an A100 memory/runtime pilot must be checked
in the training environment before making hardware performance claims.

## Standardized GPU allocator memory profiling

The common `custom` training loop profiles `constant` (fixed k), `linear`,
`exponential`, and `adaptive_mass` with identical epoch boundaries. Immediately
before each epoch's scheduling/calibration, it synchronizes the model's CUDA
device and resets peak counters once. After training, validation/test where
scheduled, metric computation, and epoch bookkeeping/checkpointing, it
synchronizes again and reads both `max_memory_allocated()` and
`max_memory_reserved()`. It never empties the CUDA cache. Both peaks are
absolute allocator usage, not increments relative to the start of the epoch.

Structured files under each seed's `<run_dir>/memory/` are:

- `epochs.jsonl`: one record per completed epoch with `peak_allocated_mib`,
  `peak_reserved_mib`, requested `k`, evaluation splits actually scheduled,
  device/CUDA status, and schedule/dataset/seed/split metadata. Adaptive
  calibration epochs additionally have `calibration_performed: true` and
  `calibration_peak_allocated_mib` / `calibration_peak_reserved_mib`.
- `run.json`: separate `run_peak_allocated_mib` and `run_peak_reserved_mib`,
  maintained as maxima across observed epoch windows. Per-epoch resets cannot
  erase these run peaks. `finished` indicates a normally completed invocation;
  `recorded_epochs` and `complete_epoch_history` expose missing earlier data.

Values are in **MiB (2^20 bytes)**. CPU runs write null memory values with
`cuda_enabled: false` and make no CUDA synchronization/reset/peak calls.
Allocated memory measures live tensor allocations; reserved memory measures
PyTorch's allocator pool, including cached blocks. Neither is total GPU VRAM
usage reported by NVML/nvidia-smi, and allocations outside the PyTorch allocator
are not captured. Existing `gpu_memory` snapshot fields and timing/metric logs
are unchanged.

Calibration is first in its epoch, so its synchronized peak snapshot occurs
before training, without resetting the common epoch counters. Its JSON
additionally exposes `peak_allocated_mib` and `peak_reserved_mib`, while
retaining the existing `peak_gpu_memory_bytes` field for compatibility.
Standalone calibration, outside the training-loop profiler, still owns its
own reset window.

On resume, canonical epoch records at/after the checkpoint's next epoch are
replayed without duplicate epoch rows. The observed run high-water mark is
retained across sessions, including previously measured replayed work.
Resuming a run that predates this instrumentation cannot recover missing
peaks; the incomplete epoch history is flagged. Forced termination before an
epoch record is written can also lose that window's peak. The measured scope
excludes model/dataset initialization and standalone inference/profiling tools.

Compare matched hardware, batch size, dataset/model, precision, evaluation
schedule and backend. Reserved peaks reflect cache retention, intentionally
without method-specific cache clearing. Hardware/CUDA validation remains a
separate GPU smoke test; boundary/reset/logging tests run with mocked counters.

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


## 7. Independent RunPod study

The dedicated [Study200 notebook](colab/RunPod_Study200.ipynb) has 14 independently
selectable experiment cells, persistent global adaptive review gates, incremental
publication exports and separate historical-predictive imports. Both datasets use
200 epochs in study-only configs. The six VOC scheduling repetitions and six fixed50
runs remain full runs. A separate opt-in A100 CUDA/KeOps smoke workflow must pass
before full study launches. Optional subset profiling never replaces full-run results.
See [study/README.md](study/README.md) for recovery, checkpoint audits, output cohorts,
commands and measurement limitations. Original LRGB configs are unchanged.
