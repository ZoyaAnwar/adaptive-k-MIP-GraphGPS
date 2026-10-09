# Colab (free tier) — Epoch-wise adaptive k-MIP

Train **fixed / linear / exponential** *k* curricula on Peptides-func (same *k* for all layers within an epoch; *k* changes over epochs).

## Preferred: clone this GitHub repo

Repo: https://github.com/ZoyaAnwar/adaptive-k-MIP-GraphGPS

1. Open `colab/Adaptive_k_MIP_Colab.ipynb` in Colab (File → Upload, or Open from GitHub).
2. Runtime → **T4 GPU**.
3. Run cells in order. The notebook **`git clone`s** this repo into `/content/kmip` (not the upstream paper repo).
4. Mount Drive for datasets + checkpoints under `MyDrive/kmip_adaptive_k/`.
5. `MODE=quick` (20 ep) → `MODE=standard` (100) with resume → optional `MODE=full` (250).

```bash
MODE=standard RUNS=fixed,linear,exponential \
  OUT_ROOT=/content/drive/MyDrive/kmip_adaptive_k/results \
  bash colab/run_ablation.sh
```

After you push new commits: re-run the clone cell or `!git -C /content/kmip pull`.

## Files

| Path | Role |
|------|------|
| `Adaptive_k_MIP_Colab.ipynb` | Main free-tier notebook (GitHub clone) |
| `setup_env.sh` | Pip install on Colab |
| `run_ablation.sh` | Runs fixed/linear/exponential |
| `../configs/.../colab/*.yaml` | Free-tier configs (PE off, batch 64, resume on) |

## Notes

- Upstream paper GitHub lacks epoch-*k* — always clone **this** fork.
- LapPE/RWSE stay **disabled** so preprocess fits free RAM.
- Checkpoints: `MyDrive/kmip_adaptive_k/results/`.
