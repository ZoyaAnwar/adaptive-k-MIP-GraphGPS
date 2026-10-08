# Colab (free tier) — Adaptive k-MIP

Train **fixed / linear / exponential** `k` schedules on Peptides-func using a free Colab T4.

## On your Mac (once)

```bash
cd "/Users/zoyaanwar/Documents/kMLR570 Project"
bash colab/pack_for_colab.sh
# → ~/Desktop/kmip-adaptive-k-colab.zip
```

Also open `colab/Adaptive_k_MIP_Colab.ipynb` in Colab:
- GitHub upload, or
- Drive upload of the notebook

## In Colab

1. Runtime → **T4 GPU**
2. Run cells in order (Drive mount → upload zip → setup → smoke → ablation)
3. Progress:
   - `MODE=quick` (20 epochs) first
   - then `MODE=standard` (100) with same Drive `OUT_ROOT` (resumes)
   - optional `MODE=full` (250)

## Files

| Path | Role |
|------|------|
| `Adaptive_k_MIP_Colab.ipynb` | Main free-tier notebook |
| `setup_env.sh` | Pip install on Colab |
| `run_ablation.sh` | Runs fixed/linear/exponential |
| `pack_for_colab.sh` | Zip project for upload |
| `../configs/.../colab/*.yaml` | Free-tier configs (PE off, batch 64, resume on) |

## Notes

- Upstream `origin` is the paper repo — **upload your zip** (or push a personal fork). Do not `git clone` upstream if you need adaptive-k.
- Keep LapPE/RWSE **disabled** so preprocess fits free RAM.
- Checkpoints land under `MyDrive/kmip_adaptive_k/results/`.
