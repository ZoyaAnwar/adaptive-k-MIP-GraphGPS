# Colab (free tier) — Epoch-wise adaptive k-MIP

Research defaults to **linear 50 to 10** for **PascalVOC-SP or Peptides-func**, using **paper-derived per-dataset training settings**. Every SparseAttention layer shares the requested k within an epoch.

## Preferred: clone this GitHub repo

Repo: https://github.com/ZoyaAnwar/adaptive-k-MIP-GraphGPS

1. Open `colab/Adaptive_k_MIP_Colab.ipynb` in Colab (File → Upload, or Open from GitHub).
2. Runtime → **T4 GPU**.
3. Run cells in order. The notebook **`git clone`s** this repo into `/content/kmip` (not the upstream paper repo).
4. Mount Drive for datasets + checkpoints under `MyDrive/kmip_adaptive_k/`.
5. Select `DATASET=PascalVOC-SP` or `DATASET=Peptides-func`; run `MODE=full RUNS=linear`. Full mode preserves the selected config's max_epoch and batch size.
6. Resume with the same dataset, mode, config, seed and OUT_ROOT; never change max_epoch midway.
7. `MODE=quick` (20 epochs) is an independent short/debug run; `MODE=standard` (100 epochs) is an independent exploratory run. Neither resumes into full research training.

```bash
DATASET=Peptides-func MODE=full RUNS=linear \
  OUT_ROOT=/content/drive/MyDrive/kmip_adaptive_k/results \
  bash colab/run_ablation.sh
```

Full settings: PascalVOC-SP uses batch 50 / 200 epochs; Peptides-func uses batch 200 / 250 epochs. PascalVOC-SP maps to `configs/LRGB/PascalVOC/`. Quick/standard override only the epoch horizon.

After you push new commits: re-run the clone cell or `!git -C /content/kmip pull`.

## Files

| Path | Role |
|------|------|
| `Adaptive_k_MIP_Colab.ipynb` | Main free-tier notebook (GitHub clone) |
| `setup_env.sh` | Pip install on Colab |
| `run_ablation.sh` | Two-dataset selection; defaults to full linear research |
| `../configs/LRGB/{PascalVOC,Peptides-func}/defaults/GPS+kmip-linear-50-to-10.yaml` | Paper-derived research configs |
| `../configs/LRGB/Peptides-func/colab/*.yaml` | Existing exploratory configs |

## Notes

- Upstream paper GitHub lacks epoch-*k* — always clone **this** fork.
- Preserve each dataset's paper positional-encoding settings; both selected configs disable LapPE/RWSE.
- Outputs: `MyDrive/kmip_adaptive_k/results/<DATASET>/<MODE>/<RUN>/`.
- `RUNS=fixed` remains available for both datasets. `RUNS=exponential` retains the existing Peptides-func exploratory config unchanged. Neither is a default run.
- Other repository dataset support is untouched and excluded from this workflow.
- No automatic resume-horizon validator is added.
