#!/usr/bin/env bash
# Install deps on a Colab (Linux + CUDA) runtime for Peptides + k-MIP.
# Safe to re-run. Prefer naive top-k if pykeops fails (Peptides is small enough).
set -euo pipefail

echo "== GPU / CUDA =="
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "== Core Python packages =="
pip -q install -U pip
pip -q install \
  yacs tqdm pyyaml ogb rdkit einops \
  tensorboardX torchmetrics wandb performer-pytorch \
  pytorch_lightning \
  scikit-learn pandas scipy h5py networkx matplotlib psutil

# Match Colab's installed torch major.minor for PyG wheels when possible.
TORCH_MM=$(python - <<'PY'
import torch
v = torch.__version__.split("+")[0]
print(".".join(v.split(".")[:2]))
PY
)
CUDA_TAG=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("cpu")
else:
    # Colab usually ships cu121 / cu124 wheels; fall back to cpu index textually.
    print("cu121")
PY
)

echo "== torch-geometric / scatter (torch=${TORCH_MM}, tag=${CUDA_TAG}) =="
pip -q install torch-geometric
pip -q install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_MM}.0+${CUDA_TAG}.html" \
  || pip -q install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_MM}.0+cpu.html" \
  || pip -q install torch-scatter

echo "== Optional pykeops (best-effort) =="
pip -q install pykeops || echo "pykeops install failed — will use naive top-k fallback"

python - <<'PY'
import torch, torch_geometric, torch_scatter
print("OK torch", torch.__version__)
print("OK pyg", torch_geometric.__version__)
print("OK scatter", torch_scatter.__version__)
try:
    import pykeops
    print("OK pykeops", pykeops.__version__)
except Exception as e:
    print("pykeops unavailable:", e)
PY

echo "setup_env.sh done."
