#!/usr/bin/env bash
# Run fixed / linear / exponential adaptive-k on Peptides-func.
#
# Usage (from repo root on Colab):
#   bash colab/run_ablation.sh                 # all three, 100 epochs (config default)
#   MODE=quick bash colab/run_ablation.sh      # 20 epochs smoke
#   MODE=full  bash colab/run_ablation.sh      # 250 epochs (may need several sessions)
#   RUNS=fixed,linear bash colab/run_ablation.sh
#   OUT_ROOT=/content/drive/MyDrive/kmip_runs bash colab/run_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${MODE:-standard}"   # quick | standard | full
RUNS="${RUNS:-fixed,linear,exponential}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/colab}"
SEED="${SEED:-0}"

case "$MODE" in
  quick)    EPOCHS=20 ;;
  standard) EPOCHS=100 ;;
  full)     EPOCHS=250 ;;
  *) echo "Unknown MODE=$MODE (use quick|standard|full)"; exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "ROOT=$ROOT"
echo "OUT_ROOT=$OUT_ROOT  MODE=$MODE  EPOCHS=$EPOCHS  RUNS=$RUNS"

declare -A CFG=(
  [fixed]=configs/LRGB/Peptides-func/colab/GPS+kmip-fixed.yaml
  [linear]=configs/LRGB/Peptides-func/colab/GPS+kmip-linear.yaml
  [exponential]=configs/LRGB/Peptides-func/colab/GPS+kmip-exponential.yaml
)

IFS=',' read -r -a RUN_LIST <<< "$RUNS"
for name in "${RUN_LIST[@]}"; do
  name="$(echo "$name" | xargs)"
  cfg="${CFG[$name]:-}"
  if [[ -z "$cfg" ]]; then
    echo "Unknown run: $name"
    exit 1
  fi
  run_out="$OUT_ROOT/$name"
  mkdir -p "$run_out"
  echo ""
  echo "======== RUN $name ========"
  echo "cfg=$cfg  out_dir=$run_out"
  python main.py \
    --cfg "$cfg" \
    --repeat 1 \
    wandb.use False \
    seed "$SEED" \
    optim.max_epoch "$EPOCHS" \
    out_dir "$run_out" \
    train.auto_resume True \
    train.enable_ckpt True \
    train.ckpt_best True \
    name_tag "$name" \
    2>&1 | tee "$run_out/train_${name}.log"
  echo "Finished $name"
done

echo ""
echo "All requested runs finished. Logs under $OUT_ROOT"
