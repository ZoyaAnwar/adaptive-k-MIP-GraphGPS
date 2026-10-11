#!/usr/bin/env bash
# Run linear research with paper-derived per-dataset training settings.
# (Same k for all layers within an epoch; k anneals over epochs.)
#
# Usage (from repo root on Colab):
#   bash colab/run_ablation.sh                 # Peptides-func linear, paper settings
#   DATASET=PascalVOC-SP bash colab/run_ablation.sh
#   MODE=quick bash colab/run_ablation.sh      # independent 20-epoch short/debug run
#   MODE=standard bash colab/run_ablation.sh   # independent 100-epoch exploratory
#   SEEDS=1 bash colab/run_ablation.sh         # one seed only
#   RUNS=fixed,linear bash colab/run_ablation.sh
#   OUT_ROOT=/content/drive/MyDrive/kmip_runs bash colab/run_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-Peptides-func}"
MODE="${MODE:-full}"   # quick | standard | full
RUNS="${RUNS:-linear}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/colab}"
SEEDS="${SEEDS:-0,1,2}"

# Explicit seeds; each gets one independent Python process.
if [[ ! "$SEEDS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
  echo "Invalid SEEDS=$SEEDS (use comma-separated nonnegative integers)"
  exit 1
fi
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
declare -A SEEN_SEEDS=()
for seed in "${SEED_LIST[@]}"; do
  if [[ -n "${SEEN_SEEDS[$seed]:-}" ]]; then
    echo "Duplicate seed=$seed in SEEDS=$SEEDS"
    exit 1
  fi
  SEEN_SEEDS[$seed]=1
done

REQUIRE_KEOPS=0
MODE_ARGS=()
case "$MODE" in
  quick)    MODE_ARGS=(optim.max_epoch 20) ;;
  standard) MODE_ARGS=(optim.max_epoch 100) ;;
  full)     REQUIRE_KEOPS=1 ;;  # Keep config settings; require KeOps/CUDA.
  *) echo "Unknown MODE=$MODE (use quick|standard|full)"; exit 1 ;;
esac

mkdir -p "$OUT_ROOT"
echo "ROOT=$ROOT"
echo "OUT_ROOT=$OUT_ROOT  DATASET=$DATASET  MODE=$MODE  RUNS=$RUNS  SEEDS=$SEEDS"

case "$DATASET" in
  Peptides-func) CONFIG_DATASET=Peptides-func ;;
  PascalVOC-SP) CONFIG_DATASET=PascalVOC ;;
  *) echo "Unsupported DATASET=$DATASET (use PascalVOC-SP|Peptides-func)"; exit 1 ;;
esac

CONFIG_DIR="configs/LRGB/$CONFIG_DATASET/defaults"
declare -A CFG=(
  [fixed]="$CONFIG_DIR/GPS+kmip.yaml"
  [linear]="$CONFIG_DIR/GPS+kmip-linear-50-to-10.yaml"
)
# Keep the existing Peptides-func exponential config available unchanged.
# It is exploratory and outside the current linear research workflow.
if [[ "$DATASET" == "Peptides-func" ]]; then
  CFG[exponential]=configs/LRGB/Peptides-func/colab/GPS+kmip-exponential.yaml
fi

# Resume with the same dataset, mode, config, seed and OUT_ROOT.
# Never change max_epoch midway or promote quick/standard checkpoints to full.
IFS=',' read -r -a RUN_LIST <<< "$RUNS"
for name in "${RUN_LIST[@]}"; do
  name="$(echo "$name" | xargs)"
  cfg="${CFG[$name]:-}"
  if [[ -z "$cfg" ]]; then
    echo "Unsupported run=$name for DATASET=$DATASET (exponential: Peptides-func only)"
    exit 1
  fi
  run_out="$OUT_ROOT/$DATASET/$MODE/$name"
  mkdir -p "$run_out"
  echo ""
  echo "======== RUN $name ========"
  echo "cfg=$cfg  out_dir=$run_out"
  for seed in "${SEED_LIST[@]}"; do
    echo "-------- seed=$seed (repeat=1) --------"
    KMIP_REQUIRE_KEOPS="$REQUIRE_KEOPS" python main.py \
      --cfg "$cfg" \
      --repeat 1 \
      wandb.use False \
      seed "$seed" \
      "${MODE_ARGS[@]}" \
      out_dir "$run_out" \
      train.auto_resume True \
      train.enable_ckpt True \
      train.ckpt_best True \
      name_tag "$name" \
      2>&1 | tee -a "$run_out/train_${name}_seed${seed}.log"
    echo "Finished $name seed=$seed"
  done
  echo "Finished $name"
done

echo ""
echo "All requested runs finished. Logs under $OUT_ROOT"
