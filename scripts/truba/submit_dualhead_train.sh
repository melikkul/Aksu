#!/bin/bash
# Train DualHead atomizer on Orfoz CPU (patched from akya-cuda).
# Training on CPU directly validates the README claim "no GPU required".
#
# Accepts one positional argument: data version (v1 or v2, default v1).
#   v1 → uses data/gold/tr_gold_morph_v1.jsonl  (80K tokens, available now)
#   v2 → uses data/tr_gold_morph/v2/silver.jsonl (2.5M, available after autolabel)
#
# Expected wall-clock: ~4-8h for v1 baseline, ~12h+ for v2.
#
# Usage:
#   sbatch scripts/truba/submit_dualhead_train.sh [v1|v2]

#SBATCH --job-name=aksu-dualhead-train-cpu
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 112
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/dualhead_train_cpu_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/dualhead_train_cpu_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
DATA_VERSION="${1:-v1}"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
cd "$PROJECT"

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=112
export MKL_NUM_THREADS=112

echo "=== DualHead Training (CPU/Orfoz) — data=$DATA_VERSION ==="
echo "Date: $(date) | Node: $(hostname) | CPUs: $(nproc)"
lscpu | grep "Model name" | head -1

# Pre-flight check
python -m aksu.train.train_v4_master --help | grep -q -- '--model' || {
    echo "FAIL: train_v4_master entry point not wired"; exit 1
}

if [ "$DATA_VERSION" = "v1" ]; then
    DATA_PATH="data/gold/tr_gold_morph_v1.jsonl"
    EVAL_PATH="data/splits/val.jsonl"
    OUT_DIR="models/dualhead_v1_cpu"
else
    DATA_PATH="data/tr_gold_morph/v2/silver.jsonl"
    EVAL_PATH="data/splits/val.jsonl"
    OUT_DIR="models/dualhead_v2"
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "FAIL: training data not found at $DATA_PATH"; exit 1
fi

/usr/bin/time -v python -m aksu.train.train_v4_master \
    --model dual_head \
    --training-data "$DATA_PATH" \
    --eval-data "$EVAL_PATH" \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab  models/vocabs/tag_vocab.json \
    --device cpu \
    --seed 42 \
    --output-dir "$OUT_DIR"

echo "=== Training complete: $(date) ==="

# Write checkpoint sidecar
python scripts/data/write_checkpoint_metadata.py "$OUT_DIR/best_model.pt"
echo "Checkpoint sidecar written."

echo ""
echo "Next: sbatch scripts/truba/submit_dualhead_eval.sh $DATA_VERSION"
