#!/bin/bash
# Evaluate a trained DualHead checkpoint on the test split (Orfoz CPU).
#
# Accepts one positional argument: data version (v1 or v2, default v1).
#   v1 → checkpoint dir: models/dualhead_v1_cpu
#   v2 → checkpoint dir: models/dualhead_v2
#
# Usage:
#   sbatch scripts/truba/submit_dualhead_eval.sh [v1|v2]

#SBATCH --job-name=aksu-dualhead-eval-cpu
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=01:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/dualhead_eval_cpu_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/dualhead_eval_cpu_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
DATA_VERSION="${1:-v1}"

module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
cd "$PROJECT"

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=56

echo "=== DualHead Evaluation (CPU/Orfoz) — version=$DATA_VERSION ==="
echo "Date:     $(date)"
echo "Node:     $(hostname)"
lscpu | grep "Model name" | head -1

if [ "$DATA_VERSION" = "v2" ]; then
    CKPT_DIR="models/dualhead_v2"
else
    CKPT_DIR="models/dualhead_v1_cpu"
fi

CKPT="$CKPT_DIR/best_model.pt"
OUTPUT="$CKPT_DIR/eval_results.json"

if [ ! -f "$CKPT" ]; then
    echo "FAIL: checkpoint not found at $CKPT"
    echo "Run sbatch scripts/truba/submit_dualhead_train.sh $DATA_VERSION first."
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Output:     $OUTPUT"

/usr/bin/time -v python scripts/eval_dualhead.py \
    --ckpt       "$CKPT" \
    --test       data/splits/test.jsonl \
    --char-vocab models/vocabs/char_vocab.json \
    --tag-vocab  models/vocabs/tag_vocab.json \
    --output     "$OUTPUT"

echo ""
echo "=== Evaluation complete: $(date) ==="
echo "Results written to $OUTPUT"
