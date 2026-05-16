#!/bin/bash
# Preprocess OSCAR-tr shard → tokens.jsonl + sentences.jsonl + token_sentences.jsonl
# Then deduplicate → unique_tokens.jsonl (input for autolabel pilot).
#
# Run AFTER data/external/ sources are downloaded (or in streaming mode via HF).
# Must complete before submit_autolabel_pilot.sh.
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_preprocess_aksu.sh

#SBATCH --job-name=aksu-preprocess
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=12:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/preprocess_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/preprocess_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
mkdir -p "$PROJECT/data/intermediate"

cd "$PROJECT"

echo "=== Preprocessing OSCAR-tr shard ==="
echo "Date: $(date) | Node: $(hostname) | CPUs: $(nproc)"

# Stream from HuggingFace — requires internet access on compute node.
# If offline, mount data/external/oscar-tr/ and update sources.py URL to a local path.
python -m aksu.data.build.preprocess \
    --shard oscar-tr \
    --max-tokens 12000000 \
    --output-dir data/intermediate

echo "Preprocessing done: $(date)"

echo "=== Deduplicating tokens ==="
python scripts/data/dedup_tokens.py \
    --input  data/intermediate/tokens.jsonl \
    --output data/intermediate/unique_tokens.jsonl

echo "Dedup done: $(date)"
echo "unique_tokens.jsonl ready for autolabel pilot"
echo ""
echo "Next step:"
echo "  sbatch scripts/truba/submit_autolabel_pilot.sh"
