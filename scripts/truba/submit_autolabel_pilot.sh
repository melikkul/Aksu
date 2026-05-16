#!/bin/bash
# Pilot autolabel run: 100K SENTENCES (~1.5M tokens) from the preprocessed corpus.
# §H+ B: pilot unit is sentences, not tokens, for stronger yield extrapolation.
# Must complete before submit_autolabel.sh (full array) is submitted.
# Outputs data/intermediate/autolabel_pilot.jsonl for yield extrapolation.
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_autolabel_pilot.sh

#SBATCH --job-name=aksu-autolabel-pilot
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/autolabel_pilot_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/autolabel_pilot_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
mkdir -p "$PROJECT/data/intermediate"
mkdir -p "$PROJECT/data/tr_gold_morph/v2"

cd "$PROJECT"

# 100K sentences ≈ 1.5M tokens (§H+ B: stronger yield extrapolation signal).
# --sentence-limit reads first 100K sentences, filters unique_tokens to those
# that appear in at least one of those sentences.
python -m aksu.data.build.autolabel \
    --unique-tokens   data/intermediate/unique_tokens.jsonl \
    --token-sentences data/intermediate/token_sentences.jsonl \
    --sentences       data/intermediate/sentences.jsonl \
    --output          data/intermediate/autolabel_pilot.jsonl \
    --sentence-limit  100000 \
    --seed 42

echo "Pilot complete. Extrapolate yield with:"
echo "python scripts/data/extrapolate_yield.py \\"
echo "    --pilot-output data/intermediate/autolabel_pilot.jsonl \\"
echo "    --target 2500000 \\"
echo "    --output audit/benchmark_results/yield_extrapolation.json"
