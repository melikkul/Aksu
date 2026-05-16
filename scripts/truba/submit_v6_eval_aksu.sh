#!/bin/bash
# Evaluate v6 disambiguator ensemble on Orfoz CPU (patched from akya-cuda).
# CPU is the correct measurement partition: the library markets itself as
# CPU-friendly; GPU throughput would be misleading for end-user expectations.
# EM result is numerically identical regardless of device.
#
# Expected wall-clock: ~60-90 min (5 seeds × ~8140 test tokens × BERTurk CPU)
#
# Usage:
#   sbatch scripts/truba/submit_v6_eval_aksu.sh

#SBATCH --job-name=aksu-v6-eval-cpu
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=04:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/v6_eval_cpu_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/v6_eval_cpu_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs
cd "$PROJECT"

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56

echo "=== v6 Disambiguator Eval (CPU/Orfoz) ==="
echo "Date: $(date) | Node: $(hostname) | CPUs: $(nproc)"
lscpu | grep "Model name" | head -1

python scripts/eval_disambiguator.py \
    --ckpts \
      models/v6/disambiguator/best_model.pt \
      models/v6/disambiguator_s123/best_model.pt \
      models/v6/disambiguator_s456/best_model.pt \
      models/v6/disambiguator_s789/best_model.pt \
      models/v6/disambiguator_s1337/best_model.pt \
    --test data/splits/test.jsonl \
    --val  data/splits/val.jsonl \
    --output models/v6/eval_results.json

echo "=== Eval complete: $(date) ==="
cat models/v6/eval_results.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
e = d.get('ensemble', {})
print(f'em_argmax={e.get(\"em_argmax_mean\",\"?\"):.4f} em_string={e.get(\"em_string_mean\",\"?\"):.4f}')
" || true
echo ""
echo "Next: python scripts/ingest_metrics.py --source models/v6/eval_results.json"
