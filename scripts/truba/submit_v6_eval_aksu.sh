#!/bin/bash
# Evaluate v6 disambiguator ensemble on akya-cuda.
# Must be submitted from /arf/scratch/scolakoglu/
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_v6_eval_aksu.sh

#SBATCH --job-name=aksu-v6-eval
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/v6_eval_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/v6_eval_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs

cd "$PROJECT"
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

echo "Eval complete. Results at models/v6/eval_results.json"
