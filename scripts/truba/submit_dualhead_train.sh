#!/bin/bash
# Train DualHead atomizer on akya-cuda (E-Step 1).
# Must be submitted from /arf/scratch/scolakoglu/
#
# Pre-flight: verify entry point works first:
#   python -m aksu.train.train_v4_master --help | grep -q -- '--model'
#
# Usage:
#   sbatch /arf/home/scolakoglu/NLP_Project/scripts/truba/submit_dualhead_train.sh

#SBATCH --job-name=aksu-dualhead-train
#SBATCH --partition=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/dualhead_train_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/dualhead_train_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p /arf/scratch/scolakoglu/logs

cd "$PROJECT"

# Pre-flight check
python -m aksu.train.train_v4_master --help | grep -q -- '--model' || {
    echo "FAIL: train_v4_master entry point not wired"; exit 1
}

/usr/bin/time -v python -m aksu.train.train_v4_master \
    --model dualhead \
    --data data/splits/train.jsonl \
    --val  data/splits/val.jsonl \
    --config configs/train/dualhead_v2.yaml \
    --seed 42 \
    --output models/dualhead_v2/

echo "Training complete. Writing checkpoint metadata sidecar..."
python scripts/data/write_checkpoint_metadata.py models/dualhead_v2/
echo "Done."
