#!/bin/bash
# Stage 8: Tier-assign + boundary-annotate (post-autolabel, Orfoz SLURM).
#
# Reads: data/intermediate/autolabeled_shard_*.jsonl
# Writes: data/tr_gold_morph/v2/{gold,silver-confident,silver-auto,
#                                  silver-marginal,bronze,oov,v1_compat}.jsonl
#         data/tr_gold_morph/v2/stats.json
#
# §H+ D boundary halt thresholds: <80% WARN, <60% HALT.
#
# Usage:
#   sbatch scripts/truba/submit_tier_boundary.sh
#   # Run with --dependency=afterok:<autolabel_array_job_id> to chain after Stage 7

#SBATCH --job-name=aksu-tier-boundary
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 56
#SBATCH --time=03:00:00
#SBATCH --output=/arf/scratch/scolakoglu/logs/tier_boundary_%j.out
#SBATCH --error=/arf/scratch/scolakoglu/logs/tier_boundary_%j.err

set -euo pipefail

PROJECT=/arf/home/scolakoglu/NLP_Project
SCRATCH=/arf/scratch/scolakoglu
module load comp/python/miniconda3
source "$PROJECT/.venv/bin/activate"

mkdir -p "$SCRATCH/logs"
mkdir -p "$PROJECT/data/intermediate"
mkdir -p "$PROJECT/data/tr_gold_morph/v2"

cd "$PROJECT"

echo "=== Stage 8: Tier-assign + boundary-annotate ==="
echo "Start: $(date)"

# --------------------------------------------------------------------------
# Step 1: Concatenate all autolabeled shards
# --------------------------------------------------------------------------
AUTOLABELED="$PROJECT/data/intermediate/autolabeled.jsonl"
echo "Concatenating autolabeled shards..."
cat "$PROJECT"/data/intermediate/autolabeled_shard_*.jsonl > "$AUTOLABELED"
TOTAL_LINES=$(wc -l < "$AUTOLABELED")
echo "Total autolabeled lines: $TOTAL_LINES"

# --------------------------------------------------------------------------
# Step 2: Tier assignment (5-tier v2 policy + v1-compat)
# --------------------------------------------------------------------------
echo "Running tier assignment..."
python -m aksu.data.build.tier_assignment \
    --input       "$AUTOLABELED" \
    --output-dir  "$PROJECT/data/tr_gold_morph/v2" \
    --v1-gold     "$PROJECT/data/gold/tr_gold_morph_v1.jsonl" \
    --v1-test-surfaces "$PROJECT/data/intermediate/v1_test_surface_forms.txt"

echo "Tier assignment complete."

# --------------------------------------------------------------------------
# Step 3: Boundary annotation per tier file
# --------------------------------------------------------------------------
TIER_FILES=(
    "gold.jsonl"
    "silver-confident.jsonl"
    "silver-auto.jsonl"
    "silver-marginal.jsonl"
    "bronze.jsonl"
)

V2_DIR="$PROJECT/data/tr_gold_morph/v2"
OVERALL_TOTAL=0
OVERALL_COVERED=0

for TFILE in "${TIER_FILES[@]}"; do
    INPATH="$V2_DIR/$TFILE"
    if [[ ! -f "$INPATH" ]]; then
        echo "WARNING: $TFILE not found, skipping boundary annotation"
        continue
    fi

    TMPOUT="$V2_DIR/${TFILE%.jsonl}_with_boundaries.jsonl"
    echo "Annotating boundaries: $TFILE ..."

    python -m aksu.data.build.boundaries \
        --input  "$INPATH" \
        --output "$TMPOUT"

    # Replace original with boundary-annotated version
    mv "$TMPOUT" "$INPATH"

    # Accumulate coverage stats from the log (last boundary coverage line)
    # We parse from the stats computed below
done

# --------------------------------------------------------------------------
# Step 4: Compute dataset stats (includes boundary coverage check)
# --------------------------------------------------------------------------
echo "Computing dataset stats..."
python scripts/data/dataset_stats.py \
    --input-dir "$V2_DIR" \
    --output    "$V2_DIR/stats.json"

# --------------------------------------------------------------------------
# Step 5: Boundary coverage halt check (§H+ D)
# --------------------------------------------------------------------------
COVERAGE=$(python - <<'PYEOF'
import json, sys
p = "data/tr_gold_morph/v2/stats.json"
try:
    s = json.loads(open(p).read())
    cov = s.get("boundary_coverage", s.get("overall_boundary_coverage", None))
    if cov is None:
        print("UNKNOWN")
    else:
        print(f"{float(cov):.4f}")
except Exception as e:
    print(f"ERROR: {e}")
PYEOF
)

echo "Boundary coverage: $COVERAGE"

if [[ "$COVERAGE" == "UNKNOWN" || "$COVERAGE" == ERROR* ]]; then
    echo "WARNING: Could not read boundary coverage from stats.json"
elif python -c "import sys; sys.exit(0 if float('$COVERAGE') >= 0.60 else 1)" 2>/dev/null; then
    if ! python -c "import sys; sys.exit(0 if float('$COVERAGE') >= 0.80 else 1)" 2>/dev/null; then
        echo "WARNING: Boundary coverage $COVERAGE < 80% (§H+ D threshold). Proceeding."
    else
        echo "Boundary coverage OK: $COVERAGE"
    fi
else
    echo "HALT: Boundary coverage $COVERAGE < 60% — boundary extractor bug (§H+ D)"
    mkdir -p "$PROJECT/audit/halt_reports"
    cat > "$PROJECT/audit/halt_reports/$(date +%Y-%m-%d)-v2-stage8.md" <<REPORT
# Halt Report — Stage 8 — $(date +%Y-%m-%d)

## Trigger
Boundary coverage $COVERAGE < 60% (§H+ D halt threshold).

## Action Required
1. Check \`src/aksu/ariturk/boundaries.py\` for BoundaryExtractor bugs
2. Sample failures: \`python -m aksu.data.build.boundaries --input data/tr_gold_morph/v2/silver-auto.jsonl --output /tmp/test_bounds.jsonl\`
3. Re-run Stage 8 after fix
REPORT
    echo "Halt report: $PROJECT/audit/halt_reports/$(date +%Y-%m-%d)-v2-stage8.md"
    exit 1
fi

echo "=== Stage 8 complete: $(date) ==="
echo "Outputs: $V2_DIR/"
ls -lh "$V2_DIR"/*.jsonl "$V2_DIR/stats.json" 2>/dev/null || true
