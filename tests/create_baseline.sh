#!/bin/bash
# Create baseline outputs for regression testing
# Run this BEFORE any optimization changes to capture reference outputs
# Uses subset of exampleApplication data for faster testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="$SCRIPT_DIR/regression_baseline"
EXAMPLE_DIR="$PROJECT_ROOT/exampleApplication"
DATA_DIR="$SCRIPT_DIR/regression_data"

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate diploshic_dev

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Verify example data exists
if [ ! -d "$EXAMPLE_DIR" ]; then
    echo "ERROR: exampleApplication directory not found at $EXAMPLE_DIR"
    echo "Please extract exampleApplication.tar.gz first"
    exit 1
fi

# Create output directories
mkdir -p "$BASELINE_DIR"
mkdir -p "$BASELINE_DIR/ms_diploid_stats"
mkdir -p "$BASELINE_DIR/ms_haploid_stats"
mkdir -p "$DATA_DIR"

# Create smaller test data if not exists
MS_TEST_FILE="$DATA_DIR/neut_5reps.msOut.gz"
if [ ! -f "$MS_TEST_FILE" ]; then
    echo "Creating 5-rep subset of neut.msOut.gz for faster testing..."
    python "$SCRIPT_DIR/extract_ms_subset.py" \
        "$EXAMPLE_DIR/neut.msOut.gz" \
        "$MS_TEST_FILE" \
        -n 5
fi

echo "Creating regression test baselines..."
echo "======================================="
echo "Using data from: $EXAMPLE_DIR"
echo "Test MS file: $MS_TEST_FILE (5 reps)"

# 1. Diploid MS Mode using 50-rep subset (162 samples, 55000 bp)
echo ""
echo "[1/4] Generating diploid MS mode baseline..."
time diploSHIC fvecSim diploid "$MS_TEST_FILE" "$BASELINE_DIR/ms_diploid.fvec" \
  --totalPhysLen=55000 --outStatsDir="$BASELINE_DIR/ms_diploid_stats"

# 2. Haploid MS Mode using 50-rep subset
echo ""
echo "[2/4] Generating haploid MS mode baseline..."
time diploSHIC fvecSim haploid "$MS_TEST_FILE" "$BASELINE_DIR/ms_haploid.fvec" \
  --totalPhysLen=55000 --outStatsDir="$BASELINE_DIR/ms_haploid_stats"

# Create VCF subset if not exists (50 samples, 110kb region for 1 window)
VCF_TEST_FILE="$DATA_DIR/ag1000g_subset_50samples_110kb.vcf.gz"
if [ ! -f "$VCF_TEST_FILE" ]; then
    echo "Creating VCF subset for faster testing..."
    python "$SCRIPT_DIR/create_vcf_subset.py" \
        "$EXAMPLE_DIR/ag1000g.phase1.ar3.pass.biallelic.3R.vcf.28000000-29000000.gz" \
        "$VCF_TEST_FILE" \
        --num-samples=50 --start=28000000 --end=28110000
fi

# 3. Diploid VCF Mode (50 samples, 110kb region = 1 window with 11 subwindows)
echo ""
echo "[3/4] Generating diploid VCF mode baseline..."
time diploSHIC fvecVcf diploid "$VCF_TEST_FILE" \
  3R 28110000 "$BASELINE_DIR/vcf_diploid.fvec" \
  --statFileName="$BASELINE_DIR/vcf_diploid.stats" \
  --segmentStart=28000000 --segmentEnd=28110000 \
  --winSize=110000

# 4. Haploid VCF Mode
echo ""
echo "[4/4] Generating haploid VCF mode baseline..."
time diploSHIC fvecVcf haploid "$VCF_TEST_FILE" \
  3R 28110000 "$BASELINE_DIR/vcf_haploid.fvec" \
  --statFileName="$BASELINE_DIR/vcf_haploid.stats" \
  --segmentStart=28000000 --segmentEnd=28110000 \
  --winSize=110000

echo ""
echo "======================================="
echo "Baseline generation complete!"
echo "Output files in: $BASELINE_DIR"
echo ""
echo "Files created:"
ls -la "$BASELINE_DIR"
