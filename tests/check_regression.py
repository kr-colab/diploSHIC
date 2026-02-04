#!/usr/bin/env python3
"""
Regression test script for diploSHIC optimization.

Compares current output against baseline to detect numerical regressions.
Uses numpy.allclose() with configurable tolerances.

Usage:
    python tests/check_regression.py [--generate] [--rtol=1e-10] [--atol=1e-12]

Options:
    --generate    Generate new outputs for comparison (don't use baselines)
    --rtol        Relative tolerance (default: 1e-10)
    --atol        Absolute tolerance (default: 1e-12)
    --verbose     Print detailed comparison info
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_fvec(filepath):
    """Parse a feature vector file and return data as numpy array.

    Handles two formats:
    - MS mode: all columns are numeric (pi_win0, pi_win1, ...)
    - VCF mode: first 4 columns are (chrom, start, end, range), rest are numeric

    Returns:
        tuple: (header_line, data_array) where data_array is float64
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return None, np.array([])

    header = lines[0].strip()
    header_parts = header.split('\t')

    # Detect format: VCF has "chrom" as first column, MS starts with stats
    if header_parts[0] == 'chrom':
        # VCF format: skip first 4 columns (chrom, start, end, range)
        skip_cols = 4
    else:
        # MS format: all columns are numeric
        skip_cols = 0

    data = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        try:
            values = [float(x) if x != 'nan' else np.nan for x in parts[skip_cols:]]
            data.append(values)
        except (ValueError, IndexError):
            # Handle lines that don't parse cleanly
            continue

    return header, np.array(data, dtype=np.float64)


def parse_stats_file(filepath):
    """Parse a stats file and return as numpy array.

    Handles two formats:
    - MS mode: all columns are numeric
    - VCF mode: first 3 columns are (chrom, start, end), rest are numeric
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return None, np.array([])

    header = lines[0].strip()
    header_parts = header.split('\t')

    # Detect format: VCF has "chrom" as first column
    if header_parts[0] == 'chrom':
        skip_cols = 3  # Skip chrom, start, end
    else:
        skip_cols = 0

    data = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        try:
            values = [float(x) if x != 'nan' else np.nan for x in parts[skip_cols:]]
            data.append(values)
        except (ValueError, IndexError):
            continue

    return header, np.array(data, dtype=np.float64)


def compare_arrays(baseline, current, name, rtol, atol, verbose=False):
    """Compare two arrays and report differences.

    Returns:
        tuple: (passed, message)
    """
    if baseline.shape != current.shape:
        return False, f"{name}: Shape mismatch - baseline {baseline.shape} vs current {current.shape}"

    if baseline.size == 0:
        return True, f"{name}: Empty arrays (OK)"

    # Handle NaN values: they should match in the same positions
    baseline_nan = np.isnan(baseline)
    current_nan = np.isnan(current)

    if not np.array_equal(baseline_nan, current_nan):
        nan_diff = np.sum(baseline_nan != current_nan)
        return False, f"{name}: NaN position mismatch ({nan_diff} positions differ)"

    # Compare non-NaN values
    mask = ~baseline_nan
    if not np.any(mask):
        return True, f"{name}: All NaN (OK)"

    baseline_vals = baseline[mask]
    current_vals = current[mask]

    if np.allclose(baseline_vals, current_vals, rtol=rtol, atol=atol, equal_nan=True):
        return True, f"{name}: PASSED (max diff: {np.max(np.abs(baseline_vals - current_vals)):.2e})"

    # Find the differences
    diff = np.abs(baseline_vals - current_vals)
    rel_diff = diff / (np.abs(baseline_vals) + atol)

    max_diff_idx = np.argmax(diff)
    max_rel_diff_idx = np.argmax(rel_diff)

    msg = (f"{name}: FAILED\n"
           f"  Max absolute diff: {diff[max_diff_idx]:.6e} at index {max_diff_idx}\n"
           f"    baseline: {baseline_vals[max_diff_idx]:.15e}\n"
           f"    current:  {current_vals[max_diff_idx]:.15e}\n"
           f"  Max relative diff: {rel_diff[max_rel_diff_idx]:.6e} at index {max_rel_diff_idx}")

    if verbose:
        # Show statistics about differences
        msg += (f"\n  Diff stats: mean={np.mean(diff):.2e}, "
                f"median={np.median(diff):.2e}, "
                f"std={np.std(diff):.2e}")

        # Count how many values exceed tolerance
        exceeds = np.sum((diff > atol) & (rel_diff > rtol))
        msg += f"\n  Values exceeding tolerance: {exceeds}/{len(diff)}"

    return False, msg


def run_command(cmd, description):
    """Run a shell command and return success status."""
    print(f"  Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Command timed out")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def generate_current_outputs(output_dir, project_root):
    """Generate current outputs for comparison using exampleApplication data."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ms_diploid_stats"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ms_haploid_stats"), exist_ok=True)

    os.chdir(project_root)
    example_dir = os.path.join(project_root, "exampleApplication")
    tests_dir = os.path.join(project_root, "tests")
    data_dir = os.path.join(tests_dir, "regression_data")

    # Verify exampleApplication exists
    if not os.path.isdir(example_dir):
        print(f"    ERROR: exampleApplication directory not found at {example_dir}")
        return False

    # Use 5-rep subset for faster testing
    ms_file = os.path.join(data_dir, "neut_5reps.msOut.gz")
    if not os.path.exists(ms_file):
        print(f"    ERROR: Test MS file not found: {ms_file}")
        print(f"    Run 'bash tests/create_baseline.sh' first to generate test data.")
        return False

    # VCF subset file (50 samples, 110kb region)
    vcf_file = os.path.join(data_dir, "ag1000g_subset_50samples_110kb.vcf.gz")
    if not os.path.exists(vcf_file):
        print(f"    ERROR: Test VCF file not found: {vcf_file}")
        print(f"    Run 'bash tests/create_baseline.sh' first to generate test data.")
        return False

    commands = [
        # Diploid MS mode (5 reps subset, 162 samples, 55000 bp)
        (f'diploSHIC fvecSim diploid "{ms_file}" "{output_dir}/ms_diploid.fvec" '
         f'--totalPhysLen=55000 --outStatsDir="{output_dir}/ms_diploid_stats"',
         "Diploid MS mode"),

        # Haploid MS mode
        (f'diploSHIC fvecSim haploid "{ms_file}" "{output_dir}/ms_haploid.fvec" '
         f'--totalPhysLen=55000 --outStatsDir="{output_dir}/ms_haploid_stats"',
         "Haploid MS mode"),

        # Diploid VCF mode (50 samples, 110kb region = 1 window with 11 subwindows)
        (f'diploSHIC fvecVcf diploid "{vcf_file}" 3R 28110000 '
         f'"{output_dir}/vcf_diploid.fvec" '
         f'--statFileName="{output_dir}/vcf_diploid.stats" '
         f'--segmentStart=28000000 --segmentEnd=28110000 '
         f'--winSize=110000',
         "Diploid VCF mode"),

        # Haploid VCF mode
        (f'diploSHIC fvecVcf haploid "{vcf_file}" 3R 28110000 '
         f'"{output_dir}/vcf_haploid.fvec" '
         f'--statFileName="{output_dir}/vcf_haploid.stats" '
         f'--segmentStart=28000000 --segmentEnd=28110000 '
         f'--winSize=110000',
         "Haploid VCF mode"),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False

    return True


def check_regression(baseline_dir, current_dir, rtol, atol, verbose=False):
    """Compare baseline and current outputs.

    Returns:
        tuple: (all_passed, results_dict)
    """
    results = {}
    all_passed = True

    # Define files to compare
    comparisons = [
        ("ms_diploid.fvec", "Diploid MS fvec", parse_fvec),
        ("ms_haploid.fvec", "Haploid MS fvec", parse_fvec),
        ("vcf_diploid.fvec", "Diploid VCF fvec", parse_fvec),
        ("vcf_haploid.fvec", "Haploid VCF fvec", parse_fvec),
        ("vcf_diploid.stats", "Diploid VCF stats", parse_stats_file),
        ("vcf_haploid.stats", "Haploid VCF stats", parse_stats_file),
    ]

    for filename, description, parser in comparisons:
        baseline_path = os.path.join(baseline_dir, filename)
        current_path = os.path.join(current_dir, filename)

        if not os.path.exists(baseline_path):
            results[description] = (False, f"Baseline file not found: {baseline_path}")
            all_passed = False
            continue

        if not os.path.exists(current_path):
            results[description] = (False, f"Current file not found: {current_path}")
            all_passed = False
            continue

        baseline_header, baseline_data = parser(baseline_path)
        current_header, current_data = parser(current_path)

        # Check headers match
        if baseline_header != current_header:
            results[description] = (False, f"Header mismatch:\n  baseline: {baseline_header}\n  current:  {current_header}")
            all_passed = False
            continue

        passed, msg = compare_arrays(baseline_data, current_data, description, rtol, atol, verbose)
        results[description] = (passed, msg)
        if not passed:
            all_passed = False

    # Compare stats directories for MS modes
    for mode in ["diploid", "haploid"]:
        stats_dir = f"ms_{mode}_stats"
        baseline_stats_dir = os.path.join(baseline_dir, stats_dir)
        current_stats_dir = os.path.join(current_dir, stats_dir)

        if os.path.isdir(baseline_stats_dir) and os.path.isdir(current_stats_dir):
            baseline_files = set(os.listdir(baseline_stats_dir))
            current_files = set(os.listdir(current_stats_dir))

            if baseline_files != current_files:
                results[f"{mode.capitalize()} MS stats dir"] = (
                    False,
                    f"File list mismatch:\n  baseline: {baseline_files}\n  current:  {current_files}"
                )
                all_passed = False
                continue

            for stat_file in baseline_files:
                baseline_stat = os.path.join(baseline_stats_dir, stat_file)
                current_stat = os.path.join(current_stats_dir, stat_file)

                _, baseline_data = parse_stats_file(baseline_stat)
                _, current_data = parse_stats_file(current_stat)

                name = f"{mode.capitalize()} MS {stat_file}"
                passed, msg = compare_arrays(baseline_data, current_data, name, rtol, atol, verbose)
                results[name] = (passed, msg)
                if not passed:
                    all_passed = False

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Check for numerical regressions in diploSHIC")
    parser.add_argument("--generate", action="store_true",
                        help="Generate new outputs (use with baseline creation)")
    parser.add_argument("--rtol", type=float, default=1e-10,
                        help="Relative tolerance (default: 1e-10)")
    parser.add_argument("--atol", type=float, default=1e-12,
                        help="Absolute tolerance (default: 1e-12)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed comparison info")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Path to baseline directory (default: tests/regression_baseline)")
    parser.add_argument("--current-dir", type=str, default=None,
                        help="Path to current output directory (default: temp directory)")

    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    baseline_dir = args.baseline_dir or (script_dir / "regression_baseline")

    print("=" * 60)
    print("diploSHIC Regression Test")
    print("=" * 60)
    print(f"Tolerance: rtol={args.rtol}, atol={args.atol}")
    print(f"Baseline directory: {baseline_dir}")

    # Check baseline exists
    if not os.path.exists(baseline_dir):
        print(f"\nERROR: Baseline directory not found: {baseline_dir}")
        print("Run 'bash tests/create_baseline.sh' first to generate baselines.")
        return 1

    # Generate current outputs
    if args.current_dir:
        current_dir = Path(args.current_dir)
        os.makedirs(current_dir, exist_ok=True)
    else:
        current_dir = Path(tempfile.mkdtemp(prefix="diploshic_regression_"))

    print(f"Current output directory: {current_dir}")
    print()

    print("Generating current outputs...")
    print("-" * 40)
    if not generate_current_outputs(str(current_dir), str(project_root)):
        print("\nERROR: Failed to generate current outputs")
        return 1
    print()

    print("Comparing outputs...")
    print("-" * 40)
    all_passed, results = check_regression(
        str(baseline_dir), str(current_dir),
        args.rtol, args.atol, args.verbose
    )

    # Print results
    print()
    for name, (passed, msg) in sorted(results.items()):
        status = "✓" if passed else "✗"
        print(f"[{status}] {msg}")

    print()
    print("=" * 60)
    if all_passed:
        print("RESULT: All regression tests PASSED")
        return 0
    else:
        print("RESULT: Some regression tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
