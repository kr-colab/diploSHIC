#!/usr/bin/env python
"""Build DAF feature training and testing sets from ms simulation output.

Mirrors the structure of diploshic/training/ and diploshic/testing/ but with
DAF histogram + distance features. Uses multiprocessing to process ms files
in parallel and vectorized numpy operations throughout.

Usage:
    python scripts/build_daf_training_sets.py
    python scripts/build_daf_training_sets.py --workers 16
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from diploshic.msTools import (
    openMsOutFileForSequentialReading,
    readNextMsRepToHaplotypeArrayIn,
    closeMsOutFile,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO_ROOT, "exampleApplication")
N_DAF_BINS = 20
N_DIST_FEATURES = 4
TOTAL_PHYS_LEN = 55000
NUM_SUB_WINS = 11
SUB_WIN_LEN = TOTAL_PHYS_LEN // NUM_SUB_WINS
N_FEATURES = N_DAF_BINS * NUM_SUB_WINS + N_DIST_FEATURES * NUM_SUB_WINS

# Pre-compute sub-window boundaries (1-based, inclusive)
SUB_WIN_EDGES = np.arange(NUM_SUB_WINS + 1) * SUB_WIN_LEN
SUB_WIN_EDGES[0] = 1
SUB_WIN_EDGES[-1] = TOTAL_PHYS_LEN
# boundaries: win i covers [SUB_WIN_EDGES[i], SUB_WIN_EDGES[i+1]]

# Pre-compute DAF bin edges
DAF_BIN_EDGES = np.linspace(0.0, 1.0, N_DAF_BINS + 1)

# Pre-allocate uniform DAF vector (used when no SNPs)
UNIFORM_DAF = np.full(N_DAF_BINS, 1.0 / N_DAF_BINS)


def _assign_snps_to_windows(positions):
    """Vectorized assignment of SNP positions to sub-windows.

    Returns an integer array of length n_snps with the sub-window index for each SNP.
    """
    pos = np.asarray(positions)
    # searchsorted on the window edges; subtract 1 to get 0-based window index
    # positions are 1-based integers, edges are [1, 5001, 10001, ..., 55000]
    win_ids = np.searchsorted(SUB_WIN_EDGES[1:], pos, side="left")
    return np.clip(win_ids, 0, NUM_SUB_WINS - 1)


def _process_rep(hap_array, positions):
    """Compute DAF histogram + distance features for one rep, fully vectorized.

    Returns a flat array of length N_FEATURES.
    """
    if len(positions) == 0:
        daf_block = np.tile(UNIFORM_DAF, NUM_SUB_WINS)
        dist_block = np.zeros(N_DIST_FEATURES * NUM_SUB_WINS)
        return np.concatenate((daf_block, dist_block))

    hap_array = np.asarray(hap_array, dtype=np.float32)
    pos_arr = np.asarray(positions, dtype=np.float64)
    n_samples = hap_array.shape[1]

    # DAFs for all SNPs at once
    dafs = hap_array.sum(axis=1) / n_samples

    # Assign each SNP to a sub-window
    win_ids = _assign_snps_to_windows(pos_arr)

    # Output arrays
    daf_out = np.empty((NUM_SUB_WINS, N_DAF_BINS), dtype=np.float64)
    dist_out = np.zeros((NUM_SUB_WINS, N_DIST_FEATURES), dtype=np.float64)

    for w in range(NUM_SUB_WINS):
        mask = win_ids == w
        n_snps = mask.sum()

        if n_snps == 0:
            daf_out[w] = UNIFORM_DAF
            # dist_out[w] already zeros
            continue

        # DAF histogram — vectorized via np.histogram
        hist, _ = np.histogram(dafs[mask], bins=DAF_BIN_EDGES)
        total = hist.sum()
        daf_out[w] = hist / total if total > 0 else UNIFORM_DAF

        # Distance stats
        if n_snps >= 2:
            win_pos = pos_arr[mask]  # already sorted since positions are sorted
            dists = np.diff(win_pos) / SUB_WIN_LEN
            dist_out[w, 0] = dists.mean()
            dist_out[w, 1] = dists.var()
            dist_out[w, 2] = dists.min()
            dist_out[w, 3] = dists.max()

    # Flatten in feature-major order: dafBin0_win0..win10, dafBin1_win0..win10, ...
    return np.concatenate((daf_out.T.ravel(), dist_out.T.ravel()))


def extract_daf_features_from_ms(ms_file):
    """Extract DAF+distance features from all reps in an ms file.

    Returns array of shape (n_reps, N_FEATURES).
    """
    file_obj, sample_size, num_instances = openMsOutFileForSequentialReading(ms_file)

    out = np.empty((num_instances, N_FEATURES), dtype=np.float64)

    for rep_idx in range(num_instances):
        hap_array, positions = readNextMsRepToHaplotypeArrayIn(
            file_obj, sample_size, TOTAL_PHYS_LEN
        )
        out[rep_idx] = _process_rep(hap_array, positions)

    closeMsOutFile(file_obj)
    return out


def _worker(ms_file):
    """Worker function for ProcessPoolExecutor. Returns (filename, features_array)."""
    basename = os.path.basename(ms_file)
    t0 = time.perf_counter()
    features = extract_daf_features_from_ms(ms_file)
    elapsed = time.perf_counter() - t0
    return basename, features, elapsed


def build_header():
    parts = np.empty(N_FEATURES, dtype=object)
    idx = 0
    for b in range(N_DAF_BINS):
        for w in range(NUM_SUB_WINS):
            parts[idx] = f"dafBin{b}_win{w}"
            idx += 1
    for feat_name in ("snpDistMean", "snpDistVar", "snpDistMin", "snpDistMax"):
        for w in range(NUM_SUB_WINS):
            parts[idx] = f"{feat_name}_win{w}"
            idx += 1
    return "\t".join(parts)


def write_fvec(filepath, header, data):
    """Write fvec file using numpy's fast text writer."""
    with open(filepath, "w") as f:
        f.write(header + "\n")
    with open(filepath, "ab") as f:
        np.savetxt(f, data, delimiter="\t", fmt="%.18g")


def main():
    parser = argparse.ArgumentParser(description="Build DAF training/testing feature sets")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    n_per_class = np.loadtxt(
        os.path.join(REPO_ROOT, "diploshic", "training", "hard.fvec"),
        skiprows=1, max_rows=1
    ).shape[0]
    # That just reads one row to get ncols; we need nrows
    with open(os.path.join(REPO_ROOT, "diploshic", "training", "hard.fvec")) as f:
        n_per_class = sum(1 for _ in f) - 1  # subtract header
    print(f"Target: {n_per_class} examples per class")
    print(f"Using {args.workers} workers")

    header = build_header()
    train_dir = os.path.join(REPO_ROOT, "diploshic", "training")
    test_dir = os.path.join(REPO_ROOT, "diploshic", "testing")

    # Collect all ms files to process
    ms_files = {}
    for sweep_type in ("hard", "soft"):
        for i in range(11):
            path = os.path.join(SIM_DIR, f"{sweep_type}_{i}.msOut.gz")
            if os.path.exists(path):
                ms_files[f"{sweep_type}_{i}"] = path
    neut_path = os.path.join(SIM_DIR, "neut.msOut.gz")
    if os.path.exists(neut_path):
        ms_files["neut"] = neut_path

    print(f"Processing {len(ms_files)} ms files in parallel...\n")
    t_start = time.perf_counter()

    # Process all files in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, path): key for key, path in ms_files.items()}
        for future in as_completed(futures):
            key = futures[future]
            basename, features, elapsed = future.result()
            results[key] = features
            print(f"  {basename}: {features.shape[0]} reps in {elapsed:.1f}s")

    # Assemble and write output files
    for sweep_type in ("hard", "soft"):
        # Concatenate all sweep files in index order
        parts = []
        for i in range(11):
            key = f"{sweep_type}_{i}"
            if key in results:
                parts.append(results[key])
        all_features = np.concatenate(parts)
        print(f"\n{sweep_type}: {all_features.shape[0]} total reps")

        linked_name = f"linked{sweep_type.capitalize()}"

        write_fvec(os.path.join(train_dir, f"{sweep_type}.daf.fvec"),
                   header, all_features[:n_per_class])
        write_fvec(os.path.join(test_dir, f"{sweep_type}.daf.fvec"),
                   header, all_features[n_per_class:2 * n_per_class])
        write_fvec(os.path.join(train_dir, f"{linked_name}.daf.fvec"),
                   header, all_features[2 * n_per_class:3 * n_per_class])
        write_fvec(os.path.join(test_dir, f"{linked_name}.daf.fvec"),
                   header, all_features[3 * n_per_class:4 * n_per_class])
        print(f"  Wrote {sweep_type} + {linked_name} ({n_per_class} train, {n_per_class} test each)")

    # Neutral — only 2000 reps available, reuse for both train and test
    # (matches the behavior of the original fvec pipeline)
    neut_features = results["neut"]
    print(f"\nneutral: {neut_features.shape[0]} total reps")
    write_fvec(os.path.join(train_dir, "neut.daf.fvec"),
               header, neut_features[:n_per_class])
    write_fvec(os.path.join(test_dir, "neut.daf.fvec"),
               header, neut_features[:n_per_class])
    print(f"  Wrote neutral ({n_per_class} train, {n_per_class} test — same reps, matching original pipeline)")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s total")


if __name__ == "__main__":
    main()
