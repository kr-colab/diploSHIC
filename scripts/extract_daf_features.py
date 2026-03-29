#!/usr/bin/env python
"""Extract DAF histogram and inter-SNP distance features from ms-format simulation output.

Produces .daf.fvec files with the same number of rows as the corresponding .fvec files.
These are loaded alongside the standard stats fvec files during training.

Usage:
    python scripts/extract_daf_features.py exampleApplication/neut.msOut.gz output/neut.daf.fvec
    python scripts/extract_daf_features.py exampleApplication/hard_5.msOut.gz output/hard_5.daf.fvec
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from diploshic.msTools import openMsOutFileForSequentialReading, readNextMsRepToHaplotypeArrayIn, closeMsOutFile
from diploshic.fvTools import compute_daf_histogram, compute_snp_distance_stats

N_DAF_BINS = 20
N_DIST_FEATURES = 4
TOTAL_PHYS_LEN = 55000
NUM_SUB_WINS = 11


def get_sub_win_bounds(sub_win_len, total_phys_len):
    """Get inclusive sub-window bounds (1-based positions)."""
    bounds = []
    start = 1
    for i in range(total_phys_len // sub_win_len):
        end = start + sub_win_len - 1
        if i == total_phys_len // sub_win_len - 1:
            end = total_phys_len
        bounds.append((start, end))
        start = end + 1
    return bounds


def get_snp_indices_in_sub_wins(sub_win_bounds, snp_locs):
    """Assign SNP indices to sub-windows based on position."""
    indices = [[] for _ in sub_win_bounds]
    win_idx = 0
    for i, loc in enumerate(snp_locs):
        while not (sub_win_bounds[win_idx][0] <= loc <= sub_win_bounds[win_idx][1]):
            win_idx += 1
        indices[win_idx].append(i)
    return indices


def process_file(ms_file, output_file, total_phys_len=TOTAL_PHYS_LEN,
                 num_sub_wins=NUM_SUB_WINS, n_daf_bins=N_DAF_BINS):
    sub_win_len = total_phys_len // num_sub_wins
    sub_win_bounds = get_sub_win_bounds(sub_win_len, total_phys_len)

    file_obj, sample_size, num_instances = openMsOutFileForSequentialReading(ms_file)

    # Build header
    header_parts = []
    for b in range(n_daf_bins):
        for w in range(num_sub_wins):
            header_parts.append(f"dafBin{b}_win{w}")
    for feat_name in ["snpDistMean", "snpDistVar", "snpDistMin", "snpDistMax"]:
        for w in range(num_sub_wins):
            header_parts.append(f"{feat_name}_win{w}")

    all_rows = []
    start = time.perf_counter()

    for rep_idx in range(num_instances):
        if rep_idx % 200 == 0:
            sys.stderr.write(f"  rep {rep_idx}/{num_instances}\n")

        hap_array, positions = readNextMsRepToHaplotypeArrayIn(
            file_obj, sample_size, total_phys_len
        )

        if len(positions) == 0:
            # No SNPs — uniform DAF histogram, zero distance stats
            daf_row = np.tile(np.full(n_daf_bins, 1.0 / n_daf_bins), num_sub_wins)
            dist_row = np.zeros(N_DIST_FEATURES * num_sub_wins)
            all_rows.append(np.concatenate([daf_row, dist_row]))
            continue

        hap_array = np.asarray(hap_array)
        snp_indices = get_snp_indices_in_sub_wins(sub_win_bounds, positions)

        daf_vals = []  # n_daf_bins values per sub-window
        dist_vals = []  # 4 values per sub-window

        for win_idx in range(num_sub_wins):
            idx = snp_indices[win_idx]
            if len(idx) > 0:
                haps_in_win = hap_array[idx, :]
                daf_hist = compute_daf_histogram(haps_in_win, n_bins=n_daf_bins)
                win_positions = [positions[i] for i in idx]
                dist_stats = compute_snp_distance_stats(win_positions, sub_win_len)
            else:
                daf_hist = np.full(n_daf_bins, 1.0 / n_daf_bins)
                dist_stats = np.zeros(N_DIST_FEATURES)
            daf_vals.append(daf_hist)
            dist_vals.append(dist_stats)

        # Interleave: for each feature, list all windows (matching fvec convention)
        daf_matrix = np.array(daf_vals)  # (num_sub_wins, n_daf_bins)
        dist_matrix = np.array(dist_vals)  # (num_sub_wins, 4)

        # Flatten in feature-major order: dafBin0_win0, dafBin0_win1, ..., dafBin0_win10, dafBin1_win0, ...
        row = np.concatenate([
            daf_matrix.T.flatten(),   # (n_daf_bins, num_sub_wins) flattened
            dist_matrix.T.flatten(),  # (4, num_sub_wins) flattened
        ])
        all_rows.append(row)

    closeMsOutFile(file_obj)

    elapsed = time.perf_counter() - start
    sys.stderr.write(f"  processed {num_instances} reps in {elapsed:.1f}s\n")

    with open(output_file, "w") as f:
        f.write("\t".join(header_parts) + "\n")
        for row in all_rows:
            f.write("\t".join(f"{v}" for v in row) + "\n")

    sys.stderr.write(f"  wrote {output_file}\n")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ms_file> <output_fvec>", file=sys.stderr)
        sys.exit(1)
    process_file(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
