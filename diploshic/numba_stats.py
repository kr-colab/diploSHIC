"""
Numba-optimized replacements for C extension functions in shicstats.
These avoid potential memory leaks in the Fortran/C code.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def r2_pair(haps, i, j, n_samps):
    """Compute R² between SNP i and SNP j."""
    pi = 0.0
    pj = 0.0
    pij = 0.0
    count = 0.0

    for k in range(n_samps):
        hi = haps[i, k]
        hj = haps[j, k]
        if (hi == 0 or hi == 1) and (hj == 0 or hj == 1):
            if hi == 1:
                pi += 1
            if hj == 1:
                pj += 1
            if hi == 1 and hj == 1:
                pij += 1
            count += 1

    if count == 0:
        return -1.0

    pi /= count
    pj /= count
    pij /= count

    Dij = pij - (pi * pj)
    denom = (pi * (1.0 - pi)) * (pj * (1.0 - pj))

    if denom == 0:
        return -1.0

    return (Dij * Dij) / denom


@njit(parallel=True, cache=True)
def compute_r2_matrix(haps):
    """
    Compute upper-triangular R² matrix for haplotype data.

    Parameters
    ----------
    haps : ndarray, shape (n_snps, n_samples)
        Haplotype matrix with values 0, 1, or -1 (missing)

    Returns
    -------
    r2_matrix : ndarray, shape (n_snps, n_snps)
        Upper triangular R² matrix
    """
    n_snps, n_samps = haps.shape
    r2_matrix = np.zeros((n_snps, n_snps), dtype=np.float64)

    for i in prange(n_snps - 1):
        for j in range(i + 1, n_snps):
            r2_matrix[i, j] = r2_pair(haps, i, j, n_samps)

    return r2_matrix


@njit(cache=True)
def zns(r2_matrix):
    """
    Compute Kelly's ZnS (mean pairwise R²).

    Parameters
    ----------
    r2_matrix : ndarray, shape (n_snps, n_snps)
        Upper triangular R² matrix

    Returns
    -------
    zns : float
        Mean R² over all valid pairs
    """
    n_snps = r2_matrix.shape[0]

    if n_snps < 2:
        return 0.0

    r2_sum = 0.0
    r2_count = 0

    for i in range(n_snps - 1):
        for j in range(i + 1, n_snps):
            val = r2_matrix[i, j]
            if val >= 0.0:
                r2_sum += val
                r2_count += 1

    if r2_count == 0:
        return 0.0

    return r2_sum / r2_count


@njit(cache=True)
def omega_naive(r2_matrix):
    """Naive O(n³) omega for correctness verification."""
    n_snps = r2_matrix.shape[0]

    if n_snps < 5:
        return 0.0

    omega_max = 0.0

    for l in range(3, n_snps - 2):
        cross_sum = 0.0
        cross_count = 0
        left_sum = 0.0
        left_count = 0
        right_sum = 0.0
        right_count = 0

        for i in range(n_snps - 1):
            for j in range(i + 1, n_snps):
                val = r2_matrix[i, j]
                if val >= 0.0:
                    if i < l and j >= l:
                        cross_sum += val
                        cross_count += 1
                    elif i < l and j < l:
                        left_sum += val
                        left_count += 1
                    elif i >= l and j >= l:
                        right_sum += val
                        right_count += 1

        if cross_count > 0 and (left_count + right_count) > 0:
            mean_cross = cross_sum / cross_count
            if mean_cross > 0:
                mean_within = (left_sum + right_sum) / (left_count + right_count)
                omega_val = mean_within / mean_cross
                if omega_val > omega_max:
                    omega_max = omega_val

    return omega_max


@njit(cache=True)
def omega_fast(r2_matrix):
    """
    Compute omega statistic using O(n²) algorithm with incremental updates.

    For partition at position l:
    - Left pairs: i < l and j < l (since i<j always, this means j < l)
    - Right pairs: i >= l and j >= l (since i<j always, this means i >= l)
    - Cross pairs: i < l and j >= l

    omega(l) = mean(left + right) / mean(cross)

    As l increases from l to l+1:
    - Pairs (i, l) where i < l move from CROSS to LEFT
    - Pairs (l, j) where j > l move from RIGHT to CROSS
    """
    n_snps = r2_matrix.shape[0]

    if n_snps < 5:
        return 0.0

    # Initialize for l = 3 by computing all sums from scratch
    left_sum = 0.0
    left_count = 0
    cross_sum = 0.0
    cross_count = 0
    right_sum = 0.0
    right_count = 0

    l = 3

    for i in range(n_snps - 1):
        for j in range(i + 1, n_snps):
            val = r2_matrix[i, j]
            if val >= 0.0:
                if i < l and j < l:
                    left_sum += val
                    left_count += 1
                elif i >= l:  # implies j >= l since j > i
                    right_sum += val
                    right_count += 1
                else:  # i < l and j >= l
                    cross_sum += val
                    cross_count += 1

    omega_max = 0.0

    # Evaluate and update incrementally
    for l in range(3, n_snps - 2):
        # Evaluate omega at current l
        if cross_count > 0 and (left_count + right_count) > 0:
            mean_cross = cross_sum / cross_count
            if mean_cross > 0:
                mean_within = (left_sum + right_sum) / (left_count + right_count)
                omega_val = mean_within / mean_cross
                if omega_val > omega_max:
                    omega_max = omega_val

        # Update for l -> l+1:
        # Pairs (i, l) where i < l: were CROSS, now LEFT
        for i in range(l):
            val = r2_matrix[i, l]
            if val >= 0.0:
                cross_sum -= val
                cross_count -= 1
                left_sum += val
                left_count += 1

        # Pairs (l, j) where j > l: were RIGHT, now CROSS
        for j in range(l + 1, n_snps):
            val = r2_matrix[l, j]
            if val >= 0.0:
                right_sum -= val
                right_count -= 1
                cross_sum += val
                cross_count += 1

    return omega_max


@njit(cache=True)
def omega(r2_matrix):
    """
    Compute omega statistic (max ratio of within-partition to cross-partition LD).

    This uses an O(n²) algorithm instead of the naive O(n³) approach.

    Parameters
    ----------
    r2_matrix : ndarray, shape (n_snps, n_snps)
        Upper triangular R² matrix

    Returns
    -------
    omega_max : float
        Maximum omega value across all partition points
    """
    return omega_fast(r2_matrix)


@njit(parallel=True, cache=True)
def pairwise_diffs(haps):
    """
    Compute pairwise differences between all sample pairs.

    Parameters
    ----------
    haps : ndarray, shape (n_snps, n_samples)
        Haplotype matrix

    Returns
    -------
    diffs : ndarray, shape (n_samples * (n_samples - 1) // 2,)
        Pairwise difference counts
    """
    n_snps, n_samps = haps.shape
    n_pairs = n_samps * (n_samps - 1) // 2
    diffs = np.zeros(n_pairs, dtype=np.float64)

    pair_idx = 0
    for i in range(n_samps - 1):
        for j in range(i + 1, n_samps):
            diff_count = 0
            for k in range(n_snps):
                hi = haps[k, i]
                hj = haps[k, j]
                if 0 <= hi <= 1 and 0 <= hj <= 1:
                    if hi != hj:
                        diff_count += 1
            diffs[pair_idx] = diff_count
            pair_idx += 1

    return diffs


@njit(parallel=True, cache=True)
def pairwise_diffs_diplo(genos):
    """
    Compute pairwise differences for diploid genotypes.

    Parameters
    ----------
    genos : ndarray, shape (n_snps, n_samples)
        Genotype matrix with values 0, 1, 2, or -1 (missing)

    Returns
    -------
    diffs : ndarray, shape (n_samples * (n_samples - 1) // 2,)
        Pairwise difference counts
    """
    n_snps, n_samps = genos.shape
    n_pairs = n_samps * (n_samps - 1) // 2
    diffs = np.zeros(n_pairs, dtype=np.float64)

    pair_idx = 0
    for i in range(n_samps - 1):
        for j in range(i + 1, n_samps):
            diff_count = 0
            for k in range(n_snps):
                gi = genos[k, i]
                gj = genos[k, j]
                if 0 <= gi <= 2 and 0 <= gj <= 2:
                    if gi != gj:
                        diff_count += 1
            diffs[pair_idx] = diff_count
            pair_idx += 1

    return diffs


# Wrapper functions that match the C extension interface
def ZnS(r2_matrix):
    """Wrapper to match shicstats.ZnS interface."""
    return (zns(r2_matrix),)


def Omega(r2_matrix):
    """Wrapper to match shicstats.omega interface."""
    return (omega(r2_matrix),)


def pairwiseDiffs(haps):
    """Wrapper to match shicstats.pairwiseDiffs interface."""
    # Ensure contiguous array
    if not haps.flags['C_CONTIGUOUS']:
        haps = np.ascontiguousarray(haps)
    return pairwise_diffs(haps)


def pairwiseDiffsDiplo(genos):
    """Wrapper to match shicstats.pairwiseDiffsDiplo interface."""
    if not genos.flags['C_CONTIGUOUS']:
        genos = np.ascontiguousarray(genos)
    return pairwise_diffs_diplo(genos)


def computeR2Matrix(haps):
    """Wrapper to match shicstats.computeR2Matrix interface."""
    if not haps.flags['C_CONTIGUOUS']:
        haps = np.ascontiguousarray(haps)
    return compute_r2_matrix(haps)


@njit(cache=True)
def get_haplotype_freq_spec(haps):
    """
    Compute haplotype frequency spectrum.

    Groups samples into distinct haplotypes (treating -1 as matching any value),
    then returns the frequency spectrum.

    Parameters
    ----------
    haps : ndarray, shape (n_snps, n_samples)
        Haplotype/genotype matrix

    Returns
    -------
    hap_counts : ndarray, shape (n_samples + 1,)
        hap_counts[i-1] = number of haplotypes occurring exactly i times
        hap_counts[n_samples] = total number of unique haplotypes
    """
    n_snps, n_samps = haps.shape

    # Output array
    hap_counts = np.zeros(n_samps + 1, dtype=np.int32)

    # Track unique haplotypes and their occurrences
    # We'll store haplotypes as rows in a 2D array
    haplotypes = np.zeros((n_samps, n_snps), dtype=np.int32)
    haplotype_occurrences = np.zeros(n_samps, dtype=np.int32)
    n_haps = 0

    for i in range(n_samps):
        haplotype_found = False

        # Check against existing haplotypes
        for j in range(n_haps):
            allsame = True
            for k in range(n_snps):
                h_jk = haplotypes[j, k]
                h_ik = haps[k, i]
                # Match if same value, or if either is -1 (missing)
                if h_jk != h_ik and h_jk != -1 and h_ik != -1:
                    allsame = False
                    break

            if allsame:
                haplotype_found = True
                haplotype_occurrences[j] += 1
                break

        if not haplotype_found:
            # Add new haplotype
            for k in range(n_snps):
                haplotypes[n_haps, k] = haps[k, i]
            haplotype_occurrences[n_haps] = 1
            n_haps += 1

    # Build frequency spectrum
    for i in range(n_haps):
        freq = haplotype_occurrences[i]
        if 0 < freq <= n_samps:
            hap_counts[freq - 1] += 1

    # Store total number of unique haplotypes
    hap_counts[n_samps] = n_haps

    return hap_counts


def getHaplotypeFreqSpec(haps):
    """Wrapper to match shicstats.getHaplotypeFreqSpec interface."""
    if not haps.flags['C_CONTIGUOUS']:
        haps = np.ascontiguousarray(haps)
    return get_haplotype_freq_spec(haps)
