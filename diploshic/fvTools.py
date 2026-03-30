import allel
import sys
import math
from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim
from scipy.spatial.distance import squareform
# Import optimized Numba stats (replaces former C extension functions)
from diploshic.numba_stats import (
    zns as numba_zns,
    omega as numba_omega,
    pairwise_diffs as numba_pairwise_diffs,
    pairwise_diffs_diplo as numba_pairwise_diffs_diplo,
    getHaplotypeFreqSpec as numba_getHaplotypeFreqSpec,
)
import numpy as np
import random
import gzip
import scipy.stats


def _normalize_genotypes(gn):
    """
    Center and normalize genotype matrix for correlation computation.

    Returns normalized matrix and monomorphic site mask.
    """
    gn = np.asarray(gn, dtype=np.float64)
    n_snps, n_samples = gn.shape

    means = gn.mean(axis=1, keepdims=True)
    gn_centered = gn - means
    std = gn_centered.std(axis=1, keepdims=True, ddof=0)

    # Handle zero variance (monomorphic sites)
    mono_mask = (std.flatten() == 0)
    std_safe = np.where(std == 0, 1, std)
    gn_norm = gn_centered / std_safe

    return gn_norm, mono_mask, n_samples


def fast_r2_for_ld(gn):
    """
    Compute R (correlation) in condensed form using BLAS matrix multiplication.

    Fast replacement for allel.stats.ld.rogers_huff_r (5-10x speedup).
    """
    gn = np.asarray(gn, dtype=np.float64)
    if gn.shape[0] <= 1:
        return np.array([], dtype=np.float64)

    gn_norm, mono_mask, n_samples = _normalize_genotypes(gn)

    # Correlation matrix via BLAS matmul
    r = (gn_norm @ gn_norm.T) / n_samples
    r[mono_mask, :] = 0
    r[:, mono_mask] = 0

    return squareform(r, checks=False)


def fast_r2_matrix_diploid(gn):
    """
    Compute R² matrix for diploid data using BLAS matrix multiplication.

    Returns full symmetric R² matrix with diagonal zeroed.
    Faster than condensed form when you need the full matrix for ZnS/Omega.
    """
    gn = np.asarray(gn, dtype=np.float64)
    n_snps = gn.shape[0]
    if n_snps <= 1:
        return np.zeros((n_snps, n_snps), dtype=np.float64)

    gn_norm, mono_mask, n_samples = _normalize_genotypes(gn)

    # Correlation matrix via BLAS matmul
    r = (gn_norm @ gn_norm.T) / n_samples
    r[mono_mask, :] = 0
    r[:, mono_mask] = 0

    r2 = r ** 2
    np.fill_diagonal(r2, 0)
    return r2


def fast_zns_omega_diploid(gn):
    """Compute ZnS and Omega from diploid genotypes in a single pass.

    Computes the correlation matrix via BLAS, then:
    - ZnS: mean of off-diagonal r² (computed from Frobenius norm, no r² matrix needed)
    - Omega: from the r² matrix (computed in-place from r)

    For n_snps <= 1, returns (0, 0).
    For n_snps < 5, returns (ZnS, 0) since Omega needs at least 5 SNPs.

    Parameters
    ----------
    gn : array-like, shape (n_snps, n_individuals)
        Diploid genotype alt-allele counts (0, 1, or 2).

    Returns
    -------
    zns : float
    omega_val : float
    """
    gn = np.asarray(gn, dtype=np.float64)
    n_snps = gn.shape[0]
    if n_snps <= 1:
        return 0.0, 0.0

    gn_norm, mono_mask, n_samples = _normalize_genotypes(gn)

    # Correlation matrix via BLAS
    r = (gn_norm @ gn_norm.T) / n_samples
    r[mono_mask, :] = 0
    r[:, mono_mask] = 0

    # ZnS from Frobenius norm — no need to materialize r²
    # ZnS = mean of all elements in the r² matrix with zero diagonal
    # = (sum(r²) - sum(diag(r)²)) / n_snps²
    frob_sq = np.sum(r * r)
    diag_sq = np.sum(np.diag(r) ** 2)
    zns = (frob_sq - diag_sq) / (n_snps * n_snps)

    if n_snps < 5:
        return float(zns), 0.0

    # Omega needs the r² matrix — compute in-place
    np.square(r, out=r)
    np.fill_diagonal(r, 0)
    omega_val = numba_omega(r)

    return float(zns), float(omega_val)


def fast_r2_matrix_haploid(haps):
    """
    Compute R² matrix for haploid data using BLAS matrix multiplication.

    Handles missing data (-1) by computing per-pair statistics over jointly
    valid samples. Returns upper-triangular matrix with -1 for invalid pairs.
    """
    haps_arr = np.asarray(haps)
    n_snps, n_samples = haps_arr.shape

    if n_snps <= 1:
        return np.zeros((n_snps, n_snps), dtype=np.float64)

    has_missing = np.any(haps_arr < 0)

    if not has_missing:
        # Fast path: use standard correlation approach
        haps_norm, mono_mask, n_samples = _normalize_genotypes(haps_arr)
        r = (haps_norm @ haps_norm.T) / n_samples
        r2 = np.triu(r ** 2, k=1)
        r2[mono_mask, :] = -1.0
        r2[:, mono_mask] = -1.0
        return r2

    # Missing data path: compute per-pair statistics via matrix operations
    # V = validity mask, M = masked values (0 where missing)
    V = (haps_arr >= 0).astype(np.float64)
    M = np.where(haps_arr >= 0, haps_arr, 0).astype(np.float64)

    # Compute sums over jointly valid samples via BLAS matmul
    n = V @ V.T                    # n[i,j] = count of jointly valid samples
    S1 = M @ V.T                   # S1[i,j] = sum of SNP i over valid pairs
    S2 = S1.T                      # S2[i,j] = sum of SNP j over valid pairs
    SS = M @ M.T                   # SS[i,j] = sum of products

    # R² = (cov)² / (var_i * var_j), using Bernoulli variance formula
    # Scaled to avoid divisions: R² = (SS*n - S1*S2)² / (S1*(n-S1) * S2*(n-S2))
    cov_scaled = SS * n - S1 * S2
    var_i_scaled = S1 * (n - S1)
    var_j_scaled = S2 * (n - S2)
    denominator = var_i_scaled * var_j_scaled

    valid_mask = (n >= 2) & (var_i_scaled > 0) & (var_j_scaled > 0)

    r2 = np.full((n_snps, n_snps), -1.0, dtype=np.float64)
    r2[valid_mask] = (cov_scaled[valid_mask] ** 2) / denominator[valid_mask]

    return np.triu(r2, k=1)


def misPolarizeAlleleCounts(ac, pMisPol):
    pMisPolInv = 1 - pMisPol
    mapping = []
    for i in range(len(ac)):
        if random.random() >= pMisPolInv:
            mapping.append([1, 0])  # swap
        else:
            mapping.append([0, 1])  # no swap
    return ac.map_alleles(mapping)


def calledGenoFracAtSite(genosAtSite):
    # Vectorized: check if any allele in each individual is missing (< 0)
    has_missing = np.any(genosAtSite < 0, axis=1)
    return np.sum(~has_missing) / len(has_missing)


def isHaploidVcfGenoArray(genos):
    return all(0 > genos[:, :, 1].flat)


def diploidizeGenotypeArray(genos):
    numSnps, numSamples, numAlleles = genos.shape
    if numSamples % 2 != 0:
        sys.stderr.write(
            "Diploidizing an odd-numbered sample. The last genome will be truncated.\n"
        )
        numSamples -= 1
    newGenos = []
    for i in range(numSnps):
        currSnp = []
        for j in range(0, numSamples, 2):
            currSnp.append([genos[i, j, 0], genos[i, j + 1, 0]])
        newGenos.append(currSnp)
    newGenos = np.array(newGenos)
    return allel.GenotypeArray(newGenos)


# contains some bits modified from scikit-allel by Alistair Miles


def readStatsDafsComputeStandardizationBins(
    statAndDafFileName, nBins=50, pMisPol=0.0
):
    stats = {}
    dafs = []
    pMisPolInv = 1 - pMisPol
    misPolarizedSnps, totalSnps = 0, 0
    with open(statAndDafFileName) as statAndDafFile:
        first = True
        for line in statAndDafFile:
            line = line.strip().split()
            if first:
                first = False
                header = line
                assert header[0] == "daf"
                for i in range(1, len(header)):
                    stats[header[i]] = []
            else:
                totalSnps += 1
                if random.random() >= pMisPolInv:
                    dafs.append(1 - float(line[0]))
                    misPolarizedSnps += 1
                else:
                    dafs.append(float(line[0]))
                for i in range(1, len(line)):
                    stats[header[i]].append(float(line[i]))

    statInfo = {}
    for statName in stats.keys():
        stats[statName] = np.array(stats[statName])
        nonan = ~np.isnan(stats[statName])
        score_nonan = stats[statName][nonan]
        daf_nonan = np.array(dafs)[nonan]
        bins = allel.stats.selection.make_similar_sized_bins(daf_nonan, nBins)
        mean_score, _, _ = scipy.stats.binned_statistic(
            daf_nonan, score_nonan, statistic=np.mean, bins=bins
        )
        std_score, _, _ = scipy.stats.binned_statistic(
            daf_nonan, score_nonan, statistic=np.std, bins=bins
        )
        statInfo[statName] = (mean_score, std_score, bins)
        sys.stderr.write(
            "mispolarized %d of %d (%f%%) "
            "SNPs when standardizing scores in %s\n"
            % (
                misPolarizedSnps,
                totalSnps,
                100 * misPolarizedSnps / float(totalSnps),
                statAndDafFileName,
            )
        )
    return statInfo


# includes a snippet copied from scikit-allel


def standardize_by_allele_count_from_precomp_bins(
    score, dafs, standardizationInfo
):
    score_standardized = np.empty_like(score)
    mean_score, std_score, bins = standardizationInfo
    dafs = np.array(dafs)
    for i in range(len(bins) - 1):
        x1 = bins[i]
        x2 = bins[i + 1]
        if i == 0:
            # first bin
            loc = dafs < x2
        elif i == len(bins) - 2:
            # last bin
            loc = dafs >= x1
        else:
            # middle bins
            loc = (dafs >= x1) & (dafs < x2)
        m = mean_score[i]
        s = std_score[i]
        score_standardized[loc] = (score[loc] - m) / s
    return score_standardized


def readFaArm(armFileName, armName=False):
    if armFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(armFileName, "rt") as armFile:
        reading = False
        seq = ""
        for line in armFile:
            if line.startswith(">"):
                if armName:
                    if reading:
                        break
                    elif line.strip()[1:] == armName:
                        reading = True
                else:
                    assert not reading
                    reading = True
            elif reading:
                seq += line.strip()
    return seq


def polarizeSnps(unmasked, positions, refAlleles, altAlleles, ancArm):
    assert len(unmasked) == len(ancArm)
    assert len(positions) == len(refAlleles)
    assert len(positions) == len(altAlleles)
    isSnp = {}
    for i in range(len(positions)):
        isSnp[positions[i]] = i

    mapping = []
    for i in range(len(ancArm)):
        if ancArm[i] in "ACGT":
            if i + 1 in isSnp:
                ref, alt = refAlleles[isSnp[i + 1]], altAlleles[isSnp[i + 1]]
                if ancArm[i] == ref:
                    mapping.append([0, 1])  # no swap
                elif ancArm[i] == alt:
                    mapping.append([1, 0])  # swap
                else:
                    mapping.append([0, 1])  # no swap -- failed to polarize
                    unmasked[i] = False
        elif ancArm[i] == "N":
            unmasked[i] = False
            if i + 1 in isSnp:
                mapping.append([0, 1])  # no swap -- failed to polarize
        else:
            sys.exit(
                "Found a character in ancestral chromosome "
                "that is not 'A', 'C', 'G', 'T' or 'N' (all upper case)!\n"
            )
    assert len(mapping) == len(positions)
    return mapping, unmasked


def getAccessibilityInWins(isAccessibleArm, winLen, subWinLen, cutoff):
    wins = []
    badWinCount = 0
    lastWinEnd = len(isAccessibleArm) - len(isAccessibleArm) % winLen
    for i in range(0, lastWinEnd, winLen):
        currWin = isAccessibleArm[i : i + winLen]
        goodWin = True
        for subWinStart in range(0, winLen, subWinLen):
            unmaskedFrac = currWin[
                subWinStart : subWinStart + subWinLen
            ].count(True) / float(subWinLen)
            if unmaskedFrac < cutoff:
                goodWin = False
        if goodWin:
            wins.append(currWin)
        else:
            badWinCount += 1
    return wins


def windowVals(
    vals, subWinBounds, positionArray, keepNans=False, absVal=False
):
    assert len(vals) == len(positionArray)

    subWinIndex = 0
    winStart, winEnd = subWinBounds[subWinIndex]
    # windowedVals = [[]]
    windowedVals = [[] for x in range(len(subWinBounds))]
    for i in range(len(positionArray)):
        currPos = positionArray[i]
        while currPos > winEnd:
            subWinIndex += 1
            winStart, winEnd = subWinBounds[subWinIndex]
            # windowedVals.append([])
        assert currPos >= winStart and currPos <= winEnd
        if keepNans is True or not math.isnan(vals[i]):
            # windowedVals[-1].append(vals[i])
            windowedVals[subWinIndex].append(vals[i])
    assert len(windowedVals) == len(subWinBounds)
    if absVal:
        return [np.absolute(win) for win in windowedVals]
    else:
        return [np.array(win) for win in windowedVals]


def readFa(faFileName, upper=False):
    seqData = {}
    seq = ""
    if faFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(faFileName, "rt") as faFile:
        reading = False
        currChr = None
        for line in faFile:
            if line.startswith(">"):
                if reading:
                    if upper:
                        # AK: currChr, seq undefined
                        seqData[currChr] = seq.upper()
                    else:
                        # AK: currChr, seq undefined
                        seqData[currChr] = seq
                else:
                    reading = True
                currChr = line[1:].strip()
                seq = ""
            else:
                seq += line.strip()
    if upper:
        seqData[currChr] = seq.upper()
    else:
        seqData[currChr] = seq
    return seqData


def readMaskAndAncDataForTraining(
    maskFileName,
    ancFileName,
    totalPhysLen,
    subWinLen,
    chrArmsForMasking,
    shuffle=True,
    cutoff=0.25,
):
    isAccessible = []
    maskData, ancData = readFa(maskFileName, upper=True), readFa(
        ancFileName, upper=True
    )
    if "all" in chrArmsForMasking:
        chrArmsForMasking = sorted(maskData)
    for currChr in chrArmsForMasking:
        assert len(maskData[currChr]) == len(ancData[currChr])
        isAccessibleArm = []
        for i in range(len(maskData[currChr])):
            if "N" in [ancData[currChr][i], maskData[currChr][i]]:
                isAccessibleArm.append(False)
            else:
                isAccessibleArm.append(True)
        windowedAccessibility = getAccessibilityInWins(
            isAccessibleArm, totalPhysLen, subWinLen, cutoff
        )
        if windowedAccessibility:
            isAccessible += windowedAccessibility
    if shuffle:
        random.shuffle(isAccessible)
    count = 0
    for i in range(len(isAccessible)):
        assert len(isAccessible[i]) == totalPhysLen
        count += 1
    assert count
    return isAccessible


def getGenoMaskInfoInWins(
    isAccessibleArm,
    genos,
    positions,
    positions2SnpIndices,
    winLen,
    subWinLen,
    cutoff,
    genoCutoff,
):
    windowedAcc, windowedGenoMask = [], []
    badWinCount = 0
    lastWinEnd = len(isAccessibleArm) - len(isAccessibleArm) % winLen
    posIdx = 0
    snpIndicesInWins = []
    sys.stderr.write(
        "about to get geno masks from arm; "
        "len: %d, genos shape: %s, num snps: %d\n"
        % (len(isAccessibleArm), genos.shape, len(positions))
    )
    calledFracs = []
    for winOffset in range(0, lastWinEnd, winLen):
        firstPos = winOffset + 1
        lastPos = winOffset + winLen
        snpIndicesInWin = []
        assert (
            len(positions) == 0
            or posIdx >= len(positions)
            or positions[posIdx] >= firstPos
        )
        while posIdx < len(positions) and positions[posIdx] <= lastPos:
            if isAccessibleArm[positions[posIdx] - 1]:
                calledFrac = calledGenoFracAtSite(genos[posIdx])
                calledFracs.append(calledFrac)
                if calledFrac >= genoCutoff:
                    snpIndicesInWin.append(posIdx)
                else:
                    isAccessibleArm[positions[posIdx] - 1] = False
            posIdx += 1
        snpIndicesInWins.append(snpIndicesInWin)
    if len(calledFracs) > 0:
        sys.stderr.write(
            "min calledFrac: %g; max calledFrac: %g; "
            "mean: %g; median: %g\n"
            % (
                min(calledFracs),
                max(calledFracs),
                np.median(calledFracs),
                np.mean(calledFracs),
            )
        )
    else:
        sys.stderr.write("no SNPs in chromosome!\n")
    winIndex = 0
    for winOffset in range(0, lastWinEnd, winLen):
        currWin = isAccessibleArm[winOffset : winOffset + winLen]
        if len(snpIndicesInWins[winIndex]) > 0:
            currGenos = genos.subset(sel0=snpIndicesInWins[winIndex])
            goodWin = True
            for subWinStart in range(0, winLen, subWinLen):
                unmaskedFrac = currWin[
                    subWinStart : subWinStart + subWinLen
                ].count(True) / float(subWinLen)
                if unmaskedFrac < cutoff:
                    goodWin = False
            if goodWin:
                windowedAcc.append(currWin)
                windowedGenoMask.append(currGenos)
            else:
                badWinCount += 1
        winIndex += 1
    if windowedAcc:
        sys.stderr.write(
            "returning %d geno arrays, "
            "with an avg of %f snps\n"
            % (
                len(windowedGenoMask),
                sum(
                    [
                        len(windowedGenoMask[i])
                        for i in range(len(windowedGenoMask))
                    ]
                )
                / float(len(windowedGenoMask)),
            )
        )  # NOQA
    else:
        sys.stderr.write("returning 0 geno arrays\n")
    return windowedAcc, windowedGenoMask


def readSampleToPopFile(sampleToPopFileName):
    table = {}
    with open(sampleToPopFileName) as sampleToPopFile:
        for line in sampleToPopFile:
            sample, pop = line.strip().split()
            table[sample] = pop
    return table


def extractGenosAndPositionsForArm(
    vcfFile, chroms, currChr, sampleIndicesToKeep
):
    # sys.stderr.write("extracting vcf info for arm %s\n" %(currChr))

    rawgenos = np.take(
        vcfFile["calldata/GT"],
        [i for i in range(len(chroms)) if chroms[i] == currChr],
        axis=0,
    )  # NOQA
    if len(rawgenos) > 0:
        genos = allel.GenotypeArray(rawgenos).subset(sel1=sampleIndicesToKeep)
        if isHaploidVcfGenoArray(genos):
            sys.stderr.write(
                "Detected haploid input for %s. "
                "Converting into diploid individuals "
                "(combining haplotypes in order).\n" % (currChr)
            )
            genos = diploidizeGenotypeArray(genos)
            sys.stderr.write("Done diploidizing %s\n" % (currChr))
        positions = np.extract(chroms == currChr, vcfFile["variants/POS"])
        if len(positions) > 0:
            genos = allel.GenotypeArray(
                genos.subset(sel0=range(len(positions)))
            )

            positions2SnpIndices = {}
            for i in range(len(positions)):
                positions2SnpIndices[positions[i]] = i

            assert len(positions) == len(positions2SnpIndices) and len(
                positions
            ) == len(genos)
            return (
                genos,
                positions,
                positions2SnpIndices,
                genos.count_alleles().is_biallelic(),
            )  # NOQA
    return np.array([]), [], {}, np.array([])


def readMaskDataForTraining(
    maskFileName,
    totalPhysLen,
    subWinLen,
    chrArmsForMasking,
    shuffle=True,
    cutoff=0.25,
    genoCutoff=0.75,
    vcfForMaskFileName=None,
    sampleToPopFileName=None,
    pop=None,
):
    if vcfForMaskFileName:
        sys.stderr.write(
            "reading geno mask info from %s\n" % (vcfForMaskFileName)
        )
        vcfFile = allel.read_vcf(vcfForMaskFileName)
        sys.stderr.write("done with read\n")
        chroms = vcfFile["variants/CHROM"]
        samples = vcfFile["samples"]
        if sampleToPopFileName:
            sampleToPop = readSampleToPopFile(sampleToPopFileName)
            sampleIndicesToKeep = [
                i
                for i in range(len(samples))
                if sampleToPop.get(samples[i], "popNotFound!") == pop
            ]  # NOQA
        else:
            sampleIndicesToKeep = [i for i in range(len(samples))]
    if maskFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open

    genosChecked = 0
    sys.stderr.write("reading %s\n" % (maskFileName))
    readingMasks = False
    isAccessible, isAccessibleArm = [], []
    genoMaskInfo = []
    currChr = None
    genos = None
    positions = None
    positions2SnpIndices = None
    with fopen(maskFileName, "rt") as maskFile:
        for line in maskFile:
            if line.startswith(">"):
                if readingMasks and len(isAccessibleArm) >= totalPhysLen:
                    if vcfForMaskFileName:
                        sys.stderr.write(
                            "processing sites "
                            "and genos for %s\n" % (currChr)
                        )
                        (
                            windowedAccessibility,
                            windowedGenoMask,
                        ) = getGenoMaskInfoInWins(
                            isAccessibleArm,
                            genos,
                            positions,
                            positions2SnpIndices,
                            totalPhysLen,
                            subWinLen,
                            cutoff,
                            genoCutoff,
                        )
                        if windowedAccessibility:
                            isAccessible += windowedAccessibility
                            genoMaskInfo += windowedGenoMask
                    else:
                        windowedAccessibility = getAccessibilityInWins(
                            isAccessibleArm, totalPhysLen, subWinLen, cutoff
                        )
                        if windowedAccessibility:
                            isAccessible += windowedAccessibility

                currChr = line[1:].strip()
                currPos = 0
                # sys.stderr.write("chrom: " + currChr + "\n")
                if "all" in chrArmsForMasking or currChr in chrArmsForMasking:
                    readingMasks = True
                else:
                    readingMasks = False
                isAccessibleArm = []
                if vcfForMaskFileName and readingMasks:
                    sys.stderr.write(
                        "checking geno mask "
                        "info from %s for %s\n" % (vcfForMaskFileName, currChr)
                    )
                    (
                        genos,
                        positions,
                        positions2SnpIndices,
                        isBiallelic,
                    ) = extractGenosAndPositionsForArm(
                        vcfFile, chroms, currChr, sampleIndicesToKeep
                    )
            else:
                if readingMasks:
                    for char in line.strip().upper():
                        if char == "N":
                            isAccessibleArm.append(False)
                        elif (
                            vcfForMaskFileName
                            and currPos in positions2SnpIndices
                        ):
                            genosChecked += 1
                            if (
                                isBiallelic[positions2SnpIndices[currPos]]
                                and calledGenoFracAtSite(
                                    genos[positions2SnpIndices[currPos]]
                                )
                                >= genoCutoff
                            ):  # NOQA
                                isAccessibleArm.append(True)
                            else:
                                isAccessibleArm.append(False)
                        else:
                            isAccessibleArm.append(True)
                        currPos += 1
    if readingMasks and len(isAccessibleArm) >= totalPhysLen:
        if vcfForMaskFileName:
            sys.stderr.write("processing sites and genos for %s\n" % (currChr))
            windowedAccessibility, windowedGenoMask = getGenoMaskInfoInWins(
                isAccessibleArm,
                genos,
                positions,
                positions2SnpIndices,
                totalPhysLen,
                subWinLen,
                cutoff,
                genoCutoff,
            )
            if windowedAccessibility:
                isAccessible += windowedAccessibility
                genoMaskInfo += windowedGenoMask
        else:
            windowedAccessibility = getAccessibilityInWins(
                isAccessibleArm, totalPhysLen, subWinLen, cutoff
            )
            if windowedAccessibility:
                isAccessible += windowedAccessibility
    if shuffle:
        if vcfForMaskFileName:
            indices = np.array([i for i in range(len(isAccessible))])
            np.random.shuffle(indices)
            isAccessible = [isAccessible[i] for i in indices]
            genoMaskInfo = [genoMaskInfo[i] for i in indices]
        else:
            random.shuffle(isAccessible)

    if len(isAccessible) == 0:
        sys.exit(
            "Error: Couldn't find a single window in our "
            "real data for masking that survived filters. May have to "
            "disable masking.\n"
        )
    for i in range(len(isAccessible)):
        assert len(isAccessible[i]) == totalPhysLen
    sys.stderr.write("checked genotypes at %d sites\n" % (genosChecked))
    if vcfForMaskFileName:
        return isAccessible, genoMaskInfo
    else:
        return isAccessible


def maskGeno():
    return np.array([-1, -1])


def isMaskedGeno(genoMask):
    for allele in genoMask:
        if allele < 0:
            return True
    return False


def maskGenos(genosInWin, genoMaskForWin):
    for snpIndex in range(len(genosInWin)):
        # if we run out of snps we just bring it around for another pass!
        maskIndex = snpIndex % len(genoMaskForWin)
        for j in range(len(genosInWin[snpIndex])):
            if isMaskedGeno(genoMaskForWin[maskIndex, j]):
                genosInWin[snpIndex, j] = maskGeno()
    return genosInWin


def readMaskDataForScan(maskFileName, chrArm):
    isAccessible = []
    readingMasks = False
    if maskFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(maskFileName, "rt") as maskFile:
        for line in maskFile:
            if line.startswith(">"):
                currChr = line[1:].strip()
                if currChr == chrArm:
                    readingMasks = True
                elif readingMasks:
                    break
            else:
                if readingMasks:
                    for char in line.strip().upper():
                        if char == "N":
                            isAccessible.append(False)
                        else:
                            isAccessible.append(True)
    return isAccessible


def normalizeFeatureVec(statVec):
    statVec = np.asarray(statVec, dtype=np.float64)
    min_val = statVec.min()
    if min_val < 0:
        statVec = statVec - min_val
    stat_sum = statVec.sum()
    if stat_sum == 0 or np.any(np.isinf(statVec)) or np.any(np.isnan(statVec)):
        return list(np.full(len(statVec), 1.0 / len(statVec)))
    return list(statVec / stat_sum)


def maxFDA(pos, ac, start=None, stop=None, is_accessible=None):
    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    ac = asarray_ndim(ac, 2)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # deal with subregion
    if start is not None or stop is not None:
        loc = pos.locate_range(start, stop)
        pos = pos[loc]
        ac = ac[loc]
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]

    # calculate values of the stat
    dafs = []
    for i in range(len(ac)):
        p1 = ac[i, 1]
        n = p1 + ac[i, 0]
        dafs.append(p1 / float(n))
    return max(dafs)


def calcAllStatsForSubWin(
    alleleCounts,
    snpLocs,
    subWinStart,
    subWinEnd,
    statVals,
    instanceIndex,
    subWinIndex,
    hapsInSubWin,
    unmasked,
    precomputedStats,
):
    """
    Compute all haploid stats for a subwindow in one pass.

    This is an optimized replacement for calling calcAndAppendStatVal once per stat.
    By computing all stats in a single function call, we eliminate:
    - 14 of 15 function call overhead
    - String-based stat dispatch (if/elif chain)
    - Repeated dictionary lookups

    The stats computed are: pi, thetaW, tajD, thetaH, fayWuH, maxFDA, HapCount,
    H1, H12, H2/H1, ZnS, Omega, distVar, distSkew, distKurt
    """
    # Pre-lookup all storage lists to avoid repeated dict access
    pi_list = statVals["pi"][instanceIndex]
    thetaW_list = statVals["thetaW"][instanceIndex]
    tajD_list = statVals["tajD"][instanceIndex]
    thetaH_list = statVals["thetaH"][instanceIndex]
    fayWuH_list = statVals["fayWuH"][instanceIndex]
    maxFDA_list = statVals["maxFDA"][instanceIndex]
    HapCount_list = statVals["HapCount"][instanceIndex]
    H1_list = statVals["H1"][instanceIndex]
    H12_list = statVals["H12"][instanceIndex]
    H2H1_list = statVals["H2/H1"][instanceIndex]
    ZnS_list = statVals["ZnS"][instanceIndex]
    Omega_list = statVals["Omega"][instanceIndex]
    distVar_list = statVals["distVar"][instanceIndex]
    distSkew_list = statVals["distSkew"][instanceIndex]
    distKurt_list = statVals["distKurt"][instanceIndex]

    # Compute diversity stats
    pi_val = allel.stats.diversity.sequence_diversity(
        snpLocs, alleleCounts, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
    )
    pi_list.append(pi_val)

    thetaW_list.append(
        allel.stats.diversity.watterson_theta(
            snpLocs, alleleCounts, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
        )
    )

    tajD_list.append(
        allel.stats.diversity.tajima_d(alleleCounts, pos=snpLocs, start=subWinStart, stop=subWinEnd)
    )

    thetaH_val = thetah(snpLocs, alleleCounts, start=subWinStart, stop=subWinEnd, is_accessible=unmasked)
    thetaH_list.append(thetaH_val)

    # fayWuH depends on thetaH and pi
    fayWuH_list.append(thetaH_val - pi_val)

    # maxFDA
    maxFDA_list.append(
        maxFDA(snpLocs, alleleCounts, start=subWinStart, stop=subWinEnd, is_accessible=unmasked)
    )

    # Haplotype stats
    HapCount_list.append(len(hapsInSubWin.distinct()))

    # Garud's H stats (computed together)
    h1, h12, h123, h21 = allel.stats.selection.garud_h(hapsInSubWin)
    H1_list.append(h1)
    H12_list.append(h12)
    H2H1_list.append(h21)

    # LD stats (ZnS and Omega computed together)
    r2Matrix = fast_r2_matrix_haploid(hapsInSubWin)
    ZnS_list.append(numba_zns(r2Matrix))
    Omega_list.append(numba_omega(r2Matrix))

    # Pairwise distance distribution stats (computed together)
    unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
    if hasattr(unmasked_slice, 'count'):
        n_unmasked = unmasked_slice.count(True)
    else:
        n_unmasked = np.sum(unmasked_slice)
    dists = numba_pairwise_diffs(np.ascontiguousarray(hapsInSubWin)) / float(n_unmasked)
    distVar_list.append(np.var(dists, ddof=1))
    distSkew_list.append(scipy.stats.skew(dists))
    distKurt_list.append(scipy.stats.kurtosis(dists))


def calcAndAppendStatVal(
    alleleCounts,
    snpLocs,
    statName,
    subWinStart,
    subWinEnd,
    statVals,
    instanceIndex,
    subWinIndex,
    hapsInSubWin,
    unmasked,
    precomputedStats,
):
    if statName == "tajD":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.tajima_d(  # NOQA
                alleleCounts, pos=snpLocs, start=subWinStart, stop=subWinEnd
            )
        )
    elif statName == "pi":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.sequence_diversity(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaW":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.watterson_theta(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaH":
        statVals[statName][instanceIndex].append(
            thetah(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "fayWuH":
        statVals[statName][instanceIndex].append(
            statVals["thetaH"][instanceIndex][subWinIndex]
            - statVals["pi"][instanceIndex][subWinIndex]
        )
    elif statName == "HapCount":
        statVals[statName][instanceIndex].append(len(hapsInSubWin.distinct()))
    elif statName == "maxFDA":
        statVals[statName][instanceIndex].append(
            maxFDA(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "H1":
        h1, h12, h123, h21 = allel.stats.selection.garud_h(hapsInSubWin)
        statVals["H1"][instanceIndex].append(h1)
        if "H12" in statVals:
            statVals["H12"][instanceIndex].append(h12)
        if "H123" in statVals:
            statVals["H123"][instanceIndex].append(h123)
        if "H2/H1" in statVals:
            statVals["H2/H1"][instanceIndex].append(h21)
    elif statName == "ZnS":
        r2Matrix = fast_r2_matrix_haploid(hapsInSubWin)
        statVals["ZnS"][instanceIndex].append(numba_zns(r2Matrix))
        statVals["Omega"][instanceIndex].append(numba_omega(r2Matrix))
    elif statName == "RH":
        rMatrixFlat = fast_r2_for_ld(
            hapsInSubWin.to_genotypes(ploidy=2).to_n_alt()
        )
        rhAvg = rMatrixFlat.mean()
        statVals["RH"][instanceIndex].append(rhAvg)
        r2Matrix = squareform(rMatrixFlat ** 2)
        statVals["Omega"][instanceIndex].append(numba_omega(r2Matrix))
    elif statName == "iHSMean":
        vals = [
            x
            for x in precomputedStats["iHS"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            statVals["iHSMean"][instanceIndex].append(0.0)
        else:
            statVals["iHSMean"][instanceIndex].append(
                sum(vals) / float(len(vals))
            )
    elif statName == "nSLMean":
        vals = [
            x
            for x in precomputedStats["nSL"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            statVals["nSLMean"][instanceIndex].append(0.0)
        else:
            statVals["nSLMean"][instanceIndex].append(
                sum(vals) / float(len(vals))
            )
    elif statName == "iHSMax":
        vals = [
            x
            for x in precomputedStats["iHS"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            maxVal = 0.0
        else:
            maxVal = max(vals)
        statVals["iHSMax"][instanceIndex].append(maxVal)
    elif statName == "nSLMax":
        vals = [
            x
            for x in precomputedStats["nSL"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            maxVal = 0.0
        else:
            maxVal = max(vals)
        statVals["nSLMax"][instanceIndex].append(maxVal)
    elif statName == "iHSOutFrac":
        statVals["iHSOutFrac"][instanceIndex].append(
            getOutlierFrac(precomputedStats["iHS"][subWinIndex])
        )
    elif statName == "nSLOutFrac":
        statVals["nSLOutFrac"][instanceIndex].append(
            getOutlierFrac(precomputedStats["nSL"][subWinIndex])
        )
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = numba_pairwise_diffs(np.ascontiguousarray(hapsInSubWin)) / float(n_unmasked)
        statVals["distVar"][instanceIndex].append(np.var(dists, ddof=1))
        statVals["distSkew"][instanceIndex].append(scipy.stats.skew(dists))
        statVals["distKurt"][instanceIndex].append(scipy.stats.kurtosis(dists))
    elif statName in [
        "H12",
        "H123",
        "H2/H1",
        "Omega",
        "distVar",
        "distSkew",
        "distKurt",
    ]:
        assert len(statVals[statName][instanceIndex]) == subWinIndex + 1


def calcAndAppendStatValDiplo(
    alleleCounts,
    snpLocs,
    statName,
    subWinStart,
    subWinEnd,
    statVals,
    instanceIndex,
    subWinIndex,
    genosInSubWin,
    unmasked,
    genosNAlt=None,
):
    if genosNAlt is None:
        genosNAlt = genosInSubWin.to_n_alt()
    if statName == "tajD":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.tajima_d(
                alleleCounts, pos=snpLocs, start=subWinStart, stop=subWinEnd
            )
        )
    elif statName == "pi":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.sequence_diversity(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaW":
        statVals[statName][instanceIndex].append(
            allel.stats.diversity.watterson_theta(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaH":
        statVals[statName][instanceIndex].append(
            thetah(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "fayWuH":
        statVals[statName][instanceIndex].append(
            statVals["thetaH"][instanceIndex][subWinIndex]
            - statVals["pi"][instanceIndex][subWinIndex]
        )
    elif statName == "HapCount":
        statVals[statName][instanceIndex].append(len(genosInSubWin.distinct()))
    elif statName == "nDiplos":
        diplotypeCounts = numba_getHaplotypeFreqSpec(genosNAlt)
        nDiplos = diplotypeCounts[genosNAlt.shape[1]]
        statVals["nDiplos"][instanceIndex].append(nDiplos)
        diplotypeCounts = diplotypeCounts[:-1]
        dh1 = garudH1(diplotypeCounts)
        dh2 = garudH2(diplotypeCounts)
        dh12 = garudH12(diplotypeCounts)
        if "diplo_H1" in statVals:
            statVals["diplo_H1"][instanceIndex].append(dh1)
        if "diplo_H12" in statVals:
            statVals["diplo_H12"][instanceIndex].append(dh12)
        if "diplo_H2/H1" in statVals:
            statVals["diplo_H2/H1"][instanceIndex].append(dh2 / dh1)
    elif statName == "diplo_ZnS":
        if genosNAlt.shape[0] == 1:
            statVals["diplo_ZnS"][instanceIndex].append(0.0)
            statVals["diplo_Omega"][instanceIndex].append(0.0)
        else:
            zns_val, omega_val = fast_zns_omega_diploid(genosNAlt)
            statVals["diplo_ZnS"][instanceIndex].append(zns_val)
            statVals["diplo_Omega"][instanceIndex].append(omega_val)
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = numba_pairwise_diffs_diplo(np.ascontiguousarray(genosNAlt)) / float(n_unmasked)
        statVals["distVar"][instanceIndex].append(np.var(dists, ddof=1))
        statVals["distSkew"][instanceIndex].append(scipy.stats.skew(dists))
        statVals["distKurt"][instanceIndex].append(scipy.stats.kurtosis(dists))
    elif statName in [
        "diplo_H12",
        "diplo_H123",
        "diplo_H2/H1",
        "distVar",
        "distSkew",
        "distKurt",
        "diplo_Omega",
    ]:
        if not len(statVals[statName][instanceIndex]) == subWinIndex + 1:
            print(statName, instanceIndex, subWinIndex + 1)
            print(
                statVals["diplo_H1"][instanceIndex],
                statVals["diplo_H12"][instanceIndex],
            )
            sys.exit()


def calcAndAppendStatValForScanDiplo(
    alleleCounts,
    snpLocs,
    statName,
    subWinStart,
    subWinEnd,
    statVals,
    subWinIndex,
    genosInSubWin,
    unmasked,
    genosNAlt=None,
):
    if genosNAlt is None:
        genosNAlt = genosInSubWin.to_n_alt()
    if statName == "tajD":
        statVals[statName].append(
            allel.stats.diversity.tajima_d(
                alleleCounts, pos=snpLocs, start=subWinStart, stop=subWinEnd
            )
        )
    elif statName == "pi":
        statVals[statName].append(
            allel.stats.diversity.sequence_diversity(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaW":
        statVals[statName].append(
            allel.stats.diversity.watterson_theta(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "thetaH":
        statVals[statName].append(
            thetah(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "fayWuH":
        statVals[statName].append(
            statVals["thetaH"][subWinIndex] - statVals["pi"][subWinIndex]
        )
    elif statName == "HapCount":
        # AK: undefined variables
        statVals[statName].append(len(genosInSubWin.distinct()))
    elif statName == "nDiplos":
        diplotypeCounts = numba_getHaplotypeFreqSpec(genosNAlt)
        nDiplos = diplotypeCounts[genosNAlt.shape[1]]
        statVals["nDiplos"].append(nDiplos)
        diplotypeCounts = diplotypeCounts[:-1]
        dh1 = garudH1(diplotypeCounts)
        dh2 = garudH2(diplotypeCounts)
        dh12 = garudH12(diplotypeCounts)
        if "diplo_H1" in statVals:
            statVals["diplo_H1"].append(dh1)
        if "diplo_H12" in statVals:
            statVals["diplo_H12"].append(dh12)
        if "diplo_H2/H1" in statVals:
            statVals["diplo_H2/H1"].append(dh2 / dh1)
    elif statName == "diplo_ZnS":
        if genosNAlt.shape[0] == 1:
            statVals["diplo_ZnS"].append(0.0)
            statVals["diplo_Omega"].append(0.0)
        else:
            # VCF scan uses condensed r (not r²) for ZnS — preserves legacy behavior
            r2Matrix = fast_r2_for_ld(genosNAlt)
            r2Matrix2 = squareform(r2Matrix ** 2)
            statVals["diplo_ZnS"].append(np.nanmean(r2Matrix))
            statVals["diplo_Omega"].append(numba_omega(r2Matrix2))
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = numba_pairwise_diffs_diplo(np.ascontiguousarray(genosNAlt)) / float(n_unmasked)
        statVals["distVar"].append(np.var(dists, ddof=1))
        statVals["distSkew"].append(scipy.stats.skew(dists))
        statVals["distKurt"].append(scipy.stats.kurtosis(dists))
    elif statName in [
        "diplo_H12",
        "diplo_H123",
        "diplo_H2/H1",
        "distVar",
        "distSkew",
        "distKurt",
        "diplo_Omega",
    ]:
        if not len(statVals[statName]) == subWinIndex + 1:
            print(statName, subWinIndex + 1)
            print(statVals["diplo_H1"], statVals["diplo_H12"])
            sys.exit()


def getOutlierFrac(vals, cutoff=2.0):
    if len(vals) == 0:
        return 0.0
    else:
        num, denom = 0, 0
        for val in vals:
            assert val >= 0
            if not math.isnan(val):
                denom += 1
                if val > cutoff:
                    num += 1
        if denom == 0:
            return 0.0
        else:
            return num / float(denom)


def appendAllStatsForMonomorphic(statVals, instanceIndex):
    """
    Append values for all haploid stats when a subwindow has no SNPs.

    This is an optimized batched version of appendStatValsForMonomorphic.
    """
    statVals["pi"][instanceIndex].append(0.0)
    statVals["thetaW"][instanceIndex].append(0.0)
    statVals["tajD"][instanceIndex].append(0.0)
    statVals["thetaH"][instanceIndex].append(0.0)
    statVals["fayWuH"][instanceIndex].append(0.0)
    statVals["maxFDA"][instanceIndex].append(0.0)
    statVals["HapCount"][instanceIndex].append(1)
    statVals["H1"][instanceIndex].append(1.0)
    statVals["H12"][instanceIndex].append(1.0)
    statVals["H2/H1"][instanceIndex].append(0.0)
    statVals["ZnS"][instanceIndex].append(0.0)
    statVals["Omega"][instanceIndex].append(0.0)
    statVals["distVar"][instanceIndex].append(0.0)
    statVals["distSkew"][instanceIndex].append(0.0)
    statVals["distKurt"][instanceIndex].append(0.0)


def appendStatValsForMonomorphic(
    statName, statVals, instanceIndex, subWinIndex
):
    if statName == "tajD":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "pi":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "thetaW":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "thetaH":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "fayWuH":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "maxFDA":
        statVals[statName][instanceIndex].append(0.0)
    elif statName == "nDiplos":
        statVals[statName][instanceIndex].append(1)
    elif statName in ["diplo_H1"]:
        statVals["diplo_H1"][instanceIndex].append(1.0)
        if "diplo_H12" in statVals:
            statVals["diplo_H12"][instanceIndex].append(1.0)
        if "diplo_H123" in statVals:
            statVals["diplo_H123"][instanceIndex].append(1.0)
        if "diplo_H2/H1" in statVals:
            statVals["diplo_H2/H1"][instanceIndex].append(0.0)
    elif statName == "diplo_ZnS":
        statVals["diplo_ZnS"][instanceIndex].append(0.0)
        statVals["diplo_Omega"][instanceIndex].append(0.0)
    elif statName == "HapCount":
        statVals[statName][instanceIndex].append(1)
    elif statName in ["H1"]:
        statVals["H1"][instanceIndex].append(1.0)
        if "H12" in statVals:
            statVals["H12"][instanceIndex].append(1.0)
        if "H123" in statVals:
            statVals["H123"][instanceIndex].append(1.0)
        if "H2/H1" in statVals:
            statVals["H2/H1"][instanceIndex].append(0.0)
    elif statName == "ZnS":
        statVals["ZnS"][instanceIndex].append(0.0)
        statVals["Omega"][instanceIndex].append(0.0)
    elif statName == "RH":
        statVals["RH"][instanceIndex].append(0.0)
        statVals["Omega"][instanceIndex].append(0.0)
    elif statName == "iHSMean":
        statVals["iHSMean"][instanceIndex].append(0.0)
    elif statName == "nSLMean":
        statVals["nSLMean"][instanceIndex].append(0.0)
    elif statName == "iHSMax":
        statVals["iHSMax"][instanceIndex].append(0.0)
    elif statName == "nSLMax":
        statVals["nSLMax"][instanceIndex].append(0.0)
    elif statName in [
        "H12",
        "H123",
        "H2/H1",
        "diplo_H12",
        "diplo_H123",
        "diplo_H2/H1",
        "Omega",
        "diplo_Omega",
    ]:
        # print(statName, statVals[statName][instanceIndex], subWinIndex+1)
        assert len(statVals[statName][instanceIndex]) == subWinIndex + 1
    else:
        statVals[statName][instanceIndex].append(0.0)


def calcAndAppendStatValForScan(
    alleleCounts,
    snpLocs,
    statName,
    subWinStart,
    subWinEnd,
    statVals,
    subWinIndex,
    hapsInSubWin,
    unmasked,
    precomputedStats,
):
    if statName == "tajD":
        statVals[statName].append(
            allel.stats.diversity.tajima_d(
                alleleCounts, pos=snpLocs, start=subWinStart, stop=subWinEnd
            )
        )
    elif statName == "pi":
        statVals[statName].append(
            allel.stats.diversity.sequence_diversity(  # NOQA
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )  # NOQA
    elif statName == "thetaW":
        statVals[statName].append(
            allel.stats.diversity.watterson_theta(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )  # NOQA
    elif statName == "thetaH":
        statVals[statName].append(
            thetah(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )  # NOQA
    elif statName == "fayWuH":
        statVals[statName].append(
            statVals["thetaH"][subWinIndex] - statVals["pi"][subWinIndex]
        )
    elif statName == "maxFDA":
        # AK: undefined variables
        statVals[statName].append(
            maxFDA(
                snpLocs,
                alleleCounts,
                start=subWinStart,
                stop=subWinEnd,
                is_accessible=unmasked,
            )
        )
    elif statName == "HapCount":
        statVals[statName].append(len(hapsInSubWin.distinct()))
    elif statName == "H1":
        h1, h12, h123, h21 = allel.stats.selection.garud_h(hapsInSubWin)
        statVals["H1"].append(h1)
        if "H12" in statVals:
            statVals["H12"].append(h12)
        if "H123" in statVals:
            statVals["H123"].append(h123)
        if "H2/H1" in statVals:
            statVals["H2/H1"].append(h21)
    elif statName == "ZnS":
        r2Matrix = fast_r2_matrix_haploid(hapsInSubWin)
        statVals["ZnS"].append(numba_zns(r2Matrix))
        statVals["Omega"].append(numba_omega(r2Matrix))
    elif statName == "RH":
        rMatrixFlat = fast_r2_for_ld(
            hapsInSubWin.to_genotypes(ploidy=2).to_n_alt()
        )
        rhAvg = rMatrixFlat.mean()
        statVals["RH"].append(rhAvg)
        r2Matrix = squareform(rMatrixFlat ** 2)
        statVals["Omega"].append(numba_omega(r2Matrix))
    elif statName == "iHSMean":
        vals = [
            x
            for x in precomputedStats["iHS"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            statVals["iHSMean"].append(0.0)
        else:
            statVals["iHSMean"].append(sum(vals) / float(len(vals)))
    elif statName == "nSLMean":
        vals = [
            x
            for x in precomputedStats["nSL"][subWinIndex]
            if not (math.isnan(x) or math.isnan(x))
        ]
        if len(vals) == 0:
            statVals["nSLMean"].append(0.0)
        else:
            statVals["nSLMean"].append(sum(vals) / float(len(vals)))
    elif statName == "iHSMax":
        vals = [
            x
            for x in precomputedStats["iHS"][subWinIndex]
            if not (math.isnan(x) or math.isinf(x))
        ]
        if len(vals) == 0:
            maxVal = 0.0
        else:
            maxVal = max(vals)
        statVals["iHSMax"].append(maxVal)
    elif statName == "nSLMax":
        vals = [
            x
            for x in precomputedStats["nSL"][subWinIndex]
            if not (math.isnan(x) or math.isnan(x))
        ]
        if len(vals) == 0:
            maxVal = 0.0
        else:
            maxVal = max(vals)
        statVals["nSLMax"].append(maxVal)
    elif statName == "iHSOutFrac":
        statVals["iHSOutFrac"].append(
            getOutlierFrac(precomputedStats["iHS"][subWinIndex])
        )
    elif statName == "nSLOutFrac":
        statVals["nSLOutFrac"].append(
            getOutlierFrac(precomputedStats["nSL"][subWinIndex])
        )
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = numba_pairwise_diffs(np.ascontiguousarray(hapsInSubWin)) / float(n_unmasked)
        statVals["distVar"].append(np.var(dists, ddof=1))
        statVals["distSkew"].append(scipy.stats.skew(dists))
        statVals["distKurt"].append(scipy.stats.kurtosis(dists))
    elif statName in [
        "H12",
        "H123",
        "H2/H1",
        "Omega",
        "distVar",
        "distSkew",
        "distKurt",
    ]:
        assert len(statVals[statName]) == subWinIndex + 1


def appendStatValsForMonomorphicForScan(statName, statVals, subWinIndex):
    if statName == "tajD":
        statVals[statName].append(0.0)
    elif statName == "pi":
        statVals[statName].append(0.0)
    elif statName == "thetaW":
        statVals[statName].append(0.0)
    elif statName == "thetaH":
        statVals[statName].append(0.0)
    elif statName == "fayWuH":
        statVals[statName].append(0.0)
    elif statName == "maxFDA":
        statVals[statName].append(0.0)
    elif statName == "nDiplos":
        statVals[statName].append(1)
    elif statName in ["diplo_H1"]:
        statVals["diplo_H1"].append(1.0)
        if "diplo_H12" in statVals:
            statVals["diplo_H12"].append(1.0)
        if "diplo_H123" in statVals:
            statVals["diplo_H123"].append(1.0)
        if "diplo_H2/H1" in statVals:
            statVals["diplo_H2/H1"].append(0.0)
    elif statName == "diplo_ZnS":
        statVals["diplo_ZnS"].append(0.0)
        statVals["diplo_Omega"].append(0.0)
    elif statName == "HapCount":
        statVals[statName].append(1)
    elif statName in ["H1"]:
        statVals["H1"].append(1.0)
        if "H12" in statVals:
            statVals["H12"].append(1.0)
        if "H123" in statVals:
            statVals["H123"].append(1.0)
        if "H2/H1" in statVals:
            statVals["H2/H1"].append(0.0)
    elif statName == "ZnS":
        statVals["ZnS"].append(0.0)
        statVals["Omega"].append(0.0)
    elif statName == "RH":
        statVals["RH"].append(0.0)
        statVals["Omega"].append(0.0)
    elif statName == "iHSMean":
        statVals["iHSMean"].append(0.0)
    elif statName == "nSLMean":
        statVals["nSLMean"].append(0.0)
    elif statName == "iHSMax":
        statVals["iHSMax"].append(0.0)
    elif statName == "nSLMax":
        statVals["nSLMax"].append(0.0)
    elif statName in [
        "H12",
        "H123",
        "H2/H1",
        "diplo_H12",
        "diplo_H123",
        "diplo_H2/H1",
        "Omega",
        "diplo_Omega",
    ]:
        # print(statName, statVals[statName][instanceIndex], subWinIndex+1)
        assert len(statVals[statName]) == subWinIndex + 1
    else:
        statVals[statName].append(0.0)


"""
WARNING: this code assumes that the second column of ac gives the derived alleles;
please ensure that this is the case (and that you are using polarized data) if
are going to use values of this statistic for the classifier!!
"""  # NOQA


def thetah(pos, ac, start=None, stop=None, is_accessible=None):
    # check inputs
    if not isinstance(pos, SortedIndex):
        pos = SortedIndex(pos, copy=False)
    ac = asarray_ndim(ac, 2)
    is_accessible = asarray_ndim(is_accessible, 1, allow_none=True)

    # deal with subregion
    if start is not None or stop is not None:
        loc = pos.locate_range(start, stop)
        pos = pos[loc]
        ac = ac[loc]
    if start is None:
        start = pos[0]
    if stop is None:
        stop = pos[-1]

    # calculate values of the stat
    h = 0
    for i in range(len(ac)):
        p1 = ac[i, 1]
        n = p1 + ac[i, 0]
        if n > 1:
            h += (p1 * p1) / (n * (n - 1.0))
    h *= 2

    # calculate value per base
    if is_accessible is None:
        n_bases = stop - start + 1
    else:
        n_bases = np.count_nonzero(is_accessible[start - 1 : stop])

    h = h / n_bases
    return h


def garudH1(hapCounts):
    h1 = 0.0

    for hapFreq in range(len(hapCounts), 0, -1):
        pi = hapFreq / float(len(hapCounts))
        h1 += hapCounts[hapFreq - 1] * pi * pi

    return h1


def garudH2(hapCounts):
    h2 = 0.0
    first = True

    for hapFreq in range(len(hapCounts), 0, -1):
        pi = hapFreq / float(len(hapCounts))
        if hapCounts[hapFreq - 1] > 0:
            if first:
                first = False
                h2 += (hapCounts[hapFreq - 1] - 1) * pi * pi
            else:
                h2 += hapCounts[hapFreq - 1] * pi * pi

    return h2


def garudH12(hapCounts):
    part1, part2 = 0.0, 0.0
    totalAdded = 0

    for hapFreq in range(len(hapCounts), 0, -1):
        pi = hapFreq / float(len(hapCounts))
        for i in range(hapCounts[hapFreq - 1]):
            if totalAdded < 2:
                part1 += pi
            else:
                part2 += pi * pi
            totalAdded += 1

    part1 = part1 * part1

    return part1 + part2


# ---------------------------------------------------------------------------
# FAST-NN-inspired features: DAF histograms and inter-SNP distance summaries
# ---------------------------------------------------------------------------

def compute_daf_histogram(hap_array, n_bins=20):
    """Compute a histogram of derived allele frequencies from a haplotype array.

    Parameters
    ----------
    hap_array : array-like, shape (n_snps, n_samples)
        Haplotype array (0/1 values). Each row is a SNP, each column a sample.
    n_bins : int
        Number of histogram bins spanning (0, 1]. Bin edges are evenly spaced.

    Returns
    -------
    numpy.ndarray, shape (n_bins,)
        Counts of SNPs falling into each DAF bin, normalized to sum to 1.
        Returns uniform vector if no SNPs.
    """
    hap_array = np.asarray(hap_array, dtype=np.float64)
    if hap_array.size == 0 or hap_array.shape[0] == 0:
        return np.full(n_bins, 1.0 / n_bins)

    n_samples = hap_array.shape[1]
    dafs = hap_array.sum(axis=1) / n_samples

    # Bin into (0, 1] — exclude monomorphic sites at DAF=0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(dafs, bins=bin_edges)

    total = hist.sum()
    if total == 0:
        return np.full(n_bins, 1.0 / n_bins)
    return hist.astype(np.float64) / total


def compute_daf_histogram_diploid(genos_n_alt, n_bins=20):
    """Compute a DAF histogram from diploid genotype alt-allele counts.

    Parameters
    ----------
    genos_n_alt : array-like, shape (n_snps, n_individuals)
        Number of alt alleles per individual per SNP (0, 1, or 2).
    n_bins : int
        Number of histogram bins spanning [0, 1].

    Returns
    -------
    numpy.ndarray, shape (n_bins,)
        Normalized histogram of DAFs.
    """
    genos_n_alt = np.asarray(genos_n_alt, dtype=np.float64)
    if genos_n_alt.size == 0 or genos_n_alt.shape[0] == 0:
        return np.full(n_bins, 1.0 / n_bins)

    n_alleles = 2 * genos_n_alt.shape[1]
    dafs = genos_n_alt.sum(axis=1) / n_alleles

    bin_edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(dafs, bins=bin_edges)

    total = hist.sum()
    if total == 0:
        return np.full(n_bins, 1.0 / n_bins)
    return hist.astype(np.float64) / total


def compute_snp_distance_stats(positions, sub_win_len):
    """Compute summary statistics of inter-SNP distances within a sub-window.

    Parameters
    ----------
    positions : array-like
        Sorted physical positions of SNPs within a sub-window.
    sub_win_len : int
        Physical length of the sub-window (used for normalization).

    Returns
    -------
    numpy.ndarray, shape (4,)
        [mean, variance, min, max] of inter-SNP distances, each normalized
        by sub_win_len. Returns zeros if fewer than 2 SNPs.
    """
    if len(positions) < 2:
        return np.zeros(4, dtype=np.float64)

    positions = np.asarray(positions, dtype=np.float64)
    dists = np.diff(positions)

    # Normalize by sub-window length
    dists = dists / sub_win_len

    return np.array([
        dists.mean(),
        dists.var(),
        dists.min(),
        dists.max(),
    ], dtype=np.float64)


def compute_daf_features_for_subwin(data_array, positions, sub_win_len,
                                    n_bins=20, diploid=False):
    """Compute DAF histogram, distance stats, and RAiSD stats for a sub-window.

    Parameters
    ----------
    data_array : array-like
        For haploid: shape (n_snps, n_samples), 0/1 haplotype values.
        For diploid: shape (n_snps, n_individuals), alt allele counts (0/1/2).
    positions : array-like
        Sorted physical positions of SNPs in this sub-window.
    sub_win_len : int
        Physical length of the sub-window.
    n_bins : int
        Number of DAF histogram bins.
    diploid : bool
        If True, treat data_array as diploid genotype alt counts.

    Returns
    -------
    daf_hist : numpy.ndarray, shape (n_bins,)
    extra_stats : numpy.ndarray, shape (7,)
        [distMean, distVar, distMin, distMax, mu_var, mu_sfs, mu_ld]
    """
    if diploid:
        daf_hist = compute_daf_histogram_diploid(data_array, n_bins=n_bins)
    else:
        daf_hist = compute_daf_histogram(data_array, n_bins=n_bins)
    dist_stats = compute_snp_distance_stats(positions, sub_win_len)
    data_array = np.asarray(data_array)
    n_samples = 2 * data_array.shape[1] if diploid else data_array.shape[1]
    raisd_stats = compute_raisd_stats(data_array, positions, sub_win_len, n_samples, diploid=diploid)
    scan_stats = raisd_scan(data_array, positions, sub_win_len, n_samples, diploid=diploid)
    return daf_hist, np.concatenate((dist_stats, raisd_stats, scan_stats))


# ---------------------------------------------------------------------------
# RAiSD-inspired features: μ_VAR, μ_SFS, μ_LD
# ---------------------------------------------------------------------------

def compute_mu_var(n_snps, sub_win_len):
    """Compute μ_VAR: SNP density within a sub-window.

    Higher values indicate more SNPs per unit length (higher diversity).
    Near a sweep, diversity drops, so μ_VAR decreases.

    Parameters
    ----------
    n_snps : int
        Number of SNPs in the sub-window.
    sub_win_len : int or float
        Physical length of the sub-window.

    Returns
    -------
    float
        SNP density (SNPs per bp).
    """
    if sub_win_len <= 0:
        return 0.0
    return n_snps / sub_win_len


def compute_mu_sfs(dafs, n_samples, slack=1):
    """Compute μ_SFS: fraction of SNPs at the edges of the SFS.

    Counts SNPs with derived allele count <= slack (singletons) or
    >= n_samples - slack (near-fixed), normalized by total SNPs.
    Near a sweep, hitchhiking creates excess rare and near-fixed variants.

    Parameters
    ----------
    dafs : array-like
        Derived allele frequencies (0 to 1) for SNPs in the sub-window.
    n_samples : int
        Number of haploid samples (or 2 * n_individuals for diploid).
    slack : int
        How many allele count classes to include at each SFS edge.
        Default 1 means singletons + (n-1)-tons only.

    Returns
    -------
    float
        Fraction of edge SNPs. Returns 0 if no SNPs.
    """
    dafs = np.asarray(dafs)
    if dafs.size == 0:
        return 0.0
    # Convert DAFs to allele counts
    counts = np.round(dafs * n_samples).astype(np.int64)
    n_edge = np.sum((counts <= slack) | (counts >= n_samples - slack))
    return float(n_edge) / len(dafs)


def compute_mu_ld(hap_array):
    """Compute μ_LD: haplotype pattern exclusivity between left/right halves.

    Splits the sub-window's SNPs into left and right halves, identifies
    distinct column patterns (haplotype configurations) in each half,
    and measures how many patterns are exclusive to one side. High
    exclusivity indicates different LD structure on each side of the
    split — a signature of a sweep boundary.

    Fully vectorized using numpy void-dtype views for row hashing
    and numpy set operations (setdiff1d, isin) for pattern comparison.

    Parameters
    ----------
    hap_array : array-like, shape (n_snps, n_samples)
        Haplotype array (0/1 values).

    Returns
    -------
    float
        Pattern exclusivity score. Returns 0 if fewer than 2 SNPs.
    """
    hap_array = np.ascontiguousarray(hap_array, dtype=np.uint8)
    n_snps = hap_array.shape[0]
    if n_snps < 2:
        return 0.0

    mid = n_snps // 2
    left = np.ascontiguousarray(hap_array[:mid])
    right = np.ascontiguousarray(hap_array[mid:])

    # View each row as a single void element for vectorized comparison
    row_bytes = hap_array.dtype.itemsize * hap_array.shape[1]
    void_dt = np.dtype((np.void, row_bytes))

    left_v = left.view(void_dt).ravel()
    right_v = right.view(void_dt).ravel()

    left_unique = np.unique(left_v)
    right_unique = np.unique(right_v)

    n_left = len(left_unique)
    n_right = len(right_unique)

    if n_left == 0 or n_right == 0:
        return 0.0

    excl_left = np.setdiff1d(left_unique, right_unique)
    excl_right = np.setdiff1d(right_unique, left_unique)

    # Count SNPs whose pattern is exclusive to their half
    n_excl_snps_left = np.isin(left_v, excl_left).sum() if len(excl_left) > 0 else 0
    n_excl_snps_right = np.isin(right_v, excl_right).sum() if len(excl_right) > 0 else 0

    mu_ld = (len(excl_left) * int(n_excl_snps_left) +
             len(excl_right) * int(n_excl_snps_right)) / (n_left * n_right)
    return float(mu_ld)


def compute_raisd_stats(hap_array, positions, sub_win_len, n_samples, diploid=False):
    """Compute all three RAiSD-inspired statistics for a sub-window.

    Parameters
    ----------
    hap_array : array-like, shape (n_snps, n_samples)
        For haploid: 0/1 haplotype values.
        For diploid: alt allele counts (0/1/2) per individual.
    positions : array-like
        SNP positions within the sub-window.
    sub_win_len : int or float
        Physical length of the sub-window.
    n_samples : int
        Number of haploid samples (or 2 * n_individuals for diploid).
    diploid : bool
        If True, treat hap_array as diploid genotype counts.

    Returns
    -------
    numpy.ndarray, shape (3,)
        [mu_var, mu_sfs, mu_ld]
    """
    hap_array = np.asarray(hap_array)
    n_snps = hap_array.shape[0]

    mu_var = compute_mu_var(n_snps, sub_win_len)

    if n_snps == 0:
        return np.array([mu_var, 0.0, 0.0], dtype=np.float64)

    # Compute DAFs for μ_SFS
    if diploid:
        n_alleles = 2 * hap_array.shape[1]
        dafs = hap_array.sum(axis=1) / n_alleles
    else:
        dafs = hap_array.sum(axis=1) / hap_array.shape[1]
        n_alleles = hap_array.shape[1]

    mu_sfs = compute_mu_sfs(dafs, n_alleles)
    mu_ld = compute_mu_ld(hap_array)

    return np.array([mu_var, mu_sfs, mu_ld], dtype=np.float64)


def _vectorized_mu_var_sfs(positions, dafs, n_samples, sub_win_len, win_snps):
    """Vectorized μ_VAR and μ_SFS for all sliding windows at once.

    No Python loops — uses numpy array slicing and rolling operations.
    """
    n_windows = len(positions) - win_snps + 1

    # μ_VAR: span of each window / (sub_win_len * win_snps)
    # span[w] = positions[w + win_snps - 1] - positions[w]
    spans = positions[win_snps - 1:] - positions[:n_windows]
    mu_var_all = spans / (sub_win_len * win_snps) if sub_win_len > 0 else np.zeros(n_windows)

    # μ_SFS: count of edge SNPs in each window
    # Pre-compute per-SNP edge status
    counts = np.round(dafs * n_samples).astype(np.int64)
    is_edge = ((counts <= 1) | (counts >= n_samples - 1)).astype(np.float64)
    # Cumulative sum for sliding window counts
    cum_edge = np.concatenate(([0.0], np.cumsum(is_edge)))
    edge_counts = cum_edge[win_snps:] - cum_edge[:n_windows]
    mu_sfs_all = edge_counts / win_snps

    # Midpoints for each window
    midpoints = (positions[:n_windows] + positions[win_snps - 1:]) / 2.0

    return mu_var_all, mu_sfs_all, midpoints


from numba import njit as _njit  # noqa: E402


@_njit(cache=True)
def _mu_ld_scan_numba(hap_flat, n_snps, n_cols, win_snps, step, n_windows):
    """Numba-accelerated μ_LD for evenly-spaced sliding windows.

    Parameters
    ----------
    hap_flat : uint8 array, length n_snps * n_cols
        Row-major flattened haplotype matrix.
    n_snps : int
    n_cols : int
        Number of samples per SNP.
    win_snps : int
        Window size in SNPs.
    step : int
        Step size between consecutive window starts.
    n_windows : int
        Number of windows to evaluate.
    """
    result = np.empty(n_windows, dtype=np.float64)
    half = win_snps // 2

    for wi in range(n_windows):
        w = wi * step  # actual start index

        left_start = w
        left_end = w + half
        right_start = w + half
        right_end = w + win_snps

        # Collect unique patterns for left half
        n_left_unique = 0
        left_unique_ids = np.empty(half, dtype=np.int64)
        left_row_ids = np.empty(half, dtype=np.int64)

        for i in range(left_start, left_end):
            is_new = True
            row_i = i * n_cols
            for u in range(n_left_unique):
                row_u = left_unique_ids[u] * n_cols
                match = True
                for c in range(n_cols):
                    if hap_flat[row_i + c] != hap_flat[row_u + c]:
                        match = False
                        break
                if match:
                    is_new = False
                    left_row_ids[i - left_start] = u
                    break
            if is_new:
                left_unique_ids[n_left_unique] = i
                left_row_ids[i - left_start] = n_left_unique
                n_left_unique += 1

        # Collect unique patterns for right half
        rhs = win_snps - half
        n_right_unique = 0
        right_unique_ids = np.empty(rhs, dtype=np.int64)
        right_row_ids = np.empty(rhs, dtype=np.int64)

        for i in range(right_start, right_end):
            is_new = True
            row_i = i * n_cols
            for u in range(n_right_unique):
                row_u = right_unique_ids[u] * n_cols
                match = True
                for c in range(n_cols):
                    if hap_flat[row_i + c] != hap_flat[row_u + c]:
                        match = False
                        break
                if match:
                    is_new = False
                    right_row_ids[i - right_start] = u
                    break
            if is_new:
                right_unique_ids[n_right_unique] = i
                right_row_ids[i - right_start] = n_right_unique
                n_right_unique += 1

        if n_left_unique == 0 or n_right_unique == 0:
            result[wi] = 0.0
            continue

        # Count exclusive left patterns
        n_excl_lp = 0
        n_excl_ls = 0
        for lu in range(n_left_unique):
            row_l = left_unique_ids[lu] * n_cols
            found = False
            for ru in range(n_right_unique):
                row_r = right_unique_ids[ru] * n_cols
                match = True
                for c in range(n_cols):
                    if hap_flat[row_l + c] != hap_flat[row_r + c]:
                        match = False
                        break
                if match:
                    found = True
                    break
            if not found:
                n_excl_lp += 1
                for s in range(half):
                    if left_row_ids[s] == lu:
                        n_excl_ls += 1

        # Count exclusive right patterns
        n_excl_rp = 0
        n_excl_rs = 0
        for ru in range(n_right_unique):
            row_r = right_unique_ids[ru] * n_cols
            found = False
            for lu in range(n_left_unique):
                row_l = left_unique_ids[lu] * n_cols
                match = True
                for c in range(n_cols):
                    if hap_flat[row_l + c] != hap_flat[row_r + c]:
                        match = False
                        break
                if match:
                    found = True
                    break
            if not found:
                n_excl_rp += 1
                for s in range(rhs):
                    if right_row_ids[s] == ru:
                        n_excl_rs += 1

        result[wi] = (n_excl_lp * n_excl_ls + n_excl_rp * n_excl_rs) / (n_left_unique * n_right_unique)

    return result


def _sliding_mu_ld_stepped(hap_array, win_snps, step, n_windows):
    """Compute μ_LD for evenly-spaced sliding window positions."""
    hap_array = np.ascontiguousarray(hap_array, dtype=np.uint8)
    n_snps, n_cols = hap_array.shape
    return _mu_ld_scan_numba(hap_array.ravel(), n_snps, n_cols, win_snps, step, n_windows)


def raisd_scan(hap_array, positions, sub_win_len, n_samples, win_snps=50,
               max_windows=50, diploid=False):
    """Run a fine-grained RAiSD-style μ scan within a sub-window.

    Vectorized μ_VAR and μ_SFS computation, numba-accelerated μ_LD scan.
    When n_snps >> win_snps, uses a step size to cap the number of
    sliding windows at max_windows for computational efficiency.

    Returns
    -------
    numpy.ndarray, shape (3,)
        [mu_product, max_mu_scan, peak_position_frac]
    """
    hap_array = np.ascontiguousarray(hap_array, dtype=np.uint8)
    positions = np.asarray(positions, dtype=np.float64)
    n_snps = hap_array.shape[0]

    if n_snps == 0:
        return np.array([0.0, 0.0, 0.5], dtype=np.float64)

    if diploid:
        dafs = hap_array.astype(np.float64).sum(axis=1) / (2 * hap_array.shape[1])
    else:
        dafs = hap_array.astype(np.float64).sum(axis=1) / hap_array.shape[1]

    # Sub-window-level composite μ
    mu_var_sw = compute_mu_var(n_snps, sub_win_len)
    mu_sfs_sw = compute_mu_sfs(dafs, n_samples)
    mu_ld_sw = compute_mu_ld(hap_array)
    mu_product = mu_var_sw * mu_sfs_sw * mu_ld_sw

    if n_snps < win_snps:
        peak_frac = 0.5
        if n_snps > 1 and positions[-1] > positions[0]:
            mid_pos = (positions[0] + positions[-1]) / 2.0
            peak_frac = np.clip((mid_pos - positions[0]) / (positions[-1] - positions[0]), 0, 1)
        return np.array([mu_product, mu_product, peak_frac], dtype=np.float64)

    # Determine step size to cap computation
    n_possible = n_snps - win_snps + 1
    step = max(1, n_possible // max_windows)

    # Sample window start positions evenly
    window_starts = np.arange(0, n_possible, step)
    n_windows = len(window_starts)

    # Vectorized μ_VAR: span / (sub_win_len * win_snps) for each sampled window
    spans = positions[window_starts + win_snps - 1] - positions[window_starts]
    mu_var_all = spans / (sub_win_len * win_snps) if sub_win_len > 0 else np.zeros(n_windows)

    # Vectorized μ_SFS: use cumsum trick for edge counts
    counts = np.round(dafs * n_samples).astype(np.int64)
    is_edge = ((counts <= 1) | (counts >= n_samples - 1)).astype(np.float64)
    cum_edge = np.concatenate(([0.0], np.cumsum(is_edge)))
    edge_counts = cum_edge[window_starts + win_snps] - cum_edge[window_starts]
    mu_sfs_all = edge_counts / win_snps

    # Midpoints
    midpoints = (positions[window_starts] + positions[window_starts + win_snps - 1]) / 2.0

    # Numba μ_LD: subsample using the step
    mu_ld_all = _sliding_mu_ld_stepped(hap_array, win_snps, step, n_windows)

    # Composite μ
    mu_scan = mu_var_all * mu_sfs_all * mu_ld_all

    max_idx = np.argmax(mu_scan)
    max_mu = mu_scan[max_idx]

    if positions[-1] > positions[0]:
        peak_frac = (midpoints[max_idx] - positions[0]) / (positions[-1] - positions[0])
    else:
        peak_frac = 0.5

    return np.array([mu_product, max_mu, np.clip(peak_frac, 0, 1)], dtype=np.float64)


# Pre-computed constants for DAF features
DAF_N_BINS = 20
DAF_N_RAISD = 3
DAF_N_SCAN = 3   # mu_product, max_mu_scan, peak_position_frac
DAF_N_DIST = 4 + DAF_N_RAISD + DAF_N_SCAN  # 4 distance + 3 RAiSD + 3 scan = 10
DAF_UNIFORM = np.full(DAF_N_BINS, 1.0 / DAF_N_BINS)
DAF_ZERO_DIST = np.zeros(DAF_N_DIST, dtype=np.float64)


def build_daf_header(n_bins=20, num_sub_wins=11):
    """Build the column header for a .daf.fvec file.

    Returns
    -------
    str
        Tab-separated header string.
    """
    parts = []
    for b in range(n_bins):
        for w in range(num_sub_wins):
            parts.append("dafBin%d_win%d" % (b, w))
    for feat_name in ("snpDistMean", "snpDistVar", "snpDistMin", "snpDistMax",
                       "muVar", "muSFS", "muLD",
                       "muProduct", "maxMuScan", "peakPosFrac"):
        for w in range(num_sub_wins):
            parts.append("%s_win%d" % (feat_name, w))
    return "\t".join(parts)


def flatten_daf_features(daf_hists, dist_stats):
    """Flatten per-subwindow DAF histograms and distance stats into a feature row.

    Parameters
    ----------
    daf_hists : list of arrays, each shape (n_bins,)
        One DAF histogram per sub-window.
    dist_stats : list of arrays, each shape (10,)
        One combined distance + RAiSD + scan stats vector per sub-window.

    Returns
    -------
    numpy.ndarray, shape (n_bins * n_subwins + 10 * n_subwins,)
        Feature-major order: dafBin0_win0..winN, dafBin1_win0..winN, ..., distMean_win0..winN, ...
    """
    daf_matrix = np.array(daf_hists)    # (n_subwins, n_bins)
    dist_matrix = np.array(dist_stats)  # (n_subwins, 4)
    return np.concatenate((daf_matrix.T.ravel(), dist_matrix.T.ravel()))
