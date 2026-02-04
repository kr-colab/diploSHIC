import allel
import sys
import math
from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim
from scipy.spatial.distance import squareform
import diploshic.shicstats as dps
import numpy as np
import random
import gzip
import scipy.stats


def fast_r2_for_ld(gn):
    """
    Compute R (correlation) matrix using BLAS-optimized matrix multiplication.

    This is a fast replacement for allel.stats.ld.rogers_huff_r that uses
    numpy's BLAS-optimized matrix multiplication instead of nested loops.
    Provides 5-10x speedup while maintaining numerical accuracy.

    Parameters
    ----------
    gn : array-like, shape (n_snps, n_samples)
        Genotype values (0, 1, or 2 for diploid; 0 or 1 for haploid)

    Returns
    -------
    r_flat : ndarray, shape (n_snps * (n_snps - 1) / 2,)
        R (correlation) values in condensed form (same as allel.rogers_huff_r output)
    """
    gn = np.asarray(gn, dtype=np.float64)
    n_snps, n_samples = gn.shape

    if n_snps <= 1:
        return np.array([], dtype=np.float64)

    # Center and normalize
    means = gn.mean(axis=1, keepdims=True)
    gn_centered = gn - means
    std = gn_centered.std(axis=1, keepdims=True, ddof=0)

    # Handle zero variance (monomorphic sites)
    std_safe = np.where(std == 0, 1, std)
    gn_norm = gn_centered / std_safe

    # Compute correlation matrix via BLAS matmul
    r = (gn_norm @ gn_norm.T) / n_samples

    # Zero out rows/cols for monomorphic sites
    mono_mask = (std.flatten() == 0)
    r[mono_mask, :] = 0
    r[:, mono_mask] = 0

    # Extract upper triangle (condensed form like squareform expects)
    r_flat = squareform(r, checks=False)

    return r_flat


def fast_r2_matrix_haploid(haps):
    """
    Compute R² matrix for haploid data using BLAS-optimized matrix multiplication.

    This is a fast replacement for dps.computeR2Matrix that provides 10-50x speedup
    while correctly handling missing data per-pair using vectorized operations.

    For binary haplotype data (0/1), we compute R² using the formula:
        R² = Cov(X_i, X_j)² / (Var(X_i) * Var(X_j))
    where statistics are computed only over jointly valid samples for each pair.

    The key insight is that all necessary sums can be computed via matrix multiplication:
        - V = validity indicator (1 if valid, 0 if missing)
        - M = masked values (X if valid, 0 if missing)
        - n_ij = (V @ V.T)[i,j] = count of jointly valid samples
        - S_i_given_j = (M @ V.T)[i,j] = sum of X_i over jointly valid samples
        - SS_ij = (M @ M.T)[i,j] = sum of X_i * X_j over jointly valid samples

    Parameters
    ----------
    haps : array-like, shape (n_snps, n_haplotypes)
        Haplotype values (0, 1, or -1 for missing)

    Returns
    -------
    r2_matrix : ndarray, shape (n_snps, n_snps)
        R² matrix with upper triangle filled (compatible with dps.ZnS and dps.omega).
        Pairs involving monomorphic sites or insufficient samples are set to -1.0.
    """
    haps_arr = np.asarray(haps)
    n_snps, n_samples = haps_arr.shape

    if n_snps <= 1:
        return np.zeros((n_snps, n_snps), dtype=np.float64)

    # Check for missing data
    has_missing = np.any(haps_arr < 0)

    if not has_missing:
        # Fast path: no missing data - simple BLAS approach
        haps_float = haps_arr.astype(np.float64)
        means = haps_float.mean(axis=1, keepdims=True)
        haps_centered = haps_float - means
        var = haps_centered.var(axis=1, keepdims=True, ddof=0)

        mono_mask = (var.flatten() == 0)
        var_safe = np.where(var == 0, 1, var)
        haps_norm = haps_centered / np.sqrt(var_safe)

        r = (haps_norm @ haps_norm.T) / n_samples
        r2 = r ** 2
        r2 = np.triu(r2, k=1)

        r2[mono_mask, :] = -1.0
        r2[:, mono_mask] = -1.0
        return r2

    # Missing data path: vectorized computation with per-pair sample counts
    # V[i,k] = 1 if haps[i,k] >= 0, else 0 (validity indicator)
    # M[i,k] = haps[i,k] if valid, else 0 (masked values)
    V = (haps_arr >= 0).astype(np.float64)
    M = np.where(haps_arr >= 0, haps_arr, 0).astype(np.float64)

    # Compute via BLAS matrix multiplication:
    # n[i,j] = sum_k V[i,k] * V[j,k] = count of jointly valid samples
    # S1[i,j] = sum_k M[i,k] * V[j,k] = sum of X_i over jointly valid samples for pair (i,j)
    # SS[i,j] = sum_k M[i,k] * M[j,k] = sum of X_i * X_j over jointly valid samples
    n = V @ V.T
    S1 = M @ V.T  # S1[i,j] = sum of X_i when both i,j valid
    SS = M @ M.T  # SS[i,j] = sum of X_i * X_j when both valid

    # Note: S2[i,j] = sum of X_j when both valid = S1[j,i] = S1.T[i,j]
    S2 = S1.T

    # For binary data X in {0,1}: X² = X, so sum(X²) = sum(X)
    # Mean of X_i over jointly valid samples: mean_i = S1 / n
    # Variance of X_i over jointly valid samples: var_i = mean_i * (1 - mean_i)
    #   (using property of Bernoulli: var = p(1-p))

    # Covariance: cov = E[XY] - E[X]E[Y] = SS/n - (S1/n)(S2/n) = (SS*n - S1*S2) / n²
    # R² = cov² / (var_i * var_j)
    #    = (SS*n - S1*S2)² / n⁴ / (S1/n * (1-S1/n) * S2/n * (1-S2/n))
    #    = (SS*n - S1*S2)² / (S1*(n-S1) * S2*(n-S2))

    # Compute numerator and denominator
    # Numerator: (SS * n - S1 * S2)²
    cov_num = SS * n - S1 * S2  # Covariance * n² (scaled)
    numerator = cov_num ** 2

    # Denominator: S1*(n-S1) * S2*(n-S2)
    # This equals n² * var_i * n² * var_j = n⁴ * var_i * var_j
    var_i_scaled = S1 * (n - S1)  # = n² * var_i
    var_j_scaled = S2 * (n - S2)  # = n² * var_j
    denominator = var_i_scaled * var_j_scaled

    # Compute R² with protection against division by zero
    # Invalid cases: n < 2 (not enough samples), or either variance = 0 (monomorphic)
    valid_mask = (n >= 2) & (var_i_scaled > 0) & (var_j_scaled > 0)

    r2 = np.zeros((n_snps, n_snps), dtype=np.float64)
    r2[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Set invalid pairs to -1.0 (filtered by dps.ZnS/omega)
    r2[~valid_mask] = -1.0

    # Zero out diagonal and lower triangle (match C extension format)
    r2 = np.triu(r2, k=1)

    return r2


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
    minVal = min(statVec)
    if minVal < 0:
        statVec = [x - minVal for x in statVec]
    normStatVec = []
    statSum = float(sum(statVec))
    if statSum == 0 or any(np.isinf(statVec)) or any(np.isnan(statVec)):
        normStatVec = [1.0 / len(statVec)] * len(statVec)
    else:
        for k in range(len(statVec)):
            normStatVec.append(statVec[k] / statSum)
    return normStatVec


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
        statVals["ZnS"][instanceIndex].append(dps.ZnS(r2Matrix)[0])
        statVals["Omega"][instanceIndex].append(dps.omega(r2Matrix)[0])
    elif statName == "RH":
        rMatrixFlat = fast_r2_for_ld(
            hapsInSubWin.to_genotypes(ploidy=2).to_n_alt()
        )
        rhAvg = rMatrixFlat.mean()
        statVals["RH"][instanceIndex].append(rhAvg)
        r2Matrix = squareform(rMatrixFlat ** 2)
        statVals["Omega"][instanceIndex].append(dps.omega(r2Matrix)[0])
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
        dists = dps.pairwiseDiffs(hapsInSubWin) / float(n_unmasked)
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
        diplotypeCounts = dps.getHaplotypeFreqSpec(genosNAlt)
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
            r2Matrix = fast_r2_for_ld(genosNAlt)
            r2Matrix2 = squareform(r2Matrix ** 2)
            statVals["diplo_ZnS"][instanceIndex].append(np.nanmean(r2Matrix2))
            statVals["diplo_Omega"][instanceIndex].append(
                dps.omega(r2Matrix2)[0]
            )
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = dps.pairwiseDiffsDiplo(genosNAlt) / float(n_unmasked)
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
        diplotypeCounts = dps.getHaplotypeFreqSpec(genosNAlt)
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
            r2Matrix = fast_r2_for_ld(genosNAlt)
            r2Matrix2 = squareform(r2Matrix ** 2)
            statVals["diplo_ZnS"].append(np.nanmean(r2Matrix))
            statVals["diplo_Omega"].append(dps.omega(r2Matrix2)[0])
    elif statName == "distVar":
        # Support both list and numpy array for unmasked
        unmasked_slice = unmasked[subWinStart - 1 : subWinEnd]
        if hasattr(unmasked_slice, 'count'):
            n_unmasked = unmasked_slice.count(True)
        else:
            n_unmasked = np.sum(unmasked_slice)
        dists = dps.pairwiseDiffsDiplo(genosNAlt) / float(n_unmasked)
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
        statVals["ZnS"].append(dps.ZnS(r2Matrix)[0])
        statVals["Omega"].append(dps.omega(r2Matrix)[0])
    elif statName == "RH":
        rMatrixFlat = fast_r2_for_ld(
            hapsInSubWin.to_genotypes(ploidy=2).to_n_alt()
        )
        rhAvg = rMatrixFlat.mean()
        statVals["RH"].append(rhAvg)
        r2Matrix = squareform(rMatrixFlat ** 2)
        statVals["Omega"].append(dps.omega(r2Matrix)[0])
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
        dists = dps.pairwiseDiffs(hapsInSubWin) / float(n_unmasked)
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
