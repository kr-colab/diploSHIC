import os
import allel
from allel.model.ndarray import SortedIndex
from allel.util import asarray_ndim
import h5py
import numpy as np
import sys
import time
from diploshic.fvTools import *

if not len(sys.argv) in [13, 15]:
    sys.exit(
        "usage:\npython makeFeatureVecsForChrArmFromVcfDiploid.py vcfFileName chrArm chrLen targetPop winSize numSubWins maskFileName unmaskedFracCutoff unmaskedGenoFracCutoff sampleToPopFileName statFileName outFileName [segmentStart segmentEnd]\n"
    )
if len(sys.argv) == 15:
    (
        vcfFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        unmaskedGenoFracCutoff,
        sampleToPopFileName,
        statFileName,
        outfn,
        segmentStart,
        segmentEnd,
    ) = sys.argv[1:]
    segmentStart, segmentEnd = int(segmentStart), int(segmentEnd)
else:
    (
        vcfFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        unmaskedGenoFracCutoff,
        sampleToPopFileName,
        statFileName,
        outfn,
    ) = sys.argv[1:]
    segmentStart = None

unmaskedFracCutoff = float(unmaskedFracCutoff)
if unmaskedFracCutoff < 0.0 or unmaskedFracCutoff > 1.0:
    sys.exit(
        "unmaskedFracCutoff=%s but must be within [0, 1]. AAAAARRRRGHHHHHH!!!\n"
        % (unmaskedFracCutoff)
    )
unmaskedGenoFracCutoff = float(unmaskedGenoFracCutoff)
if unmaskedGenoFracCutoff < 0.0 or unmaskedGenoFracCutoff > 1.0:
    sys.exit(
        "unmaskedGenoFracCutoff=%s but must be within [0, 1]. AAAAARRRRGHHHHHH!!!\n"
        % (unmaskedGenoFracCutoff)
    )
chrLen, winSize, numSubWins = int(chrLen), int(winSize), int(numSubWins)
assert winSize % numSubWins == 0 and numSubWins > 1
subWinSize = int(winSize / numSubWins)


def getSubWinBounds(chrLen, subWinSize):
    lastSubWinEnd = chrLen - chrLen % subWinSize
    lastSubWinStart = lastSubWinEnd - subWinSize + 1
    subWinBounds = []
    for subWinStart in range(1, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        subWinBounds.append((subWinStart, subWinEnd))
    return subWinBounds


def getSnpIndicesInSubWins(subWinSize, lastSubWinEnd, snpLocs, firstSubWinStart=1):
    subWinStart = firstSubWinStart
    subWinEnd = subWinStart + subWinSize - 1
    snpIndicesInSubWins = [[]]
    for i in range(len(snpLocs)):
        while snpLocs[i] <= lastSubWinEnd and not (
            snpLocs[i] >= subWinStart and snpLocs[i] <= subWinEnd
        ):
            subWinStart += subWinSize
            subWinEnd += subWinSize
            snpIndicesInSubWins.append([])
        if snpLocs[i] <= lastSubWinEnd:
            snpIndicesInSubWins[-1].append(i)
    while subWinEnd < lastSubWinEnd:
        snpIndicesInSubWins.append([])
        subWinStart += subWinSize
        subWinEnd += subWinSize
    return snpIndicesInSubWins


def readSampleToPopFile(sampleToPopFileName):
    table = {}
    with open(sampleToPopFileName) as sampleToPopFile:
        for line in sampleToPopFile:
            sample, pop = line.strip().split()
            table[sample] = pop
    return table


vcfFile = allel.read_vcf(vcfFileName)
chroms = vcfFile["variants/CHROM"]
positions = np.extract(chroms == chrArm, vcfFile["variants/POS"])

if maskFileName.lower() in ["none", "false"]:
    sys.stderr.write(
        "Warning: a mask.fa file for the chr arm with all masked sites N'ed out is strongly recommended"
        + " (pass in the reference to remove Ns at the very least)!\n"
    )
    unmasked = np.ones(chrLen, dtype=bool)
else:
    unmasked = np.array(readMaskDataForScan(maskFileName, chrArm), dtype=bool)
    assert len(unmasked) == chrLen

if statFileName.lower() in ["none", "false"]:
    statFileName = None

samples = vcfFile["samples"]
if not sampleToPopFileName.lower() in ["none", "false"]:
    sampleToPop = readSampleToPopFile(sampleToPopFileName)
    sampleIndicesToKeep = [
        i
        for i in range(len(samples))
        if sampleToPop.get(samples[i], "popNotFound!") == targetPop
    ]
else:
    sampleIndicesToKeep = [i for i in range(len(samples))]
rawgenos = np.take(
    vcfFile["calldata/GT"],
    [i for i in range(len(chroms)) if chroms[i] == chrArm],
    axis=0,
)
genos = allel.GenotypeArray(rawgenos).subset(sel1=sampleIndicesToKeep)

if segmentStart != None:
    snpIndicesToKeep = [
        i
        for i in range(len(positions))
        if segmentStart <= positions[i] <= segmentEnd
    ]
    if len(snpIndicesToKeep) == 0:
        sys.exit(
            "Error: no SNPs in the given segment of the chr arm; exiting\n"
        )
    positions = [positions[i] for i in snpIndicesToKeep]
    genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))

if isHaploidVcfGenoArray(genos):
    sys.stderr.write(
        "Detected haploid input. Converting into diploid individuals (combining haplotypes in order).\n"
    )
    genos = diploidizeGenotypeArray(genos)

alleleCounts = genos.count_alleles()

# remove all but mono/biallelic unmasked sites
isBiallelic = alleleCounts.is_biallelic()
for i in range(len(isBiallelic)):
    if not (
        isBiallelic[i]
        and calledGenoFracAtSite(genos[i]) >= unmaskedGenoFracCutoff
    ):
        unmasked[positions[i] - 1] = False
snpIndicesToKeep = [
    i for i in range(len(positions)) if unmasked[positions[i] - 1]
]
genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))
positions = np.array([positions[i] for i in snpIndicesToKeep])
alleleCounts = allel.AlleleCountsArray(
    [[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snpIndicesToKeep]
)

# Pre-convert data structures to avoid repeated conversions in allel stats functions
# This is a major optimization - allel functions call np.asarray on every call
positions = SortedIndex(positions, copy=False)
alleleCounts = asarray_ndim(alleleCounts, 2)

statNames = [
    "pi",
    "thetaW",
    "tajD",
    "distVar",
    "distSkew",
    "distKurt",
    "nDiplos",
    "diplo_H1",
    "diplo_H12",
    "diplo_H2/H1",
    "diplo_ZnS",
    "diplo_Omega",
]

subWinBounds = getSubWinBounds(chrLen, subWinSize)

header = "chrom classifiedWinStart classifiedWinEnd bigWinRange".split()
statHeader = "chrom start end".split()
for statName in statNames:
    statHeader.append(statName)
    for i in range(numSubWins):
        header.append("%s_win%d" % (statName, i))
statHeader = "\t".join(statHeader)
header = "\t".join(header)
outFile = open(outfn, "w")
outFile.write(header + "\n")
statVals = {}
for statName in statNames:
    statVals[statName] = []

nDafBins = DAF_N_BINS
dafHistsAll = []
dafDistsAll = []

# DAF output file
dafOutfn = outfn.replace(".fvec", ".daf.fvec")
if dafOutfn == outfn:
    dafOutfn = outfn + ".daf.fvec"
dafHeader = "chrom\tclassifiedWinStart\tclassifiedWinEnd\tbigWinRange\t" + build_daf_header(n_bins=nDafBins, num_sub_wins=numSubWins)
dafOutFile = open(dafOutfn, "w")
dafOutFile.write(dafHeader + "\n")

startTime = time.perf_counter()
goodSubWins = []

# Determine iteration bounds based on segment (if specified)
if segmentStart is not None:
    # Align firstSubWinStart to subWinSize boundary at or after segmentStart
    firstSubWinStart = ((segmentStart - 1) // subWinSize) * subWinSize + 1
    # Align lastSubWinEnd to subWinSize boundary at or before segmentEnd
    lastSubWinEnd = (segmentEnd // subWinSize) * subWinSize
else:
    firstSubWinStart = 1
    lastSubWinEnd = chrLen - chrLen % subWinSize

snpIndicesInSubWins = getSnpIndicesInSubWins(
    subWinSize, lastSubWinEnd, positions, firstSubWinStart
)
subWinIndex = 0
lastSubWinStart = lastSubWinEnd - subWinSize + 1
if statFileName:
    statFile = open(statFileName, "w")
    statFile.write(statHeader + "\n")
for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
    subWinEnd = subWinStart + subWinSize - 1
    unmaskedFrac = np.sum(unmasked[subWinStart - 1 : subWinEnd]) / float(
        subWinEnd - subWinStart + 1
    )
    if (
        segmentStart == None
        or subWinStart >= segmentStart
        and subWinEnd <= segmentEnd
    ):
        sys.stderr.write(
            "%d-%d num unmasked snps: %d; unmasked frac: %f\n"
            % (
                subWinStart,
                subWinEnd,
                len(snpIndicesInSubWins[subWinIndex]),
                unmaskedFrac,
            )
        )
    if (
        len(snpIndicesInSubWins[subWinIndex]) > 0
        and unmaskedFrac >= unmaskedFracCutoff
    ):
        genosInSubWin = allel.GenotypeArray(
            genos.subset(sel0=snpIndicesInSubWins[subWinIndex])
        )
        # Pre-compute genosNAlt once per subwindow to avoid repeated conversions
        genosNAlt = genosInSubWin.to_n_alt()
        statValStr = []
        for statName in statNames:
            calcAndAppendStatValForScanDiplo(
                alleleCounts,
                positions,
                statName,
                subWinStart,
                subWinEnd,
                statVals,
                subWinIndex,
                genosInSubWin,
                unmasked,
                genosNAlt=genosNAlt,
            )
        goodSubWins.append(True)
        # DAF features for this sub-window
        posInSubWin = np.asarray(positions)[snpIndicesInSubWins[subWinIndex]]
        dafH, dafD = compute_daf_features_for_subwin(
            np.asarray(genosNAlt), posInSubWin, subWinSize, n_bins=nDafBins, diploid=True
        )
        dafHistsAll.append(dafH)
        dafDistsAll.append(dafD)
        if statFileName:
            statFile.write(
                "\t".join(
                    [chrArm, str(subWinStart), str(subWinEnd)]
                    + [str(statVals[statName][-1]) for statName in statNames]
                )
                + "\n"
            )
    else:
        for statName in statNames:
            appendStatValsForMonomorphicForScan(
                statName, statVals, subWinIndex
            )
        goodSubWins.append(False)
        dafHistsAll.append(DAF_UNIFORM.copy())
        dafDistsAll.append(DAF_ZERO_DIST.copy())
    if goodSubWins[-numSubWins:].count(True) == numSubWins:
        outVec = []
        for statName in statNames:
            outVec += normalizeFeatureVec(statVals[statName][-numSubWins:])
        midSubWinEnd = int(subWinEnd - subWinSize * (numSubWins // 2))
        midSubWinStart = midSubWinEnd - subWinSize + 1
        outFile.write(
            "%s\t%d\t%d\t%d-%d\t"
            % (
                chrArm,
                midSubWinStart,
                midSubWinEnd,
                subWinEnd - winSize + 1,
                subWinEnd,
            )
            + "\t".join([str(x) for x in outVec])
        )
        outFile.write("\n")
        # Write DAF features for this complete window
        dafRow = flatten_daf_features(dafHistsAll[-numSubWins:], dafDistsAll[-numSubWins:])
        dafOutFile.write(
            "%s\t%d\t%d\t%d-%d\t"
            % (chrArm, midSubWinStart, midSubWinEnd, subWinEnd - winSize + 1, subWinEnd)
            + "\t".join(["%.18g" % v for v in dafRow])
        )
        dafOutFile.write("\n")
    subWinIndex += 1
if statFileName:
    statFile.close()
outFile.close()
dafOutFile.close()
sys.stderr.write("wrote DAF features to %s\n" % dafOutfn)
sys.stderr.write(
    "completed in %g seconds\n" % (time.perf_counter() - startTime)
)
