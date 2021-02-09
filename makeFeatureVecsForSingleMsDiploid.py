import sys, os
import allel
import random
import numpy as np
from msTools import *
from fvTools import *
import time

trainingDataFileName, totalPhysLen, numSubWins, maskFileName, vcfForMaskFileName, popForMask, sampleToPopFileName, unmaskedGenoFracCutoff, chrArmsForMasking, unmaskedFracCutoff, outStatsDir, fvecFileName = sys.argv[1:]
totalPhysLen = int(totalPhysLen)
numSubWins = int(numSubWins)
subWinLen = totalPhysLen//numSubWins
assert totalPhysLen % numSubWins == 0 and numSubWins > 1

sys.stderr.write("file name='%s'" %(trainingDataFileName))

trainingDataFileObj, sampleSize, numInstances = openMsOutFileForSequentialReading(trainingDataFileName)

if maskFileName.lower() in ["none", "false"]:
    sys.stderr.write("maskFileName='%s': not masking any sites!\n" %(maskFileName))
    maskFileName = False
    unmaskedFracCutoff = 1.0
else:
    chrArmsForMasking = chrArmsForMasking.split(",")
    unmaskedFracCutoff = float(unmaskedFracCutoff)
    if unmaskedFracCutoff > 1.0 or unmaskedFracCutoff < 0.0:
        sys.exit("unmaskedFracCutoff must lie within [0, 1]. AAARRRRGGGGHHHHH!!!!\n")

if vcfForMaskFileName.lower() in ["none", "false"]:
    sys.stderr.write("vcfForMaskFileName='%s': not masking any genotypes!\n" %(vcfForMaskFileName))
    vcfForMaskFileName = False
else:
    if not maskFileName:
        sys.exit("Cannot mask genotypes without also supplying a file for masking entire sites (can use reference genome with Ns if desired). AAARRRGHHHHH!!!!!!\n")
    if sampleToPopFileName.lower() in ["none", "false"] or popForMask.lower() in ["none", "false"]:
        sampleToPopFileName = None
        sys.stderr.write("No sampleToPopFileName specified. Using all individuals for masking genotypes.\n")
    unmaskedGenoFracCutoff = float(unmaskedGenoFracCutoff)
    if unmaskedGenoFracCutoff > 1.0 or unmaskedGenoFracCutoff < 0.0:
        sys.exit("unmaskedGenoFracCutoff must lie within [0, 1]. AAARRRRGGGGHHHHH!!!!\n")

def getSubWinBounds(subWinLen, totalPhysLen): # get inclusive subwin bounds
    subWinStart = 1
    subWinEnd = subWinStart + subWinLen - 1
    subWinBounds = [(subWinStart, subWinEnd)]
    numSubWins = totalPhysLen//subWinLen
    for i in range(1, numSubWins-1):
        subWinStart += subWinLen
        subWinEnd += subWinLen
        subWinBounds.append((subWinStart, subWinEnd))
    subWinStart += subWinLen
    # if our subwindows are 1 bp too short due to rounding error, the last window picks up all of the slack
    subWinEnd = totalPhysLen
    subWinBounds.append((subWinStart, subWinEnd))
    return subWinBounds

if not maskFileName:
    unmasked = [True] * totalPhysLen
else:
    drawWithReplacement = False
    sys.stderr.write("reading masking data...")
    maskData = readMaskDataForTraining(maskFileName, totalPhysLen, subWinLen, chrArmsForMasking, shuffle=True, cutoff=unmaskedFracCutoff,
                                                     genoCutoff=unmaskedGenoFracCutoff, vcfForMaskFileName=vcfForMaskFileName, pop=popForMask,
                                                     sampleToPopFileName=sampleToPopFileName)
    if vcfForMaskFileName:
        maskData, genoMaskData = maskData
    else:
        genoMaskData = [None]*len(maskData)
    sys.stderr.write("done!\n")

    if len(maskData) < numInstances:
        sys.stderr.write("Warning: didn't get enough windows from masked data (needed %d; got %d); will draw with replacement!!\n" %(numInstances, len(maskData)))
        drawWithReplacement = True
    else:
        sys.stderr.write("Got enough windows from masked data (needed %d; got %d); will draw without replacement.\n" %(numInstances, len(maskData)))

def getSnpIndicesInSubWins(subWinBounds, snpLocs):
    snpIndicesInSubWins = []
    for subWinIndex in range(len(subWinBounds)):
        snpIndicesInSubWins.append([])

    subWinIndex = 0
    for i in range(len(snpLocs)):
        while not (snpLocs[i] >= subWinBounds[subWinIndex][0] and snpLocs[i] <= subWinBounds[subWinIndex][1]):
            subWinIndex += 1
        snpIndicesInSubWins[subWinIndex].append(i)
    return snpIndicesInSubWins

subWinBounds = getSubWinBounds(subWinLen, totalPhysLen)
#statNames = ["pi", "thetaW", "tajD", "nDiplos","diplo_H2","diplo_H12","diplo_H2/H1","diplo_ZnS","diplo_Omega"]
statNames = ["pi", "thetaW", "tajD", "distVar","distSkew","distKurt","nDiplos","diplo_H1","diplo_H12","diplo_H2/H1","diplo_ZnS","diplo_Omega"]
header = []
for statName in statNames:
    for i in range(numSubWins):
        header.append("%s_win%d" %(statName, i))
header = "\t".join(header)


statVals = {}
for statName in statNames:
    statVals[statName] = []
start = time.perf_counter()
numInstancesDone = 0
sys.stderr.write("ready to process sim reps. here we go!\n")
for instanceIndex in range(numInstances):
    sys.stderr.write("starting rep %d of %d\n" %(instanceIndex, numInstances))
    hapArrayIn, positionArray = readNextMsRepToHaplotypeArrayIn(trainingDataFileObj, sampleSize, totalPhysLen)

    haps = allel.HaplotypeArray(hapArrayIn, dtype='i1')
    if maskFileName:
        if drawWithReplacement:
            randIndex = random.randint(0, len(maskData)-1)
            unmasked, genoMasks = maskData[randIndex], genoMaskData[randIndex]
        else:
            unmasked, genoMasks = maskData[instanceIndex], genoMaskData[instanceIndex]
        assert len(unmasked) == totalPhysLen
    if haps.shape[1] % 2 == 1:
        haps = haps[:,:-1]
    genos = haps.to_genotypes(ploidy=2)
    unmaskedSnpIndices = [i for i in range(len(positionArray)) if unmasked[positionArray[i]-1]]
    if len(unmaskedSnpIndices) == 0:
        sys.stderr.write("no snps for rep %d\n" %(instanceIndex))
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            for statName in statNames:
                appendStatValsForMonomorphic(statName, statVals, instanceIndex, subWinIndex)
    else:
        sys.stderr.write("processing snps for rep %d\n" %(instanceIndex))
        if maskFileName:
            preMaskCount = np.sum(genos.count_alleles())
            if genoMasks:
                sys.stderr.write("%d snps in the masking window for rep %d\n" %(len(genoMasks), instanceIndex))
                genos = maskGenos(genos.subset(sel0=unmaskedSnpIndices), genoMasks)
            else:
                genos = genos.subset(sel0=unmaskedSnpIndices)
            alleleCountsUnmaskedOnly = genos.count_alleles()
            maskedCount = preMaskCount - np.sum(alleleCountsUnmaskedOnly)
            sys.stderr.write("%d of %d genotypes (%.2f%%) masked for rep %d\n" %(maskedCount, preMaskCount, 100*maskedCount/preMaskCount, instanceIndex))
        else:
            alleleCountsUnmaskedOnly = genos.count_alleles()
        positionArrayUnmaskedOnly = [positionArray[i] for i in unmaskedSnpIndices]
        snpIndicesInSubWins = getSnpIndicesInSubWins(subWinBounds, positionArrayUnmaskedOnly)
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            subWinStart, subWinEnd = subWinBounds[subWinIndex]
            unmaskedFrac = unmasked[subWinStart-1:subWinEnd].count(True)/float(subWinLen)
            assert unmaskedFrac >= unmaskedFracCutoff
            snpIndicesInSubWinUnmasked = snpIndicesInSubWins[subWinIndex]
            sys.stderr.write("examining subwindow %d which has %d unmasked SNPs\n" %(subWinIndex, len(snpIndicesInSubWinUnmasked)))
            if len(snpIndicesInSubWinUnmasked) > 0:
                genosInSubWin = genos.subset(sel0=snpIndicesInSubWinUnmasked)
                for statName in statNames:
                    calcAndAppendStatValDiplo(alleleCountsUnmaskedOnly, positionArrayUnmaskedOnly, statName, subWinStart, \
                                 subWinEnd, statVals, instanceIndex, subWinIndex, genosInSubWin, unmasked)
            else:
                for statName in statNames:
                    appendStatValsForMonomorphic(statName, statVals, instanceIndex, subWinIndex)
    numInstancesDone += 1
    sys.stderr.write("finished %d reps after %f seconds\n" %(numInstancesDone, time.perf_counter()-start))

if numInstancesDone != numInstances:
    sys.exit("Expected %d reps but only processed %d. Perhaps we are using malformed simulation output!\n" %(numInstancesDone, numInstances))

statFiles = []
if outStatsDir.lower() != "none":
    for subWinIndex in range(numSubWins):
        statFileName = "%s/%s.%d.stats" %(outStatsDir, trainingDataFileName.split("/")[-1].rstrip(".gz"), subWinIndex)
        statFiles.append(open(statFileName, "w"))
        statFiles[-1].write("\t".join(statNames) + "\n")
with open(fvecFileName, "w") as fvecFile:
    fvecFile.write(header + "\n")
    for i in range(numInstancesDone):
        statLines = []
        for subWinIndex in range(numSubWins):
            statLines.append([])
        outVec = []
        for statName in statNames:
            outVec += normalizeFeatureVec(statVals[statName][i])
            for subWinIndex in range(numSubWins):
                statLines[subWinIndex].append(statVals[statName][i][subWinIndex])
        if statFiles:
            for subWinIndex in range(numSubWins):
                statFiles[subWinIndex].write("\t".join([str(x) for x in statLines[subWinIndex]]) + "\n")
        fvecFile.write("\t".join([str(x) for x in outVec]) + "\n")

if statFiles:
    for subWinIndex in range(numSubWins):
        statFiles[subWinIndex].close()

sys.stderr.write("total time spent calculating summary statistics and generating feature vectors: %f secs\n" %(time.perf_counter()-start))
closeMsOutFile(trainingDataFileObj)
