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
    sys.stderr.write("vcfForMaskFileName='%s': not masking any genotypes!" %(vcfForMaskFileName))
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
    maskData, genoMaskData = readMaskDataForTraining(maskFileName, totalPhysLen, subWinLen, chrArmsForMasking, shuffle=True, cutoff=unmaskedFracCutoff,
                                                     genoCutoff=unmaskedGenoFracCutoff, vcfForMaskFileName=vcfForMaskFileName, pop=popForMask,
                                                     sampleToPopFileName=sampleToPopFileName)
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
start = time.clock()
numInstancesDone = 0
for instanceIndex in range(numInstances):
    hapArrayIn, positionArray = readNextMsRepToHaplotypeArrayIn(trainingDataFileObj, sampleSize, totalPhysLen)

    snpIndicesInSubWins = getSnpIndicesInSubWins(subWinBounds, positionArray)
    haps = allel.HaplotypeArray(hapArrayIn, dtype='i1')
    if maskFileName:
        if drawWithReplacement:
            randIndex = random.choice(len(maskData))
            unmasked = maskData[randIndex], genoMaskData[randIndex]
        else:
            unmasked = maskData[instanceIndex], genoMaskData[instanceIndex]
        assert len(unmasked) == totalPhysLen
    genos = haps.to_genotypes(ploidy=2)
    unmaskedSnpIndices = [i for i in range(len(positionArray)) if unmasked[positionArray[i]-1]]
    if len(unmaskedSnpIndices) == 0:
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            for statName in statNames:
                appendStatValsForMonomorphic(statName, statVals, instanceIndex, subWinIndex)
    else:
        positionArrayUnmaskedOnly = [positionArray[i] for i in unmaskedSnpIndices]
        if maskFileName:
            preMaskCount = np.sum(genos.count_alleles())
            genos = maskGenos(genos.subset(sel0=unmaskedSnpIndices), genoMaskData[instanceIndex])
            ac = genos.count_alleles()
            sys.stderr.write("%d genotypes masked for rep %d\n" %(preMaskCount - np.sum(ac), instanceIndex))
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            subWinStart, subWinEnd = subWinBounds[subWinIndex]
            unmaskedFrac = unmasked[subWinStart-1:subWinEnd].count(True)/float(subWinLen)
            assert unmaskedFrac >= unmaskedFracCutoff
            snpIndicesInSubWinUnmasked = [x for x in snpIndicesInSubWins[subWinIndex] if unmasked[positionArray[x]-1]]
            if len(snpIndicesInSubWinUnmasked) > 0:
                genosInSubWin = genos.subset(sel0=snpIndicesInSubWinUnmasked)
                for statName in statNames:
                    calcAndAppendStatValDiplo(alleleCountsUnmaskedOnly, positionArrayUnmaskedOnly, statName, subWinStart, \
                                 subWinEnd, statVals, instanceIndex, subWinIndex, genosInSubWin, unmasked)
            else:
                for statName in statNames:
                    appendStatValsForMonomorphic(statName, statVals, instanceIndex, subWinIndex)
    numInstancesDone += 1

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

sys.stderr.write("total time spent calculating summary statistics and generating feature vectors: %f secs\n" %(time.clock()-start))
closeMsOutFile(trainingDataFileObj)
