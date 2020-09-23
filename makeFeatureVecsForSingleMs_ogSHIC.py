import sys
import allel
import random
import numpy as np
import msTools
import fvTools
import time

'''usage example
python makeFeatureVecsForSingleMsFileDiploid.py /san/data/dan/simulations/discoal_multipopStuff/spatialSVMSims/trainingSets/equilibNeut.msout.gz 110000 11 /san/data/ag1kg/accessibility/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP3.accessible.fa /san/data/ag1kg/outgroups/anc.meru_mela.fa 2L,2R,3L,3R 0.25 0.01 trainingSetsStats/ trainingSetsFeatureVecs/equilibNeut.msout.gz.fvec
''' # NOQA

trainingDataFileName, totalPhysLen, numSubWins, maskFileName, \
    ancFileName, chrArmsForMasking, unmaskedFracCutoff, pMisPol, \
    outStatsDir, fvecFileName = sys.argv[1:]
totalPhysLen = int(totalPhysLen)
numSubWins = int(numSubWins)
pMisPol = float(pMisPol)
# below was the old call to get the iHS normalizations
# standardizationInfo = readStatsDafsComputeStandardizationBins(statAndDafFileName, nBins=50, pMisPol=pMisPol) # NOQA
subWinLen = totalPhysLen//numSubWins
assert totalPhysLen % numSubWins == 0 and numSubWins > 1
chrArmsForMasking = chrArmsForMasking.split(",")

sys.stderr.write("file name='%s'" % (trainingDataFileName))

trainingDataFileObj, sampleSize, numInstances =\
        msTools.openMsOutFileForSequentialReading(trainingDataFileName)

if maskFileName.lower() in ["none", "false"]:
    sys.stderr.write(
        "maskFileName='%s': not doing any masking!\n" % (maskFileName))
    maskFileName = False
    unmaskedFracCutoff = 1.0
else:
    unmaskedFracCutoff = float(unmaskedFracCutoff)
    if unmaskedFracCutoff > 1.0:
        sys.exit(
            "unmaskedFracCutoff must lie within [0, 1].\n")


def getSubWinBounds(subWinLen, totalPhysLen):  # get inclusive subwin bounds
    subWinStart = 1
    subWinEnd = subWinStart + subWinLen - 1
    subWinBounds = [(subWinStart, subWinEnd)]
    numSubWins = totalPhysLen//subWinLen
    for i in range(1, numSubWins-1):
        subWinStart += subWinLen
        subWinEnd += subWinLen
        subWinBounds.append((subWinStart, subWinEnd))
    subWinStart += subWinLen
    # if our subwindows are 1 bp too short due to rounding error,
    # the last window picks up all of the slack
    subWinEnd = totalPhysLen
    subWinBounds.append((subWinStart, subWinEnd))
    return subWinBounds


if not maskFileName:
    unmasked = [True] * totalPhysLen
else:
    drawWithReplacement = False
    if ancFileName.lower() in ["none", "false"]:
        maskData = fvTools.readMaskDataForTraining(
            maskFileName, totalPhysLen, subWinLen,
            chrArmsForMasking, shuffle=True, cutoff=unmaskedFracCutoff)
    else:
        maskData = fvTools.readMaskAndAncDataForTraining(
            maskFileName, ancFileName, totalPhysLen, subWinLen,
            chrArmsForMasking, shuffle=True, cutoff=unmaskedFracCutoff)
    if len(maskData) < numInstances:
        sys.stderr.write("Warning: didn't get enough windows from masked data (needed %d; got %d); will draw with replacement!!\n" % ( # NOQA
            numInstances, len(maskData)))
        drawWithReplacement = True
    else:
        sys.stderr.write("Got enough windows from masked data (needed %d; got %d); will draw without replacement.\n" % ( # NOQA
            numInstances, len(maskData)))


def getSnpIndicesInSubWins(subWinBounds, snpLocs):
    snpIndicesInSubWins = []
    for subWinIndex in range(len(subWinBounds)):
        snpIndicesInSubWins.append([])

    subWinIndex = 0
    for i in range(len(snpLocs)):
        while not (snpLocs[i] >= subWinBounds[subWinIndex][0] and
                   snpLocs[i] <= subWinBounds[subWinIndex][1]):
            subWinIndex += 1
        snpIndicesInSubWins[subWinIndex].append(i)
    return snpIndicesInSubWins


subWinBounds = getSubWinBounds(subWinLen, totalPhysLen)
statNames = ["pi", "thetaW", "tajD", "thetaH", "fayWuH", "maxFDA", "HapCount",
             "H1", "H12", "H2/H1", "ZnS", "Omega", "distVar",
             "distSkew", "distKurt"]
header = []
for statName in statNames:
    for i in range(numSubWins):
        header.append("%s_win%d" % (statName, i))
header = "\t".join(header)


statVals = {}
for statName in statNames:
    statVals[statName] = []
start = time.perf_counter()
numInstancesDone = 0
for instanceIndex in range(numInstances):
    hapArrayIn, positionArray = msTools.readNextMsRepToHaplotypeArrayIn(
        trainingDataFileObj, sampleSize, totalPhysLen)

    snpIndicesInSubWins = getSnpIndicesInSubWins(subWinBounds, positionArray)
    haps = allel.HaplotypeArray(hapArrayIn, dtype='i1')
    if maskFileName:
        if drawWithReplacement:
            unmasked = random.choice(maskData)
        else:
            unmasked = maskData[instanceIndex]
        assert len(unmasked) == totalPhysLen
    genos = haps.to_genotypes(ploidy=2)
    unmaskedSnpIndices = [i for i in range(
        len(positionArray)) if unmasked[positionArray[i]-1]]
    if len(unmaskedSnpIndices) == 0:
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            for statName in statNames:
                fvTools.appendStatValsForMonomorphic(
                    statName, statVals, instanceIndex, subWinIndex)
    else:
        positionArrayUnmaskedOnly = [positionArray[i]
                                     for i in unmaskedSnpIndices]
        ac = genos.count_alleles()
        alleleCountsUnmaskedOnly = allel.AlleleCountsArray(
            np.array([ac[i] for i in unmaskedSnpIndices]))
        sampleSizes = [sum(x) for x in alleleCountsUnmaskedOnly]
        assert len(set(sampleSizes)) == 1 and sampleSizes[0] == sampleSize
        if pMisPol > 0:
            alleleCountsUnmaskedOnly = fvTools.misPolarizeAlleleCounts(
                alleleCountsUnmaskedOnly, pMisPol)
        # dafs = alleleCountsUnmaskedOnly[:,1]/float(sampleSizes[0])
        unmaskedHaps = haps.subset(sel0=unmaskedSnpIndices)
        unmaskedGenos = genos.subset(sel0=unmaskedSnpIndices)
        precomputedStats = {}
        for statName in statNames:
            statVals[statName].append([])
        for subWinIndex in range(numSubWins):
            subWinStart, subWinEnd = subWinBounds[subWinIndex]
            unmaskedFrac = unmasked[subWinStart -
                                    1:subWinEnd].count(True)/float(subWinLen)
            assert unmaskedFrac >= unmaskedFracCutoff
            snpIndicesInSubWinUnmasked = [
                x for x in snpIndicesInSubWins[subWinIndex] if unmasked[positionArray[x]-1]] # NOQA
            if len(snpIndicesInSubWinUnmasked) > 0:
                hapsInSubWin = haps.subset(sel0=snpIndicesInSubWinUnmasked)
                genosInSubWin = genos.subset(sel0=snpIndicesInSubWinUnmasked)
                for statName in statNames:
                    fvTools.calcAndAppendStatVal(alleleCountsUnmaskedOnly,
                                                 positionArrayUnmaskedOnly,
                                                 statName, subWinStart,
                                                 subWinEnd, statVals,
                                                 instanceIndex, subWinIndex,
                                                 hapsInSubWin, unmasked,
                                                 precomputedStats)
            else:
                for statName in statNames:
                    fvTools.appendStatValsForMonomorphic(
                        statName, statVals, instanceIndex, subWinIndex)
    numInstancesDone += 1

if numInstancesDone != numInstances:
    sys.exit("Expected %d reps but only processed %d. Perhaps we are using malformed simulation output!\n" % ( # NOQA
        numInstancesDone, numInstances))

statFiles = []
if outStatsDir.lower() != "none":
    for subWinIndex in range(numSubWins):
        statFileName = "%s/%s.%d.stats" % (
            outStatsDir, trainingDataFileName.split("/")[-1].rstrip(".gz"),
            subWinIndex)
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
            # print(statName)
            outVec += fvTools.normalizeFeatureVec(statVals[statName][i])
            for subWinIndex in range(numSubWins):
                statLines[subWinIndex].append(
                    statVals[statName][i][subWinIndex])
        if statFiles:
            for subWinIndex in range(numSubWins):
                statFiles[subWinIndex].write(
                    "\t".join([str(x) for x in statLines[subWinIndex]]) + "\n")
        fvecFile.write("\t".join([str(x) for x in outVec]) + "\n")

if statFiles:
    for subWinIndex in range(numSubWins):
        statFiles[subWinIndex].close()

sys.stderr.write("total time spent calculating summary statistics and generating feature vectors: %f secs\n" % ( # NOQA
    time.perf_counter()-start))
msTools.closeMsOutFile(trainingDataFileObj)
