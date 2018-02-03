import sys, gzip, bisect, subprocess, os
from io import StringIO

def getSnpsOverflowingChr(newPositions, totalPhysLen):
    overflowers = []
    for i in reversed(range(len(newPositions))):
        if newPositions[i] > totalPhysLen:
            overflowers.append(newPositions[i])
    return overflowers

def fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers):
    posH = {}
    for pos in newPositions:
        posH[pos]=1
    for i in range(len(overflowers)):
        del newPositions[-1]
    for pos in reversed(range(1, totalPhysLen+1)):
        if not pos in posH:
            bisect.insort_left(newPositions, pos)
            overflowers.pop()
            if len(overflowers) == 0:
                break

def msPositionsToIntegerPositions(positions, totalPhysLen):
    snpNum = 1
    prevPos = -1
    prevIntPos = -1
    newPositions = []
    for position in positions:
        assert position >= prevPos
        origPos = position
        if position == prevPos:
            position += 0.000001
        prevPos = origPos

        intPos = int(totalPhysLen*position)
        if intPos == 0:
            intPos = 1
        if intPos <= prevIntPos:
            intPos = prevIntPos + 1
        prevIntPos = intPos
        newPositions.append(intPos)
    overflowers = getSnpsOverflowingChr(newPositions, totalPhysLen)
    if overflowers:
        fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers)
    #sys.stderr.write("number of snps: %d (new) vs. %d (old)\n" %(len(newPositions), len(positions)))
    assert len(newPositions) == len(positions)
    assert all(newPositions[i] <= newPositions[i+1] for i in range(len(newPositions)-1))
    assert newPositions[-1] <= totalPhysLen
    return newPositions

def msRepToHaplotypeArrayIn(samples, positions, totalPhysLen):
    for i in range(len(samples)):
        assert len(samples[i]) == len(positions)

    positions = msPositionsToIntegerPositions(positions, totalPhysLen)

    hapArrayIn = []
    for j in range(len(positions)):
        hapArrayIn.append([])
        for i in range(len(samples)):
            hapArrayIn[j].append(samples[i][j])
    return hapArrayIn, positions

def msOutToHaplotypeArrayIn(msOutputFileName, totalPhysLen):
    if msOutputFileName == "stdin":
        isFile = False
        msStream = sys.stdin
    else:
        isFile = True
        if msOutputFileName.endswith(".gz"):
            msStream = gzip.open(msOutputFileName,'rt')
        else:
            msStream = open(msOutputFileName)

    header = msStream.readline()
    program,numSamples,numSims = header.strip().split()[:3]
    numSamples,numSims = int(numSamples),int(numSims)

    hapArraysIn = []
    positionArrays = []
    #advance to first simulation
    line = msStream.readline()
    while line.strip() != "//":
        line = msStream.readline()
    while line:
        if line.strip() != "//":
            sys.exit("Malformed ms-style output file: read '%s' instead of '//'. AAAARRRRGGHHH!!!!!\n" %(line.strip()))
        segsitesBlah,segsites = msStream.readline().strip().split()
        segsites = int(segsites)
        if segsitesBlah != "segsites:":
            sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
        if segsites == 0:
            positions = []
            hapArrayIn = []
            for i in range(numSamples):
                hapArrayIn.append([])
        else:
            positionsLine = msStream.readline().strip().split()
            if not positionsLine[0] == "positions:":
                sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
            positions = [float(x) for x in positionsLine[1:]]

            samples = []
            for i in range(numSamples):
                sampleLine = msStream.readline().strip()
                if len(sampleLine) != segsites:
                    sys.exit("Malformed ms-style output file %s segsites but %s columns in line: %s; line %s of %s samples AAAARRRRGGHHH!!!!!\n" %(segsites,len(sampleLine),sampleLine,i,numSamples))
                samples.append(sampleLine)
            if len(samples) != numSamples:
                raise Exception
            hapArrayIn, positions = msRepToHaplotypeArrayIn(samples, positions, totalPhysLen)
        hapArraysIn.append(hapArrayIn)
        positionArrays.append(positions)
        line = msStream.readline()
        #advance to the next non-empty line or EOF
        while line and line.strip() == "":
            line = msStream.readline()
        #sys.stderr.write("finished rep %d\n" %(len(hapArraysIn)))
    if len(hapArraysIn) != numSims:
        sys.exit("Malformed ms-style output file: %s of %s sims processed. AAAARRRRGGHHH!!!!!\n" %(len(hapArraysIn), numSims))

    if isFile:
        msStream.close()
    return hapArraysIn, positionArrays

def openMsOutFileForSequentialReading(msOutputFileName):
    if msOutputFileName == "stdin":
        isFile = False
        msStream = sys.stdin
    else:
        isFile = True
        if msOutputFileName.endswith(".gz"):
            msStream = gzip.open(msOutputFileName,'rt')
        else:
            msStream = open(msOutputFileName)

    header = msStream.readline()
    program,numSamples,numSims = header.strip().split()[:3]
    numSamples,numSims = int(numSamples),int(numSims)

    return (msStream, isFile), numSamples, numSims

def closeMsOutFile(fileInfoTuple):
    msStream, isFile = fileInfoTuple
    if isFile:
        msStream.close()

def readNextMsRepToHaplotypeArrayIn(fileInfoTuple, numSamples, totalPhysLen):
    msStream, isFile = fileInfoTuple

    #advance to next simulation
    line = msStream.readline()
    while not line.strip().startswith("//"):
        line = msStream.readline()

    segsitesBlah,segsites = msStream.readline().strip().split()
    segsites = int(segsites)
    if segsitesBlah != "segsites:":
        sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
    if segsites == 0:
        positions = []
        hapArrayIn = []
        for i in range(numSamples):
            hapArrayIn.append([])
    else:
        positionsLine = msStream.readline().strip().split()
        if not positionsLine[0] == "positions:":
            sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
        positions = [float(x) for x in positionsLine[1:]]

        samples = []
        for i in range(numSamples):
            sampleLine = msStream.readline().strip()
            if len(sampleLine) != segsites:
                sys.exit("Malformed ms-style output file %s segsites but %s columns in line: %s; line %s of %s samples AAAARRRRGGHHH!!!!!\n" %(segsites,len(sampleLine),sampleLine,i,numSamples))
            samples.append(sampleLine)
        if len(samples) != numSamples:
            raise Exception
        hapArrayIn, positions = msRepToHaplotypeArrayIn(samples, positions, totalPhysLen)

    line = msStream.readline()

    return hapArrayIn, positions
