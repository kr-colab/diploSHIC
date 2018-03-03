import sys, os, random

neutTrainingFileName, softTrainingFilePrefix, hardTrainingFilePrefix, sweepTrainingWindows, linkedTrainingWindows, outDir = sys.argv[1:]
#sweepTrainingWindows and linkedTrainingWindows are comma-separated lists

sweepFilePaths, linkedFilePaths = {}, {}
for trainingFilePrefix in [softTrainingFilePrefix, hardTrainingFilePrefix]:
    trainingSetDir = "/".join(trainingFilePrefix.split("/")[:-1])
    trainingFilePrefixDirless = trainingFilePrefix.split("/")[-1]
    linkedWins = [int(x) for x in linkedTrainingWindows.split(",")]
    sweepWins = [int(x) for x in sweepTrainingWindows.split(",")]
    linkedFilePaths[trainingFilePrefix] = []
    sweepFilePaths[trainingFilePrefix] = []

    for fileName in os.listdir(trainingSetDir):
        if fileName.startswith(trainingFilePrefixDirless):
            winNum = int(fileName.split("_")[1].split(".")[0])
            if winNum in linkedWins:
                linkedFilePaths[trainingFilePrefix].append(trainingSetDir + "/" + fileName)
            elif winNum in sweepWins:
                sweepFilePaths[trainingFilePrefix].append(trainingSetDir + "/" + fileName)

def getExamplesFromFVFile(simFileName):
    try:
        simFile = open(simFileName,'rt')
        lines = [line.strip() for line in simFile.readlines() if not "nan" in line]
        header = lines[0]
        examples = lines[1:]
        simFile.close()
        return header, examples
    except Exception:
        return "", []

def getExamplesFromFVFileLs(simFileLs):
    examples = []
    keptHeader = ""
    for filePath in simFileLs:
        header, currExamples = getExamplesFromFVFile(filePath)
        if header:
            keptHeader = header
        examples += currExamples
    return keptHeader, examples

def getMinButNonZeroExamples(lsLs):
    counts = []
    for ls in lsLs:
        if len(ls) > 0:
            counts.append(len(ls))
    if not counts:
        raise Exception
    return min(counts)

header, neutExamples = getExamplesFromFVFile(neutTrainingFileName)
linkedSoftHeader, linkedSoftExamples = getExamplesFromFVFileLs(linkedFilePaths[softTrainingFilePrefix])
softHeader, softExamples = getExamplesFromFVFileLs(sweepFilePaths[softTrainingFilePrefix])
linkedHardHeader, linkedHardExamples = getExamplesFromFVFileLs(linkedFilePaths[hardTrainingFilePrefix])
hardHeader, hardExamples = getExamplesFromFVFileLs(sweepFilePaths[hardTrainingFilePrefix])
trainingSetLs = [linkedSoftExamples, softExamples, linkedHardExamples, hardExamples,neutExamples]
numExamplesToKeep = getMinButNonZeroExamples(trainingSetLs)
for i in range(len(trainingSetLs)):
    random.shuffle(trainingSetLs[i])
    trainingSetLs[i] = trainingSetLs[i][:numExamplesToKeep]
linkedSoftExamples, softExamples, linkedHardExamples, hardExamples, neutExamples = trainingSetLs

outFileNames = ["neut.fvec", "linkedSoft.fvec", "soft.fvec", "linkedHard.fvec", "hard.fvec"]
outExamples = [neutExamples, linkedSoftExamples, softExamples, linkedHardExamples, hardExamples]
for i in range(len(outFileNames)):
    if outExamples[i]:
        outFile = open(outDir +"/"+ outFileNames[i], "w")
        outFile.write(hardHeader+"\n")
        for example in outExamples[i]:
            outFile.write("%s\n" %(example))
        outFile.close()
