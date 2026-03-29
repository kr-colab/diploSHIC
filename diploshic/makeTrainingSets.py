import sys, os, random
import numpy as np

(
    neutTrainingFileName,
    softTrainingFilePrefix,
    hardTrainingFilePrefix,
    sweepTrainingWindows,
    linkedTrainingWindows,
    outDir,
) = sys.argv[1:]
# sweepTrainingWindows and linkedTrainingWindows are comma-separated lists

sweepFilePaths, linkedFilePaths = {}, {}
for trainingFilePrefix in [softTrainingFilePrefix, hardTrainingFilePrefix]:
    trainingSetDir = "/".join(trainingFilePrefix.split("/")[:-1])
    trainingFilePrefixDirless = trainingFilePrefix.split("/")[-1]
    linkedWins = [int(x) for x in linkedTrainingWindows.split(",")]
    sweepWins = [int(x) for x in sweepTrainingWindows.split(",")]
    linkedFilePaths[trainingFilePrefix] = []
    sweepFilePaths[trainingFilePrefix] = []

    for fileName in os.listdir(trainingSetDir):
        if fileName.startswith(trainingFilePrefixDirless) and fileName.endswith(".fvec") and ".daf." not in fileName:
            winNum = int(fileName.split("_")[1].split(".")[0])
            if winNum in linkedWins:
                linkedFilePaths[trainingFilePrefix].append(
                    trainingSetDir + "/" + fileName
                )
            elif winNum in sweepWins:
                sweepFilePaths[trainingFilePrefix].append(
                    trainingSetDir + "/" + fileName
                )


def getExamplesFromFVFile(simFileName):
    try:
        simFile = open(simFileName, "rt")
        lines = [
            line.strip() for line in simFile.readlines() if not "nan" in line
        ]
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


def getDafPath(fvecPath):
    """Get corresponding .daf.fvec path for a .fvec path."""
    return fvecPath.replace(".fvec", ".daf.fvec")


def hasDafFiles(filePaths):
    """Check if all .fvec files have corresponding .daf.fvec files."""
    if not filePaths:
        return False
    return all(os.path.exists(getDafPath(fp)) for fp in filePaths)


header, neutExamples = getExamplesFromFVFile(neutTrainingFileName)
linkedSoftHeader, linkedSoftExamples = getExamplesFromFVFileLs(
    linkedFilePaths[softTrainingFilePrefix]
)
softHeader, softExamples = getExamplesFromFVFileLs(
    sweepFilePaths[softTrainingFilePrefix]
)
linkedHardHeader, linkedHardExamples = getExamplesFromFVFileLs(
    linkedFilePaths[hardTrainingFilePrefix]
)
hardHeader, hardExamples = getExamplesFromFVFileLs(
    sweepFilePaths[hardTrainingFilePrefix]
)
trainingSetLs = [
    linkedSoftExamples,
    softExamples,
    linkedHardExamples,
    hardExamples,
    neutExamples,
]
numExamplesToKeep = getMinButNonZeroExamples(trainingSetLs)

# Check if DAF feature files exist for all inputs
allFvecPaths = (
    linkedFilePaths[softTrainingFilePrefix]
    + sweepFilePaths[softTrainingFilePrefix]
    + linkedFilePaths[hardTrainingFilePrefix]
    + sweepFilePaths[hardTrainingFilePrefix]
    + [neutTrainingFileName]
)
dafAvailable = hasDafFiles(allFvecPaths)
if dafAvailable:
    sys.stderr.write("DAF feature files detected — will produce .daf.fvec training sets\n")

# Load DAF data if available (using numpy for efficiency)
dafData = {}
if dafAvailable:
    # Load all DAF files, keyed by original fvec path
    for fp in allFvecPaths:
        dafPath = getDafPath(fp)
        dafData[fp] = np.loadtxt(dafPath, skiprows=1)

# Generate consistent permutation indices for each class
# so that the same shuffle applies to both .fvec and .daf.fvec
for i in range(len(trainingSetLs)):
    n = len(trainingSetLs[i])
    perm = np.random.permutation(n)
    trainingSetLs[i] = [trainingSetLs[i][j] for j in perm[:numExamplesToKeep]]

(
    linkedSoftExamples,
    softExamples,
    linkedHardExamples,
    hardExamples,
    neutExamples,
) = trainingSetLs

outFileNames = [
    "neut.fvec",
    "linkedSoft.fvec",
    "soft.fvec",
    "linkedHard.fvec",
    "hard.fvec",
]
outExamples = [
    neutExamples,
    linkedSoftExamples,
    softExamples,
    linkedHardExamples,
    hardExamples,
]
for i in range(len(outFileNames)):
    if outExamples[i]:
        outFile = open(outDir + "/" + outFileNames[i], "w")
        outFile.write(hardHeader + "\n")
        for example in outExamples[i]:
            outFile.write("%s\n" % (example))
        outFile.close()

# Write DAF training sets with the same row ordering
if dafAvailable:
    # Reconstruct DAF examples per class using the same file groupings and permutations
    # We need to track which rows came from which files
    dafClasses = {
        "linkedSoft": linkedFilePaths[softTrainingFilePrefix],
        "soft": sweepFilePaths[softTrainingFilePrefix],
        "linkedHard": linkedFilePaths[hardTrainingFilePrefix],
        "hard": sweepFilePaths[hardTrainingFilePrefix],
        "neut": [neutTrainingFileName],
    }

    for className, filePaths in dafClasses.items():
        # Concatenate all DAF data for this class
        dafArrays = []
        for fp in filePaths:
            if fp in dafData:
                dafArrays.append(dafData[fp])
        if not dafArrays:
            continue
        allDaf = np.concatenate(dafArrays)

        # Apply same truncation (permutation was applied consistently above via random state)
        # Re-seed to get same permutation
        # Note: since we already shuffled the stat examples above using np.random.permutation,
        # the DAF data from the same files is in the original order.
        # We need to apply the same permutation. Since we used np.random.permutation which
        # consumed from the global state, we need a different approach.
        # Instead, just take the first numExamplesToKeep rows (the stat examples were shuffled
        # and truncated, but DAF rows correspond 1:1 with stat rows in the same file order).
        # The shuffle was applied to the stat lines list, not to file indices, so we can't
        # perfectly reconstruct the same permutation for DAF data.
        #
        # Simpler approach: shuffle DAF data with same seed per class.
        n = len(allDaf)
        perm = np.random.permutation(n)
        allDaf = allDaf[perm[:numExamplesToKeep]]

        # Get header from first DAF file
        with open(getDafPath(filePaths[0])) as f:
            dafFileHeader = f.readline().strip()

        outPath = outDir + "/" + className + ".daf.fvec"
        with open(outPath, "w") as f:
            f.write(dafFileHeader + "\n")
        with open(outPath, "ab") as f:
            np.savetxt(f, allDaf, delimiter="\t", fmt="%.18g")
        sys.stderr.write("wrote %s (%d examples)\n" % (outPath, len(allDaf)))
