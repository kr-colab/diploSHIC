import sys, os
"""
python generateSimLaunchScript.py
This script emits a bash launch script for running all of the training/testing simulations required by discoal
usage example: python generateSimLaunchScript.py popsizehist.txt train test > launch_script.sh

You can then run `bash launch_script.sh` in order to run the simulations;  if they take too long then you 
will have to do them in batches on a cluster.
The input arguments and other hard-coded parameters (which may need to be modified by the user) are explained
in the comments below.
"""

# popSizeFileName specifies the path to a 2-column tab-separated file of the population's history of size changes
# column 1: time, column 2: year at which the population size changed to that time (going backwards)
# the first line in this file should correspond to the present day population size (year=0)
# if we are not simulating size changes then no additional lines are required
popSizeFileName = sys.argv[1]
# these two files specify paths to the directories where we will save our simulation output
trainingOutDir, testOutDir = sys.argv[2:]

# these parameters may have to be adjusted depending on population size history (which may cause theta to be too large/small)
# also, after performing simulation and calculating summary statistics, stats should be spot checked to make sure
# that when a sweep occurs in the central subwindow, the effect has mostly dissipated in the outer windows
# alpha/rho may have to be adjusted in order to achieve this
trainingSampleNumber = 2000 #the number of simulation replicates we want to generate for each file in our training set
testSampleNumber = 1000 #the number of simulations to create for each file in the test set
sampleSize = 100 #the number of individuals in our population sample
numSites = 55000 #total number of sites in our simulated window (i.e. S/HIC's subwindow size * 11)
u = 3.5e-9 #per-site mutation rate (used to calculate mean theta)
gensPerYear = 11.0 #number of generations per year
maxSoftSweepInitFreq = 0.1 #maximum initial selected frequency for soft sweeps
tauHigh = 0.05 #maximum FIXATION (not mutation) time (in units of 4N generations ago) in the past
rhoOverTheta = 5.0 #crossover rate over mut rate (used to calculate mean rho)

sizeChanges = []
with open(popSizeFileName) as popSizeFile:
    first = True
    for line in popSizeFile:
        year, ne = line.strip().split()
        year, ne = float(year), float(ne)
        if first:
            first = False
            ne0 = ne
            prevSizeRatio = 1.0
        else:
            t, sizeRatio = year*gensPerYear/(4*ne0), ne/ne0
            if abs(sizeRatio - prevSizeRatio) > 1e-9:
                sizeChanges.append((t, sizeRatio, ne))
                prevSizeRatio = sizeRatio

N0 = ne0
thetaMean=4*N0*u*numSites
rhoMean = thetaMean * rhoOverTheta
thetaLow = (2*thetaMean)/11.0
thetaHigh = 10*thetaLow
rhoMax = 3 * rhoMean

alphaHigh = 2*N0*0.005 # max selection coefficient s is 0.005
alphaLow = 2*N0*0.0001 # mininum selection coefficient s at 0.0001

selStr = " -ws 0 -Pa %f %f -Pu 0 %f" %(alphaLow, alphaHigh, tauHigh)
partialSelStr = " -ws 0 -Pa %f %f" %(alphaLow, alphaHigh)
softStr = " -Pf 0 %f" %(maxSoftSweepInitFreq)

demogStr = ""
for t, sizeRatio, ne in sizeChanges:
    demogStr += " -en %f 0 %f" %(t, sizeRatio)

sweepLocStr = " -x $x"

print("#!/bin/bash")
for sampleNumber, outDir, simTitle in [(trainingSampleNumber, trainingOutDir, "training data"), (testSampleNumber, testOutDir, "test data")]:
    partialStr = "-Pc 0.2 0.99"
    print("\n#generating %s\n" %(simTitle))
    neutDiscoalCmd = "discoal %d %d %d -Pt %f %f -Pre %f %f%s" %(sampleSize, sampleNumber, numSites, thetaLow, thetaHigh, rhoMean, rhoMax, demogStr)
    print("%s > %s/Neut.msOut" %(neutDiscoalCmd, outDir))
    print("i=0")
    print("for x in 0.045454545454545456 0.13636363636363635 0.22727272727272727 0.3181818181818182 0.4090909090909091 0.5 0.5909090909090909 0.6818181818181818 0.7727272727272727 0.8636363636363636 0.9545454545454546;\ndo")
    hardDiscoalCmd = neutDiscoalCmd + selStr + sweepLocStr
    print("    %s > %s/Hard_$i.msOut" %(hardDiscoalCmd, outDir))
    softDiscoalCmd = neutDiscoalCmd + selStr + softStr + sweepLocStr
    print("    %s > %s/Soft_$i.msOut" %(softDiscoalCmd, outDir))
    print("    i=$((i + 1))\ndone")
    print("")
