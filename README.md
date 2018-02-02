# diploS/HIC
This repo contains the implementation for `diploS/HIC` as described in Kern and Schrider (2018), along 
with its associated support scripts. `diploS/HIC` uses a deep convolutional neural network to identify
hard and soft selective sweep in population genomic data. 

The workflow for analysis using `diploS/HIC` consists of four basic parts. 1) Generation of a training set for `diploS/HIC` 
using simulation. 2) `diploS/HIC` training and performance evaluation. 3) Calculation of `dipoS/HIC` feature vectors from genomic data.
4) prediction on empirical data using the trained network. The software provided here can handle the last three parts; population
genetic simulations must be performed using separate software such as discoal (https://github.com/kern-lab/discoal) 

## Installation
`diploS/HIC` has a number of dependencies that should be straightforward to install using python package managers
such as `conda` or `pip`. The complete list of dependencies looks like this:

- numpy
- scipy
- scikit-allel
- scikit-learn
- tensorflow
- keras

## Install on linux
I'm going to focus on the steps involved to install on a linux machine using Anaconda as our python source / main
package manager. First download and install Anaconda

```
$ wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
$ bash Anaconda3-5.0.1-Linux-x86_64.sh
```
That will give us the basics (numpy, scipy, scikit-learn, etc). Next lets install scikit-allel using `conda`
```
$ conda install -c conda-forge scikit-allel
```
That's easy. Installing tensorflow and keras can be slightly more touchy. You will need to determine if 
you want to use a CPU-only implementation (probably) or a GPU implementation of tensorflow. See
https://www.tensorflow.org/install/install_linux for install instructions. I'm going to install the
CPU version for simplicity. tensorflow and keras are the libraries which handle the deep learning
portion of things so it is important to make sure the versions of these libraries play well together 
```
$ pip install tensorflow 
$ pip install keras
```
Note that because I'm using the Anaconda version of python, pip will only install this in the anaconda directory
which is a good thing. Okay that should be the basics of dependencies. Now we are ready to install `diploS/HIC` itself
```
$ git clone https://github.com/kern-lab/diploSHIC.git
$ cd diploSHIC 
$ python setup.py install
```
Assuming all the dependencies were installed this should be all set

## Usage
The main program that you will interface with is `diploSHIC.py`. This script has four run modes that allow the user to 
perform each of the main steps in the supervised machine learning process. We will briefly lay out the modes of use
and then will provide a complete example of how to use the program for fun and profit.

`diploSHIC.py` uses the `argparse` module in python to try to give the user a complete, command line based help menu. 
We can see the top level of this help by typing
```
$ python diploSHIC.py -h
usage: diploSHIC.py [-h] {train,predict,fvecSim,fvecVcf} ...

calculate feature fectors, train, or predict with diploSHIC

possible modes (enter 'python diploSHIC.py modeName -h' for modeName's help message:
  {train,predict,fvecSim,fvecVcf}
                        sub-command help
    train               train and test a shic CNN
    predict             perform prediction using an already-trained SHIC CNN
    fvecSim             Generate feature vectors from simulated data
    fvecVcf             Generate feature vectors from data in a VCF file

optional arguments:
  -h, --help            show this help message and exit
```
### feature vector generation modes
The first task in our pipeline is generating feature vectors from simulation data (or empirical data) to
use with the CNN that we will train and then use for prediction. The `diploSHIC.py` script eases this 
process with two run modes

#### fvecSim mode
The fvecSim run mode is used for turning ms-style output into feature vectors compatible with `diploSHIC.py`. The
help message from this mode looks like this
```
$ python diploSHIC.py fvecSim -h
usage: diploSHIC.py fvecSim [-h] [--totalPhysLen TOTALPHYSLEN]
                            [--numSubWins NUMSUBWINS]
                            [--maskFileName MASKFILENAME]
                            [--chrArmsForMasking CHRARMSFORMASKING]
                            [--unmaskedFracCutoff UNMASKEDFRACCUTOFF]
                            [--outStatsDir OUTSTATSDIR]
                            [--ancFileName ANCFILENAME] [--pMisPol PMISPOL]
                            shicMode msOutFile fvecFileName

required arguments:
  shicMode              specifies whether to use original haploid SHIC (use
                        'haploid') or diploSHIC ('diploid')
  msOutFile             path to simulation output file (must be same format
                        used by Hudson's ms)
  fvecFileName          path to file where feature vectors will be written

optional arguments:
  -h, --help            show this help message and exit
   --totalPhysLen TOTALPHYSLEN
                        Length of simulated chromosome for converting infinite
                        sites ms output to finite sites (default=1100000)
  --numSubWins NUMSUBWINS
                        The number of subwindows that our chromosome will be
                        divided into (default=11)
  --maskFileName MASKFILENAME
                        Path to a fasta-formatted file that contains masking
                        information (marked by 'N'). If specified, simulations
                        will be masked in a manner mirroring windows drawn
                        from this file.
  --chrArmsForMasking CHRARMSFORMASKING
                        A comma-separated list (no spaces) of chromosome arms
                        from which we want to draw masking information (or
                        'all' if we want to use all arms. Ignored if
                        maskFileName is not specified.
  --unmaskedFracCutoff UNMASKEDFRACCUTOFF
                        Minimum fraction of unmasked sites, if masking
                        simulated data
  --outStatsDir OUTSTATSDIR
                        Path to a directory where values of each statistic in
                        each subwindow are recorded for each rep
  --ancFileName ANCFILENAME
                        Path to a fasta-formatted file that contains inferred
                        ancestral states ('N' if unknown). This is used for
                        masking, as sites that cannot be polarized are masked,
                        and we mimic this in the simulted data. Ignored in
                        diploid mode which currently does not use ancestral
                        state information
  --pMisPol PMISPOL     The fraction of sites that will be intentionally
                        polarized to better approximate real data
```
This mode takes three arguments and then offers many options. The arguments are the "shicMode", i.e. whether
to calculate the haploid or diploid summary statistics, the name of the input file, and the name of the output file. 
The various options allow one to account for missing data (via masking), unfolding the site frequency spectrum via the ancestral
states file (haploid only), and a mis-polarization rate of that unfolded site frequency spectrum. Please see the example usage below
for a fleshed out example of how to use these features.

#### fvecVcf mode
The fvecVcf mode is used for calculating feature vectors from data that is stored as a VCF file. 
The help message from this mode is as follows
```
$ python diploSHIC.py fvecVcf -h
usage: diploSHIC.py fvecVcf [-h] [--targetPop TARGETPOP]
                            [--sampleToPopFileName SAMPLETOPOPFILENAME]
                            [--winSize WINSIZE] [--numSubWins NUMSUBWINS]
                            [--maskFileName MASKFILENAME]
                            [--unmaskedFracCutoff UNMASKEDFRACCUTOFF]
                            [--ancFileName ANCFILENAME]
                            [--statFileName STATFILENAME]
                            [--segmentStart SEGMENTSTART]
                            [--segmentEnd SEGMENTEND]
                            shicMode chrArmVcfFile chrArm chrLen

required arguments:
  shicMode              specifies whether to use original haploid SHIC (use
                        'haploid') or diploSHIC ('diploid')
  chrArmVcfFile         VCF format file containing data for our chromosome arm
                        (other arms will be ignored)
  chrArm                Exact name of the chromosome arm for which feature
                        vectors will be calculated
  chrLen                Length of the chromosome arm

optional arguments:
  -h, --help            show this help message and exit
  --targetPop TARGETPOP
                        Population ID of samples we wish to include
  --sampleToPopFileName SAMPLETOPOPFILENAME
                        Path to tab delimited file with population
                        assignments; format: SampleID popID
  --winSize WINSIZE     Length of the large window (default=1100000)
  --numSubWins NUMSUBWINS
                        Number of sub-windows within each large window
                        (default=11)
  --maskFileName MASKFILENAME
                        Path to a fasta-formatted file that contains masking
                        information (marked by 'N'); must have an entry with
                        title matching chrArm
  --unmaskedFracCutoff UNMASKEDFRACCUTOFF
                        Fraction of unmasked sites required to retain a
                        subwindow
  --ancFileName ANCFILENAME
                        Path to a fasta-formatted file that contains inferred
                        ancestral states ('N' if unknown); must have an entry
                        with title matching chrArm. Ignored for diploid mode
                        which currently does not use ancestral state
                        information.
  --statFileName STATFILENAME
                        Path to a file where statistics will be written for
                        each subwindow that is not filtered out
  --segmentStart SEGMENTSTART
                        Left boundary of region in which feature vectors are
                        calculated (whole arm if omitted)
  --segmentEnd SEGMENTEND
                        Right boundary of region in which feature vectors are
                        calculated (whole arm if omitted)
```
This mode takes four arguments and again has many options. The required arguments are the "shicMode", i.e. whether
to calculate the haploid or diploid summary statistics, the name of the input file, which chromosome to arm to calculate
statistics for, and the length of that chromosome.

### train mode
Individual run mode help menus can be brought up in a similar way. For instance here is the help
for the train mode
```
$ python diploSHIC.py train -h
usage: diploSHIC.py train [-h] [--epochs EPOCHS] [--numSubWins NUMSUBWINS]
                          nDims trainDir testDir outputModel

required arguments:
  nDims                 dimensionality of the feature vector
  trainDir              path to training set files
  testDir               path to test set files, can be same as trainDir
  outputModel           file name for output model, will create two files one
                        with structure one with weights

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       max epochs for training CNN (default = 100)
  --numSubWins NUMSUBWINS
                        number of subwindows that our chromosome is divided
                        into (default = 11)
```
As you will see in a moment train mode is used for training the deep learning classifier. Its required
arguments are nDims (12 for diploid, 11 for haploid) , trainDir (the directory where the training feature vectors
are kept), testDir (the directory where the testing feature vectors are kept), and outputModel the file name for the trained
network. One note -- `diploSHIC.py` expects five files named `hard.fvec`, `soft.fvec`, `neut.fvec`, `linkedSoft.fvec`, and 
`linkedHard.fvec` in the training and testing directories. The training and testing directory can be the same directory in 
which case 20% of the training examples are held out for use in testing and validation.

train mode has two options, the number of subwindows used for the feature vectors and the number of training epochs for the
network.

### predict mode
Once a classifier has been trained, one uses the predict mode of `diploSHIC.py` to classify empirical data. Here is the help
statement
```
$ python diploSHIC.py predict -h
usage: diploSHIC.py predict [-h] [--numSubWins NUMSUBWINS]
                            nDims modelStructure modelWeights predictFile
                            predictFileOutput

required arguments:
  nDims                 dimensionality of the feature vector
  modelStructure        path to CNN structure .json file
  modelWeights          path to CNN weights .h5 file
  predictFile           input file to predict
  predictFileOutput     output file name

optional arguments:
  -h, --help            show this help message and exit
  --numSubWins NUMSUBWINS
                        number of subwindows that our chromosome is divided
                        into (default = 11)
```
The predict mode takes as input nDims (as above), the two model files output by the train mode, an input file of empirical feature 
vectors, and a file name for the prediction output. 


