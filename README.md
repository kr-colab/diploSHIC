# diploS/HIC
This repo contains the implementation for `diploS/HIC` as described in Kern and Schrider (2018; https://doi.org/10.1534/g3.118.200262), along 
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
- pandas
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

calculate feature vectors, train, or predict with diploSHIC

possible modes (enter 'python diploSHIC.py modeName -h' for modeName's help message:
  {fvecSim,makeTrainingSets,train,fvecVcf,predict}
                        sub-command help
    fvecSim             Generate feature vectors from simulated data
    makeTrainingSets    Combine feature vectors from muliple fvecSim runs into
                        5 balanced training sets
    train               train and test a shic CNN
    fvecVcf             Generate feature vectors from data in a VCF file
    predict             perform prediction using an already-trained SHIC CNN

optional arguments:
  -h, --help            show this help message and exit
```
### before running diploSHIC: simulating training/testing data
All flavors of S/HIC require simulated data for training (and ideally, testing). Users can select whatever simulator 
they prefer and parameterize them however they wish. We have included an example script in this respository 
(generateSimLaunchScript.py) which demonstrates how a training set can be simulated with discoal (available at 
https://github.com/kern-lab/discoal).

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
  fvecFileName          path to file where feature vectors will be written

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
This mode takes five arguments and again has many options. The required arguments are the "shicMode", i.e. whether
to calculate the haploid or diploid summary statistics, the name of the input file, which chromosome to arm to calculate
statistics for, the length of that chromosome, and the name of the output file.

### training the CNN and prediction
Once we have feature vector files ready to go we can train and test our CNN and then finally do prediction on empirical data.

### formatting our training set
Before entering train mode we need to consolidate our training set into 5 files, one for each class. This is done using the
makeTrainingSets mode whose help message is as follows:
```
$ python diploSHIC.py makeTrainingSets -h
usage: diploSHIC.py makeTrainingSets [-h]
                                     neutTrainingFileName
                                     softTrainingFilePrefix
                                     hardTrainingFilePrefix
                                     sweepTrainingWindows
                                     linkedTrainingWindows outDir

required arguments:
  neutTrainingFileName  Path to our neutral feature vectors
  softTrainingFilePrefix
                        Prefix (including higher-level path) of files
                        containing soft training examples; files must end with
                        '_$i.$ext' where $i is the subwindow index of the
                        sweep and $ext is any extension.
  hardTrainingFilePrefix
                        Prefix (including higher-level path) of files
                        containing hard training examples; files must end with
                        '_$i.$ext' where $i is the subwindow index of the
                        sweep and $ext is any extension.
  sweepTrainingWindows  comma-separated list of windows to classify as sweeps
                        (usually just '5' but without the quotes)
  linkedTrainingWindows
                        list of windows to treat as linked to sweeps (usually
                        '0,1,2,3,4,6,7,8,9,10' but without the quotes)
  outDir                path to directory where the training sets will be
                        written

optional arguments:
  -h, --help            show this help message and exit
```
#### train mode
Here is the help message for the train mode of `diploSHIC.py`
```
$ python diploSHIC.py train -h
usage: diploSHIC.py train [-h] [--epochs EPOCHS] [--numSubWins NUMSUBWINS]
                          trainDir testDir outputModel

required arguments:
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
arguments are trainDir (the directory where the training feature vectors
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
                            modelStructure modelWeights predictFile
                            predictFileOutput

required arguments:
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
The predict mode takes as input the two model files output by the train mode, an input file of empirical feature 
vectors, and a file name for the prediction output. 

#### a quick example of the train/predict cycle
We have supplied in the repo some example data that can give you a quick run through the train/predict cycle (we will also
shortly provide a soup-to-nuts example that starts by calculating feature vectors from simulations and ends with prediction of 
genomic data). Let's quickly give that code a spin. The directories `testing/` and `training/` each contain appropriately
formatted diploid feature vectors that are ready to be fed into diploSHIC. First we will train the diploSHIC CNN, but we will
restrict the number of training epochs to 10 to keep things relatively brief (this runs in less than 5 minutes on our server). 
```
$ python diploSHIC.py train training/ testing/ fooModel --epochs 10
```
as it runs a bunch of information monitoring the training of the network will apear. We are tracking the loss and accuracy in the
validation set. When optimization is complete our trained network will be contained in two files, `fooModel.json` and 
`fooModel.weights.hdf5`. The last bit of output from `diploSHIC.py` gives us information about the loss and accuracy on
the held out test data. From the above run my looks like this:
```
evaluation on test set:
diploSHIC loss: 0.404791
diploSHIC accuracy: 0.846800
```
Not bad. In practice I would set the `--epochs` value much higher than 10- the default setting of 100 should suffice in most cases.
Now that we have a trained model we can make predictions on some empirical data. In the repo there is a file called `testEmpirical.fvec`
that we will use as input
```
$ python diploSHIC.py predict fooModel.json fooModel.weights.hdf5 testEmpirical.fvec testEmpirical.preds
```
the output predictions will be saved in `testEmpirical.preds` and should be straightforward to interpret.

### A complete test case
In the interest of showing the user the whole enchilada when it comes to the workflow, I've provided the user
with a more detailed example on the wiki of this repo. That example can be found here: https://github.com/kern-lab/diploSHIC/wiki/A-soup-to-nuts-example

