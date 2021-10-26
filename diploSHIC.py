import argparse,time,sys,subprocess

pyExec = sys.executable
if "/" in sys.argv[0]:
    diploShicDir = "/".join(sys.argv[0].split("/")[:-1]) + "/"
else:
    diploShicDir = ""

################# use argparse to get CL args and make sure we are kosher
parser = argparse.ArgumentParser(description='calculate feature vectors, train, or predict with diploSHIC')
parser._positionals.title = 'possible modes (enter \'python diploSHIC.py modeName -h\' for modeName\'s help message'
subparsers = parser.add_subparsers(help='sub-command help')
parser_c = subparsers.add_parser('fvecSim', help='Generate feature vectors from simulated data')
parser_e = subparsers.add_parser('makeTrainingSets', help='Combine feature vectors from muliple fvecSim runs into 5 balanced training sets')
parser_a = subparsers.add_parser('train', help='train and test a shic CNN')
parser_d = subparsers.add_parser('fvecVcf', help='Generate feature vectors from data in a VCF file')
parser_b = subparsers.add_parser('predict', help='perform prediction using an already-trained SHIC CNN')
#parser_a.add_argument('nDims', metavar='nDims', type=int, 
#                   help='dimensionality of the feature vector')
parser_a.add_argument('trainDir', help='path to training set files')
parser_a.add_argument('testDir', help='path to test set files, can be same as trainDir')
parser_a.add_argument('outputModel', help='file name for output model, will create two files one with structure one with weights')
parser_a.add_argument('--epochs', type=int, help='max epochs for training CNN (default = 100)', default=100)
parser_a.add_argument('--numSubWins', type=int, help='number of subwindows that our chromosome is divided into (default = 11)', default=11)
parser_a.add_argument('--confusionFile', help='optional file to which confusion matrix plot will be written (default = None)', default=None)
parser_a.set_defaults(mode='train')
parser_a._positionals.title = 'required arguments'

#parser_b.add_argument('nDims', metavar='nDims', type=int, 
#                   help='dimensionality of the feature vector')
parser_b.add_argument('modelStructure', help='path to CNN structure .json file')
parser_b.add_argument('modelWeights', help='path to CNN weights .h5 file')
parser_b.add_argument('predictFile', help='input file to predict')
parser_b.add_argument('predictFileOutput', help='output file name')
parser_b.add_argument('--numSubWins', type=int, help='number of subwindows that our chromosome is divided into (default = 11)', default=11)
parser_b.add_argument('--simData', help='Are we using simulated input data wihout coordinates?', action="store_true")
parser_b.set_defaults(mode='predict')
parser_b._positionals.title = 'required arguments'

parser_c.add_argument('shicMode', help='specifies whether to use original haploid SHIC (use \'haploid\') or diploSHIC (\'diploid\')',
                          default='diploid')
parser_c.add_argument('msOutFile', help='path to simulation output file (must be same format used by Hudson\'s ms)')
parser_c.add_argument('fvecFileName', help='path to file where feature vectors will be written')
parser_c.add_argument('--totalPhysLen', type=int, help='Length of simulated chromosome for converting infinite sites ms output to finite sites (default=1100000)',
                          default=1100000)
parser_c.add_argument('--numSubWins', type=int, help='The number of subwindows that our chromosome will be divided into (default=11)', default=11)
parser_c.add_argument('--maskFileName', help=('Path to a fasta-formatted file that contains masking information (marked by \'N\'). '
                          'If specified, simulations will be masked in a manner mirroring windows drawn from this file.'), default="None")
parser_c.add_argument('--vcfForMaskFileName', help=('Path to a VCF file that contains genotype information. This will be used to mask genotypes '
                          'in a manner that mirrors how the true data are masked.'), default=None)
parser_c.add_argument('--popForMask', help='The label of the population for which we should draw genotype information from the VCF for masking purposes.',
                          default=None)
parser_c.add_argument('--sampleToPopFileName', help=('Path to tab delimited file with population assignments (used for genotype masking); format: '
                          'SampleID\tpopID'), default="None")
parser_c.add_argument('--unmaskedGenoFracCutoff', type=float, help='Fraction of unmasked genotypes required to retain a site (default=0.75)', default=0.75)
parser_c.add_argument('--chrArmsForMasking', help=('A comma-separated list (no spaces) of chromosome arms from which we want to draw masking '
                          'information (or \'all\' if we want to use all arms. Ignored if maskFileName is not specified.'), default="all")
parser_c.add_argument('--unmaskedFracCutoff', type=float, help='Minimum fraction of unmasked sites, if masking simulated data (default=0.25)', default=0.25)
parser_c.add_argument('--outStatsDir', help='Path to a directory where values of each statistic in each subwindow are recorded for each rep',
                          default="None")
parser_c.add_argument('--ancFileName', help=('Path to a fasta-formatted file that contains inferred ancestral states (\'N\' if unknown).'
                          ' This is used for masking, as sites that cannot be polarized are masked, and we mimic this in the simulted data.'
                          ' Ignored in diploid mode which currently does not use ancestral state information'),
                          default="None")
parser_c.add_argument('--pMisPol', type=float, help='The fraction of sites that will be intentionally polarized to better approximate real data (default=0.0)',
                          default=0.0)
parser_c.set_defaults(mode='fvecSim')
parser_c._positionals.title = 'required arguments'

parser_d.add_argument('shicMode', help='specifies whether to use original haploid SHIC (use \'haploid\') or diploSHIC (\'diploid\')')
parser_d.add_argument('chrArmVcfFile', help='VCF format file containing data for our chromosome arm (other arms will be ignored)')
parser_d.add_argument('chrArm', help='Exact name of the chromosome arm for which feature vectors will be calculated')
parser_d.add_argument('chrLen', type=int, help='Length of the chromosome arm')
parser_d.add_argument('fvecFileName', help='path to file where feature vectors will be written')
parser_d.add_argument('--targetPop', help='Population ID of samples we wish to include', default="None")
parser_d.add_argument('--sampleToPopFileName', help=('Path to tab delimited file with population assignments; format: '
                          'SampleID\tpopID'), default="None")
parser_d.add_argument('--winSize', type=int, help='Length of the large window (default=1100000)', default=1100000)
parser_d.add_argument('--numSubWins', type=int, help='Number of sub-windows within each large window (default=11)', default=11)
parser_d.add_argument('--maskFileName', help=('Path to a fasta-formatted file that contains masking information (marked by \'N\'); '
                          'must have an entry with title matching chrArm'), default="None")
parser_d.add_argument('--unmaskedFracCutoff', type=float, help='Fraction of unmasked sites required to retain a subwindow (default=0.25)', default=0.25)
parser_d.add_argument('--unmaskedGenoFracCutoff', type=float, help='Fraction of unmasked genotypes required to retain a site (default=0.75)', default=0.75)
parser_d.add_argument('--ancFileName', help=('Path to a fasta-formatted file that contains inferred ancestral states (\'N\' if unknown); '
                          'must have an entry with title matching chrArm. Ignored for diploid mode which currently does not use ancestral '
                          'state information.'), default="None")
parser_d.add_argument('--statFileName', help='Path to a file where statistics will be written for each subwindow that is not filtered out',
                         default="None")
parser_d.add_argument('--segmentStart', help='Left boundary of region in which feature vectors are calculated (whole arm if omitted)',
                         default="None")
parser_d.add_argument('--segmentEnd', help='Right boundary of region in which feature vectors are calculated (whole arm if omitted)',
                         default="None")
parser_d.set_defaults(mode='fvecVcf')
parser_d._positionals.title = 'required arguments'

parser_e.add_argument('neutTrainingFileName', help='Path to our neutral feature vectors')
parser_e.add_argument('softTrainingFilePrefix', help=('Prefix (including higher-level path) of files containing soft training examples'
                          '; files must end with \'_$i.$ext\' where $i is the subwindow index of the sweep and $ext is any extension.'))
parser_e.add_argument('hardTrainingFilePrefix', help=('Prefix (including higher-level path) of files containing hard training examples'
                          '; files must end with \'_$i.$ext\' where $i is the subwindow index of the sweep and $ext is any extension.'))
parser_e.add_argument('sweepTrainingWindows', type=int, help=('comma-separated list of windows to classify as sweeps (usually just \'5\''
                          ' but without the quotes)'))
parser_e.add_argument('linkedTrainingWindows', help=('list of windows to treat as linked to sweeps (usually \'0,1,2,3,4,6,7,8,9,10\' but'
                          ' without the quotes)'))
parser_e.add_argument('outDir', help='path to directory where the training sets will be written')
parser_e.set_defaults(mode='makeTrainingSets')
parser_e._positionals.title = 'required arguments'
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
args = parser.parse_args()
argsDict = vars(args)

if argsDict['mode'] in ['train', 'predict']:
###########################################################
# Import a bunch of libraries if everything checks out
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    from keras.models import Sequential, Model
    from keras import optimizers
    from keras.layers import Dense, Dropout, Activation, Flatten, Input
    from keras.layers import Conv2D, MaxPooling2D,concatenate
    from keras.utils import np_utils
    from sklearn.model_selection import train_test_split
    from keras.utils.layer_utils import convert_all_kernels_in_model
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    import keras.backend as K
    import fnmatch

    #nDims = argsDict['nDims']
    numSubWins = argsDict['numSubWins']

if argsDict['mode'] == 'train':
    trainingDir=argsDict['trainDir']
    testingDir=argsDict['testDir']
    epochOption = argsDict['epochs']
    outputModel = argsDict['outputModel']
    confusionFile = argsDict['confusionFile']
    #nCores = 12
    print("loading data now...")
    #training data
    
    hard = np.loadtxt(trainingDir+"hard.fvec",skiprows=1)
    nDims = int(hard.shape[1] / numSubWins)
    h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
    neut = np.loadtxt(trainingDir+"neut.fvec",skiprows=1)
    n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
    soft = np.loadtxt(trainingDir+"soft.fvec",skiprows=1)
    s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))
    lsoft = np.loadtxt(trainingDir+"linkedSoft.fvec",skiprows=1)
    ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,numSubWins))
    lhard = np.loadtxt(trainingDir+"linkedHard.fvec",skiprows=1)
    lh1 = np.reshape(lhard,(lhard.shape[0],nDims,numSubWins))

    both=np.concatenate((h1,n1,s1,ls1,lh1))
    y=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))

    #reshape both to explicitly set depth image. need for theanno not sure with tensorflow
    both = both.reshape(both.shape[0],nDims,numSubWins,1)
    if (trainingDir==testingDir):
        X_train, X_test, y_train, y_test = train_test_split(both, y, test_size=0.2)
    else:
        X_train = both
        y_train = y
        #testing data
        hard = np.loadtxt(testingDir+"hard.fvec",skiprows=1)
        h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
        neut = np.loadtxt(testingDir+"neut.fvec",skiprows=1)
        n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
        soft = np.loadtxt(testingDir+"soft.fvec",skiprows=1)
        s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))
        lsoft = np.loadtxt(testingDir+"linkedSoft.fvec",skiprows=1)
        ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,numSubWins))
        lhard = np.loadtxt(testingDir+"linkedHard.fvec",skiprows=1)
        lh1 = np.reshape(lhard,(lhard.shape[0],nDims,numSubWins))

        both2=np.concatenate((h1,n1,s1,ls1,lh1))
        X_test = both2.reshape(both2.shape[0],nDims,numSubWins,1)
        y_test=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))


    Y_train = np_utils.to_categorical(y_train, 5)
    Y_test = np_utils.to_categorical(y_test, 5)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)

    validation_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)
    test_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)


    #print(X_train.shape)
    print("training set has %d examples" % X_train.shape[0])
    print("validation set has %d examples" % X_valid.shape[0])
    print("test set has %d examples" % X_test.shape[0])
    
    model_in = Input(X_train.shape[1:])
    h = Conv2D(128, 3, activation='relu',padding="same", name='conv1_1')(model_in)
    h = Conv2D(64, 3, activation='relu',padding="same", name='conv1_2')(h)
    h = MaxPooling2D(pool_size=3, name='pool1',padding="same")(h)
    h = Dropout(0.15, name='drop1')(h)
    h = Flatten(name='flaten1')(h)

    dh = Conv2D(128, 2, activation='relu',dilation_rate=[1,3],padding="same", name='dconv1_1')(model_in)
    dh = Conv2D(64, 2, activation='relu',dilation_rate=[1,3],padding="same", name='dconv1_2')(dh)
    dh = MaxPooling2D(pool_size=2, name='dpool1')(dh)
    dh = Dropout(0.15, name='ddrop1')(dh)
    dh = Flatten(name='dflaten1')(dh)

    dh1 = Conv2D(128, 2, activation='relu',dilation_rate=[1,4],padding="same", name='dconv4_1')(model_in)
    dh1 = Conv2D(64, 2, activation='relu',dilation_rate=[1,4],padding="same", name='dconv4_2')(dh1)
    dh1 = MaxPooling2D(pool_size=2, name='d1pool1')(dh1)
    dh1 = Dropout(0.15, name='d1drop1')(dh1)
    dh1 = Flatten(name='d1flaten1')(dh1)

    h =  concatenate([h,dh,dh1])
    h = Dense(512,name="512dense",activation='relu')(h)
    h = Dropout(0.2, name='drop7')(h)
    h = Dense(128,name="last_dense",activation='relu')(h)
    h = Dropout(0.1, name='drop8')(h)
    output = Dense(5,name="out_dense",activation='softmax')(h)
    model = Model(inputs=[model_in], outputs=[output])





    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # define early stopping callback
    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, \
                              verbose=1, mode='auto')
    
    model_json = model.to_json()
    with open(outputModel+".json", "w") as json_file:
        json_file.write(model_json)
    modWeightsFilepath=outputModel+".weights.hdf5"
    checkpoint = ModelCheckpoint(modWeightsFilepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

    callbacks_list = [earlystop,checkpoint]
    #callbacks_list = [earlystop] #turning off checkpointing-- just want accuracy assessment

    datagen.fit(X_train)
    validation_gen.fit(X_valid)
    test_gen.fit(X_test)
    start = time.time()
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), \
                        steps_per_epoch=len(X_train) / 32, epochs=epochOption,verbose=1, \
                        callbacks=callbacks_list, \
                        validation_data=validation_gen.flow(X_valid,Y_valid, batch_size=32), \
                        validation_steps=len(X_test)/32)
    #model.fit(X_train, Y_train, batch_size=32, epochs=100,validation_data=(X_test,Y_test),callbacks=callbacks_list, verbose=1)
    score = model.evaluate_generator(test_gen.flow(X_test,Y_test, batch_size=32),len(Y_test)/32)
    sys.stderr.write("total time spent fitting and evaluating: %f secs\n" %(time.time()-start))

    print("evaluation on test set:")
    print("diploSHIC loss: %f" % score[0])
    print("diploSHIC accuracy: %f" % score[1])
    if confusionFile:
        from misc import plot_confusion_matrix
        import matplotlib.pyplot as plt
        plot_confusion_matrix(model, test_gen.standardize(X_test), Y_test, labels=[0,4,2,3,1], display_labels=['Hard', 'Hard-linked', 'Soft', 'Soft-linked', 'Neutral'], cmap=plt.cm.Blues, normalize='true')
        plt.savefig(confusionFile, bbox_inches='tight')

elif argsDict['mode'] == 'predict':
    import pandas as pd
    from keras.models import model_from_json
    
    #import data from predictFile
    x_df=pd.read_table(argsDict['predictFile'])
    if argsDict['simData']:
        testX = x_df[list(x_df)[:]].to_numpy()
    else:
        testX = x_df[list(x_df)[4:]].to_numpy()
    nDims = int(testX.shape[1]/numSubWins)
    np.reshape(testX,(testX.shape[0],nDims,numSubWins))
    #add channels
    testX = testX.reshape(testX.shape[0],nDims,numSubWins,1)
    #set up generator for normalization 
    validation_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)
    validation_gen.fit(testX)
    
    #import model
    json_file = open(argsDict['modelStructure'], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(argsDict['modelWeights'])
    print("Loaded model from disk")
    
    #get predictions
    preds = model.predict(validation_gen.standardize(testX))
    predictions = np.argmax(preds,axis=1)
    
    #np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1)
    classDict = {0:'hard',1:'neutral',2:'soft',3:'linkedSoft',4:'linkedHard'}
    
    #output the predictions
    outputFile = open(argsDict['predictFileOutput'],'w')
    if argsDict['simData']:
        outputFile.write('predClass\tprob(neutral)\tprob(likedSoft)\tprob(linkedHard)\tprob(soft)\tprob(hard)\n')
    else:
        outputFile.write('chrom\tclassifiedWinStart\tclassifiedWinEnd\tbigWinRange\tpredClass\tprob(neutral)\tprob(likedSoft)\tprob(linkedHard)\tprob(soft)\tprob(hard)\n')

    for index, row in x_df.iterrows():
        if argsDict['simData']:
            outputFile.write('{}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n'.format(classDict[predictions[index]],preds[index][1],preds[index][3],preds[index][4], \
                preds[index][2],preds[index][0]))
        else:
            outputFile.write('{}\t{}\t{}\t{}\t{}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\n'.format( row['chrom'],row['classifiedWinStart'],row['classifiedWinEnd'],row['bigWinRange'], \
                classDict[predictions[index]],preds[index][1],preds[index][3],preds[index][4],preds[index][2],preds[index][0]))
    outputFile.close()
    print("{} predictions complete".format(index+1))
elif argsDict['mode'] == 'fvecSim':
    if argsDict['shicMode'].lower() == 'diploid':
        cmdArgs = [argsDict['msOutFile'], argsDict['totalPhysLen'], argsDict['numSubWins'], argsDict['maskFileName'],
                   argsDict['vcfForMaskFileName'], argsDict['popForMask'], argsDict['sampleToPopFileName'], argsDict['unmaskedGenoFracCutoff'],
                   argsDict['chrArmsForMasking'], argsDict['unmaskedFracCutoff'], argsDict['outStatsDir'], argsDict['fvecFileName']]
        cmd = pyExec + " " + diploShicDir + "makeFeatureVecsForSingleMsDiploid.py " + " ".join([str(x) for x in cmdArgs])
    elif argsDict['shicMode'].lower() == 'haploid':
        cmdArgs = [argsDict['msOutFile'], argsDict['totalPhysLen'], argsDict['numSubWins'], argsDict['maskFileName'], argsDict['ancFileName'],
               argsDict['chrArmsForMasking'], argsDict['unmaskedFracCutoff'], argsDict['pMisPol'], argsDict['outStatsDir'],
               argsDict['fvecFileName']]
        cmd = pyExec + " " + diploShicDir + "makeFeatureVecsForSingleMs_ogSHIC.py " + " ".join([str(x) for x in cmdArgs])
    else:
        sys.exit("'shicMode' must be set to either 'diploid' or 'haploid'")
    print(cmd)
    subprocess.call(cmd.split())
elif argsDict['mode'] == 'fvecVcf':
    if argsDict['shicMode'].lower() == 'diploid':
        cmdArgs = [argsDict['chrArmVcfFile'], argsDict['chrArm'], argsDict['chrLen'], argsDict['targetPop'], argsDict['winSize'],
                   argsDict['numSubWins'], argsDict['maskFileName'], argsDict['unmaskedFracCutoff'], argsDict['unmaskedGenoFracCutoff'],
                   argsDict['sampleToPopFileName'], argsDict['statFileName'], argsDict['fvecFileName']]
        cmd = pyExec + " " + diploShicDir + "makeFeatureVecsForChrArmFromVcfDiploid.py " + " ".join([str(x) for x in cmdArgs])
    elif argsDict['shicMode'].lower() == 'haploid':
        cmdArgs = [argsDict['chrArmVcfFile'], argsDict['chrArm'], argsDict['chrLen'], argsDict['targetPop'], argsDict['winSize'],
               argsDict['numSubWins'], argsDict['maskFileName'], argsDict['unmaskedFracCutoff'], argsDict['sampleToPopFileName'],
               argsDict['ancFileName'], argsDict['statFileName'], argsDict['fvecFileName']]
        cmd = pyExec + " " + diploShicDir + "makeFeatureVecsForChrArmFromVcf_ogSHIC.py " + " ".join([str(x) for x in cmdArgs])
    else:
        sys.exit("'shicMode' must be set to either 'diploid' or 'haploid'")
    additionalArgs = []
    if argsDict['segmentStart'] != "None":
        additionalArgs += [argsDict['segmentStart'], argsDict['segmentEnd']]
        cmd += " " + " ".join(additionalArgs)
    #cmd += " > " + argsDict['fvecFileName']
    print(cmd)
    subprocess.call(cmd.split())
elif argsDict['mode'] == 'makeTrainingSets':
    cmdArgs = [argsDict['neutTrainingFileName'], argsDict['softTrainingFilePrefix'], argsDict['hardTrainingFilePrefix'],
               argsDict['sweepTrainingWindows'], argsDict['linkedTrainingWindows'], argsDict['outDir']]
    cmd = pyExec + " " + diploShicDir + "makeTrainingSets.py " + " ".join([str(x) for x in cmdArgs])
    print(cmd)
    subprocess.call(cmd.split())
