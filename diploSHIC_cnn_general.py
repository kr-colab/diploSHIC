import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys, os
np.random.seed(123)
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
import time
import fnmatch
if len(sys.argv)<3:
    print("diploSHIC_cnn_general.py nDims trainDir testDir heatmapFileName")
    sys.exit()
print("loading data now...")
#hard.fvec  linkedHard.fvec  linkedPartialHard.fvec  linkedPartialSoft.fvec  linkedSoft.fvec  neut.fvec  partialHard.fvec  partialSoft.fvec  soft.fvec
nDims = int(sys.argv[1])
trainingDir=sys.argv[2]
testingDir=sys.argv[3]
figFileName = sys.argv[4]

nCores = 12

#training data
finalCol = (11 * nDims)+1
hard = np.loadtxt(trainingDir+"hard.fvec",skiprows=1,usecols=list(range(1,finalCol)))
h1 = np.reshape(hard,(hard.shape[0],nDims,11))
neut = np.loadtxt(trainingDir+"neut.fvec",skiprows=1,usecols=list(range(1,finalCol)))
n1 = np.reshape(neut,(neut.shape[0],nDims,11))
soft = np.loadtxt(trainingDir+"soft.fvec",skiprows=1,usecols=list(range(1,finalCol)))
s1 = np.reshape(soft,(soft.shape[0],nDims,11))
lsoft = np.loadtxt(trainingDir+"linkedSoft.fvec",skiprows=1,usecols=list(range(1,finalCol)))
ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,11))
lhard = np.loadtxt(trainingDir+"linkedHard.fvec",skiprows=1,usecols=list(range(1,finalCol)))
lh1 = np.reshape(lhard,(lhard.shape[0],nDims,11))

both=np.concatenate((h1,n1,s1,ls1,lh1))
y=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))

#reshape both to explicitly set depth image. need for theanno not sure with tensorflow
both = both.reshape(both.shape[0],nDims,11,1)
if (trainingDir==testingDir):
    X_train, X_test, y_train, y_test = train_test_split(both, y, test_size=0.2)
else:
    X_train = both
    y_train = y
    #testing data
    hard = np.loadtxt(testingDir+"hard.fvec",skiprows=1,usecols=list(range(1,finalCol)))
    h1 = np.reshape(hard,(hard.shape[0],nDims,11))
    neut = np.loadtxt(testingDir+"neut.fvec",skiprows=1,usecols=list(range(1,finalCol)))
    n1 = np.reshape(neut,(neut.shape[0],nDims,11))
    soft = np.loadtxt(testingDir+"soft.fvec",skiprows=1,usecols=list(range(1,finalCol)))
    s1 = np.reshape(soft,(soft.shape[0],nDims,11))
    lsoft = np.loadtxt(testingDir+"linkedSoft.fvec",skiprows=1,usecols=list(range(1,finalCol)))
    ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,11))
    lhard = np.loadtxt(testingDir+"linkedHard.fvec",skiprows=1,usecols=list(range(1,finalCol)))
    lh1 = np.reshape(lhard,(lhard.shape[0],nDims,11))

    both2=np.concatenate((h1,n1,s1,ls1,lh1))
    X_test = both2.reshape(both2.shape[0],nDims,11,1)
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


print(X_train.shape)
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
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                          verbose=1, mode='auto')
filepath="diploSHIC_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

#callbacks_list = [earlystop,checkpoint]
callbacks_list = [earlystop] #turning off checkpointing-- just want accuracy assessment

datagen.fit(X_train)
validation_gen.fit(X_valid)
test_gen.fit(X_test)
start = time.clock()
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), \
                    steps_per_epoch=len(X_train) / 32, epochs=100,verbose=1, \
                    callbacks=callbacks_list, \
                    validation_data=validation_gen.flow(X_valid,Y_valid, batch_size=32), \
                    validation_steps=len(X_test)/32, \
                    workers=nCores)
#model.fit(X_train, Y_train, batch_size=32, epochs=100,validation_data=(X_test,Y_test),callbacks=callbacks_list, verbose=1)
score = model.evaluate_generator(test_gen.flow(X_test,Y_test, batch_size=32),len(Y_test)/32,workers=nCores)
sys.stderr.write("total time spent fitting and evaluating: %f secs\n" %(time.clock()-start))

print(model.metrics_names)
print(score)
#model.save("annoSHIC_trainedCNN.h5")


############# Heatmap
labelToClassName = {0:"hard",1:"neut",2:"soft",3:"linkedSoft",4:"linkedHard"}

def getSelTypeFromFilePrefix(filePrefix):
    if "PartialHard" in filePrefix:
        return "PartialHard"
    elif "PartialSoft" in filePrefix:
        return "PartialSoft"
    elif "hard" in filePrefix:
        return "Hard"
    elif "soft" in filePrefix:
        return "Soft"
    elif "neut" in filePrefix:
        return "Neut"
    else:
        raise Exception

def getSelTypeForDisplay(selType):
    selType = selType.lower()
    if "partialhard" in selType:
        return "Hard partial"
    elif "partialsoft" in selType:
        return "Soft partial"
    elif "soft" in selType:
        return "Soft"
    elif "hard" in selType:
        return "Hard"
    else:
        raise ValueError

def keyToLabel(tupleKey):
#    print(tupleKey)
    if (tupleKey[0] == "Neut"):
        return(1)
    elif (tupleKey[0]== "Hard"):
        if (tupleKey[1] != 5):
            return(4)
        else: 
            return(0)
    elif (tupleKey[0] == "Soft"):
        if (tupleKey[1] != 5):
            return(3)
        else:
            return(2)
    else:
        raise Exception


testExampleCount = 0
testData = []
testExamplesInFile = {}
testY = []
for testSetFileName in fnmatch.filter(os.listdir(testingDir),'*diploid.fvec'):
#    print(testSetFileName)
    testSetFile = open(testingDir + testSetFileName)
    currTestData = testSetFile.readlines()
    testSetFile.close()

    testExamplesInFile[testSetFileName] = []
    testHeader = currTestData[0].strip().split("\t")
    currTestData = currTestData[1:]#remove the header from the test data file
    testSetFilePrefix = testSetFileName.split(".")[0]
    if "_" in testSetFilePrefix:
        testSetFilePrefix = testSetFilePrefix.split("_")
        selType, selWin = getSelTypeFromFilePrefix(testSetFilePrefix[0]), testSetFilePrefix[1]
        selWin = int(selWin)
        key = (selType, selWin)
    else:
        key = (getSelTypeFromFilePrefix(testSetFilePrefix), -1)
    
    for testExample in currTestData:
        if not "nan" in testExample:
            testData.append(testExample)
            testExamplesInFile[testSetFileName].append(testExampleCount)
            testExampleCount += 1
            testY.append(keyToLabel(key))	
testX = []
for i in range(len(testData)):
    testData[i] = testData[i].strip().split("\t")
    currVector = []
    for j in range(len(testData[i])):
        currVector.append(float(testData[i][j]))
    testX.append(currVector)

testX = np.array(testX)
testX=np.reshape(testX,(testX.shape[0],nDims,11,1))


preds = model.predict(validation_gen.standardize(testX))
predictions = np.argmax(preds,axis=1)

outlinesH = {}
heatmap = []
classOrder = "hard linkedHard soft linkedSoft neut".split()
for testSetFileName in sorted(testExamplesInFile):
    denom = float(len(testExamplesInFile[testSetFileName]))
    currPreds = {}
    currExamples = []
    for className in classOrder:
        currPreds[className] = 0
    #for cl in range(len(classOrder)):
    #    currPreds[cl] = 0
    for testExampleIndex in testExamplesInFile[testSetFileName]:
        currExamples.append(testX[testExampleIndex])
#        print(testSetFileName)
#        print("testExIndex: ",testExampleIndex)
#        print("predictions[testExampleIndex]: ",predictions[testExampleIndex])
#        print("labelToClassName[predictions[testExampleIndex]]: ",labelToClassName[predictions[testExampleIndex]])
        predictedClass = labelToClassName[predictions[testExampleIndex]]
        #predictedClass = predictions[testExampleIndex]
        currPreds[predictedClass] += 1/denom
    #equilibHard_0.fvec
    #equilibSoft_f0_0.1.fvec
    testSetFilePrefix = testSetFileName.split(".")[0]
    if "_" in testSetFilePrefix:
        testSetFilePrefix = testSetFilePrefix.split("_")
        selType, selWin = getSelTypeFromFilePrefix(testSetFilePrefix[0]), testSetFilePrefix[1]
        selWin = int(selWin)
        key = (selType, selWin)
    else:
        key = (getSelTypeFromFilePrefix(testSetFilePrefix), -1)
    #print(testSetFileName, key, currPreds)
    outlinesH[key] = (testSetFileName, [currPreds[className] for className in classOrder])
rowLabels, data = [], []
keys = list(outlinesH.keys())


selVals = {"Hard":0, "Soft":1, "Neut":2}
keys.sort(key=lambda x: (selVals[x[0]], x[1]))
for selType, selWin in keys:
    fileName, vec = outlinesH[(selType, selWin)]
    #print fileName, selType, selWin
    if "Neut" in selType:
        rowLabels.append("Neutral")
    else:
        if selWin == 5:
            rowLabels.append("%s sweep in focal window" %(getSelTypeForDisplay(selType)))
        else:
            if selWin < 5:
                direction = "left"
            else:
                direction = "right"
            diff = abs(selWin-5)
            if diff == 1:
                plural = ""
            else:
                plural = "s"
            rowLabels.append("%s sweep %s window%s to %s" %(getSelTypeForDisplay(selType), diff, plural, direction))
    data.append(vec)
data = np.array(data)
myFig = plt.figure()
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues)
cbar.set_label('Fraction of simulations assigned to class', rotation=270, labelpad=20)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.axis('tight')

plt.tick_params(axis='y', which='both', right='off')
plt.tick_params(axis='x', which='both', direction='out')
ax.set_xticklabels(["Hard", "Hard-linked", "Soft", "Soft-linked", "Neutral"], minor=False, fontsize=9, rotation=45, ha="left")


ax.set_yticklabels(rowLabels, minor=False, fontsize=7)

for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        val = data[y, x]
        val *= 100
        if val > 50:
            c = '0.9'
        else:
            c = 'black'
        ax.text(x + 0.5, y + 0.5, '%.1f%%' % val, horizontalalignment='center', verticalalignment='center', color=c, fontsize=6)

plt.savefig(figFileName, bbox_inches='tight', dpi=600)



