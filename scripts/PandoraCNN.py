import dircache
import keras
import math
import numpy as np
import os
import random
import re

#====================

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, files, batchSize = 32, gridSize = 16, nClasses = 2, shuffle=False):
        'Initialization'
        self.gridSize = gridSize
        self.batchSize = batchSize
        self.files = files
        self.nClasses = nClasses
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.files)

    def __getitem__(self, index):
        'Generate one batch of data'
        fname = self.files[index]

        with open(fname) as f:
            content = f.readlines()

        content = [x.strip() for x in content]
        X = []
        Y = []

        for line in content:
            matrix = np.zeros((self.gridSize, self.gridSize))
            outcomeArray =  np.zeros(self.nClasses)
            pdg = int(float(line.split(",")[1]))
            if (int)(pdg) > 10000:
                continue

            if pdg in [11, -11, 22]:
                outcomeArray[0] = 1
            else:
                outcomeArray[1] = 1

            pixels = line.split(",")[2:-1]
            for idx, pixel in enumerate(pixels):
                y = int(math.floor(idx / self.gridSize))
                x = int(idx - (y * self.gridSize))
                matrix[x,y] = int(256 * float(pixel) / 10000) # Integer pixels prevents loss nan errors

            X.append(matrix)
            Y.append(outcomeArray)

        X = np.array(X).astype(np.float32)
        X = X.reshape(-1, self.gridSize, self.gridSize, 1)
        Y = np.array(Y).astype(np.int)
        Y = Y.reshape(-1, self.nClasses)
        return X, Y

#====================

def CollectData(fileDirectory, fileFormat, fileExt = 'txt', testSize = 0.25):
    allFiles = []
    allFiles.extend(dircache.listdir(fileDirectory))
    allFiles[:] = [ item for item in allFiles if re.match('.*\.' + fileExt + '$', item.lower()) ]
    allFiles.sort()

    nFiles = 0
    if allFiles:
        nFiles = len(allFiles)

    outputFiles = []
    for currentFile in allFiles:
        matchObj = re.match(fileFormat, currentFile, re.M|re.I)
        if matchObj:
            identifier = matchObj.group(1)
            outputFiles.append(os.path.join(fileDirectory, currentFile))

    random.shuffle(outputFiles)
    cut = int(testSize * len(outputFiles))
    trainingData = outputFiles[cut:]
    testingData = outputFiles[:cut]
    return trainingData, testingData

#====================

def Sample(X, Y, testFraction = 0.1):
    trainSize = int((1.0 - testFraction) * len(Y))
    testSize = int(testFraction * len(Y))
    print("Test Size  : " + str(testSize))
    print("Train Size : " + str(trainSize))

    X_train = X[:trainSize]
    Y_train = Y[:trainSize]
    X_test  = X[trainSize:]
    Y_test  = Y[trainSize:]

    return X_train, Y_train, X_test, Y_test

#====================

def DisplayImage(X, Y, number = 0, gridSize = 16):
    labels = { 0 : 'Shower', 1: 'Track'}
    img = np.reshape(X[number], (gridSize, gridSize))
    label = np.argmax(Y[number,:])
    plt.imshow(img, cmap='gray')
    plt.title("(Label: " + str(labels[label]) + ")")
    plt.show()
    return

#====================

def GetUniformClassWeights(Y):
    nTrk = 0
    nShw = 0

    for (x,y), value in np.ndenumerate(Y):
        if y == 0 and value == 1:
            nShw = nShw + 1
        if y == 1 and value == 1:
            nTrk = nTrk + 1

    weight = nShw / nTrk
    class_weights = {0: 1, 1:weight}
    return class_weights

#====================

