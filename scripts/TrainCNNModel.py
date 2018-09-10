# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import math

np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from keras import optimizers

#====================

def LoadData(fname, gridSize = 16, nOutcomes = 2):
    with open(fname) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    matrix = np.zeros((gridSize, gridSize))
    outcomeArray =  np.zeros(nOutcomes)
    X = []
    Y = []

    shwCounter = 0
    trkCounter = 0

    for line in content:
        matrix = np.zeros((gridSize, gridSize))
        outcomeArray =  np.zeros(nOutcomes)
        pdg = int(float(line.split(",")[1]))

        if (int)(pdg) > 10000:
            continue

        if pdg in [11, -11, 22]:
            shwCounter = shwCounter + 1
            outcomeArray[0] = 1
        else:
            trkCounter = trkCounter + 1
            outcomeArray[1] = 1

        pixels = line.split(",")[2:-1]

        for idx, pixel in enumerate(pixels):
            y = math.floor(idx / gridSize)
            x = idx - (y * gridSize)
            matrix[x,y] = int(256 * float(pixel) / 10000) # Integer pixels prevents loss nan errors

        X.append(matrix)
        Y.append(outcomeArray)

    return np.array(X).astype(np.float64), np.array(Y).astype(np.int)

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


if __name__=="__main__":

    # Settings ------------------------------------------------------------------------------------

    trainingFile    = '/Users/stevengreen/LAr/Jobs/protoDUNE/2018/August/Keras/TrainingData/TinySample.txt'
    svmName         = 'TrackShowerId'
    gridSize        = 16
    nClasses        = 2
    nEpochs         = 10

    saveModel       = True
    saveMetrics     = True
    jsonFileName    = 'Model.json'
    h5FileName      = 'Model.h5'

    displayInput    = False
    loadModel       = False

    #----------------------------------------------------------------------------------------------

    x, y = LoadData(trainingFile, gridSize = gridSize)
    X_train, Y_train, X_test, Y_test = Sample(x, y)

    if displayInput:
        DisplayImage(X_test, Y_test, number = 0, gridSize = gridSize)

    class_weights = GetUniformClassWeights(Y_train)

    X_train = X_train.reshape(-1, gridSize, gridSize, 1)
    X_test = X_test.reshape(-1, gridSize, gridSize, 1)
    Y_train = Y_train.reshape(-1, nClasses)
    Y_test = Y_test.reshape(-1, nClasses)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(gridSize,gridSize,1), activation=None))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nClasses))

    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=128, epochs=nEpochs, verbose=1, class_weight=class_weights)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test Score : {}'.format(score))

    if saveMetrics:
        plt.plot(history.history['acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Model_Accuracy.png')

        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('Model_Loss.png')

    if saveModel:
        model_json = model.to_json()
        with open(jsonFileName, "w") as json_file:
            json_file.write(model_json)

        model.save_weights(h5FileName)
        print("Saved Model")

