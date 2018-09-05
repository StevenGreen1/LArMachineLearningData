# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import math

#====================

def loadDate(fname, gridSize = 16, nOutcomes = 2):
    with open(fname) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    matrix = np.zeros((gridSize, gridSize))
    outcomeArray =  np.zeros(nOutcomes)
    X = []
    Y = []

    for line in content:
        pdg = int(float(line.split(",")[1]))

        if (int)(pdg) > 10000:
            continue

        if pdg in [11, -11, 22]:
            outcomeArray[0] = 1
        else:
            outcomeArray[1] = 1

        pixels = line.split(",")[2:-1]

        for idx, pixel in enumerate(pixels):
            y = math.floor(idx / gridSize)
            x = idx - (y * gridSize)
            matrix[x,y] = int(256 * float(pixel) / 10000) # Integer pixels prevents loss nan errors

        X.append(matrix)
        Y.append(outcomeArray)
        matrix = np.zeros((gridSize, gridSize))
        outcomeArray =  np.zeros(nOutcomes)

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
    labels = { 0 : 'Track', 1: 'Shower'}
    img = np.reshape(X[number], (gridSize, gridSize))
    label = np.argmax(Y[number,:])
    plt.imshow(img, cmap='gray')
    plt.title("(Label: " + str(labels[label]) + ")")
    plt.show()
    return

#====================
from keras.models import model_from_json

# load json and create model
json_file = open('simplemodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("simplemodel.h5")
print("Loaded model from disk")

fname = 'SinglePicture.txt'
gridSize = 16
n_classes = 2

X, Y = loadDate(fname, gridSize = gridSize)
DisplayImage(X, Y, 0, gridSize = gridSize)
X = X.reshape(-1, gridSize, gridSize, 1)

#for idxK, kernel in enumerate(X):
#    for idxR, row in enumerate(kernel):
#        for inxC, col in enumerate(row):
#            for idxD, depth in enumerate(col):
#                print("(" + str(idxK) + "," + str(idxR) + "," + str(inxC) + "," + str(idxD) + ') = ' + str(X[idxK][idxR][inxC][idxD]))

print(loaded_model.predict(X, batch_size=128, verbose=1))

