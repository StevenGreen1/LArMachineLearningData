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

    shwCounter = 0
    trkCounter = 0

    for line in content:
        pdg = int(float(line.split(",")[1]))

        if (int)(pdg) > 10000:
            continue

        if pdg in [11, -11, 22]:
            shwCounter = shwCounter + 1
            outcomeArray[0] = 1
        else:
            trkCounter = trkCounter + 1
            outcomeArray[1] = 1

        if (trkCounter > 1000 or shwCounter > 1000):
            continue

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
    labels = { 0 : 'Shower', 1: 'Track'}
    img = np.reshape(X[number], (gridSize, gridSize))
    label = np.argmax(Y[number,:])
    plt.imshow(img, cmap='gray')
    plt.title("(Label: " + str(labels[label]) + ")")
    plt.show()
    return

#====================

#fname = 'TestingCaloHitInfo2.txt'
fname = '../../Data/BigData.txt'
gridSize = 16
n_classes = 2

x, y = loadDate(fname, gridSize = gridSize)
X_train, Y_train, X_test, Y_test = Sample(x, y)

DisplayImage(X_test, Y_test, number = 0, gridSize = gridSize)

np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from keras import optimizers
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)

X_train = X_train.reshape(-1, gridSize, gridSize, 1)
X_test = X_test.reshape(-1, gridSize, gridSize, 1)
Y_train = Y_train.reshape(-1, n_classes)
Y_test = Y_test.reshape(-1, n_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

model = Sequential()
model.add(Conv2D(1, (3, 3), input_shape=(gridSize,gridSize,1), activation=None))
model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(n_classes))
#model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
              batch_size=128, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)

print(score)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("simplemodel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("simplemodel.h5")
print("Saved model to disk")

# later...

sys.exit()

