# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys

np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

from keras import optimizers

from PandoraCNN import *

if __name__=="__main__":

    # Settings ------------------------------------------------------------------------------------

    fileLocation    = '/home/stevegreen/LAr/2018/October/TrainingData'
    fileFormat      = 'CaloHitDump_DeepLearning_Job_Number_(.*?).txt'
    svmName         = 'TrackShowerId'
    gridSize        = 16
    nClasses        = 2
    nEpochs         = 10

    saveModel       = True
    saveMetrics     = True
    jsonFileName    = 'Model.json'
    h5FileName      = 'Model.h5'

    #----------------------------------------------------------------------------------------------

    # Parameters
    params = {'batchSize': 128,
              'gridSize' : 16,
              'nClasses' : 2,
              'shuffle'  : False}

    trainingData, testingData = CollectData(fileLocation, fileFormat, fileExt = 'txt', testSize = 0.5)

     # Generators
    training_generator = DataGenerator(trainingData, **params)
    validation_generator = DataGenerator(testingData, **params)

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

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=nEpochs,
                                  verbose=1, 
                                  shuffle=True,
                                  use_multiprocessing=False)

    score = model.evaluate_generator(generator=validation_generator)
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

