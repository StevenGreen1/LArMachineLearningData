import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json
import json
import argparse
from datetime import datetime

from PandoraMVA import *

#--------------------------------------------------------------------------------------------------

def WriteConv2DXml(modelFile, layer, indentation):
    weights = layer.get_weights()[0]
    bias = layer.get_weights()[1]
    borderMode = layer.get_config()['padding']

    nKernels = weights.shape[3]
    depth = weights.shape[2]
    nRows = weights.shape[1]
    nCols = weights.shape[0]

    indentation = OpenXmlTag(modelFile, 'LayerConv2D', indentation)
    modelFile.write((' ' * indentation) + '<LayerConv2DConfig borderMode="' + str(borderMode) + '" nKernels="' + str(nKernels) + '" nDeep="' + str(depth) + '" nRows="' + str(nRows) + '" nCols="' + str(nCols) + '"/>\n')

    for kernel in range(nKernels):
        for depth in range(depth):
            for row in range(nRows):
                for col in range(nCols):
                    modelFile.write((' ' * indentation) + '<Conv2DWeight kernel="' + str(kernel) + '" depth="' + str(depth) + '" row="' + str(row) + '" col="' + str(col) + '" value="' + str(weights[row, col, depth, kernel]) + '"/>\n')

    for idx, value in enumerate(bias):
        modelFile.write((' ' * indentation) + '<Conv2DBias kernel="' + str(kernel) + '" bias="' + str(value) + '"/>\n')

    CloseXmlTag(modelFile, 'LayerConv2D', indentation)

#--------------------------------------------------------------------------------------------------

def WriteActivationXml(modelFile, layer, indentation):
    activationType = str(layer.get_config()['activation'])
    indentation = OpenXmlTag(modelFile, 'LayerActivation', indentation)
    WriteXmlFeature(modelFile, activationType, 'ActivationFunction', indentation)
    CloseXmlTag(modelFile, 'LayerActivation', indentation)

#--------------------------------------------------------------------------------------------------

def WriteMaxPooling2DXml(modelFile, layer, indentation):
    poolSizeX = layer.get_config()['pool_size'][0]
    poolSizeY = layer.get_config()['pool_size'][1]
    indentation = OpenXmlTag(modelFile, 'LayerMaxPooling2D', indentation)
    modelFile.write((' ' * indentation) + '<MaxPooling2DConfig PoolX="' + str(poolSizeX) + '" PoolY="' + str(poolSizeY) + '"/>\n')
    CloseXmlTag(modelFile, 'LayerMaxPooling2D', indentation)

#--------------------------------------------------------------------------------------------------

def WriteFlattenXml(modelFile, layer, indentation):
#    name = str(layer.get_config()['name'])
    indentation = OpenXmlTag(modelFile, 'LayerFlatten', indentation)
#    WriteXmlFeature(modelFile, name, 'Name', indentation)
    CloseXmlTag(modelFile, 'LayerFlatten', indentation)

#--------------------------------------------------------------------------------------------------

def WriteDenseXml(modelFile, layer, indentation):
    weightList = layer.get_weights()[0]
    biasList = layer.get_weights()[1]
    inputNodes = str(weightList.shape[0])
    outputNodes = str(weightList.shape[1])

    indentation = OpenXmlTag(modelFile, 'LayerDense', indentation)
    modelFile.write((' ' * indentation) + '<LayerDenseConfig nInputNodes="' + str(inputNodes) + '" nOutputNodes="' + str(outputNodes) + '"/>\n')

    for idx, weights in enumerate(weightList):
        outputString = '<DenseWeight connection="' + str(idx) + '" '
        for idxW, value in enumerate(weights):
            outputString += 'weight' + str(idxW) + '="' + str(value) + '" '
        outputString = outputString.rstrip()
        outputString += '/>\n'
        modelFile.write((' ' * indentation) + outputString)

    outputString = '<DenseBias '
    for idx, bias in enumerate(biasList):
        outputString += 'bias' + str(idx) + '="' + str(bias) + '" '
    outputString = outputString.rstrip()
    outputString += '/>\n'
    modelFile.write((' ' * indentation) + outputString)

    CloseXmlTag(modelFile, 'LayerDense', indentation)

#--------------------------------------------------------------------------------------------------

def WriteXmlFile(architecture, model, outputFileName):
    datetimeString = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    with open(outputFileName, "a") as modelFile:
        indentation = 0
        indentation = OpenXmlTag(modelFile, 'ConvolutionalNeuralNetwork', indentation)
        WriteXmlFeature(modelFile, 'TrackShowerId', 'Name', indentation)
        WriteXmlFeature(modelFile, datetimeString, 'Timestamp', indentation)

        for idx, layer in enumerate(architecture["config"]):
            layerName = layer['class_name']
            layer = model.layers[idx]

            if layerName == 'Conv2D':
                WriteConv2DXml(modelFile, layer, indentation)
            elif layerName == 'Activation':
                WriteActivationXml(modelFile, layer, indentation)
            elif layerName == 'MaxPooling2D':
                WriteMaxPooling2DXml(modelFile, layer, indentation)
            elif layerName == 'Flatten':
                WriteFlattenXml(modelFile, layer, indentation)
            elif layerName == 'Dense':
                WriteDenseXml(modelFile, layer, indentation)

        CloseXmlTag(modelFile, 'ConvolutionalNeuralNetwork', indentation)

#--------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='This is a simple script to dump Keras model into simple format suitable for porting into pure C++ model')

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-o', '--output', help="Ouput file name", required=True)

args = parser.parse_args()

print('Read architecture from', args.architecture)
print('Read weights from', args.weights)
print('Writing to', args.output)

arch = open(args.architecture).read()
model = model_from_json(arch)
model.load_weights(args.weights)
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
arch = json.loads(arch)

WriteXmlFile(arch, model, args.output)
