import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

inputSize = 8
hiddenLayerSize = 3
outputSize = inputSize
eta = 0.1
numEpochs = 10

def generateData():
    inputs = np.array ([-1 , -1 , -1 , 1 , -1 , -1 , -1 , -1])
    inputs = inputs.reshape(inputSize , 1)
    return inputs

def generateTargets(inputs):
    targets = inputs
    return targets

def initfirstLayerWeights():
    weights = np.random.randn( hiddenLayerSize , inputSize)
    return weights

def initsecondLayerWeights():
    weights = np.random.randn( outputSize , hiddenLayerSize)
    return weights

def transferFunction( x):
   return (2 / (1 + np.exp(-x))) - 1

def transferFunctionDerivative( x):
    return ((1 + transferFunction(x)) * (1 - transferFunction(x))) / 2

def hiddenResult( inputs , inputWeights):
    result = np.dot(inputWeights , inputs)
    return result

def forwardPassHiddenLayer( hiddenResults):
    transferValues = np.zeros(hiddenLayerSize)
    for index in range(hiddenLayerSize):
        transferValues[index] = transferFunction( hiddenResults[index][0] )
    return transferValues

def forwardPassOutputLayer(transferValues, outputWeights ):
    sum = np.dot(outputWeights , transferValues)
    outputs= np.zeros(outputSize)
    for index in range(outputSize):
        outputs[index] = transferFunction( sum[index] )
    return outputs

def deltaOutputLayer(targets, outputs):
    outputs = outputs.reshape(outputSize, 1)
    difference = np.array(np.subtract(targets , outputs))
    delta = np.zeros(outputSize)
    for index in range(outputSize):
        delta[index] = transferFunctionDerivative(outputs[index][0]) * difference[index]  
    return delta

def deltahiddenLayer(OutputWeights, deltaOutputs , hiddenResults ):
    sum= np.dot(OutputWeights.T , deltaOutputs)
    delta = np.zeros(hiddenLayerSize)
    for index in range(hiddenLayerSize):
        delta[index] = sum[index] * transferFunctionDerivative ( hiddenResults[index][0])
    return delta

def hiddenWeightUpdate( hiddenUpdate , inputs):
    inputVector = np.array (np.zeros ( inputSize))
    inputVector = inputVector.reshape(inputSize, 1)
    hiddenUpdate = hiddenUpdate.reshape(hiddenLayerSize, 1)
    for row in range(inputSize):
        inputVector[row] = inputs[row][0]
    dotProduct = np.dot(hiddenUpdate , inputVector.T)
    update = eta * dotProduct
    return update

def outputWeightUpdate( hiddenResults , outputUpdate):
    hiddenResults = hiddenResults.reshape(hiddenLayerSize, 1)
    outputUpdate = outputUpdate.reshape(inputSize , 1)
    dotProduct = np.dot(outputUpdate  , hiddenResults.T)
    update = eta * dotProduct
    return update

def plotData (inputs, targets, outputs):
    plt.plot([p for p in inputs], [p for p in targets], 'b.')
    plt.plot([p for p in inputs], [p for p in outputs], 'r.')
    return plt.show()

def run():
    inputs = generateData()
    targets = generateTargets(inputs)
    inputWeights= initfirstLayerWeights()
    outputWeights = initsecondLayerWeights()
    outputs = np.zeros(outputSize)
    for epoch in range (numEpochs):
        hiddenValues = hiddenResult( inputs , inputWeights)
        hiddenLayerResults = forwardPassHiddenLayer(hiddenValues)
        outputs = forwardPassOutputLayer(hiddenLayerResults , outputWeights)

        outputUpdate = deltaOutputLayer(targets, outputs)
        hiddenUpdate = deltahiddenLayer (outputWeights , outputUpdate , hiddenValues)

        inputWeightsUpdate = hiddenWeightUpdate( hiddenUpdate , inputs)
        inputWeights = np.add(inputWeights  , inputWeightsUpdate )

        targetWeightUpdate = outputWeightUpdate(hiddenLayerResults , outputUpdate )
        outputWeights = np.add ( outputWeights  ,  targetWeightUpdate)
        outputs = forwardPassOutputLayer(hiddenLayerResults , outputWeights)
    plotData(inputs, targets , outputs)
run()