import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

inputSize = 100
hiddenLayerSize = 20
outputSize = inputSize
eta = 0.1
numEpochs = 10

def generateDataClassA():
    n = inputSize/2
    half = (int)(n/2)
    mA = [1, 0] 
    sigmaA = 0.5
    classA = np.concatenate(
            (np.random.randn(half, 2) * sigmaA + mA,
            np.random.randn(half, 2) * sigmaA + mA))   
    return classA

def generateDataClassB():
    n = inputSize/2
    half = (int)(n/2)
    mB = [-1, 0]
    sigmaB = 0.5
    classB = np.concatenate(
            (np.random.randn(half, 2) * sigmaB + mB,
            np.random.randn(half, 2) * sigmaB + mB))
    return classB

def generateTrainingInput(classA , classB):
    dataPoints = np.array(np.concatenate((classA, classB)))
    bias = np.array([np.ones(dataPoints.shape[0])])
    inputs = np.concatenate((dataPoints, bias.T), axis=1)
    return inputs

def generateTargets(classA , classB):
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    return targets

def plotTrainingData(classA , classB ):
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    return plt.show()

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
    difference = np.array(np.subtract(targets , outputs))
    delta = np.zeros(outputSize)
    for index in range(outputSize):
        delta[index] = transferFunctionDerivative(outputs[index]) * difference[index]  
    return delta

def deltahiddenLayer(OutputWeights, deltaOutputs , hiddenResults ):
    sum= np.dot(OutputWeights.T , deltaOutputs)
    delta = np.zeros(hiddenLayerSize)
    for index in range(hiddenLayerSize):
        delta[index] = sum[index] * transferFunctionDerivative ( hiddenResults[index][0])
    return delta

def hiddenWeightUpdate( hiddenUpdate , inputs):
    inputVector = np.array (np.zeros ( inputSize))
    inputVector = inputVector.reshape(100, 1)
    hiddenUpdate = hiddenUpdate.reshape(20, 1)
    for row in range(inputSize):
        inputVector[row] = inputs[row][0]
    dotProduct = np.dot(hiddenUpdate , inputVector.T)
    update = eta * dotProduct
    return update

def outputWeightUpdate( hiddenResults , outputUpdate):
    hiddenResults = hiddenResults.reshape(20, 1)
    outputUpdate = outputUpdate.reshape(100 , 1)
    dotProduct = np.dot(outputUpdate  , hiddenResults.T)
    update = eta * dotProduct
    return update

def run():
    classA = generateDataClassA()
    classB = generateDataClassB()
    inputs  = generateTrainingInput(classA , classB)
    targets = generateTargets(classA , classB)

    inputWeights= initfirstLayerWeights()
    outputWeights = initsecondLayerWeights()

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
    
run()