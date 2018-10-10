import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits import mplot3d
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

start = -5 
end = 5 
step = 0.5 
inputSize = int ( (end - start)/ step)
matrixSize = int ( inputSize*inputSize)  
hiddenLayerSize = 30
eta = 0.2
numEpochs = 20
outputSize = matrixSize

def function (x , y):
    z = np.zeros((inputSize , inputSize))
    for xIndex in range(len(x)):
        for yIndex in range((len(y))):
            z[xIndex][yIndex] = (np.exp(-(np.power(x[xIndex] , 2) + np.power(y[yIndex] , 2))* 0.1 ) ) -0.5
    return z

def generateTargets (z):
    targets = np.array(z).reshape( 1, matrixSize)
    return targets

def initfirstLayerWeights():
    weights = np.random.randn( hiddenLayerSize , matrixSize)
    return weights

def initsecondLayerWeights():
    weights = np.random.randn( matrixSize , hiddenLayerSize)
    return weights

def transferFunction( x):
   return (2 / (1 + np.exp(-x))) - 1

def transferFunctionDerivative( x):
    return ((1 + transferFunction(x)) * (1 - transferFunction(x))) / 2

def hiddenResult( inputs , inputWeights):
    result = np.dot(inputWeights , inputs.T)
    return result

def forwardPassHiddenLayer( hiddenResults):
    transferValues = np.zeros( hiddenLayerSize)
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
    outputs = outputs.reshape((1 , outputSize))
    difference = np.array(np.subtract(targets , outputs))
    delta = np.zeros(outputSize)
    for index in range(outputSize):
        delta[index] = transferFunctionDerivative(outputs[0][index]) * difference[0][index]  
    return delta

def deltahiddenLayer(OutputWeights, deltaOutputs , hiddenResults ):
    sum= np.dot(OutputWeights.T , deltaOutputs)
    delta = np.zeros(hiddenLayerSize)
    for index in range(hiddenLayerSize):
        delta[index] = sum[index] * transferFunctionDerivative ( hiddenResults[index][0])
    return delta

def hiddenWeightUpdate( hiddenUpdate , inputs):
    inputVector = np.array (np.zeros ( matrixSize))
    inputVector = inputVector.reshape(1, matrixSize)
    hiddenUpdate = hiddenUpdate.reshape(hiddenLayerSize, 1)
    for row in range(matrixSize):
        inputVector[0][row] = inputs[0][row]
    dotProduct = np.dot(hiddenUpdate , inputVector)
    update = eta * dotProduct
    return update

def outputWeightUpdate( hiddenResults , outputUpdate):
    hiddenResults = hiddenResults.reshape(hiddenLayerSize, 1)
    outputUpdate = outputUpdate.reshape(matrixSize , 1)
    dotProduct = np.dot(outputUpdate  , hiddenResults.T)
    update = eta * dotProduct
    return update

def plotData ( x , y ):
    z = function(x, y)
    X,Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, z,  rstride=1, cstride=1,cmap= 'viridis' ,edgecolor='none')
    ax.set_title("Original Bell-Shaped Gaussian")
    plt.show()

def plotResults(x, y, outputs):
    Z = outputs. reshape((inputSize , inputSize))
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap= 'viridis' ,edgecolor='none')
    ax.set_title("Trained Bell-Shaped Gaussian")
    plt.show()

def run():
    x = np.arange(-5.0,5.0,0.5)
    y = np.arange(-5.0,5.0,0.5)
    z  = function( x, y)
    targets = generateTargets(z)
    xx, yy = np.meshgrid(x, y) 
    patterns = np.concatenate((xx.reshape( 1, matrixSize).copy(), yy.reshape( 1, matrixSize).copy()))
    bias = np.ones((1, 400))
    trainingPatterns = np.concatenate ( ( patterns , bias), axis = 0  )

    inputWeights= initfirstLayerWeights()
    outputWeights = initsecondLayerWeights()
    outputs = np.zeros(outputSize)
    for epoch in range (numEpochs):
        hiddenValues = hiddenResult( trainingPatterns , inputWeights)
        hiddenLayerResults = forwardPassHiddenLayer(hiddenValues)
        outputs = forwardPassOutputLayer(hiddenLayerResults , outputWeights)

        outputUpdate = deltaOutputLayer(targets, outputs)
        hiddenUpdate = deltahiddenLayer (outputWeights , outputUpdate , hiddenValues)

        inputWeightsUpdate = hiddenWeightUpdate( hiddenUpdate , trainingPatterns)
        inputWeights = np.add(inputWeights  , inputWeightsUpdate )

        targetWeightUpdate = outputWeightUpdate(hiddenLayerResults , outputUpdate )
        outputWeights = np.add ( outputWeights  ,  targetWeightUpdate)
        outputs = forwardPassOutputLayer(hiddenLayerResults , outputWeights)
    plotData(x, y)
    plotResults(x, y , outputs)
run()