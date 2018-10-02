import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

# https://www.hackerearth.com/blog/uncategorized/radial-basis-function-network/
# https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/

lowerBound = 0
upperBound = 7
stepSize = 0.1
inputSize = (int)((upperBound-lowerBound)/stepSize)
hiddenLayerSize = 8

def generateInput (inputSize, low , step):
    inputs = np.zeros([inputSize])
    current = low
    for index in range(len(inputs)):
        inputs[index] = current
        current = current + step
    return  inputs

def sinFunctionCalculate (inputs):
    outputs= np.zeros(len(inputs))
    for inputIndex in range(len(outputs)):
        x = inputs[inputIndex]
        y = math.sin(2*x)
        outputs[inputIndex] = y
    return outputs

def squareFunctionCalculate(inputs):
    outputs= np.zeros(len(inputs))
    for inputIndex in range(len(outputs)):
        x = inputs[inputIndex]
        y = math.pow(2*x , 2)
        outputs[inputIndex] = y
    return outputs

def noiseSinCalculate (inputs):
    noise = np.random.uniform(-0.1, 0.1, inputSize)
    outputs= np.zeros(len(inputs))
    for inputIndex in range(len(outputs)):
        x = inputs[inputIndex]
        y = math.sin(2*x) + noise[inputIndex]
        outputs[inputIndex] = y
    return outputs

def noiseSquareCalculate(inputs):
    noise = np.random.uniform(-0.1, 0.1, inputSize)
    outputs= np.zeros(len(inputs))
    for inputIndex in range(len(outputs)):
        x = inputs[inputIndex]
        y = math.pow(2*x , 2) + noise[inputIndex]
        outputs[inputIndex] = y
    return outputs

def calculateCenteroids(numInputs, numCentroids , inputs):
    temp = np.random.choice(inputs, size=numCentroids, replace=False)
    centroids = np.zeros([numCentroids])
    convergence = 0
    while( convergence <= 100):
        matrix = np.zeros((numInputs , numCentroids))
        for inputIndex in range(len(inputs)):
            minDist = abs( temp[0] -  inputs[inputIndex])
            centerPosition = 0
            for centerIndex in range(len(temp)):
                distance = abs( temp[centerIndex] -  inputs[inputIndex])
                if (distance < minDist ):
                    minDist = distance
                    centerPosition = centerIndex
            matrix[inputIndex][centerPosition] = 1

        for i  in   range(numCentroids) :
            sum = 0
            counter = 0
            average = 0 
            for j in range(numInputs):
                if matrix[j][i] == 1:
                    sum = sum + inputs[j]
                    counter += 1
            average = sum / counter
            centroids[i] = average
        convergence = convergence +1 
    return centroids

def calculateSigma( numInputs,  inputs , numCentroids  , centroids):
    distanceMatrix = np.zeros((numCentroids , numCentroids))
    d = 0
    for i in range(numCentroids):
        for j in range(numCentroids):
            distanceMatrix[i][j] = np.linalg.norm(centroids[i]-centroids[j])
            if d  < distanceMatrix[i][j]:
                d = distanceMatrix[i][j]
    sigma =  d / math.sqrt(2 * numCentroids)
    return sigma 

def transferFunctionCalculate(inputs, centroids , sigma  , numCentroids  , numInputs ):
    transferMatrix = np.zeros((numInputs , numCentroids))
    for i in range(numInputs):
        for j in range(numCentroids):
            dist = np.linalg.norm(inputs[i]-centroids[j])
            numerator = - ( math.pow(dist , 2))
            denominator =  2 * math.pow(sigma, 2)
            power = numerator / denominator
            result = math.exp(power)
            transferMatrix[i][j] = result
    return transferMatrix
    
def weightCalculate(transferMatrix, realValues , inputs):
    transferMatrixTranspose = transferMatrix.transpose()    
    LH = np.dot( transferMatrixTranspose , transferMatrix )
    LHInverse = np.linalg.inv(LH)
    RH = np.dot(LHInverse, transferMatrixTranspose)
    weights = np.dot (RH , realValues )
    return weights

def approximationCalculate(weights , transferMatrix):
    approximations = np.dot(transferMatrix , weights)
    return approximations

def errorCalculate(transferMatrix , weights , realValues):
    difference = np.dot(transferMatrix , weights) - realValues
    vectorNorm = np.linalg.norm(difference)
    totalError = math.pow(vectorNorm , 2)
    return totalError

def sinFunctionLearning():
    inputs = generateInput (inputSize, lowerBound , stepSize)
    realOutputs = sinFunctionCalculate (inputs)
    centroids = calculateCenteroids(inputSize, hiddenLayerSize , inputs)
    sigma = calculateSigma( inputSize,  inputs , hiddenLayerSize  , centroids)
    transferMatrix = transferFunctionCalculate(inputs, centroids , sigma  , hiddenLayerSize  , inputSize )
    weights= weightCalculate(transferMatrix, realOutputs , inputs)
    approximatedOutputs = approximationCalculate(weights , transferMatrix)
    error = errorCalculate(transferMatrix , weights , realOutputs)
    plt.plot([x for x in inputs], [y for y in realOutputs ] , 'bo')
    plt.plot([x for x in inputs], [y for y in approximatedOutputs] , 'ro')
    print('The error is: ', error)
    return plt.show()

def squareFunctionLearning():
    inputs = generateInput (inputSize, lowerBound , stepSize)
    realOutputs = squareFunctionCalculate (inputs)
    centroids = calculateCenteroids(inputSize, hiddenLayerSize , inputs)
    sigma = calculateSigma( inputSize,  inputs , hiddenLayerSize  , centroids)
    transferMatrix = transferFunctionCalculate(inputs, centroids , sigma  , hiddenLayerSize  , inputSize )
    weights= weightCalculate(transferMatrix, realOutputs , inputs)
    approximatedOutputs = approximationCalculate(weights , transferMatrix)
    error = errorCalculate(transferMatrix , weights , realOutputs)
    plt.plot([x for x in inputs], [y for y in realOutputs ] , 'bo')
    plt.plot([x for x in inputs], [y for y in approximatedOutputs] , 'ro')
    print('The error is: ', error)
    return plt.show()

def noiseSinLearning():
    inputs = generateInput (inputSize, lowerBound , stepSize)
    realOutputs = noiseSinCalculate (inputs)
    centroids = calculateCenteroids(inputSize, hiddenLayerSize , inputs)
    sigma = calculateSigma( inputSize,  inputs , hiddenLayerSize  , centroids)
    transferMatrix = transferFunctionCalculate(inputs, centroids , sigma  , hiddenLayerSize  , inputSize )
    weights= weightCalculate(transferMatrix, realOutputs , inputs)
    approximatedOutputs = approximationCalculate(weights , transferMatrix)
    error = errorCalculate(transferMatrix , weights , realOutputs)
    plt.plot([x for x in inputs], [y for y in realOutputs ] , 'bo')
    plt.plot([x for x in inputs], [y for y in approximatedOutputs] , 'ro')
    print('The error is: ', error)
    return plt.show()

def noiseSquareLearning():
    inputs = generateInput (inputSize, lowerBound , stepSize)
    realOutputs = noiseSquareCalculate (inputs)
    centroids = calculateCenteroids(inputSize, hiddenLayerSize , inputs)
    sigma = calculateSigma( inputSize,  inputs , hiddenLayerSize  , centroids)
    transferMatrix = transferFunctionCalculate(inputs, centroids , sigma  , hiddenLayerSize  , inputSize )
    weights= weightCalculate(transferMatrix, realOutputs , inputs)
    approximatedOutputs = approximationCalculate(weights , transferMatrix)
    error = errorCalculate(transferMatrix , weights , realOutputs)
    plt.plot([x for x in inputs], [y for y in realOutputs ] , 'bo')
    plt.plot([x for x in inputs], [y for y in approximatedOutputs] , 'ro')
    print('The error is: ', error)
    return plt.show()

#sinFunctionLearning()
#squareFunctionLearning()
#noiseSinLearning()
noiseSquareLearning()
