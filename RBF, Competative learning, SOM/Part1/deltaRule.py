import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3312446/

learningRate = 0.2
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

def transferFunctionCalculate(inputIndex, centroids , sigma  , numCentroids , inputs  ):
    result = np.zeros((numCentroids))
    for i in range(numCentroids):
        dist = np.linalg.norm(inputs[inputIndex]-centroids[i])
        numerator = - ( math.pow(dist , 2))
        denominator =  2 * math.pow(sigma, 2)
        power = numerator / denominator
        result[i] = math.exp(power)
    return result

def deltaWeightCalculate(error , transferVectorPerInput ):
    delta = np.zeros(hiddenLayerSize)
    for centroidIndex in range(len(transferVectorPerInput)):
        delta[centroidIndex] = learningRate * error * transferVectorPerInput[centroidIndex]
    return delta

def weightCalculate(deltaVector):
    weightMatrix = np.zeros((hiddenLayerSize))
    for centroidIndex in range(hiddenLayerSize):
            weightMatrix[centroidIndex] = weightMatrix[centroidIndex]+ deltaVector[centroidIndex]
    return weightMatrix

def approximationCalculate(weights , transferVector):
    approximation = np.dot(weights , transferVector.T)
    return approximation

def errorCalculate(target , approximation):
    difference = target - approximation
    error = 0.5 * math.pow(difference , 2)
    return error

def sinFunctionLearning():
    inputs = generateInput (inputSize, lowerBound , stepSize)
    realOutputs = sinFunctionCalculate (inputs)
    centroids = calculateCenteroids(inputSize, hiddenLayerSize , inputs)
    sigma = calculateSigma( inputSize,  inputs , hiddenLayerSize  , centroids)   
    approximations = np.zeros(inputSize)
    iters = 0
    for inputIndex in range(len(inputs)):
        iters = 0
        while (iters <= 500):
            weights = np.random.rand(hiddenLayerSize)
            transferVector = transferFunctionCalculate(inputIndex, centroids , sigma  , hiddenLayerSize , inputs )
            approximation = approximationCalculate(weights , transferVector )
            error = errorCalculate(realOutputs[inputIndex] , approximation)
            updateValue = deltaWeightCalculate(error , transferVector)
            weights =  weights - updateValue
            approximations[inputIndex] = approximationCalculate(weights , transferVector )
            iters =iters+ 1
    plt.plot([x for x in inputs], [y for y in realOutputs ] , 'bo')
    plt.plot([x for x in inputs], [y for y in approximations] , 'ro')
    return plt.show()

sinFunctionLearning()
