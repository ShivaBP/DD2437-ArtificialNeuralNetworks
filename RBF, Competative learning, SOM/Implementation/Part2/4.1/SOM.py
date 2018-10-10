import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

stepSize = 0.2
numEpochs = 21

def readInput():
    inputs = np.loadtxt('animals.dat', delimiter= ',' , dtype=int)
    inputs = inputs.reshape(32, 84)
    return inputs

def animals():
    animals = np.loadtxt('animalnames.txt', dtype=str)
    return np.array(animals)

def initWeigths():
    weights = np.zeros((100 , 84))
    for row in range(len(weights)):
        for col in range(len(weights[0])):
            value = random.random()
            weights[row][col] = value
    return weights

def attributesPerAnimal(inputs , animalIndex):
    attributes = np.zeros(len(inputs[0]))
    for col in range(len(inputs[0])):
        attributes[col] = inputs[animalIndex][col]   
    return attributes

def closestWeigthRow(weights, attributes):
    numNodes = len(weights)
    minDistance = np.inf
    closestNodeIndex = -1 
    for nodeIndex in range(numNodes):
        a = np.subtract(attributes , weights[nodeIndex])
        b = np.subtract(attributes , weights[nodeIndex])
        distance = math.sqrt(np.dot(a.T , b))
        if ( distance < minDistance):
            minDistance = distance
            closestNodeIndex = nodeIndex
    return closestNodeIndex

def weightUpdate(attributes , weights , winnerIndex , radius):
    numNodes = len(weights)
    numAttributes = len(attributes)
    updatedWeights = weights
    minLimit = winnerIndex - radius
    if(minLimit < 0):
        minLimit = 0
    maxLimit = winnerIndex + radius
    if (maxLimit > 100):
        maxLimit = 100
    for index in range (minLimit  , maxLimit):
        stepValue = stepSize * np.subtract(attributes , weights[index])
        updatedWeights[index] = np.add(updatedWeights[index] , stepValue)
    return updatedWeights

def run():
    initRadius= 50
    inputs= readInput()
    animalList = animals()
    numAnimals = len(animalList)
    weights = initWeigths()
    animalResult = np.zeros(numAnimals)
    for epoch in range(numEpochs):
        radius = round ((initRadius - (epoch * initRadius/ (numEpochs-1)))/2 )
        for animalIndex in range(numAnimals):
            attributes = attributesPerAnimal(inputs, animalIndex)
            closestNode = closestWeigthRow(weights, attributes)
            weights = weightUpdate(attributes , weights , closestNode , radius)
            closestNode = closestWeigthRow(weights, attributes)
            animalResult[animalIndex] = closestNode
    print(animalList[np.argsort(animalResult)].reshape(32))
run()


