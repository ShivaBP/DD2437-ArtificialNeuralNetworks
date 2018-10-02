import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math

stepSize = 0.2
numEpochs = 21

def readInput():
    inputs = np.loadtxt('cities.dat', delimiter= ',' , dtype=float)
    inputs = inputs.reshape(10, 2)
    return inputs

def plotCityPoints(inputs):
    numPoints = len(inputs)
    xcoordinates= np.zeros(10)
    ycoordinates= np.zeros(10)
    for row in range(numPoints):
        for col in range(2):
            xcoordinates[row] = inputs[row][0]
            ycoordinates[row] = inputs[row][1]   
    plt.plot([x for x in xcoordinates], [y for y in ycoordinates ] , 'bo')
    return plt.show()

def plotTourPath(inputs , result):
    for index in range(len(result)):
        nextIndex = index+1
        if(nextIndex > 9):
            nextIndex = 0
        thisP = coordinatesPerPoint(inputs , int (result[index]))
        nextP = coordinatesPerPoint(inputs , int (result[nextIndex]))
        print(thisP)
        x1 = thisP[0]
        y1 = thisP[1]
        x2 = nextP[0]
        y2= nextP[1]
        plt.plot(x1, y1 , 'bo')
        plt.plot(x2, y2 , 'bo')
        plt.plot([x1,x2],[y1 , y2],'-r')
    return plt.show()

def initWeigths():
    weights = np.zeros((10 , 2))
    for row in range(len(weights)):
        for col in range(len(weights[0])):
            value = random.random()
            weights[row][col] = value
    return weights
def coordinatesPerPoint(inputs , index):
    numPoints = len(inputs)
    coordinates = np.zeros(2)
    coordinates[0] = inputs[index][0]
    coordinates[1] = inputs[index][1]
    return coordinates

def closestWeigthRow(weights, coordinates):
    numNodes = len(weights)
    minDistance = np.inf
    closestNodeIndex = -1 
    for nodeIndex in range(numNodes):
        a = np.subtract(coordinates , weights[nodeIndex])
        b = np.subtract(coordinates , weights[nodeIndex])
        distance = math.sqrt(np.dot(a.T , b))
        if ( distance < minDistance):
            minDistance = distance
            closestNodeIndex = nodeIndex
    return closestNodeIndex

def weightUpdate(coordinates , weights , winnerIndex , radius):
    numNodes = len(weights)
    updatedWeights = weights
    minLimit = winnerIndex - radius
    if(minLimit < 0):
        minLimit = 10 - abs(minLimit)
    maxLimit = winnerIndex + radius
    if (maxLimit > 10):
        maxLimit = 0 + (maxLimit -10 )
    for index in range (minLimit  , maxLimit):
        stepValue = stepSize * np.subtract(coordinates , weights[index])
        updatedWeights[index] = np.add(updatedWeights[index] , stepValue)
    return updatedWeights

def run():
    initRadius= 2
    inputs= readInput()
    numCities = len(inputs)
    weights = initWeigths()
    results = np.zeros(10)
    for epoch in range(numEpochs):
        radius = round ((initRadius - (epoch * initRadius/ (numEpochs-1)))/2 )
        for cityIndex in range(numCities):
            coordinates = coordinatesPerPoint(inputs , cityIndex)
            closestNode = closestWeigthRow(weights, coordinates)
            weights = weightUpdate(coordinates , weights , closestNode , radius)
            closestNode = closestWeigthRow(weights, coordinates)
            results[cityIndex] = closestNode
    print(results)
    plotCityPoints(inputs)
    plotTourPath(inputs, results)
run()


