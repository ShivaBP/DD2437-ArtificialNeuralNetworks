import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
from random import randint
import math

lowerBound = 0
upperBound = 7
stepSize = 0.1
inputSize = (int)((upperBound-lowerBound)/stepSize)
numberOfClassifiers = 30
etha = 0.2

def generateData ():
    inputs = np.zeros([inputSize])
    current = lowerBound
    for index in range(len(inputs)):
        inputs[index] = current
        current = current + stepSize
    return  inputs
    
def sinFunctionCalculate (inputs):
    outputs= np.zeros(len(inputs))
    for inputIndex in range(len(outputs)):
        x = inputs[inputIndex]
        y = math.sin(2*x)
        outputs[inputIndex] = y
    return outputs

def generateInput():
    inputs= generateData()
    outputs = sinFunctionCalculate(inputs)
    dataVectors = np.zeros((inputSize , 2))
    for index in range(inputSize):
        dataVectors[index][0] = inputs[index]
        dataVectors[index][1] = outputs[index]
    return dataVectors

def generateCodeBook():
    data = generateInput()
    classifiers = random.sample(list(data), numberOfClassifiers)
    result= np.array(classifiers)
    return result

def vectorCodebookDistance(vector , codeBook):
    minDistance = float('Inf')
    codeBookVectorRow  = -1
    for row in range(len(codeBook)):
        xDifference = vector[0] - codeBook[row][0]
        yDifference = vector[1] - codeBook[row][1]
        temp = math.pow(xDifference , 2)  + math.pow(yDifference , 2) 
        distance = math.sqrt(temp)
        if(distance < minDistance):
            minDistance = distance
            codeBookVectorRow = row
    return codeBookVectorRow 

def approximation(inputs ,  codeBook):
    approximations = np.zeros((inputSize , 2))
    for dataRow in range(inputSize):
        classifierIndex = vectorCodebookDistance(inputs[dataRow] , codeBook)
        approximations[dataRow][0] = codeBook[classifierIndex][0]
        approximations[dataRow][1] = codeBook[classifierIndex][1]
    return approximations

def error(inputs , approximations):
    difference = np.linalg.norm (approximations-inputs)
    error = 0.5 * math.pow(difference , 2)
    return error

def updateClassifiers( codebook, error):
    delta = np.zeros((numberOfClassifiers , 2))
    for row in range(numberOfClassifiers):
        delta[row][0] = codebook[row][0] + (etha * error )
        delta[row][1] = codebook[row ][1] + (etha * error )
    return delta

def run():
    inputs = generateInput()
    codebook = generateCodeBook()
    approximations = approximation(inputs , codebook)
    inaccuracy = error(inputs , approximations)
    updatedCodebook = updateClassifiers(codebook , inaccuracy)
    approximations = approximation(inputs , updatedCodebook)
    realX = np.zeros(inputSize)
    realY = np.zeros(inputSize)
    approxX = np.zeros(inputSize)
    approxY = np.zeros(inputSize)
    for row in range(inputSize):
        for col in range(2):
            realX[row] = inputs[row][0]
            realY[row] = inputs[row][1]
            approxX[row] = approximations[row][0]
            approxY[row] = approximations[row][1]
    plt.plot([x for x in realX], [y for y in realY ] , 'bo')
    plt.plot([x for x in approxX], [y for y in approxY] , 'ro')
    return plt.show()

run()