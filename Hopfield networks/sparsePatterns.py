import numpy as np
from random import randint
import math


numberOfNeurons = 100
activity  = 0.1
bias = 0.5

def generateInput():
    numRows = 100
    numCols = 100
    inputs = np.zeros((100,100))
    for row in range(numRows):
        for col in range (numCols):
            value = randint(0 , 2)
            if (value <= 1.5):
                inputs[row][col] = 0
            else:
                inputs[row][col] = 1
    return inputs

def inputVectorSeperation ( vectorIndex , inputs):
    vector = inputs[vectorIndex -1 ]
    return vector

def sign(value):
    return -(value)
def weightsPerVector(neuronVector):
    weights = np.zeros((numberOfNeurons , numberOfNeurons))
    for row in range(numberOfNeurons):
        for col in range(numberOfNeurons):
            if(row == col):
                weights[row][col] = 0
            else:
                weights[row][col] = (neuronVector[row]-activity) * (neuronVector[col] - activity)
                weights[col][row] = weights[row][col]
    return weights

def energy(weights, neuronVector):
    energy = - np.dot(weights  , (np.dot (neuronVector.T , neuronVector)) )
    return energy

def finalWeights(weights1 , weights2 , weights3):
    finalWeights = np.zeros((numberOfNeurons , numberOfNeurons))
    for row in range(numberOfNeurons):
        for col in range(numberOfNeurons):
            finalWeights[row][col] = (weights1[row][col] + weights2[row][col] + weights3[row][col]) / 8
    return finalWeights

def activation(finalWeights, inputVector):
    activations = np.zeros(numberOfNeurons)
    for index in range(numberOfNeurons):
        sum = 0
        opponent = 0
        while (opponent < 8):
            if (opponent == index):
                opponent = opponent + 1
            else:
                sum = sum + (finalWeights[index][opponent]* inputVector[opponent])
            opponent = opponent +1
        activations[index] = 0.5 + 0.5 * sign(sum - bias)
    return activations

def initLearning():
    inputs = generateInput()
    maxIter = 3
    iter = 0
    while(iter <= maxIter):
        input1 = inputVectorSeperation(1 , inputs)
        weights1 = weightsPerVector(np.array(input1)) 
        energy1 = energy(weights1 , input1)
        print(energy1)

        input2 =  inputVectorSeperation(2 , inputs)
        weights2 = weightsPerVector(np.array(input2))
        energy2 = energy(weights2 , input2) 
        print(energy2)

        input3 =  inputVectorSeperation(3 , inputs)   
        weights3 = weightsPerVector(np.array(input3))
        energy3 = energy(weights3 , input3)
        print(energy3)
        
        final = finalWeights(weights1, weights2 , weights3)
        activations1 = activation(final , input1)
        activations2 = activation(final , input2)
        activations3 = activation(final , input3)
        iter = iter+1  

    print (activations1)
    print (activations2)
    print (activations3)
    

initLearning()