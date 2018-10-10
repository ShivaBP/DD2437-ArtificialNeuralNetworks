import numpy as np
import random
import math

numberOfNeurons = 1024

def readInput():
    inputs = np.loadtxt('pict.dat', delimiter= ',' , dtype=int)
    inputs = inputs.reshape(11, 1024)
    return inputs

def inputVectorSeperation ( vectorIndex , inputs):
    vector = inputs[vectorIndex -1 ]
    return vector

def weightsPerVector(neuronVector):
    weights = np.zeros((numberOfNeurons , numberOfNeurons))
    for row in range(numberOfNeurons):
        for col in range(numberOfNeurons):
            if(row == col):
                weights[row][col] = 0
            else:
                weights[row][col] = neuronVector[row] * neuronVector[col]
                weights[col][row] = weights[row][col]
    return weights

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

        if (sum >= 0):
            activations[index] = 1
        elif ( sum <0):
            activations[index] = - 1  
    return activations

def initLearning():
    inputs = readInput()
    maxIter = 3
    iter = 0
    while(iter <= maxIter):
        input1 = inputVectorSeperation(1 , inputs)
        input2 =  inputVectorSeperation(2 , inputs)
        input3 =  inputVectorSeperation(3 , inputs)
        weights1 = weightsPerVector(np.array(input1))       
        weights2 = weightsPerVector(np.array(input2))    
        weights3 = weightsPerVector(np.array(input3))
        final = finalWeights(weights1, weights2 , weights3)
        activations1 = activation(final , input1)
        activations2 = activation(final , input2)
        activations3 = activation(final , input3)
        iter = iter+1  
    print (activations1)
    print (activations2)
    print (activations3)

initLearning()