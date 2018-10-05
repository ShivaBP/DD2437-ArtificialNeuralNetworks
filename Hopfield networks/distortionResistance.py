
import numpy as np
from random import randint
import math

numberOfNeurons = 1024

def readInput():
    inputs = np.loadtxt('pict.dat', delimiter= ',' , dtype=int)
    inputs = inputs.reshape(11, 1024)
    return inputs

def inputVectorSeperation ( vectorIndex , inputs):
    vector = inputs[vectorIndex -1 ]
    return vector

def addNoise(inputVector):
    maxIter = 10 
    iter = 0
    while(iter < maxIter):
        noiseIndex = randint(0, 1023)
        inputVector[noiseIndex] = - (inputVector[noiseIndex] )
        iter = iter+1
    return inputVector

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
        data1 = inputVectorSeperation(1 , inputs)
        input1 = addNoise(data1)
        weights1 = weightsPerVector(np.array(input1)) 
        energy1 = energy(weights1 , input1)
        print(energy1)

        data2 =  inputVectorSeperation(2 , inputs)
        input2= addNoise(data2)
        weights2 = weightsPerVector(np.array(input2))
        energy2 = energy(weights2 , input2) 
        print(energy2)

        data3 =  inputVectorSeperation(3 , inputs)   
        input3 = addNoise(data3)
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