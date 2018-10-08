import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random

n = 100
half = (int)(n/2)
mA = [3, 2]
mB = [-2, -1]
sigmaA = 0.5
sigmaB = 0.3

classA = np.concatenate(
        (np.random.randn(half, 2) * sigmaA + mA,
         np.random.randn(half, 2) * sigmaA + mA))

classB = np.concatenate(
        (np.random.randn(half, 2) * sigmaB + mB,
         np.random.randn(half, 2) * sigmaB + mB))

dataPoints = np.array(np.concatenate((classA, classB)))
bias = np.array([np.ones(dataPoints.shape[0])])
inputs = np.concatenate((dataPoints, bias.T), axis=1)
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

def decision_boundary():
    weights = seqDelta()
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
# https://medium.com/@thomascountz/calculate-the-decision-boundary-of-a-single-perceptron-visualizing-linear-separability-c4d77099ef38
    for x in np.linspace(np.amin(inputs[:, 0]), np.amax(inputs[:, 1])):
        slope = -(weights[0][0] / weights[0][1]) / (weights[0][0] / weights[0][1])
        intercept = -weights[0][0] / weights[0][2]
        y = slope * x + intercept
        plt.plot(x, y, 'ko')
    return plt.show()

def batchPerceptron():
    weights = np.random.normal(size = (1, len(inputs[0])))
    eta = 0.5
    numberOfEpochs = 30
    for epoch in range(numberOfEpochs):
        predictions = np.dot(weights, inputs.T)
        activations = np.where(predictions > 0, 1, 0)
        error = predictions - activations
        weightDifference = eta*np.dot(error, inputs)
        weights = weights + weightDifference
    return weights 

def batchDelta():
    weights = np.random.normal(size = (1, len(inputs[0])))
    eta = 0.01
    numberOfEpochs = 20
    for epoch in range(numberOfEpochs):        
        predictions = np.dot( weights , inputs.T)   
        error = np.subtract(predictions , targets)      
        weights = np.add(weights, (- eta * np.dot(error, inputs)))
    return weights

def seqDelta():
    weights = np.random.normal(size = (1, len(inputs[0])))
    eta = 0.0001
    numberOfEpochs = 20
    for epoch in range(numberOfEpochs):
        for inputRow in range(len(inputs)):
            sum = np.dot(weights , inputs[inputRow].T)
            error = sum - targets[inputRow]
            inputArray= inputs[inputRow].reshape(1, 3)
            update = np.dot(error ,inputArray)
            updateValue = - eta * update
            weights =  np.add(weights ,updateValue)
    return weights
    
decision_boundary()
			