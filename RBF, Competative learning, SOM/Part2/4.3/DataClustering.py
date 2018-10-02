import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import math
import matplotlib.patches as mpatches

stepSize = 0.2
numEpochs = 21

def readAllVotes():
    inputs = np.loadtxt('votes.dat', delimiter= ',' , dtype=float)
    inputs = inputs.reshape(349, 31)
    return inputs

def readDistricts():
    inputs = np.loadtxt('mpdistrict.dat' , dtype=int)
    inputs = inputs.reshape(349 , 1)
    return inputs

def readGenders():
    inputs = np.loadtxt('mpsex.dat' , dtype=int)
    inputs = inputs.reshape(349 , 1)
    return inputs

def readParties():
    inputs = np.loadtxt('mpparty.dat' , dtype=int)
    inputs = inputs.reshape(349 , 1)
    return inputs

def readNmaes():
    inputs = np.loadtxt('mpnames.txt' , dtype=str)
    inputs = inputs.reshape(349 , 1)
    return inputs

def votesPerMp(allVotes , MPIndex):
    votes = np.zeros(31)
    for index in range(31):
        votes[index] = allVotes[MPIndex][index]
    return votes

def initGrid():
    grid = np.zeros(((10,10,31)))
    for row in range(10):
        for col in range(10):
            value = np.random.rand(1,31)
            grid[row][col] = value
    return grid

def closestWeigthRow(grid,  votes):
    minDistance = np.inf
    closestNodePos = np.zeros(2)
    for row in range(10):
        for col in range(10):
            distance = np.dot( (votes-grid[row][col]).T, votes-grid[row][col] )
            if ( distance < minDistance):
                minDistance = distance
                closestNodePos[0] = row
                closestNodePos[1] = col
    return closestNodePos

def weightUpdate(votes , grid  , winnerPos , radius):
    numNodes = 100
    updatedWeights = grid
    minLimitRow= winnerPos[0] - radius
    if(minLimitRow < 0):
        minLimitRow = 0
    minLimitCol= winnerPos[1] - radius
    if(minLimitCol < 0):
        minLimitCol = 0
    maxLimitRow= winnerPos[0] + radius
    if(maxLimitRow >9):
        maxLimitRow = 9
    maxLimitCol= winnerPos[1] + radius
    if(maxLimitCol > 9):
        maxLimitCol = 9
    for row in range (int (minLimitRow)  , int(maxLimitRow)):
        for col in range(int (minLimitCol) , int (maxLimitCol)):          
            stepValue = stepSize * np.subtract(votes , grid[row][col])
            updatedWeights[row][col] = np.add(grid[row][col] , stepValue)
    return updatedWeights

def run():
    initRadius= 10
    allVotes = readAllVotes()
    parties = readParties()
    numVotes = 31
    numMembers = 349
    results = np.zeros(numMembers)
    grid = initGrid()
    for epoch in range(numEpochs):
        radius = round ((initRadius - (epoch * initRadius/ (numEpochs-1)))/2 )
        for MPIndex in range(numMembers):
            votes = votesPerMp(allVotes , MPIndex)
            closestNode = closestWeigthRow(grid , votes )
            grid = weightUpdate(votes , grid , closestNode ,radius)
            closestNode = closestWeigthRow(grid, votes)
            results[MPIndex] = closestNode[0] *10 + closestNode[1]
    print(results)



run()
