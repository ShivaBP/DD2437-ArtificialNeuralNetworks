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

def readGenders():
    inputs = np.loadtxt('mpsex.dat' , dtype=int)
    inputs = inputs.reshape(349 , 1)
    return inputs

def readParties():
    inputs = np.loadtxt('mpparty.dat' , dtype=int)
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

def plotParties(resultVector) :
    parties = readParties()
    for memberIndex in range(349):
        x = resultVector[memberIndex][0] + 1
        y = resultVector[memberIndex][1] + 1
        if parties[memberIndex] == 1:
            plt.plot([x], [y], 'bo')
        elif parties[memberIndex] == 2:
            plt.plot([x], [y], 'co')
        elif parties[memberIndex] == 3:
            plt.plot([x], [y], 'ko')
        elif parties[memberIndex] == 4:
            plt.plot([x], [y], 'ro')
        elif parties[memberIndex] == 5:
            plt.plot([x], [y], 'go')
        elif parties[memberIndex] == 6:
            plt.plot([x], [y], 'yo')
        elif parties[memberIndex] == 7:
            plt.plot([x], [y], 'mo')
        elif parties[memberIndex] == 0:
            plt.plot([x] , [y], 'wo')
    plt.grid()
    plt.axis([0, 15, 0, 15])
    m = mpatches.Patch(color='blue', label='Moderaterna')
    fp = mpatches.Patch(color='cyan', label='Liberalerna')
    s = mpatches.Patch(color='black', label='Socialdemokraterna')
    v = mpatches.Patch(color='red', label='Vänsterpartiet')
    mp = mpatches.Patch(color='green', label='Miljöpartiet')
    kd = mpatches.Patch(color='yellow', label='Kristdemokraterna')
    c = mpatches.Patch(color='magenta', label='Centerpartiet')
    noP = mpatches.Patch(color='white', label='No party')
    plt.axis('on')
    plt.legend(handles=[m, fp, s, v, mp, kd, c, noP],bbox_to_anchor=(1 , 1), loc=1, borderaxespad=0.)    
    return plt.show()

def plotGenders(resultVector) :
    genders = readGenders()
    for memberIndex in range(349):
        x = resultVector[memberIndex][0] + 1
        y = resultVector[memberIndex][1] + 1
        if genders[memberIndex] == 0:
            plt.plot([x], [y], 'bo')
        elif genders[memberIndex] == 1:
            plt.plot([x], [y], 'mo')
    plt.grid()
    plt.axis([0, 15, 0, 15])
    m = mpatches.Patch(color='blue', label='Male')
    f = mpatches.Patch(color='magenta', label='Female')
    plt.axis('on')
    plt.legend(handles=[m, f],bbox_to_anchor=(1 , 1), loc=1, borderaxespad=0.)    
    return plt.show()     

def run():
    initRadius= 8
    allVotes = readAllVotes() 
    numVotes = 31
    numMembers = 349
    results = np.zeros((numMembers, 2))
    grid = initGrid()
    for epoch in range(numEpochs):
        radius = round (initRadius - (0.4 * epoch))
        for MPIndex in range(numMembers):
            votes = votesPerMp(allVotes , MPIndex)
            closestNode = closestWeigthRow(grid , votes )
            grid = weightUpdate(votes , grid , closestNode ,radius)
            closestNode = closestWeigthRow(grid, votes)
            results[MPIndex][0] = closestNode[0] 
            results[MPIndex][1] = closestNode[1] 
    plotParties(results)
    plotGenders(results)

run()
