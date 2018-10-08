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

def plotData():
    plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
    plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
    return plt.show()
    
plotData()


			