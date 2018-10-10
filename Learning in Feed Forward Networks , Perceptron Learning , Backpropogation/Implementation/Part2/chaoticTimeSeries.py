import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits import mplot3d
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

beta = 0.2 
gamma = 0.1 
n = 10 
delay = 25
rangeStart= 300
rangeEnd = 1500
inputSize = (rangeEnd - rangeStart)

def function ( inputValue):
    result = 0
    previous = inputValue-1
    if (inputValue == 0 ):
        result = 1.5
    elif (inputValue < 0):
        reuslt= 0
    else:
        numerator = 0.2 * function(previous - 25)
        denominator = 1 + np.power(  function (previous - 25), 10 )
        result = function(previous) + (numerator / denominator) - (0.1 (function (previous)))
    return result

def generateInput():
    t = np.zeros(inputSize)
    for index in range(inputSize):
        t[index] = index+1
    return t