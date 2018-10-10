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

start = -5 
end = 5 
step = 0.5 
inputSize = int ( (end - start)/ step)
matrixSize = int ( inputSize*inputSize)  

def function (x , y):
    return np.exp(- (x ** 2  + y** 2)  / 10 ) - 0.5

def generateTargets (z):
    targets = np.array(z).reshape( inputSize , 1)
    return targets

def plotData ( x , y , z):
    Z = z.reshape(inputSize , 1)
    X,Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap= 'viridis' ,edgecolor='none')
    ax.set_title("Bell-Shaped Gaussian")
    plt.show()

def run():
    x = np.arange(-5.0,5.0,0.5)
    y = np.arange(-5.0,5.0,0.5)
    z  = function( x, y)
    targets = generateTargets(z)
    xx, yy = np.meshgrid(x, y) 
    patterns = np.concatenate((xx.reshape( 1, matrixSize).copy(), yy.reshape( 1, matrixSize).copy()))
    plotData ( x, y, z)

run()