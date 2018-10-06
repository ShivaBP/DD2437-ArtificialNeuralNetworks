import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM

# http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html


numEpochs = 4
eta = 0.1 
numComponents  = 100
numBatch = 255
classes = [18, 5, 9, 1, 6, 8, 10, 29, 27, 7]

def trainingImage():
    data  = np.loadtxt('bindigit_trn.csv', delimiter= ',' , dtype=int)
    data = data.reshape(8000, 784)
    return data

def trainingImageTarget():
    targets = np.loadtxt('targetdigit_trn.csv', dtype=int)
    return targets

def run():
    train= trainingImage()
    trainTarget = trainingImageTarget() 
    rbm = BernoulliRBM(n_components=numComponents, learning_rate=eta, batch_size=numBatch ,random_state=0, verbose= True)   
    errors = np.empty(numEpochs)

    for epoch in range(numEpochs):
        transformedTrain = rbm.fit_transform(train, trainTarget)
        reconstructedTrain = transformedTrain.dot(rbm.components_)
        errors[epoch] = np.sum(np.abs(reconstructedTrain - train))

    digts , digitAxis = plt.subplots(2, len(classes))
    for index in range(len(classes)):
        imageIndex = classes[index]
        original  = np.copy(train[imageIndex].reshape(28,28))
        reconstructed= np.copy(reconstructedTrain[imageIndex].reshape(28,28))
        digitAxis[0, index].imshow(original, extent=(0, 28, 0, 28) , aspect = 'auto') 
        digitAxis[1, index].imshow(reconstructed, extent=(0, 28, 0, 28) , aspect = 'auto' )
    plt.show()


run()