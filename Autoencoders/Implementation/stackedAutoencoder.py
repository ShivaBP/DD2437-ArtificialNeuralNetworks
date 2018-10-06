import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

# https://blog.keras.io/building-autoencoders-in-keras.html

numEpochs = 20
eta = 0.1 
encodingDimension  = 100
numBatch = 100
classes = [18, 5, 9, 1, 6, 8, 10, 29, 27, 7]

def trainingImage():
    data  = np.loadtxt('bindigit_trn.csv', delimiter= ',' , dtype=int)
    data = data.reshape(8000, 784)
    return data

def run():
    train = trainingImage()
    image = Input ( shape=(784,))

    encoded = Dense(encodingDimension,activation='relu',)(image)
    decoded = Dense(784 , activation='sigmoid')(encoded)
    autoencoder = Model(image,decoded)   
    encoder = Model(image,encoded)
    encodedInput = Input(shape = (encodingDimension, ))
    decoder = autoencoder.layers[-1]
    decoder = Model(encodedInput, decoder(encodedInput))

    autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
    autoencoder.fit( train, train, epochs=numEpochs, batch_size=numBatch, shuffle=True)
    
    encodedImages = encoder.predict(train)
    decodedImages = decoder.predict(encodedImages)

    digts , digitAxis = plt.subplots(2, len(classes))
    for index in range(len(classes)):
        imageIndex = classes[index]
        original  = np.copy(train[imageIndex].reshape(28,28))
        decoded= np.copy(decodedImages[imageIndex].reshape(28,28))
        digitAxis[0, index].imshow(original, extent=(0, 28, 0, 28) , aspect = 'auto') 
        digitAxis[1, index].imshow(decoded, extent=(0, 28, 0, 28) , aspect = 'auto' )
    plt.show()

run()

