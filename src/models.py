from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.activations import swish

top_8_accuracy = keras.metrics.TopKCategoricalAccuracy(k=8, name='top 8 accuracy')

import tensorflow_addons as tfa
focal_loss = tfa.losses.SigmoidFocalCrossEntropy()

def create_model(activation=swish,
                 loss=focal_loss,
                 optimizer='adam'):

    model = Sequential()

    # first conv layer
    # sees .5s
    model.add(Conv1D(150, #num of features extracted from istream
                     30, #number of frames filter can see at once
                     activation=activation))

    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # sees 1s
    model.add(Conv1D(80,
                     30,
                     activation=activation))

    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # sees 2s
    model.add(Conv1D(80,
                     30,
                     activation=activation))

    # sees whole 30s, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    model.add(Dense(80, activation=activation))

    model.add(Dropout(.2))

    model.add(Dense(80, activation=activation))

    model.add(Dropout(.2))

    # final output layer
    model.add(Dense(26, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

def create_beefy_model(activation=swish,
                       loss=focal_loss,
                       optimizer='adam'):

    model = Sequential()

    # first conv layer
    # sees .5s
    model.add(Conv1D(250, #num of features extracted from istream
                     30, #number of frames filter can see at once
                     activation=activation))

    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # sees 1s
    model.add(Conv1D(150,
                     30,
                     activation=activation))

    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # sees 2s
    model.add(Conv1D(150,
                     30,
                     activation=activation))

    # sees whole 30s, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    model.add(Dense(120, activation=activation))

    model.add(Dropout(.2))

    model.add(Dense(120, activation=activation))

    model.add(Dropout(.2))

    # final output layer
    model.add(Dense(26, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

def vgg_headless_type(activation=swish,
             loss=focal_loss,
             optimizer='adam'):

    model = Sequential()

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 32
    # size of filters:   15
    # sees: .25s

    model.add(Conv1D(32, 15, activation=activation))
    model.add(Conv1D(32, 15, activation=activation))
    model.add(SpatialDropout1D(.1))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 64
    # size of filters:   15
    # sees: .5s

    model.add(Conv1D(64, 15, activation=activation))
    model.add(Conv1D(64, 15, activation=activation))
    model.add(SpatialDropout1D(.1))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   15
    # sees: 1s

    model.add(Conv1D(128, 15, activation=activation))
    model.add(Conv1D(128, 15, activation=activation))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 2s
    model.add(Conv1D(256, 15, activation=activation))
    model.add(Conv1D(256, 15, activation=activation))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Conv1D(256, 15, activation=activation))
    model.add(SpatialDropout1D(.5))

    # sees whole 30s, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    # final output layer
    model.add(Dense(26, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model
