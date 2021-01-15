'''
    This file contains all of the rough drafts of different
    base models I tried out in the early days of this project. 

    custom_mk4 was what eventually ended up becoming SSBML-Base-Model
'''
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, SpatialDropout1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Activation
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

def vgg_type(activation=swish,
             loss=focal_loss,
             optimizer='nadam'):

    model = Sequential()

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 32
    # size of filters:   15
    # sees: .25s

    model.add(Conv1D(32, 15, activation=activation))
    model.add(Conv1D(32, 15, activation=activation))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 64
    # size of filters:   15
    # sees: .5s

    model.add(Conv1D(64, 15, activation=activation))
    model.add(Conv1D(64, 15, activation=activation))
    model.add(SpatialDropout1D(.2))
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

    model.add(Dense(80, activation=activation))
    model.add(Dense(80, activation=activation))

    # final output layer
    model.add(Dense(26, activation='softmax'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

def custom_type(activation=swish,
                loss=focal_loss,
                optimizer='adam',
                name='custom'):

    model = Sequential(name=name)

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 32
    # size of filters:   15
    # sees: .25s

    model.add(Conv1D(64, 15, activation=activation, name='conv1.1'))
    model.add(Conv1D(64, 15, activation=activation, name='conv1.2'))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 64
    # size of filters:   15
    # sees: .5s

    model.add(Conv1D(64, 15, activation=activation, name='conv2.1'))
    model.add(Conv1D(64, 15, activation=activation, name='conv2.2'))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   15
    # sees: 1s

    model.add(Conv1D(128, 15, activation=activation, name='conv3.1'))
    model.add(Conv1D(128, 15, activation=activation, name='conv3.2'))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 2s
    model.add(Conv1D(128, 15, activation=activation, name='conv4.1'))
    model.add(Conv1D(128, 15, activation=activation, name='conv4.2'))
    model.add(SpatialDropout1D(.2))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Conv1D(256, 30, activation=activation, name='conv5'))
    model.add(SpatialDropout1D(.5))

    # sees whole 30s, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    model.add(Dense(80, activation=activation, name='dense1'))
    model.add(Dense(80, activation=activation, name='dense2'))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

def custom_mk2(activation=swish,
               loss=focal_loss,
               optimizer='adam',
               name='custom_mk2'):

    model = Sequential(name=name)

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   30
    # sees: .5s

    model.add(Conv1D(150, 30, activation=activation, name='conv1'))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   15
    # sees: .5s

    model.add(Conv1D(128, 15, activation=activation, name='conv2'))
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 1s
    model.add(Conv1D(128, 15, activation=activation, name='conv3.1'))
    model.add(Conv1D(128, 15, activation=activation, name='conv3.2'))
    model.add(MaxPooling1D(pool_size=4))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Conv1D(256, 15, activation=activation, name='conv4'))

    # sees whole clip, takes pool
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    model.add(Dense(80, activation=activation, name='dense1'))
    model.add(Dropout(.2))

    model.add(Dense(80, activation=activation, name='dense2'))
    model.add(Dropout(.2))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

def custom_mk3(activation=swish,
               loss=focal_loss,
               optimizer='adam',
               name='custom_mk3'):

    model = Sequential(name=name)

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   30
    # sees: .5s

    model.add(Conv1D(150, 30, activation=activation, name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   15
    # sees: .5s

    model.add(Conv1D(128, 15, activation=activation, name='conv2.1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 1s
    model.add(Conv1D(128, 15, activation=activation, name='conv3.1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 2s
    model.add(Conv1D(256, 15, activation=activation, name='conv4.1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(.25))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Conv1D(512, 15, activation=activation, name='conv5.1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(.25))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 8s
    model.add(Conv1D(512, 15, activation=activation, name='conv6'))

    # sees whole clip, takes max pool
    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(.25))

    model.add(Dense(128, use_bias=False, name='dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    model.add(Dense(128, use_bias=False, name='dense2'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

# The Thickest Model
# Computationally expensive,
# but extracts lots of features
def custom_mk4(activation=swish,
               loss=focal_loss,
               optimizer='adam',
               name='custom_mk4'):

    model = Sequential(name=name)

    # -----------------------------------------------
    # number of filters: 128
    # size of filters:   30
    # sees: .5s
    model.add(Conv1D(150, 30, activation=activation, input_shape=(None, 13)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 1s
    model.add(Conv1D(256, 15, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Conv1D(512, 15, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(.25))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 8s
    model.add(Conv1D(512, 15, activation=activation))

    # sees whole clip, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model

# More slim than mk4
def custom_mk5(activation=swish,
               loss=focal_loss,
               optimizer='adam',
               name='custom_mk5'):

    model = Sequential(name=name)

    model.add(Input(shape=(None, 13)))

    # -----------------------------------------------
    # number of filters: 150
    # size of filters:   30
    # sees: .5s

    model.add(Conv1D(150, 30, activation=activation, name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # -----------------------------------------------
    # number of filters: 150
    # size of filters:   15
    # sees: .5s
    model.add(Conv1D(150, 15, activation=activation, name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 2s
    model.add(Conv1D(256, 15, activation=activation, name='conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(.25))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 8s
    model.add(Conv1D(256, 15, activation=activation, name='conv4'))
    model.add(BatchNormalization())
    model.add(Dropout(.25))

    # sees whole clip, takes max pool
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(80, use_bias=False, name='dense1'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    # Dense Layer 2
    model.add(Dense(80, use_bias=False, name='dense2'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(.25))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model
