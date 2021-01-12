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


def base_model(activation=swish,
               loss=focal_loss,
               optimizer='adam',
               name='SSBML-Base-Model'):

    model = Sequential(name=name)

    # -----------------------------------------------
    
    # number of filters: 128
    # size of filters:   30
    # sees: .5s
    model.add(Sequential([
        Conv1D(150, 30, activation=activation, input_shape=(None, 13)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
    ],  name = 'ConvCell-1'))
    
    # -----------------------------------------------
    
    # number of filters: 256
    # size of filters:   15
    # sees: 1s
    model.add(Sequential([
        Conv1D(256, 15, activation=activation),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
    ],  name = 'ConvCell-2'))

    # -----------------------------------------------
    
    # number of filters: 256
    # size of filters:   15
    # sees: 4s
    model.add(Sequential([
        Conv1D(512, 15, activation=activation),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(.25)
    ],  name = 'ConvCell-3'))

    # -----------------------------------------------
    # number of filters: 256
    # size of filters:   15
    # sees: 8s
    model.add(Sequential([
        Conv1D(512, 15, activation=activation)
    ],  name = 'ConvCell-4'))

    # sees whole clip,
    # takes average of each channel
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())

    # ----------------------------------------------
    #             HEAD
    # ----------------------------------------------
    
    # Dense Cell 1
    model.add(Sequential([
        Dense(128),
        BatchNormalization(),
        Activation(activation),
        Dropout(.25)
    ], name = 'DenseCell-1'))

    # Dense Cell 2
    model.add(Sequential([
        Dense(128),
        BatchNormalization(),
        Activation(activation),
        Dropout(.25)
    ], name = 'DenseCell-2'))

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy', top_8_accuracy])
    
    return model
