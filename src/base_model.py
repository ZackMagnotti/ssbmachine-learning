'''
Author : Zack Magnotti
Email : zack@magnotti.net
Date : 3/19/2021

Python module containing the Keras code
for the creation of SSBML-Base-Model
'''

from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, Flatten
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.activations import swish
from tensorflow_addons.losses import SigmoidFocalCrossEntropy as Focal

NAME = 'SSBML-Base-Model'

ACTIVATION = swish

OPTIMIZER = keras.optimizers.Adam()

LOSS = Focal()

METRICS = [
    metrics.CategoricalAccuracy(name='accuracy'),
    metrics.TopKCategoricalAccuracy(k=8, name='top 8 accuracy'),
]

def base_model(
        name = NAME,
        activation = ACTIVATION,
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS,
    ):
    '''
    Creates SSBML-Base-Model architecture 
    and compiles the model with the given 
    optimizer, loss, and metrics.

    Parameters
    -----------
    name (string) : name of output model
    activation : activation function for output model
    optimizer : optimizer for output model
    loss : loss function for output model
    metrics : metrics for output model
    
    Outputs (yield)
    -----------
    model (Sequential) : model with SSBML-Base-Model architecture
    '''

    model = Sequential(name = name)

    # -----------------------------------------------

    # ConvCell-1
    # number of filters: 128
    # size of filters:   30
    # sees: .5s
    model.add(Sequential([
        Conv1D(150, 30, activation=activation, input_shape=(None, 13)),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
    ],  name = 'ConvCell-1'))

    # -----------------------------------------------

    # ConvCell-2
    # number of filters: 256
    # size of filters:   15
    # sees: 1s
    model.add(Sequential([
        Conv1D(256, 15, activation=activation),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
    ],  name = 'ConvCell-2'))

    # -----------------------------------------------

    # ConvCell-3
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

    # ConvCell-4
    # number of filters: 256
    # size of filters:   15
    # sees: 8s
    model.add(Sequential([
        Conv1D(512, 15, activation=activation)
    ],  name = 'ConvCell-4'))

    # ----------------------------------------------

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

    # ----------------------------------------------

    # Dense Cell 2
    model.add(Sequential([
        Dense(128),
        BatchNormalization(),
        Activation(activation),
        Dropout(.25)
    ], name = 'DenseCell-2'))

    # ----------------------------------------------

    # final output layer
    model.add(Dense(26, activation='softmax', name='final'))
    model.compile(
        loss = loss,
        optimizer = optimizer,
        metrics = metrics
    )

    return model
