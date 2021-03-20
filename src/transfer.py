'''
Author : Zack Magnotti
Email : zack@magnotti.net
Date : 3/19/2021

Python containing the Keras code to
create SSBML-Transfer-Model by removing
and replacing the head of SSBML-Base-Model.
'''

from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model
from tensorflow_addons.losses import SigmoidFocalCrossEntropy as Focal

HEAD = Sequential([

    # dense cell 1
    Sequential([
        Dense(128),
        BatchNormalization(),
        Activation(swish),
        Dropout(.5),
    ],  name = 'DenseCell-1'),

    # dense cell 2
    Sequential([
        Dense(128),
        BatchNormalization(),
        Activation(swish),
        Dropout(.5),
    ],  name = 'DenseCell-2'),

    # final output layer
    Dense(1, activation = 'sigmoid', name = 'output'),

], name = 'Binary-Classifier')

NAME = 'SSBML-Transfer-Model'

OPTIMIZER = 'adam'

LOSS = Focal()

METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(),
    metrics.Recall(),
    
    # this is an ugly hack but it is neccessary as
    # keras does not have simply a "specificity" metric
    metrics.SpecificityAtSensitivity(
        sensitivity = .01, # this doesn't matter
        num_thresholds = 1, # so we only get score at threshold = .5
        name = 'specificity'
    )
]

def remove_head(
        base_model, 
        trainable = False
    ):
    ''' 
    Returns a copy of the base model with the head removed.

    Where the head starts is determined by the location of a "flatten" layer

    Parameters
    -----------
    base_model (Sequential) : base model to be decapitated

    Outputs (yield)
    -----------
    model (Sequential) : headless copy of base_model
    '''

    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, keras.layers.Flatten):
            head_start = i + 1
            break
    else:
        raise ValueError('base_model has no flatten layer')

    name = base_model.name
    model = Sequential(base_model.layers[:head_start], name=name)
    model.trainable = trainable
    return model

def add_new_head(
        headless_base_model,
        head = HEAD,
        name = NAME,
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS
    ):
    ''' 
    Adds a new head to headless_base_model

    Parameters
    -----------
    headless_base_model (Sequential) : base model to be re-headed
    name (string) : name of new model
    head (Sequential) : keras sequential model to act as new head
    optimizer : optimizer for output model
    loss : loss function for output model
    metrics : metrics for output model

    Outputs (yield)
    -----------
    model (Sequential) : full model with new head
    '''

    model = Sequential([headless_base_model, head], name=name)
    model.compile(optimizer, loss, metrics)
    model.build(input_shape=(None, None, 13))
    return model

def replace_head(
        base_model,
        head = HEAD,
        name = NAME,
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS,
        trainable_base = False
    ):
    ''' 
    Returns a copy of the base model with the head replaced.

    Where the head starts is determined by the location of a "flatten" layer

    Parameters
    -----------
    base_model (Sequential) : base model to be re-headed
    name (string) : name of new model
    head (Sequential) : keras sequential model to act as new head
    optimizer : optimizer for output model
    loss : loss function for output model
    metrics : metrics for output model

    Outputs (yield)
    -----------
    model (Sequential) : full model with new head
    '''

    model = remove_head(base_model, trainable_base)
    model = add_new_head(model, head, name, optimizer, loss, metrics)
    return model
