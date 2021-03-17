from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model
from tensorflow_addons.losses import SigmoidFocalCrossEntropy as Focal

OPTIMIZER = 'adam'

LOSS = Focal()

METRICS = [
    metrics.CategoricalAccuracy(name='accuracy'),
    metrics.Precision(name='player_precision', class_id=0),
    metrics.Recall(name='player_recall', class_id=0),
    metrics.Precision(name='nonplayer_precision', class_id=1),
    metrics.Recall(name='nonplayer_recall', class_id=1),
]

HEAD = Sequential([
    
    # dense cell 1
    Dense(128, name='head_dense_1'),
    BatchNormalization(name='head_batchnorm_1'),
    Activation(swish, name='head_activation_1'),
    Dropout(.5, name='head_dropout_1'),
    
    # dense cell 2
    Dense(128, name='head_dense_2'),
    BatchNormalization(name='head_batchnorm_2'),
    Activation(swish, name='head_activation_2'),
    Dropout(.5, name='head_dropout_2'),
    
    # final output layer
    Dense(2, activation='softmax', name='output')
    
], name='Binary-Classifier')

def remove_head(
        base_model, 
        trainable = False
    ):
    # use flatten layer to 
    # determine where the head
    # starts and remove it
    for i, layer in enumerate(base_model.layers):
        if 'flatten' in layer.name:
            head_start = i + 1
            break
    else:
        raise ValueError('base_model has no flatten layer')
    
    name = base_model.name
    model = Sequential(base_model.layers[:head_start], name=name)
    model.trainable = trainable
    return model

def add_new_head(
        base_model,
        name = 'transfer_model',
        head = HEAD,
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS
    ):

    model = Sequential([base_model, head], name=name)
    model.compile(optimizer, loss, metrics)
    model.build(input_shape=(None, None, 13))
    return model

def replace_head(
        base_model,
        name = 'transfer_model',
        head = HEAD,
        optimizer = OPTIMIZER,
        loss = LOSS,
        metrics = METRICS,
        trainable_base = False
    ):

    model = remove_head(base_model, trainable_base)
    model = add_new_head(model, name, head, optimizer, loss, metrics)
    return model
