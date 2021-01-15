from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model

default_optimizer_ = 'adam'
default_metrics_ = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
]

onehot_metrics = [
    metrics.CategoricalAccuracy(name='accuracy'),
    metrics.Precision(name='player_precision', class_id=0),
    metrics.Recall(name='player_recall', class_id=0),
    metrics.Precision(name='nonplayer_precision', class_id=1),
    metrics.Recall(name='nonplayer_recall', class_id=1),
]

standard_head = Sequential([
    
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
    Dense(1, activation='sigmoid', name='output')
    
], name='sigmoid_binary_classifier')

onehot_head = Sequential([
    
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
    
], name='onehot_binary_classifier')

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
    base_model = Sequential(base_model.layers[:head_start], name=name)
    base_model.trainable = trainable
    return base_model

def add_new_head(
        base_model,
        head = standard_head,
        name = 'transfer_model',
        optimizer = default_optimizer_,
        loss = 'binary_crossentropy',
        metrics = default_metrics_
    ):

    base_model = Sequential([base_model, head], name=name)
    base_model.compile(optimizer, loss, metrics)
    base_model.build(input_shape=(None, None, 13))
    return base_model

def replace_head(
        base_model,
        head = standard_head,
        name = 'transfer_model',
        trainable_base = False,
        optimizer = default_optimizer_,
        loss = 'binary_crossentropy',
        metrics = default_metrics_
    ):

    base_model = remove_head(base_model, trainable_base)
    base_model = add_new_head(base_model, head, name, optimizer, loss, metrics)
    return base_model

def ssbml_transfer_model(
        head = standard_head,
        name = 'SSBML-Transfer-Model',
        base_model = 'models/SSBML-Base-Model',
        trainable_base = False,
        optimizer = default_optimizer_,
        loss = 'binary_crossentropy',
        metrics = default_metrics_
    ):
    return replace_head(load_model(base_model), head, name, trainable_base, optimizer, loss, metrics)
