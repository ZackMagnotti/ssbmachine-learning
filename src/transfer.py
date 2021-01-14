from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.activations import swish
from tensorflow.keras.models import load_model

default_optimizer_ = 'adam'

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
    
], name='head_densex2')

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
        model,
        head = standard_head,
        name = 'transfer_model',
        optimizer = default_optimizer_,
        loss = 'binary_crossentropy',
        metrics = ['binary_accuracy']
    ):

    model = Sequential([model, head], name=name)
    model.compile(optimizer, loss, metrics)
    model.build(input_shape=(None, None, 13))
    return model

def replace_head(
        model,
        head = standard_head,
        name = 'transfer_model',
        trainable_base = False,
        optimizer = default_optimizer_,
        loss = 'binary_crossentropy',
        metrics = ['binary_accuracy']
    ):

    model = remove_head(model, trainable_base)
    model = add_new_head(model, head, name, optimizer, loss, metrics)
    return model

def transfer_model(name, base_model):
    return replace_head(load_model(base_model), name=name)
