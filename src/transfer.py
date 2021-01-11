from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.activations import swish

standard_head = [
    # dense layer 1
    Dense(128, name='head_dense_1')
    BatchNormalization(name='head_batchnorm_1')
    Activation(swish, name='head_activation_1')
    Dropout(.5, name='head_dropout_1')
    
    # dense layer 2
    Dense(128, name='head_dense_2')
    BatchNormalization(name='head_batchnorm_2')
    Activation(swish, name='head_activation_2')
    Dropout(.5, name='head_dropout_2')
    
    # final output layer
    Dense(1, activation='sigmoid', name='output')
]

def remove_head(base_model, trainable=False):
    # use flatten layer to 
    # determine where the head
    # starts and remove it
    for i, layer in enumerate(base_model.layers):
        if layer.name == 'flatten':
            head_start = i + 1
            break
    else:
        raise ValueError('base_model has no flatten layer')
    base_model = Sequential(base_model.layers[:head_start])
    headless_base_model.trainable = trainable
    return headless_base_model

def add_new_head(model, head=standard_head, optimizer='nadam'):
    model = Sequential([model, head])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['binary_accuracy'])
    return model

def replace_head(base_model):
    pass
