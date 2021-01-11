from tensorflow import keras

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

    base_layers = base_model.layers[:head_start]
    
    headless_base_model = keras.Sequential(base_layers)
    headless_base_model.build(input_shape=(None, None, 13))
    headless_base_model.trainable = trainable

    return headless_base_model

def add_new_head(base_layers):
    pass

def replace_head(base_model):
    pass
