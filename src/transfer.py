from tensorflow.keras import Sequential

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

def add_new_head(base_model, head):
    return Sequential([base_model, head])

def replace_head(base_model):
    pass
