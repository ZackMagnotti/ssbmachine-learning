# SSBMachine Learning

Use Deep Learning to analyse the variety and similarities between playstyles in Super Smash Bros Melee.

## Part 1: Character Detection
  
Use a Convolutional Neural Network to predict a player's character selection based on the raw inputs from that player's controller.

### Model

```
Model: "SSBML-Base-Model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1d (Conv1D)              (None, None, 150)         58650     
_________________________________________________________________
batch_normalization (BatchNo (None, None, 150)         600       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, None, 150)         0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, None, 256)         576256    
_________________________________________________________________
batch_normalization_1 (Batch (None, None, 256)         1024      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, None, 256)         0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, None, 512)         1966592   
_________________________________________________________________
batch_normalization_2 (Batch (None, None, 512)         2048      
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, None, 512)         0         
_________________________________________________________________
dropout (Dropout)            (None, None, 512)         0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, None, 512)         3932672   
_________________________________________________________________
global_average_pooling1d (Gl (None, 512)               0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               65664     
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512       
_________________________________________________________________
activation (Activation)      (None, 128)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
batch_normalization_4 (Batch (None, 128)               512       
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
final (Dense)                (None, 26)                3354      
=================================================================
Total params: 6,624,396
Trainable params: 6,622,048
Non-trainable params: 2,348
_________________________________________________________________
```

### Training

foo

### Results

bar

## Part 2: Player Detection

Use Transfer Learning to train a neural network to predict *who* is holding the controller, based only on the raw inputs from that controller.

### Model

```
Model: "transfer_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
base (Sequential)            (None, 512)               6537842   
_________________________________________________________________
head_dense64x2 (Sequential)  (None, 1)                 37569     
=================================================================
Total params: 6,575,411
Trainable params: 37,313
Non-trainable params: 6,538,098
_________________________________________________________________
```

### Training

foobar

### Results

barfoo
