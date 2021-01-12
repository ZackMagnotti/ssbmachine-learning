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
ConvCell-1 (Sequential)      (None, None, 150)         59250     
_________________________________________________________________
ConvCell-2 (Sequential)      (None, None, 256)         577280    
_________________________________________________________________
ConvCell-3 (Sequential)      (None, None, 512)         1968640   
_________________________________________________________________
ConvCell-4 (Sequential)      (None, None, 512)         3932672   
_________________________________________________________________
global_average_pooling1d (Gl (None, 512)               0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
DenseCell-1 (Sequential)     (None, 128)               66176     
_________________________________________________________________
DenseCell-2 (Sequential)     (None, 128)               17024     
_________________________________________________________________
final (Dense)                (None, 26)                3354      
=================================================================
Total params: 6,624,396
Trainable params: 6,622,048
Non-trainable params: 2,348
_________________________________________________________________
```

### Training Data

SSBML-Base-Model was trained on just under ***825,000*** examples of Melee gameplay taken from the [Melee Public SLP Dataset](https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f/view?usp=sharing). 

Each example is 1800 frames of input data; ***30 second clips of gameplay from one player.***

[Some EDA type things, class balance etc.]

### Results

```
Test accuracy: 95%
Test test top 8 categorical accuracy: 99%
```

#### Confusion Matrix

![SSBM-Base-Model confusion matrix](images/SSBML-Base-Model.png)

## Part 2: Player Detection

Use *Transfer Learning* to modify SSBML-Base-Model into a model that predicts *who* is holding the controller, based only on the raw inputs from that controller.

### Model

```
Model: "transfer_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
SSBML-Base-Model (Sequential (None, 512)               6537842   
_________________________________________________________________
head_dense64x2 (Sequential)  (None, 1)                 37569     
=================================================================
Total params: 6,575,411
Trainable params: 37,313
Non-trainable params: 6,538,098
_________________________________________________________________
```

### Training Data

Player detection models were trained on 30 second examples of gameplay from an individual player,
mixed with random 30 second examples that were not from that player.

While the character classification training dataset was close to 200G (before processing), 
individual players' datasets were typically closer to 1G.

### Results

pending
