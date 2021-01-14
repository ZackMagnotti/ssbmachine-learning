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

### Results

```
Test accuracy: 97%
Test top 8 accuracy: 99.6%
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
mixed with random examples from the public anonymous dataset.

While the character classification training dataset was close to 200G (before processing), 
individual players' datasets were typically closer to 1G.

### Results

---
#### Blynde

Test accuracy: **88%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.461    |	0.059    |
|nonplayer |	0.065    |	0.415    |

---
#### gh0st

Test accuracy: **87%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.510    |	0.050    |
|nonplayer |	0.065    |	0.375    |

---
#### SmashMaster9000

Test accuracy: **90%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.407    |	0.033    |
|nonplayer |	0.062    |	0.498    |

---
#### ixwonkr

Test accuracy: **81%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.431    |	0.129    |
|nonplayer |	0.079    |	0.361    |

---
#### Lie0x

Test accuracy: **80%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.44    |	0.12    |
|nonplayer |	0.06    |	0.38    |

---
#### TCBL

Test accuracy: **81%**

|	       |player	     | nonplayer |
| -------  | ----------- | --------- |
|player    |	0.291    |	0.069    |
|nonplayer |	0.093    |	0.547    |
