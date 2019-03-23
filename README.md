# Improved BiDAF with Self-Attention
## Overview
This is a final project implementation of [CS224N](http://web.stanford.edu/class/cs224n/) as Stanford University.

This work is collaborated with [Gendong Zhang](https://github.com/zgdsh29) and [Zixuan Zhou](https://github.com/lynnezixuan).

There has been extensive researches conducted on conquering SQuAD dataset and Q&A tasks in general. Bi-Directional Attention Flow (BiDAF) is one of the most popular ones. In this project, BiDAF model is provided to us as the [baseline mode](https://github.com/chrischute/squad). We investigated both networks and proposed improvements on the latter one. We performed feature engineering on the embedding layers and added self-attention layer to the model structure. In addition, we designed learnable weighted average-attention layer which itself improved the accuracy of the baseline model by nearly 5%. With the combination of these three features, the resulted F1/EM score of the improved model is increased by an average of 7% on a backbone of Gated Recurrent Unit (GRU) network.

The final results of this model are 66.241% in F1 and 62.679% in EM for SQuAD 2.0 dataset.

## Model Architecture
The overall model architecture is shown as follows
![alt text](https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention/blob/master/archi.png)

The model starts by the concatenation of word embeddings and character embeddings of the text/query. The embeddings used are the pre-trained Glove word and Glove char embeddings. Notice at the end of each word/character embedding, there is an additional vector called AvgCE/AvgWE, which is our self-designed learnable weighted average attentaion. This vector contains learnable weight to compute average embeddings of each word/character embedding. The embeddings are then feed into a Bi-Derectional RNN layer. The outputs from the RNN layer are used to compute self-attention score. The final results are computed based on the encoder of another RNN layer which takes both the attention and previous RNN outputs as its input.

## Setup
Run this command to get the SQuAD 2.0 dataset:
```
python setup.py
```
The dataset set contains train, dev and test. The dev and test are both from the official dev dataset. CS224N devide it into half for each of the dev and test used in this project.

## Train
Run this command to train the model:
```
python train_1.py -n baseline
```
Thanks to the elegant implementation of the original baseline model, the model evaluates itself periodically, and save the weight that has the best performance on dev set as *best.pth.tar*

## Test
Run this command to test the wieght on the dev/test set and generate a *.csv* file.
```
python test.py --split dev --load_path ./save/train/baseline-01/best.pth.tar --name dev_test_01
```
Note that if the *--split* is *dev*, the script will print the F1 and EM score. If it is *test*, it will only generate the *.csv* file.

## Acknowledgement
We apprieciate the help from the teaching staff of CS224N winter 2019, especially the starter code written by TA Christopher Chute. [Microsoft Azure](https://azure.microsoft.com/en-us/) generously provided financial support used during development.
