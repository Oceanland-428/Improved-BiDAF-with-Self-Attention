# Improved BiDAF with Self-Attention
## Overview
This is a final project implementation of CS224N as Stanford University.

There has been extensive researches conducted on conquering SQuAD dataset and Q&A tasks in general. Bi-Directional Attention Flow (BiDAF) is one of the most popular ones. In this project, BiDAF model is provided to us as the baseline mode. We investigated both networks and proposed improvements on the latter one. We performed feature engineering on the embedding layers and added self-attention layer to the model structure. In addition, we designed learnable weighted average-attention layer which itself improved the accuracy of the baseline model by nearly 5%. With the combination of these three features, the resulted F1/EM score of the improved model is increased by an average of 7% on a backbone of Gated Recurrent Unit (GRU) network.

## Model Architecture
The overall model architecture is shown as follows
![alt text](https://github.com/Oceanland-428/Improved-BiDAF-with-Self-Attention/blob/master/archi.png)

The model starts by the concatenation of word embeddings and character embeddings of the text/query. The embeddings used are the pre-trained Glove word and Glove char embeddings. Notice at the end of each word/character embedding, there is an additional vector called AvgCE/AvgWE, which is our self-designed learnable weighted average attentaion. This vector contains learnable weight to compute average embeddings of each word/character embedding. The embeddings are then feed into a Bi-Derectional RNN layer. The outputs from the RNN layer are used to compute self-attention score. The final results are computed based on the encoder of another RNN layer which takes both the attention and previous RNN outputs as its input.
