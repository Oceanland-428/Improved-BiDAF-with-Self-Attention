# Improved BiDAF with Self-Attention
## Overview
This is a final project implementation of CS224N as Stanford University.

There has been extensive researches conducted on conquering SQuAD dataset and Q&A tasks in general. Bi-Directional Attention Flow (BiDAF) is one of the most popular ones. In this project, BiDAF model is provided to us as the baseline mode. We investigated both networks and proposed improvements on the latter one. We performed feature engineering on the embedding layers and added self-attention layer to the model structure. In addition, we designed learnable weighted average-attention layer which itself improved the accuracy of the baseline model by nearly 5%. With the combination of these three features, the resulted F1/EM score of the improved model is increased by an average of 7% on a backbone of Gated Recurrent Unit (GRU) network.

## Model Architecture
