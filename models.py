import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.hidden_size = hidden_size * 2 # adding the char embedding, double the hidden_size. 
        
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        #input_size=self.hidden_size+2 is due to we add two extra features (avg_attention) to both char embedding
        #and word embedding to boost the performance. The avg_attention is use the attention mechanism to learn 
        #a weighted average among the vectors by the model itself.
        self.enc = layers.RNNEncoder(input_size=self.hidden_size+2,
                                     hidden_size=self.hidden_size,
#                                      num_layers=2, # The number of layer can be changed, but less or no improvement.
                                     num_layers = 1,
                                     drop_prob=drop_prob)
        
        self.att = layers.BiDAFAttention(hidden_size=2*self.hidden_size,
                                         drop_prob=drop_prob)
        
        #Add extra layer of self-attention based on the paper 'Simple and Effective Multi-Paragraph Reading Comprehension'
        #URL: https://arxiv.org/pdf/1710.10723.pdf 
        self.self_att = layers.SelfAtt(hidden_size=2*self.hidden_size,
                                      drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8*self.hidden_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        #Planned to use cosine similarity or TF-IDF to add extra feature to the embedding but need more thoughts to well 
        #implement
#         self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
       
        self.out = layers.BiDAFOutput(hidden_size=self.hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
#         print("~~~~~~", c_mask.shape)
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cc_idxs, cw_idxs)         # (batch_size, c_len, hidden_size)
#         c_avg = self.avgatt(c_emb)
#         print(c_emb.shape, c_avg.shape)
#         c_emb = torch.cat((c_emb, c_avg), dim=1)
        # cc_emb = self.char_emb(cc_idxs)
        q_emb = self.emb(qc_idxs, qw_idxs)         # (batch_size, q_len, hidden_size)

        
#         q_avg = self.avgatt(q_emb)
#         q_emb = torch.cat((q_emb, q_avg), dim=1)
        # qc_emb = self.char_emb(qc_idxs)
        # c_emb = torch.cat([cc_emb, cw_emb], dim=1)
        # q_emb = torch.cat([qc_emb, qw_emb], dim=1)
#         for c_word in range(c_emb.shape[1]):
#             for q_word in range(q_emb.shape[1]):
#                 sim = self.sim()
       
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
#         print(att.shape)

        # keep the same dimension as above, but with more feature information embeded
        att = self.self_att(att, c_mask)  # (batch_size, c_len, 8 * hidden_size)
#         print(att.shape)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
