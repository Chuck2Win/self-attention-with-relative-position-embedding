# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:59:40 2021

@author: OK
"""

# relative positional embedding

import torch
import torch.nn as nn
import math
# k : max distance for clipping

def make_relative_position(q_seq_len, seq_len, k):
    Q = torch.arange(q_seq_len)[:,None] # q_seq_len, 1
    # it might be key or value
    S = torch.arange(seq_len)[None,:] # 1, seq_len
    # max(-k,min(j-i,k)) - j is seq_len of key/value and i is seq_len of query
    rp = torch.clip(S-Q,-k,k) # q_seq_len, seq_len 
    # + k 
    out = rp + k
    return out

# batch, h와는 무관
class RelativePositionEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = args
        # 2k+1, d_k
        self.d_k = self.args.d_model // self.args.n_head
        self.emb = nn.Embedding(2*self.config.k+1,self.d_k)
        # nn.Parameter(torch.randn((2*self.config.k+1,self.config.d_k)))
        
    def forward(self, relative_position):
        # relative_position
        """
        relative position 
        shape : seq_len(query), seq_len(key or value)
        i = 3
        일 때 a_ij 연산
        seq len = 7이라고 가정
        -2, -1, 0, 1, 2, 3, 4 (j-i)
    
        """
        out = self.emb.forward(relative_position)
        
        return out

class token_embedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.token_embedding = nn.Embedding(args.n_vocab, args.d_model, padding_idx = args.padding_idx)
    def forward(self,input):
        # input : (bs, seq_len) -> (bs, seq_len, d_model)
        output = self.token_embedding(input) 
        return output

# model에선 ReLU모델을 활용했지만 , GeLU모델도 구현함. 
class gelu(nn.Module):
    def __init__(self):
        super().__init__()
    #gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.0044715x**3))
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.0044715*(x**3))))

# Multihead attention
# 1. Encoder 부 - self attention
# 2. Decoder 부 - masked self attention
# 3. Encoder - Decoder attention
# Mask만 subsequent mask + padding mask하면 됨

class relative_position_aware_multi_head_attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.d_k = self.args.d_model // self.args.n_head
        self.linear_Q = nn.Linear(args.d_model,args.d_model)
        self.linear_K = nn.Linear(args.d_model,args.d_model)
        self.linear_V = nn.Linear(args.d_model,args.d_model)
        
    def forward(self, query, key, value, mask = None, relative_position = False):
        # shape (bs, seq_len, d_model) -> (bs,seq_len,h,d_k)
        # mask shape - (bs, seq_len(q), seq_len(k))
        Q = self.linear_Q(query)
        seq_len_q = Q.size(1)
        Q = Q.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous() # bs,h,seq_len,d_k
        K = self.linear_K(key) 
        seq_len_k = K.size(1)
        K = K.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        V = self.linear_V(value) 
        V = V.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        # merge 용 
        # bs, seq_len_q, h, seq_len_k
        E_1 = torch.matmul(Q,K.transpose(2,3).contiguous())/math.sqrt(self.d_k)
        # ------------------------------------------------------------------------------------------------------------------ #
        # Relative position embedding이 들어감
        # seq_len_q, seq_len, d_k
        if relative_position:
            # Q shape : (bs, seq_len, h, d_k) -> (seq_len_q, bs*h, d_k)
            Q_ = Q.transpose(0,1).contiguous().reshape(seq_len_q,-1,self.args.d_k)
            # seq_len_q, bs*h,seq_len_k
            E_2 = torch.matmul(Q_,relative_position.transpose(1,2).contiguous())
            # seq_len_q, bs, h, seq_len_k -> bs, seq_len_q, h, seq_len_k
            E_2 = E_2.reshape(seq_len_q,-1,self.args.n_head,seq_len_k).transpose(0,1).contiguous()
            E_1 += E_2
        E_1 = E_1.transpose(1,2).contiguous()
        # padding mask
        if mask is not None:
            # mask shape : (bs, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1).expand(E_1.size()) # bs, h, seq_len(q), seq_len(k)
            E_1 = E_1.masked_fill(mask,-1e8)
        softmax = nn.Softmax(3).forward(E_1)
        output = torch.matmul(softmax,V) # bs, h, seq_len, d_k
        output = output.transpose(1,2).contiguous()
        output = output.reshape(-1,self.args.seq_len,self.args.d_model) # bs, seq_len, d_model
        return output
