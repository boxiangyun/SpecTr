# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch.nn.functional as F
from os.path import join as pjoin
import sys
#sys.path.insert(0,'./entmax')
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from scipy import ndimage

from entmax import EntmaxAlpha

from activations import sparsemax

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def swish(x):
    return x * torch.sigmoid(x)

def sharpen(x, T, eps=1e-6):
    temp = x**(1/T)
    return (temp+ eps) / (temp.sum(axis=-1, keepdims=True) + eps)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class Attention(nn.Module):
    def __init__(self,hidden_size, num_heads,attention_dropout_rate,
                 sparse_topk,use_entmax15,
                 vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
#        self.softmax = Softmax(dim=-1)
#        self.attn_fn = entmax15 if use_entmax15 else F.softmax
        self.use_entmax15 = use_entmax15
        if use_entmax15 == 'softmax':
            self.att_fn = F.softmax
        elif use_entmax15 == 'entmax_bisect':
            self.att_fn = EntmaxAlpha(1.33)
        elif use_entmax15 == 'sparsemax':
            self.att_fn = sparsemax
        elif use_entmax15 == 'adaptive_entmax':
            self.att_fn = EntmaxAlpha(self.num_attention_heads)
        else:
            raise ValueError("Oops! That was invalid attention function.Try again...")
        

        self.sparse_topk = sparse_topk
        
    def transpose_for_scores(self, x):
        
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if self.use_entmax15 == 'entmax_bisect' or 'adaptive_entmax':
            attention_probs = self.att_fn(attention_scores)
        else:
            attention_probs = self.att_fn(attention_scores,dim=-1)
#        mask_value = max_neg_value(attention_scores)
#        if self.sparse_topk is not None and self.sparse_topk < attention_scores.shape[-1]:
#            top, _ = attention_scores.topk(self.sparse_topk, dim = -1)
#            vk = top[..., -1].unsqueeze(-1).expand_as(attention_scores)
#            mask = attention_scores < vk
#            attention_scores.masked_fill_(mask, mask_value)
#            del mask
#        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, drop_out):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim,hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(drop_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size=784, seq_length=10, drop_out=0.):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_out)

        pe = torch.zeros(seq_length, hidden_size)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, hidden_size=768, seq_length=10, drop_out=0.):
        super(Embeddings, self).__init__()
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, hidden_size))
        self.dropout = Dropout(drop_out)

    def forward(self, x):
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size,mlp_dim,drop_out)
        self.attn = Attention(hidden_size, num_heads,attention_dropout_rate, sparse_topk=8, use_entmax15=use_entmax15,vis=vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    


class Transformer(nn.Module):
    def __init__(self,seq_length, num_layers, hidden_size, mlp_dim, num_heads, drop_out, 
                 attention_dropout_rate, pos_embedway, use_entmax15, vis):
        super(Transformer, self).__init__()
        self.vis = vis
        if pos_embedway == 'random':
            self.embeddings = Embeddings(hidden_size, seq_length, drop_out)
        elif pos_embedway == 'sincos':
            self.embeddings = PositionalEncoding(hidden_size, seq_length, drop_out)
        else:
            self.embeddings=None
            
        self.encoder = Encoder(num_layers, hidden_size, mlp_dim, num_heads, drop_out, attention_dropout_rate, use_entmax15, vis)

    def forward(self, input_ids):
        if self.embeddings is not None:
            embedding_output = self.embeddings(input_ids)
        else:
            embedding_output = input_ids
            
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class UTransformer(nn.Module):
    def __init__(self,input_dim=256, hidden_size=768, mlp_dim = 3072,
                 seq_length=10, num_layers=12, num_heads=8, drop_out=0.1, attention_dropout_rate=0.0, 
                 pos_embedway='random', vis=False):
        
        super(UTransformer, self).__init__()
        self.pos_embedway = pos_embedway
        self.input_head = Linear(input_dim, hidden_size)
        self.transformer = Transformer(seq_length, num_layers, hidden_size, mlp_dim, 
                                       num_heads, drop_out, attention_dropout_rate, pos_embedway, vis)
        self.head = Linear(hidden_size, input_dim)
        self.vis = vis
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_head.weight)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.input_head.bias, std=1e-6)
        nn.init.normal_(self.head.bias, std=1e-6)
        
    def forward(self, x):
        x = self.input_head(x)
        x, attn_weights = self.transformer(x)
        x = self.head(x)
        if self.vis:
            return x, attn_weights
        else:
            return x
        
    def load_from(self, pretrainedmodel_path):
#def load_transformer_pretrainedmodel(loaded_model, pretrainedmodel_path, pretrained_pos=False):

        weights = np.load(pretrainedmodel_path)
        with torch.no_grad():
#            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
#            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
#            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            if self.pos_embedway == 'random':
                posemb_new = self.transformer.embeddings.position_embeddings
                if posemb.size() == posemb_new.size():
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                else:
                    ntok_new = posemb_new.size(1)#10
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    
                    gs_old = len(posemb_grid)#197
                    gs_new = ntok_new
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
    #                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    
                    zoom = (gs_new / gs_old,  1)
                    
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    #                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                    self.transformer.embeddings.position_embeddings.copy_(np2th(posemb_grid))
            
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


if __name__ == '__main__':
    model = UTransformer()
    model.load_from(pretrainedmodel_path='./ViT-B_16.npz')
