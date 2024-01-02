import torch.nn as nn
import torch
from torchinfo import summary
from collections import OrderedDict
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Callable, Optional
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter

class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(  
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x) 
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out



class SpatialTemporalLayer(nn.Module):
    def __init__(
        self, model_dim,st_emb_size,feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionScoreLayer(st_emb_size, num_heads, mask)
        self.ln1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv = nn.Linear(num_heads,1)
        self.dgcn = DGCN(model_dim, model_dim,2,st_emb_size)

    def forward(self, x,st_emb): ###X: B,T,N,D 
        # x: B,T,N,D
        residual = x
        st_emb = st_emb.unsqueeze(0)
        attn_adj = self.attn(st_emb, st_emb, st_emb).squeeze(0)  # (batch_size, ..., length, model_dim) 1,T,N,N,K
        attn_adj = self.conv(attn_adj).squeeze(-1)##T,N,N
        attn_adj = torch.softmax(attn_adj, dim=-1)
        st_emb = st_emb.squeeze(0)
        out = self.dgcn(x,attn_adj,st_emb)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        return out


class AttentionScoreLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
        attn_score = torch.stack(torch.split(attn_score, batch_size, dim=0),dim=-1)
        return attn_score
    
class DGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k,st_emb_size):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k, dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.lambda_fc =nn.Sequential( 
                OrderedDict([('lambda1', nn.Linear(st_emb_size, 32)),
                             ('lambda2', nn.ReLU()),
                             ('lambda3', nn.Linear(32, 1))]))
        self.linear = nn.Linear(dim_out,dim_out)
    def get_actual_lambda(self,lambda_):
        return 1 + torch.relu(lambda_)
        
    def forward(self, x,adj,st_emb):
        node_num = x.shape[2]
        lambda_ = self.lambda_fc(st_emb) #B,T,N,1
        lambda_ = self.get_actual_lambda(lambda_)
        supports1 = torch.eye(node_num).to(x.device)
        supports2 = adj  
        supports1 = ((2 * lambda_ - 2) / lambda_) * supports1 ##B,T,N,N
        supports2 = (2 / lambda_) * supports2 ###B,T,N,N    
        x_g1 = torch.einsum("tnm,btmc->btnc", supports1, x)
        x_g2 = torch.einsum("tnm,btmc->btnc", supports2, x) 
        x_g = torch.stack([x_g1,x_g2],dim=3)
        x_gconv = torch.einsum('btnki,kio->btno', x_g, self.weights) + self.bias  #b, N, dim_out
        return x_gconv

  

class DSTGTN(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        st_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + st_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.st_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(in_steps, num_nodes, st_embedding_dim))
        )

        self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
   

        self.temporal_layers = nn.ModuleList(
            [
                TemporalAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.st_layers= nn.ModuleList(
            [
                SpatialTemporalLayer(self.model_dim,self.st_embedding_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        x = self.input_proj(x) 
        features = [x]
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        features.append(tod_emb)
        dow_emb = self.dow_embedding(dow.long()) 
        features.append(dow_emb)
        st_emb = self.st_embedding.expand(size=(batch_size, *self.st_embedding.shape))
        features.append(st_emb)
        x = torch.cat(features, dim=-1)  

        for layer in self.temporal_layers:
            x = layer(x, dim=1)
        
        for layer in self.st_layers:
            x = layer(x,self.st_embedding)
  
        out = x.transpose(1, 2)  
        out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
        out = self.output_proj(out).view( batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)  
        return out
