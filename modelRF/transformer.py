import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.multihead_attention import MultiheadAttention
from torch import Tensor
import math


class TransformerCrossAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=8, attn_drop=0.1, linear_drop=0.1):
        super().__init__()
        assert input_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.dropout_module = nn.Dropout(linear_drop)
        self.self_attn = MultiheadAttention(input_dim, num_heads, dropout=attn_drop, self_attention=False)
        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.activation_fn = F.relu
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, input_dim)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, input_dim * 6),
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(self, q, k, v,
                encoder_padding_mask,
                attn_mask=None,
                cond=None):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4)

        residual = q

        # condition
        cond_flag = cond is not None
        if cond_flag:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.cond_mlp(cond).chunk(6, dim=-1)
            q = modulate(self.self_attn_layer_norm(q), shift_msa, scale_msa)
        else:
            q = self.self_attn_layer_norm(q)

        x, _ = self.self_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(gate_msa * x, residual) if cond_flag else self.residual_connection(x, residual)

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp) if cond_flag else self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(gate_mlp * x, residual) if cond_flag else self.residual_connection(x, residual)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
