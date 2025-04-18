"""
Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import math


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, in_channels=128, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24, positions=None):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP
            dropout (float): Dropout rate
            T (int): Period for positional encoding
            len_max_seq (int, optional): Maximum sequence length for precomputed encodings
            positions (list, optional): Not used in init; positions passed per sample in forward
            d_model (int, optional): Dimension of the model (defaults to in_channels)
        """
        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.d_model = d_model if d_model is not None else in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.T = T
        self.len_max_seq = len_max_seq

        self.name = f'TAE_dk{self.d_k}_{self.n_head}Heads_{"|".join(map(str, n_neurons))}_T{self.T}_do{dropout}'
        if positions is not None:
            self.name += '_bespokePos'

        self.inlayernorm = nn.LayerNorm(self.d_model)
        self.outlayernorm = nn.LayerNorm(self.d_model)
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert n_neurons[0] == n_head * self.d_model
        assert n_neurons[-1] == self.d_model
        layers = []
        for i in range(len(n_neurons) - 1):
            layers.extend([
                nn.Linear(n_neurons[i], n_neurons[i + 1]),
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def get_sinusoid_encoding(self, positions, d_model):
        """Compute sinusoidal encodings for given positions."""
        def cal_angle(position, hid_idx):
            return position / np.power(self.T, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        if isinstance(positions, torch.Tensor):
            positions = positions.cpu().numpy()
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).to(self.inlayernorm.weight.device)

    def forward(self, x, positions=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            positions: List or tensor of temporal positions [batch_size, seq_len] (e.g., days)
        """
        sz_b, seq_len, d = x.shape
        x = self.inlayernorm(x)

        # Add positional encoding
        if positions is not None:
            # positions: [batch_size, seq_len]
            pos_enc = torch.stack([self.get_sinusoid_encoding(pos, self.d_model) for pos in positions])  # [sz_b, seq_len, d_model]
            x = x + pos_enc.to(x.device)
        else:
            # Default sequential positions
            src_pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(sz_b, seq_len)
            pos_enc = self.get_sinusoid_encoding(src_pos[0], self.d_model)
            x = x + pos_enc.unsqueeze(0).to(x.device)

        # Attention and MLP
        enc_output, attn = self.attention_heads(x, x, x)
        enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads
        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))
        return enc_output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k),
            nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(q).view(sz_b, seq_len, n_head, d_k)
        q = q.mean(dim=1).squeeze()  # MEAN query
        q = self.fc2(q.view(sz_b, n_head * d_k)).view(sz_b, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * sz_b, d_k)

        k = self.fc1_k(k).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)

        v = v.repeat(n_head, 1, 1)

        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, 1, d_in)
        output = output.squeeze(dim=2)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn