# Copyright (c) 2025 Jin Cao
# SPDX-License-Identifier: MIT
#
# The Transformer Model.
#
# Author: Jin Cao <aihalop@gmail.com>


import torch
import torch.nn.functional as F
import math
import random
from data import Multi30k, de_tokenize


device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx, max_token_length):
        super().__init__()
        self.input_embedding = \
            torch.nn.Embedding(num_embeddings, embedding_dim,
                               padding_idx=padding_idx)

        # As mentioned in the paper,
        #
        # "...We also experimented with using learned positional
        # embeddings instead, and found that the two versions produced
        # nearly identical results...",
        #
        # we use learned positional embeddings here.
        self.position_encoding = \
            torch.nn.Embedding(max_token_length, embedding_dim)

    def forward(self, x):
        positions = torch.arange(0, x.shape[1]).repeat(x.shape[0], 1).to(device)
        return self.position_encoding(positions) + self.input_embedding(x)


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4 * embedding_dim),
            torch.nn.ReLU(), # TODO(Jin Cao): try Tanh
            torch.nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.f(x)


def ScaledDotProductAttention(Q, K, V, mask):
    dk = Q.size(-1)
    mask = mask.unsqueeze(-3)
    return F.softmax(
        (Q @ K.transpose(-2, -1)).masked_fill(mask, -1e10) \
        / math.sqrt(dk), dim=-1
    ) @ V


def Concat(attentions, batch, len_query):
    return attentions.transpose(1, 2).contiguous().view(batch, len_query, -1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dk, dv, num_heads):
        super().__init__()
        self._dk = dk
        self._dv = dv
        self._num_heads = num_heads

    def forward(self, Q, K, V, mask=None):
        batch = Q.size(0)
        len_query = Q.size(1)
        len_key = K.size(1)
        len_value = V.size(1)

        # Seperate projected Q, K, V into h heads.
        # shape of Q, K: (batch, head, token, dv)
        # shape of V:    (batch, head, token, dk)
        Q = Q.view(batch, len_query, self._num_heads, self._dk).transpose(1, 2)
        K = K.view(batch, len_key,   self._num_heads, self._dk).transpose(1, 2)
        V = V.view(batch, len_value, self._num_heads, self._dv).transpose(1, 2)

        if mask is None:
            mask = (torch.zeros(Q.shape[-2], K.shape[-2]) == 1).to(device)

        return Concat(
            ScaledDotProductAttention(Q, K, V, mask),
            batch, len_query
        )


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_dim, dk, dv, num_heads):
        super().__init__()
        self.projector_q = torch.nn.Linear(embedding_dim, dk * num_heads)
        self.projector_k = torch.nn.Linear(embedding_dim, dk * num_heads)
        self.projector_v = torch.nn.Linear(embedding_dim, dv * num_heads)
        self.attention = MultiHeadAttention(dk, dv, num_heads)

    def forward(self, x, mask=None):
        return self.attention(
            self.projector_q(x), self.projector_k(x), self.projector_v(x),
            mask
        )

class EncoderDecoderAttention(torch.nn.Module):
    def __init__(self, embedding_dim, dk, dv, num_heads):
        super().__init__()
        self.projector_q = torch.nn.Linear(embedding_dim, dk * num_heads)
        self.projector_k = torch.nn.Linear(embedding_dim, dk * num_heads)
        self.projector_v = torch.nn.Linear(embedding_dim, dv * num_heads)
        self.attenion = MultiHeadAttention(dk, dv, num_heads)

    def forward(self, y, x):
        return self.attenion(
            self.projector_q(y),
            self.projector_k(x),
            self.projector_v(x)
        )


class EncoderBlock(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dk, dv, padding_idx,
                 max_token_length, num_heads):
        super().__init__()
        self._padding_idx = padding_idx
        self._embedding_dim = embedding_dim
        self.self_attention = SelfAttention(embedding_dim, dk, dv, num_heads)
        self.feed_forward = FeedForward(embedding_dim)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, padding_mask=None):
        attention = F.layer_norm(
            x + self.dropout(self.self_attention(x, padding_mask)),
            x.shape
        )
        output = F.layer_norm(
            attention + self.feed_forward(attention), attention.shape)
        return output


class Encoder(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings, embedding_dim, dk, dv,
                 padding_idx, max_token_length, num_heads):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [EncoderBlock(num_embeddings, embedding_dim, dk, dv, padding_idx,
                          max_token_length, num_heads)
             for i in range(num_layers)]
        )

    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dk, dv, num_heads):
        super().__init__()
        self.self_attention = SelfAttention(embedding_dim, dk, dv, num_heads)
        self.encoder_decoder_attention = \
            EncoderDecoderAttention(embedding_dim, dk, dv, num_heads)
        self.feed_forward = FeedForward(embedding_dim)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, y, x):
        # mask:
        # [* 0 0]
        # [* * 0]
        # [* * *]
        mask = (torch.tril(torch.ones(y.shape[-2], y.shape[-2])) == 0).to(device)
        self_attention = F.layer_norm(y + self.self_attention(y, mask), y.shape)
        encode_decode_attention = F.layer_norm(
            self_attention + self.dropout(
                self.encoder_decoder_attention(self_attention, x)),
            self_attention.shape
        )
        output = F.layer_norm(
            encode_decode_attention + self.feed_forward(encode_decode_attention),
            encode_decode_attention.shape
        )

        return output


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, num_embeddings,
                 embedding_dim, dk, dv, num_heads):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [DecoderBlock(num_embeddings, embedding_dim, dk, dv, num_heads)
             for i in range(num_layers)]
        )

    def forward(self, y, x):
        for block in self.blocks:
            y = block(y, x)

        return y


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, num_input_embeddings, num_output_embeddings,
                 embedding_dim, padding_idx, max_token_length, num_heads, en_vocab_size):
        super().__init__()
        self._padding_idx = padding_idx
        self._embedding_dim = embedding_dim
        self._num_heads = num_heads
        dk = int(embedding_dim / num_heads)
        dv = int(embedding_dim / num_heads)
        # Since the input and output vocabularies are different, we
        # have to use two embeddings respectively.
        self.input_embedding = Embedding(
            num_input_embeddings, embedding_dim, padding_idx, max_token_length)
        self.output_embedding = Embedding(
            num_output_embeddings, embedding_dim, padding_idx, max_token_length)

        self.encoder = Encoder(num_layers, num_input_embeddings, embedding_dim,
                               dk, dv, padding_idx, max_token_length, num_heads)

        self.decoder = Decoder(num_layers, num_output_embeddings,
                               embedding_dim, dk, dv, num_heads)
        self.linear = torch.nn.Linear(embedding_dim, en_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


    def forward(self, x, y):
        '''x, input; y, output'''
        padding_mask = (x == self._padding_idx).unsqueeze(-2)
        x = F.dropout(self.input_embedding(x), p=0.1)
        y = F.dropout(self.output_embedding(y), p=0.1)
        encode = self.encoder(x, padding_mask)
        output = self.decoder(y, encode)
        return self.linear(output)
