import math

import torch
from torch import nn
# from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.nn.modules.transformer import _get_clones

# def attention(query, key, value, mask=None, dropout=None):
#     "Compute 'Scaled Dot Product Attention'"
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) \
#              / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim = -1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn
#
#
# class MultiheadAttention(nn.Module):
#     def __init__(self, d_model, h, dropout=0.1):
#         "Take in model size and number of heads."
#         super(MultiheadAttention, self).__init__()
#         assert d_model % h == 0
#         # We assume d_v always equals d_k
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, query, key, value, key_padding_mask):
#         query = query.transpose(0, 1)
#         key = key.transpose(0, 1)
#         value = value.transpose(0, 1)
#         key_padding_mask = 1 - key_padding_mask.long()
#         "Implements Figure 2"
#         if key_padding_mask is not None:
#             # Same mask applied to all h heads.
#             key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.h, 1).unsqueeze(-1)
#         nbatches = query.size(0)
#
#         # 1) Do all the linear projections in batch from d_model => h x d_k
#         query, key, value = \
#             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#              for l, x in zip(self.linears, (query, key, value))]
#
#         # 2) Apply attention on all the projected vectors in batch.
#         x, self.attn = attention(query, key, value, mask=key_padding_mask,
#                                  dropout=self.dropout)
#
#         # 3) "Concat" using a view and apply a final linear.
#         x = x.transpose(1, 2).contiguous() \
#             .view(nbatches, -1, self.h * self.d_k)
#         final = self.linears[-1](x).transpose(0, 1)
#         return final, self.attn

class PositionEncoder(nn.Module):
    """Implement the PE function."""

    def __init__(self, input_dim, max_len=5000):
        super().__init__()

        # Compute the positional encodings once in log space.
        pos_encoding = torch.zeros(max_len, input_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., input_dim, 2) * -(math.log(10000.0) / input_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).clone().detach().requires_grad_(False)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, length):
        return self.pos_encoding[:, :length]


class ResidualBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_inputs, new_inputs):
        return raw_inputs + self.dropout(new_inputs)


class SkipNormBlock(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_inputs, new_inputs):
        return self.norm(raw_inputs + self.dropout(new_inputs))


class LayerScaleBlock(nn.Module):
    def __init__(self, model_dim, dropout, init_value=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(init_value * torch.ones(model_dim), requires_grad=True)

    def forward(self, raw_inputs, new_inputs):
        return raw_inputs + self.dropout(self.gamma * new_inputs)


class FeedForwardBlock(nn.Module):
    def __init__(self, model_dim, dim_feedforward, dropout, activation):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, model_dim)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.linear2(self.dropout(self.activation(self.linear1(inputs))))


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, head_num, dim_feedforward=2048,
                 dropout=0.1, activation="relu", need_pos=True, **kwargs):
        super().__init__()
        self.need_pos = need_pos
        self.self_attn = MultiheadAttention(model_dim, head_num)
        self.pre_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 2)
        self.post_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 2)
        self.layer_scale_block = nn.ModuleList([LayerScaleBlock(model_dim, dropout)] * 2)
        self.feed_forward = FeedForwardBlock(model_dim, dim_feedforward, dropout, activation)
        self.pos_encoder = PositionEncoder(model_dim)

    def forward(self, inputs, padding_mask=None, pos=None):
        """
        :param inputs: Tensor, (B, L, D), Default: Batch First
        :param padding_mask: ByteTensor, (B, L), NOTICE: Invalid bit is True
        :param pos: Tensor, (B, L, D)
        :return:
        """
        if self.need_pos:
            value = self.pre_norm[0](inputs)
            query = key = value + (pos if pos is not None else self.pos_encoder(value.size(1)))
        else:
            query = key = value = self.pre_norm[0](inputs)
        new_inputs, self_attn_weight = self.self_attn(query=query.transpose(0, 1),
                                                      key=key.transpose(0, 1),
                                                      value=value.transpose(0, 1),
                                                      key_padding_mask=padding_mask)
        inputs = self.layer_scale_block[0](inputs, self.post_norm[0](new_inputs.transpose(0, 1)))
        inputs = self.layer_scale_block[1](inputs, self.post_norm[1](self.feed_forward(self.pre_norm[1](inputs))))
        return inputs, self_attn_weight


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, head_num, dim_feedforward=2048,
                 dropout=0.1, activation="relu", need_pos=True, **kwargs):
        super().__init__()
        self.need_pos = need_pos
        self.self_attn = MultiheadAttention(model_dim, head_num, dropout)
        self.cross_attn = MultiheadAttention(model_dim, head_num, dropout)
        self.pre_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 3)
        self.post_norm = nn.ModuleList([nn.LayerNorm(model_dim)] * 3)
        self.layer_scale_block = nn.ModuleList([LayerScaleBlock(model_dim, dropout)] * 3)
        self.feed_forward = FeedForwardBlock(model_dim, dim_feedforward, dropout, activation)
        self.pos_encoder = PositionEncoder(model_dim)

    def forward(self, inputs, memory, input_padding_mask=None, memory_padding_mask=None, input_attn_mask=None,
                input_pos=None, memory_pos=None):
        """
        :param input_pos: Tensor or None, (B, S, D)
        :param memory_pos: Tensor or None, (B, T, D)
        :param inputs: Tensor, (B, S, D)
        :param memory: Tensor, (B, T, D)
        :param input_padding_mask:  ByteTensor, (B, S), Invalid bit is True
        :param memory_padding_mask: ByteTensor, (B, T), Invalid bit is True
        :param input_attn_mask: ByteTensor, (T, T), M(i, j) = True: j-th pos won't attend to i-th pos
        :return:
        """
        if self.need_pos:
            v_inputs = self.pre_norm[0](inputs)
            q_inputs = k_inputs = v_inputs + (input_pos if input_pos is not None else self.pos_encoder(inputs.size(1)))
        else:
            q_inputs = k_inputs = v_inputs = self.pre_norm[0](inputs)
        new_inputs, self_attn_weight = self.self_attn(query=q_inputs.transpose(0, 1),
                                                      key=k_inputs.transpose(0, 1),
                                                      value=v_inputs.transpose(0, 1),
                                                      key_padding_mask=input_padding_mask,
                                                      attn_mask=input_attn_mask)
        inputs = self.layer_scale_block[0](inputs, self.post_norm[0](new_inputs.transpose(0, 1)))
        new_inputs = self.pre_norm[1](inputs)
        if self.need_pos:
            q_inputs = new_inputs + (input_pos if input_pos is not None else self.pos_encoder(inputs.size(1)))
            k_memory = memory + (memory_pos if memory_pos is not None else self.pos_encoder(memory.size(1)))
        else:
            q_inputs, k_memory = new_inputs, memory
        new_inputs, cross_attn_weight = self.cross_attn(query=q_inputs.transpose(0, 1),
                                                        key=k_memory.transpose(0, 1),
                                                        value=memory.transpose(0, 1),
                                                        key_padding_mask=memory_padding_mask)
        inputs = self.layer_scale_block[1](inputs, self.post_norm[1](new_inputs.transpose(0, 1)))
        inputs = self.layer_scale_block[2](inputs, self.post_norm[2](self.feed_forward(self.pre_norm[2](inputs))))
        return inputs, self_attn_weight, cross_attn_weight


def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        This mask is consistent with the parameter definition for Transformer / Multi-head-Attention
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoderFrame(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderFrame, self).__init__()
        assert isinstance(encoder_layer, EncoderLayer), "Invalid Encoder Layer Type!"
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask=None, src_pos=None):
        output = src
        for layer in self.layers:
            output, _ = layer(output, src_key_padding_mask, src_pos)
        return output if self.norm is None else self.norm(output)


class TransformerDecoderFrame(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoderFrame, self).__init__()
        assert isinstance(decoder_layer, DecoderLayer), "Invalid Decoder Layer Type!"
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, tgt_pos=None, memory_pos=None):
        output = tgt
        for layer in self.layers:
            output, _, _ = layer(output, memory, input_attn_mask=tgt_mask, input_padding_mask=tgt_key_padding_mask,
                                 memory_padding_mask=memory_key_padding_mask, input_pos=tgt_pos, memory_pos=memory_pos)
        return output if self.norm is None else self.norm(output)


class TransEncoder(nn.Module):
    def __init__(self, model_dim, head_num, num_layers, dim_feedforward, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = EncoderLayer(model_dim, head_num, dim_feedforward, dropout, activation, need_pos=True)
        encoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoderFrame(encoder_layer, num_layers, encoder_norm)

    def forward(self, *inputs, **kwargs):
        return self.encoder(*inputs, **kwargs)


class TransDecoder(nn.Module):
    def __init__(self, model_dim, head_num, num_layers, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        decoder_layer = DecoderLayer(model_dim, head_num, dim_feedforward, dropout, activation, need_pos=True)
        decoder_norm = nn.LayerNorm(model_dim)
        self.decoder = TransformerDecoderFrame(decoder_layer, num_layers, decoder_norm)

    def forward(self, *inputs, **kwargs):
        return self.decoder(*inputs, **kwargs)


class Transformer(nn.Module):
    def __init__(self, model_dim, head_num, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, activation="relu"):
        super().__init__()
        encoder_layer = EncoderLayer(model_dim, head_num, dim_feedforward, dropout, activation, need_pos=True)
        decoder_layer = DecoderLayer(model_dim, head_num, dim_feedforward, dropout, activation, need_pos=True)
        encoder_norm = nn.LayerNorm(model_dim)
        decoder_norm = nn.LayerNorm(model_dim)
        self.encoder = TransformerEncoderFrame(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = TransformerDecoderFrame(decoder_layer, num_decoder_layers, decoder_norm)

    def encode(self, *inputs, **kwargs):
        return self.encoder(*inputs, **kwargs)

    def decode(self, tgt, encoding, src_padding_mask=None, tgt_padding_mask=None, src_pos=None, tgt_pos=None):
        tgt_length = tgt.size(1)
        return self.decoder(tgt, encoding, tgt_mask=generate_square_subsequent_mask(tgt_length).to(tgt.device),
                            tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=src_padding_mask,
                            tgt_pos=tgt_pos, memory_pos=src_pos)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, src_pos=None, tgt_pos=None):
        encoding = self.encode(src, src_padding_mask, src_pos)
        output = self.decode(tgt, encoding, src_pos=src_pos, tgt_pos=tgt_pos, src_padding_mask=src_padding_mask,
                             tgt_padding_mask=tgt_padding_mask)
        return output


if __name__ == "__main__":
    encoder = TransEncoder(8, 2, 1, 32)
    inputs = torch.zeros(2, 4, 8)
    mask = torch.ones(2, 4).bool()
    mask[:, 0] = False
    result = encoder(inputs, mask)
    print(result)
