import copy
import math

import torch
from torch import nn

from models.dynamic_rnn import DynamicGRU
from models.gate import InternalGate, InitialGate, VanillaGate
from models.metrics import max_min_norm
from models.transformer import TransEncoder, TransDecoder
from utils.helper import right_shift, fold, masked_operation


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PivotFormer(nn.Module):
    def __init__(self, hidden_size, chunk_size):
        super().__init__()
        self.core = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=chunk_size, stride=chunk_size,
                      padding=0, groups=hidden_size, bias=True),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        )

    def forward(self, inputs):
        # inputs: (B * N, S, D)
        batch_size, dim = inputs.size(0), inputs.size(-1)
        flatten_inputs = inputs.flatten(0, 1).transpose(-2, -1)  # (B * N, D, S)
        return self.core(flatten_inputs).reshape(batch_size, -1, dim)
        # (B * N, D, 1) -> (B, N, D)

class PivotTransformer(nn.Module):
    def __init__(self, max_text_num, hidden_size, head_num, inner_layer, outer_layer, dropout, use_gate, gate_bias, use_pivot):
        super().__init__()
        self.use_pivot = use_pivot
        self.use_gate = use_gate
        self.gate_bias = gate_bias
        self.chunk_size = int(math.sqrt(max_text_num))
        self.chunk_num = math.ceil(max_text_num / self.chunk_size)
        self.hidden_size = hidden_size
        self.inner_layer = inner_layer
        self.outer_layer = outer_layer
        self.block_encoder = _get_clones(TransEncoder(hidden_size, head_num, inner_layer, hidden_size * 4,
                                                      dropout, activation='gelu'), outer_layer)
        self.internal_gate = InternalGate(hidden_size, dropout)
        self.pivot_encoder = _get_clones(TransEncoder(hidden_size, head_num, inner_layer, hidden_size * 4,
                                                      dropout, activation='gelu'), outer_layer)

    def outer_forward(self, pivot, mask, layer_idx):
        """
        :param pivot: FloatTensor, (B, N, D)
        :param mask: IntTensor, (B, N), 1: valid bit, 0: invalid bit
        :param layer_idx: Int, the index of encoder to make process
        :return: New pivot, in (B * N, 1, D)
        """
        new_pivot = self.pivot_encoder[layer_idx](pivot, mask == 0)
        dense_pivot = new_pivot.reshape(-1, 1, self.hidden_size)
        return dense_pivot

    def inner_forward(self, block_feat, block_mask, layer_idx):
        """
        :param block_feat: FloatTensor, (B * N, S + 1, D)
        :param block_mask: IntTensor, (B * N, S + 1)
        :param layer_idx: Int, the index of encoder to make process
        :return: New pivots, FloatTensor, (B, N, D)
        :return: New features, FloatTensor, (B * N, S, D)
        """
        results = self.block_encoder[layer_idx](block_feat, block_mask == 0)
        pivots = results[:, 0].reshape(-1, self.chunk_num, block_feat.size(-1))  # (B, N, D)
        return pivots, results[:, 1:]

    def process_input(self, embedding, init_logit, mask, shift):
        if shift:
            embedding = right_shift(embedding, dim=1, length=self.chunk_size // 2)
            mask = right_shift(mask, dim=1, length=self.chunk_size // 2)
            init_logit = right_shift(init_logit, dim=1, length=self.chunk_size // 2)
        blocks = fold(embedding, split_dim=1, size=self.chunk_size, stack_dim=1)  # (B, N, S, D)
        block_mask = fold(mask, split_dim=1, size=self.chunk_size, stack_dim=1)  # (B, N, S)
        init_logit = fold(init_logit, split_dim=1, size=self.chunk_size, stack_dim=1)  # (B, N, S)
        batch_size, chunk_num = block_mask.size(0), block_mask.size(1)
        pivot_inj_mask = torch.ones(batch_size, chunk_num, 1).to(blocks.device)  # (B, N, 1)
        init_weight = init_logit.masked_fill(block_mask == 0, -1e4).softmax(dim=-1)  # softmax on the S dim
        pivots = masked_operation(blocks * init_weight.unsqueeze(-1), block_mask, dim=2, operation="sum")  # (B, N, D)
        pivot_mask = block_mask.sum(-1) != 0  # (B, N)
        block_mask = torch.cat((pivot_inj_mask, block_mask), dim=-1)  # (B, N, S + 1)
        return blocks, block_mask, pivots, pivot_mask

    def calculate(self, blocks, block_mask, pivots, pivot_mask):
        """
        :param blocks: FloatTensor, (B, N, S, D)
        :param block_mask: IntTensor, (B, N, S + 1), 1: valid bit, 0: invalid bit
        :param pivots: FloatTensor, (B, N, D)
        :param pivot_mask: IntTensor, (B, N), 1: valid bit, 0: invalid bit
        :return:
        """
        batch_size, chunk_num, chunk_size, dim = blocks.size()
        all_gates = []
        block_feat = blocks.reshape(-1, self.chunk_size, dim)  # (B * N, S, D)
        block_mask = block_mask.reshape(-1, self.chunk_size + 1)  # (B * N, S + 1)
        inner_mask = block_mask[:, 1:]  # (B * N, S)
        for layer_idx in range(self.outer_layer):
            pivots = self.outer_forward(pivots, pivot_mask, layer_idx)  # (B * N, 1, D)
            if self.use_pivot:
                block_feat = torch.cat((pivots, block_feat), dim=1)  # (B * N, S + 1, D)
            else:
                dummy_pivot = torch.zeros(block_feat.size(0), 1, block_feat.size(-1)).to(block_feat.device)
                block_feat = torch.cat((dummy_pivot, block_feat), dim=1)
            pivots, block_feat = self.inner_forward(block_feat, block_mask, layer_idx) # (B, N, D) & (B * N, S, D)
            block_gate = self.internal_gate(block_feat, pivots, inner_mask)
            all_gates.append(block_gate)  # (B, L)
            block_weight = block_gate.reshape(-1, chunk_size).masked_fill(inner_mask == 0, -1e4).softmax(dim=1)
            # (B, L) -> (B * N, S)
            pivots = pivots + masked_operation(block_feat * block_weight.unsqueeze(-1),
                                               inner_mask, dim=1, operation="sum").reshape(pivots.size())
            # right: (B * N, S, D) & (B * N, S) -> (B * N, D)
        final_block = block_feat.reshape(batch_size, -1, dim)
        final_gate = torch.stack(all_gates, dim=1).mean(1)
        return pivots, final_block, final_gate  # (B, N, D) & (B, L, D) & (B, L)

    def forward(self, embedding, init_logit, mask, shift):
        inputs = self.process_input(embedding, init_logit, mask, shift)
        return self.calculate(*inputs)


class VanillaGRU(nn.Module):
    def __init__(self, hidden_size, inner_layer, outer_layer, dropout):
        super().__init__()
        self.encoder = DynamicGRU(hidden_size, hidden_size // 2, num_layers=inner_layer * outer_layer, dropout=dropout,
                                  batch_first=True, bidirectional=True)
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, embedding, mask):
        length = mask.sum(-1)
        all_embedding, _ = self.encoder(embedding, length)
        all_gate = self.gate(all_embedding).squeeze(-1)
        return all_embedding, all_gate


# class VanillaTransformer(nn.Module):
#     def __init__(self, hidden_size, head_num, inner_layer, outer_layer, dropout):
#         super().__init__()
#         self.encoder = TransEncoder(hidden_size, head_num, inner_layer * outer_layer, hidden_size * 4, dropout, activation='gelu')
#         self.gate = nn.Linear(hidden_size, 1)
#
#     def forward(self, embedding, mask):
#         all_embedding = self.encoder(embedding, mask == 0)
#         all_gate = self.gate(all_embedding).squeeze(-1)
#         return all_embedding, all_gate


class VanillaTransformer(nn.Module):
    def __init__(self, hidden_size, head_num, inner_layer, outer_layer, dropout):
        super().__init__()
        self.layer_num = inner_layer * outer_layer
        self.encoder = _get_clones(TransEncoder(hidden_size, head_num, 1, hidden_size * 4, dropout, activation='gelu'),
                                   self.layer_num)
        self.gate = _get_clones(VanillaGate(hidden_size, dropout), self.layer_num)

    def forward(self, embedding, mask):
        all_gate = []
        for layer_idx in range(self.layer_num):
            embedding = self.encoder[layer_idx](embedding, mask == 0)
            all_gate.append(self.gate[layer_idx](embedding, mask))
        final_gate = torch.stack(all_gate, dim=0).mean(0)
        return embedding, final_gate



