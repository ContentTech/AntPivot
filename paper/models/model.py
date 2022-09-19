import copy

import torch
from torch import nn
from transformers import TransfoXLModel, TransfoXLConfig

from models.dynamic_rnn import DynamicGRU
from models.gate import InitialGate
from labml_helpers.metrics.simple_state import SimpleStateModule
from models.metrics import max_min_norm
from models.pivot_transformer import PivotTransformer, VanillaTransformer, VanillaGRU
from models.other_transformer import TransformerXL, Longformer
from utils.helper import right_shift, fold, left_shift, masked_operation


class MainModel(nn.Module):
    def __init__(self, text_embd_dims, spk_num, hidden_size, dropout, num_labels,
                 head_num, inner_layer, outer_layer, max_text_num, mel_width,
                 mel_dim, chunk_size, chunk_num, gate_bias, use_gate, core, use_shift,
                 use_spk, use_mel, use_pivot, use_init, use_final, **kwargs):
        super().__init__()
        self.use_gate = use_gate
        self.core = core
        self.use_spk = use_spk
        self.use_mel = use_mel
        self.use_init = use_init
        self.use_final = use_final
        self.use_shift = use_shift
        self.gate_bias = gate_bias
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.max_text_num = max_text_num  # Actually the max num is 900 - 15 = 885
        self.spk_lut = nn.Embedding(spk_num, hidden_size)
        self.sent_input = nn.Linear(text_embd_dims, hidden_size)
        self.input_gate = InitialGate(hidden_size, dropout)
        self.transformer_xl = TransformerXL(hidden_size, head_num, inner_layer, outer_layer, chunk_size, dropout)
        self.longformer = Longformer(hidden_size, head_num, inner_layer, outer_layer, chunk_size, dropout)
        self.pivot_transformer = PivotTransformer(max_text_num, hidden_size, head_num, inner_layer,
                                                  outer_layer, dropout, use_gate, gate_bias, use_pivot)
        self.base_transformer = VanillaTransformer(hidden_size, head_num, inner_layer, outer_layer, dropout)
        self.base_gru = VanillaGRU(hidden_size, inner_layer, outer_layer, dropout)
        self.mel_input = nn.Sequential(
            nn.Linear(mel_width * mel_dim, 2 * hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        final_size = 2 * hidden_size if (self.core == "Pivot" and self.use_shift) else hidden_size
        self.split_linear = nn.Sequential(
            nn.Linear(final_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        # self.start_linear, self.end_linear, self.inside_linear = [copy.deepcopy(self.split_linear)] * 3
        self.out_linear = nn.Sequential(
            nn.Linear(final_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 4)
        )

    def align_op(self, pre_input, post_input, op="cat"):
        if op == "cat":
            return torch.cat((pre_input, left_shift(post_input, dim=1, length=self.chunk_size // 2)), dim=-1)
        elif op == "add":
            return pre_input + left_shift(post_input, dim=1, length=self.chunk_size // 2)
        elif op == "avg":
            return (pre_input + left_shift(post_input, dim=1, length=self.chunk_size // 2)) / 2.0
        else:
            raise NotImplementedError

    def transformer_model(self, embedding, init_logit, mask):
        if self.use_shift:
            # pre_half, post_half = embedding.chunk(dim=-1, chunks=2)
            pre_pivot, pre_block, pre_gate = self.pivot_transformer(embedding, init_logit, mask, shift=False)
            post_pivot, post_block, post_gate = self.pivot_transformer(embedding, init_logit, mask, shift=True)
            final_feature = self.align_op(pre_block, post_block, op="cat")
            final_gate = self.align_op(pre_gate, post_gate, op="avg")
        else:
            _, final_feature, final_gate = self.pivot_transformer(embedding, init_logit, mask, shift=False)
        return final_feature, final_gate

    def forward(self, sent_emb, spk, mask, mel):
        """
        :param sent_emb: FloatTensor, in size (B, L, D_input)
        :param spk: IntTensor, in size (B, L)
        # :param tags: IntTensor, in size (B, L), 0: other, 1: start, 2: middle, 3: end
        :param mask: IntTensor, in size (B, L), 1: valid, 0: invalid
        :param mel: FloatTensor, in size (B, L, W, D_mel)
        :return:
        """
        batch_size, max_text = spk.size()
        sent_emb, spk_emb = self.sent_input(sent_emb), self.spk_lut(spk)
        mel_emb = self.mel_input(mel.reshape(batch_size, max_text, -1))  # (B, L, W * D_mel) -> (B, L, D)
        overall_emb = sent_emb
        if self.use_spk:
            overall_emb = overall_emb + spk_emb
        if self.use_mel:
            overall_emb = overall_emb + mel_emb
        overall_emb, input_gate = self.input_gate(overall_emb, mask)
        if self.core == "Pivot":
            final_feature, final_gate = self.transformer_model(overall_emb, input_gate, mask)
        elif self.core == "GRU":
            final_feature, final_gate = self.base_gru(overall_emb, mask)
        elif self.core == "Transformer":
            final_feature, final_gate = self.base_transformer(overall_emb, mask)
        elif self.core == "TransformerXL":
            final_feature, final_gate = self.transformer_xl(overall_emb, mask)
        elif self.core == "Longformer":
            final_feature, final_gate = self.longformer(overall_emb, mask)
        else:
            raise NotImplementedError
        # pivot_gates = torch.stack((pre_pivot_gate, post_pivot_gate), dim=1).squeeze(-1)
        # block_gates = torch.stack((pre_block_gate, post_block_gate), dim=1).squeeze(-1)
        split_logit = self.split_linear(final_feature).squeeze(-1)
        out_logit = self.out_linear(final_feature).squeeze(-1)
        if not self.use_init:
            input_gate = None
        if not self.use_final:
            input_gate = None
        return {
            "split_logit": split_logit,   # (B, L)
            # "start_logit": start_logit,
            # "end_logit": end_logit,
            "out_logit": out_logit,
            "init_logit": input_gate,    # (B, L)
            "final_logit": final_gate,   # (B, L)
        }


