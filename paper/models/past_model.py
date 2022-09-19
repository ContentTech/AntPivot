import copy
import math

import torch
# from models.pivot_transformer import PivotTransformer
import torch.nn.functional as F
from torch import nn

from models.dynamic_rnn import DynamicGRU
from models.transformer import TransEncoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PivotTransformer(nn.Module):
    def __init__(self, max_text_num, hidden_size, head_num, inner_layer, outer_layer):
        super().__init__()
        self.chunk_size = int(math.sqrt(max_text_num))
        self.chunk_num = math.ceil(max_text_num / self.chunk_size)
        self.hidden_size = hidden_size
        self.inner_layer = inner_layer
        self.outer_layer = outer_layer
        self.pivot = nn.Parameter(torch.randn(1, self.chunk_num, hidden_size), requires_grad=True)
        self.block_encoder = _get_clones(TransEncoder(hidden_size, head_num, inner_layer, hidden_size * 4), outer_layer)
        self.pivot_encoder = _get_clones(TransEncoder(hidden_size, head_num, inner_layer, hidden_size * 4), outer_layer)

    def outer_forward(self, pivot, mask, layer_idx):
        """
        :param pivot: FloatTensor, (B, N, D)
        :param mask: IntTensor, (B, N), 1: valid bit, 0: invalid bit
        :param layer_idx: Int, the index of encoder to make process
        :return: New pivot, in (B * N, 1, D)
        """
        result = self.pivot_encoder[layer_idx](pivot, mask == 0)
        return result.reshape(-1, 1, self.hidden_size)


    def inner_forward(self, chunk_feat, chunk_mask, layer_idx):
        """
        :param chunk_feat: FloatTensor, (B * N, S, D)
        :param chunk_mask: IntTensor, (B * N, S)
        :param layer_idx: Int, the index of encoder to make process
        :return: New features, FloatTensor, (B * N, S, D)
        """
        results = self.block_encoder[layer_idx](chunk_feat, chunk_mask == 0)
        pivots = results[:, 0].reshape(-1, self.chunk_num, chunk_feat.size(-1))  # (B, N, D)
        return pivots, results[:, 1:]

    def forward(self, features, masks):
        """
        :param features: FloatTensor, (B, L, D)
        :param masks: IntTensor, (B, L), 1: valid bit, 0: invalid bit
        :return:
        """
        batch_size, max_num, dim = features.size()
        chunk_feat = torch.stack(features.chunk(dim=1, chunks=self.chunk_size), dim=1)  # (B, N, S, D)
        chunk_mask = torch.stack(masks.chunk(dim=1, chunks=self.chunk_size), dim=1)   # (B, N, S)
        pivots = chunk_feat.mean(dim=2)  # (B, N, D)
        outer_mask = chunk_mask.sum(-1) != 0  # (B, N)
        chunk_feat = chunk_feat.reshape(-1, self.chunk_size, dim)  # (B * N, S, D)
        chunk_mask = chunk_mask.reshape(-1, self.chunk_size)  # (B * N, S)
        pivot_mask = torch.ones(batch_size * self.chunk_num, 1).to(masks.device)  # (B * N, 1)
        # PIVOT bit must be valid!  (Otherwise there will be NAN results)
        inner_mask = torch.cat((pivot_mask, chunk_mask), dim=1)  # (B * N, S + 1)
        for layer_idx in range(self.outer_layer):
            pivots = self.outer_forward(pivots, outer_mask, layer_idx)  # (B * N, 1, D)
            chunk_feat = torch.cat((pivots, chunk_feat), dim=1)  # (B * N, S + 1, D)
            pivots, chunk_feat = self.inner_forward(chunk_feat, inner_mask, layer_idx)  # (B, N, D)
        all_feat = chunk_feat.reshape(features.size())  # (B * N, S, D) -> (B, N, S, D) -> (B, L, D)
        return all_feat


class SentenceGate(nn.Module):
    def __init__(self, text_embd_dims, hidden_size, thresh, gate_method="soft"):
        super().__init__()
        self.gate_method = gate_method
        self.thresh = thresh
        self.input_linear = nn.Linear(text_embd_dims, hidden_size)
        self.input_rnn = DynamicGRU(input_size=text_embd_dims, hidden_size=hidden_size // 2,
                                    batch_first=True, bidirectional=True, num_layers=1)
        self.input_conv = nn.Conv1d(in_channels=text_embd_dims, out_channels=hidden_size,
                                    kernel_size=5, padding=2, stride=1)
        self.hard_gate = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.soft_gate = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, sentence, mask):
        """
            :param sentence: FloatTensor, in size (B, L, D_input)
            :param mask: IntTensor, in size (B, L), 1: valid, 0: invalid
            :return: new sentence embedding and new sentence mask
        """
        self_feat = self.input_linear(sentence)
        global_feat, _ = self.input_rnn(sentence, mask.sum(-1))
        local_feat = self.input_conv(sentence.transpose(-2, -1)).transpose(-2, -1)
        if self.gate_method == "hard":
            gate_feat = torch.cat((self_feat, local_feat, global_feat), dim=-1)
            conf_logit = self.hard_gate(gate_feat)  # (B, L, 1)
            new_mask = mask * (conf_logit.sigmoid().squeeze(-1) < self.thresh)
            return sentence, new_mask, conf_logit.squeeze(-1)
        elif self.gate_method == "soft":
            gate_feat = torch.cat((local_feat, global_feat), dim=-1)
            conf_logit = self.soft_gate(gate_feat)
            new_sentence = self_feat * conf_logit.sigmoid()
            return new_sentence, mask, conf_logit.squeeze(-1)



class MainModel(nn.Module):
    def __init__(self, text_embd_dims=768, spk_num=3, hidden_size=512, dropout=0.1, num_labels=4,
                 coarse_thres=0.1, head_num=4, inner_layer=3, outer_layer=3, max_text_num=900,
                 thresh=0.2, gate_method="soft", **kwargs):
        super().__init__()
        self.coarse_thres = coarse_thres
        self.gate_method = gate_method
        self.max_text_num = max_text_num  # Actually the max num is 900 - 15 = 885
        self.half_chunk = int(math.sqrt(max_text_num)) // 2
        self.spk_lut = nn.Embedding(spk_num, hidden_size)
        self.gate = SentenceGate(text_embd_dims, hidden_size, thresh=thresh, gate_method=gate_method)
        self.pivot_transformer = PivotTransformer(max_text_num, hidden_size, head_num, inner_layer, outer_layer)
        # self.post_pivot_transformer = PivotTransformer(max_text_num, hidden_size, head_num, inner_layer, outer_layer)
        self.split_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.start_linear, self.end_linear, self.inside_linear = [copy.deepcopy(self.split_linear)] * 3
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )
        self.mel_input = nn.Sequential(
            nn.Linear(8 * 128, 2 * hidden_size), nn.ReLU(),
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, sent_emb, spk, mask, mel):
        """
        :param sent_emb: FloatTensor, in size (B, L, D_input)
        :param spk: IntTensor, in size (B, L)
        # :param tags: IntTensor, in size (B, L), 0: other, 1: start, 2: middle, 3: end
        :param mask: IntTensor, in size (B, L), 1: valid, 0: invalid
        :return:
        """
        # First Step: Filter out useless information
        batch_size, max_text = spk.size()
        sent_emb, mask, conf_logit = self.gate(sent_emb, mask)
        # Second Step: Combine speaker id
        spk_emb = self.spk_lut(spk)
        mel_emb = self.mel_input(mel.reshape(batch_size, max_text, -1))  # (B, L, W * D_mel) -> (B, L, D)
        pre_input_emb, pre_mask = sent_emb + spk_emb, mask # + mel_emb, mask  # (B, L, D)
        post_input_emb = F.pad(pre_input_emb, [0, 0, self.half_chunk, 0])[:, :self.max_text_num]
        post_mask = F.pad(mask, [self.half_chunk, 0])[:, :self.max_text_num]
        # Second Step: Use Transformer to calculate
        pre_result = self.pivot_transformer(pre_input_emb, pre_mask)
        post_result = self.pivot_transformer(post_input_emb, post_mask)
        final = pre_result + F.pad(post_result[:, self.half_chunk:], [0, 0, 0, self.half_chunk])
        start_logit = self.start_linear(final).squeeze(-1)
        end_logit = self.end_linear(final).squeeze(-1)
        inside_logit = self.inside_linear(final).squeeze(-1)
        split_logit = self.split_linear(final).squeeze(-1)
        cls_logit = self.out_linear(final)
        return {
            "init_logit": conf_logit,
            "final_logit": conf_logit,
            # "start_logit": start_logit,
            # "end_logit": end_logit,
            # "inside_logit": inside_logit,
            # "cls_logit": cls_logit,
            "split_logit": split_logit
        }
