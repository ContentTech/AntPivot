import torch
from torch import nn

from models.dynamic_rnn import DynamicGRU
from utils.helper import masked_operation


class InitialGate(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.input_rnn = DynamicGRU(input_size=hidden_size, hidden_size=hidden_size // 2,
                                    batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=15, padding=7, stride=1, groups=hidden_size, bias=True),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )


    def forward(self, sentence, mask):
        """
            :param sentence: FloatTensor, in size (B, L, D)
            :param mask: IntTensor, in size (B, L), 1: valid, 0: invalid
            :return: new sentence embedding and new sentence mask
        """
        sent_emb, all_emb = self.input_rnn(sentence, mask.sum(-1))
        global_feat = all_emb.unsqueeze(1).expand_as(sent_emb)
        local_feat = self.input_conv(sentence.transpose(-2, -1)).transpose(-2, -1)
        gate_feat = torch.cat((sentence, local_feat, global_feat), dim=-1)
        conf_logit = self.gate(gate_feat)
        return sent_emb, conf_logit.squeeze(-1)


class VanillaGate(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=15, padding=7, stride=1, groups=hidden_size, bias=True),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, sentence, mask):
        global_feat = masked_operation(sentence, mask, dim=1, operation="mean").unsqueeze(1).expand_as(sentence)
        local_feat = self.input_conv(sentence.transpose(-2, -1)).transpose(-2, -1)
        gate_feat = torch.cat((sentence, local_feat, global_feat), dim=-1)
        conf_logit = self.gate(gate_feat)
        return conf_logit.squeeze(-1)


class InternalGate(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size, hidden_size), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, block, pivot, block_mask):
        """
        :param block: FloatTensor, in size (B * N, S, D)
        :param pivot: FloatTensor, in size (B, N, D)
        :param block_mask: IntTensor, in (B * N, S)
        :return: in size (B, N * S)
        """
        batch_size, chunk_num, dim = pivot.size()
        chunk_size = block.size(-2)
        block_, block_mask_ = block.reshape(batch_size, -1, dim), block_mask.reshape(batch_size, -1)
        local_feat = pivot.unsqueeze(2).repeat(1, 1, chunk_size, 1).reshape(block_.size())
        global_feat = masked_operation(block_, block_mask_, dim=1, operation="mean").unsqueeze(1).expand_as(block_)
        gate_feat = torch.cat((block_, local_feat, global_feat), dim=-1)
        conf_logit = self.gate(gate_feat)
        return conf_logit.squeeze(-1)
