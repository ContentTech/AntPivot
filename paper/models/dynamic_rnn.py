import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_len):
        if self.batch_first:
            max_num_steps = x.size(1)
        else:
            max_num_steps = x.size(0)
        seq_len = None

        if seq_len is not None:
            sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
            _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
            if self.batch_first:
                sorted_x = x.index_select(0, sorted_idx)
            else:
                # print(sorted_idx)
                sorted_x = x.index_select(1, sorted_idx)

            packed_x = nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_seq_len.cpu().numpy(), batch_first=self.batch_first)

            self.gru.flatten_parameters()
            out, state = self.gru(packed_x)

            unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

            if self.batch_first:
                out = unpacked_x.index_select(0, original_idx)
                if out.shape[1] < max_num_steps:
                    out = F.pad(out, [0, 0, 0, max_num_steps - out.shape[1]])
            else:
                out = unpacked_x.index_select(1, original_idx)
        else:
            self.gru.flatten_parameters()
            out, state = self.gru(x)

        if self.gru.bidirectional:
            state = torch.cat((state[0], state[1]), dim=-1)
        return out, state


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=bias,
                            batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_len):
        self.lstm.flatten_parameters()
        if self.batch_first:
            max_num_steps = x.size(1)
        else:
            max_num_steps = x.size(0)

        if seq_len is not None:
            sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
            _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
            if self.batch_first:
                sorted_x = x.index_select(0, sorted_idx)
            else:
                # print(sorted_idx)
                sorted_x = x.index_select(1, sorted_idx)

            packed_x = nn.utils.rnn.pack_padded_sequence(
                sorted_x, sorted_seq_len.cpu().numpy(), batch_first=self.batch_first)

            out, state = self.lstm(packed_x)

            unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

            if self.batch_first:
                out = unpacked_x.index_select(0, original_idx)
                if out.shape[1] < max_num_steps:
                    out = F.pad(out, [0, 0, 0, max_num_steps - out.shape[1]])
            else:
                out = unpacked_x.index_select(1, original_idx)
        else:
            self.lstm.flatten_parameters()
            out, state = self.lstm(x)
            # if out.shape[0] < max_frame_num:
            #     out = F.pad(out, [0, 0, 0, 0, 0, max_frame_num - out.shape[0]])

        # state = state.transpose(0, 1).contiguous().view(out.size(0), -1)
        return out, state
