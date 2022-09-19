from torch import nn
from transformers import TransfoXLModel, TransfoXLConfig, LongformerModel, LongformerConfig


class TransformerXL(nn.Module):
    def __init__(self, hidden_size, head_num, inner_layer, outer_layer, chunk_size, dropout):
        super().__init__()
        self.core = TransfoXLModel(TransfoXLConfig(d_model=hidden_size, n_head=head_num,
                                                   n_layer=inner_layer * outer_layer,
                                                   d_inner=hidden_size * 4,
                                                   mem_len=chunk_size,
                                                   dropout=dropout, dropatt=dropout))
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, embedding, mask):
        all_embedding = self.core(inputs_embeds=embedding, output_hidden_states=True,
                                  return_dict=True).last_hidden_state
        all_gate = self.gate(all_embedding).squeeze(-1)
        return all_embedding, all_gate


class Longformer(nn.Module):
    def __init__(self, hidden_size, head_num, inner_layer, outer_layer, chunk_size, dropout):
        super().__init__()
        self.core = LongformerModel(LongformerConfig(hidden_size=hidden_size,
                                                     num_attention_heads=head_num,
                                                     num_hidden_layers=inner_layer * outer_layer,
                                                     intermediate_size=hidden_size * 4,
                                                     attention_window=chunk_size,
                                                     hidden_dropout_prob=dropout,
                                                     attention_probs_dropout_prob=dropout,
                                                     max_position_embeddings=1024))
        self.gate = nn.Linear(hidden_size, 1)

    def forward(self, embedding, mask):
        all_embedding = self.core(inputs_embeds=embedding, output_hidden_states=True,
                                  return_dict=True).last_hidden_state
        all_gate = self.gate(all_embedding).squeeze(-1)
        return all_embedding, all_gate
