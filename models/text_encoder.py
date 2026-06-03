from transformers import DebertaV2Model
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deberta  = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")

        #parameter freezing
        for param in self.deberta.parameters():
            param.requires_grad = False

    def forward(self,input_ids,attention_mask):
        output = self.deberta(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        sequence = output.last_hidden_state # [B, 24, 768]

        return sequence
