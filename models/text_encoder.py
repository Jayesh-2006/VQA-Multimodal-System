from transformers import BertModel
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")


        #projection layer
        self.fc = nn.Linear(768,512)

        # parameter freezing
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self,input_ids,attention_mask):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        # cls extraction
        # cls_embedding = output.last_hidden_state[:,0,:] #[B,768]

        # cls_embedding = self.fc(cls_embedding) #[B,512]

        # return cls_embedding
        sequence = output.last_hidden_state
        sequence = self.fc(sequence) #[B,16,512]

        return sequence
