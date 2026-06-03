from torch import nn
from transformers import Swinv2Model


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinv2 = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")

        # freeze 
        for param in self.swinv2.parameters():
            param.requires_grad = False
       
    def forward(self,x):
        outputs = self.swinv2(pixel_values=x)
        features = outputs.last_hidden_state  #[B,36,1024]
        return features