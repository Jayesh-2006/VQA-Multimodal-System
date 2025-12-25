import torch
from torch import nn

class GatedFusion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc = nn.Linear(2*dim,dim)

    def forward(self,img_feat,text_feat):
        #img fea [B,512]
        #text fea [B,512]

        concat = torch.cat([img_feat,text_feat],dim=1) #[B,1024]
        gate = torch.sigmoid(self.fc(concat))

        fused = gate*img_feat + (1-gate)*text_feat
        fused =  nn.functional.normalize(fused, dim=1)
        return fused