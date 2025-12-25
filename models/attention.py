import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,img_dim,text_dim,hidden_dim):
        super().__init__()

        # projection for same scale
        self.img_proj = nn.Linear(img_dim, hidden_dim) # [B,2048]-->[B,512]
        self.text_proj = nn.Linear(text_dim, hidden_dim) # [B,512]-->[B,512]

        # scoring layer
        self.full_proj = nn.Linear(hidden_dim,1)

    def forward(self,img_features,text_features):
        # img_features: [B,49,2048]
        # text_features: [B,512]

         # [B,512] --> [B,49,512] 
        text_features = text_features.unsqueeze(1) # [B,1,512]
        text_features = text_features.expand(-1,img_features.size(1),-1) # [B,49,512]

        img_proj = self.img_proj(img_features) # [B,49,512]
        text_features = self.text_proj(text_features) # [B,49,512]

        combined_features = torch.tanh(img_proj + text_features) # [B,49,512]

        attention = self.full_proj(combined_features) # [B,49,1]

        alpha = F.softmax(attention, dim=1) # [B,49,1] # probabilities

        img_att = (img_features * alpha).sum(dim=1) # [B,2048] weighted sum  || [B,49,2048]*[B,49,1] --> [B,49,2048] --> sum dim1 --> [B,2048]
        return img_att
