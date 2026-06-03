import torch

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion import FusionEncoder
from models.attention import CrossAttention
from torch import nn

class VQAModel(nn.Module):
    def __init__(self,num_answers):
        super().__init__()

        self.image_encoder = ImageEncoder()  ##[B,36,1024]
        self.text_encoder = TextEncoder() #[B,24, 768]

        

        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(hidden_dim=768, num_heads=8),
            CrossAttention(hidden_dim=768, num_heads=8),
            CrossAttention(hidden_dim=768, num_heads=8)
        ])

        self.image_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
        )

        self.fusion_encoder = FusionEncoder(hidden_dim=768, num_heads=12, num_layers=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_answers)
        )

        self.query_pool = nn.Linear(768, 1)
        

    def forward(self,images, input_ids, attention_mask):
        img_features = self.image_encoder(images)  #[B,36,1024]
        text_features = self.text_encoder(input_ids, attention_mask) #[B,24,768]
    
        img_features = self.image_proj(img_features)

        #t2i = [B,24,768]  #i2t = [B,36,768]

        for cross_attn in self.cross_attn_layers:
            text_features, img_features = cross_attn(img_features, text_features, attention_mask)



        fused_output = self.fusion_encoder(img_features, text_features,attention_mask)  #[B, 68, 768]

        query_tokens = fused_output[:, :8, :]  #[B, 8, 768]
        scores = self.query_pool(query_tokens)  #[B, 8,1]
        weights = torch.softmax(scores, dim=1)  #[B, 8,1]

        query_tokens = (query_tokens * weights).sum(dim=1)  #[B, 768]


        logits = self.classifier(query_tokens)  #[B, num_answers]

        return logits