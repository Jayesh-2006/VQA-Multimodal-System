import torch
from torch import nn

class FusionEncoder(nn.Module):
    def __init__(self,hidden_dim=768,num_heads=12,num_layers=4):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.img_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.txt_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.query_token = nn.Parameter(torch.randn(1, 8, hidden_dim))

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self,img_feat,text_feat,attention_mask):

        img_feat = img_feat + self.img_type_embed  # [B, 36, 768]
        text_feat = text_feat + self.txt_type_embed  # [B, 24, 768]

        query_tokens = self.query_token.expand(img_feat.size(0), 8, -1)  # [B, 8, 768]

        img_mask = torch.ones(img_feat.size(0), img_feat.size(1),device=img_feat.device)  # [B, 36]
        query_mask = torch.ones(img_feat.size(0), 8,device=img_feat.device) # [B, 8]

        combined_mask = torch.cat([query_mask, img_mask, attention_mask], dim=1)  # [B, 8+36+24]
        combined_mask = ~combined_mask.bool()  # Convert to boolean mask

        fusion_input = torch.cat([query_tokens,img_feat, text_feat], dim=1)  # [B, 8+36+24, 768]

        fused_output = self.encoder(fusion_input, src_key_padding_mask=combined_mask)  # [B, 68, 768]
        return fused_output