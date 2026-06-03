import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self,hidden_dim = 768, num_heads = 8):
        super().__init__()

        
        # Attn layers
        self.text_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.image_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # Attn norms
        self.text_norm1= nn.LayerNorm(hidden_dim)
        self.image_norm1 = nn.LayerNorm(hidden_dim)

        # Text FFN
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Image FFN
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # FFN Norms
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        self.image_norm2 = nn.LayerNorm(hidden_dim)


    def forward(self,image_features, text_features,attention_mask):
        
        text_padding_mask = ~attention_mask.bool()

        # Text -> Image
        t2i, _ = self.text_to_image(
            query=text_features,
            key=image_features,
            value=image_features
        )
        t2i = self.text_norm1(t2i + text_features)
        t2i = self.text_ffn(t2i) + t2i
        t2i = self.text_norm2(t2i)


        # Image -> Text
        i2t, _ = self.image_to_text(
            query=image_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_padding_mask
        )
        i2t = self.image_norm1(i2t + image_features)
        i2t = self.image_ffn(i2t) + i2t
        i2t = self.image_norm2(i2t)

        return t2i, i2t  #t2i = [B,24,768]  #i2t = [B,36,768]

