from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.gate_fusion import GatedFusion
from models.attention import Attention
from torch import nn

class VQAModel(nn.Module):
    def __init__(self,num_answers):
        super().__init__()

        self.image_encoder = ImageEncoder()  #[B,49,2048]
        self.text_encoder = TextEncoder() #[B,512]

        self.img_attn = Attention(query_dim=512,context_dim=2048,embed_dim=512)
        self.text_attn = Attention(query_dim=2048,context_dim=512,embed_dim=512)

        self.img_projection = nn.Linear(2048,512) #[B,512]
        self.fusion = GatedFusion(dim = 512) #[B,512]

        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,num_answers)
        )

    def forward(self,images, input_ids, attention_mask):
        img_features_grid = self.image_encoder(images)  #[B,49,2048]
        text_features = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask) #[B,16,512]
        cls_embeddings = text_features[:,0,:]  #[B,512]

        img_attention = self.img_attn(context=img_features_grid, query=cls_embeddings) #[B,2048]
        text_attention = self.text_attn(context = text_features[:,1:,:], query =img_attention)

        img_attention = self.img_projection(img_attention)  #[B,512]

        fused = self.fusion(img_attention,text_attention)  #[B,512]

        logits = self.classifier(fused)

        return logits