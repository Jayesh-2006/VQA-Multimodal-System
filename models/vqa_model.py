import torch
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.gate_fusion import GatedFusion
from torch import nn

class VQAModel(nn.Module):
    def __init__(self,num_answers):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.fusion = GatedFusion(dim = 512)

        self.classifier = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,num_answers)
        )

    def forward(self,images, input_ids, attention_mask):
        img_features = self.image_encoder(images) #[B,512]
        text_features = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask) #[B,512]

        fused = self.fusion(img_features,text_features)  #[B,512]
        fused = nn.functional.normalize(fused, dim=1)


        logits = self.classifier(fused)

        return logits