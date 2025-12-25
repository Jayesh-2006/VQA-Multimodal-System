from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


        #freeze all resnet parameters layer 4
        for name,param in self.resnet.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False
        
        #remove last 2 layers of resnet(ie fc and avg pooling)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        
       
    def forward(self,x):
        features = self.resnet(x)  # [B, 2048, 7, 7]
        features = features.permute(0,2,3,1) # [B, 7, 7, 2048]
        features = features.view(features.size(0), -1, features.size(-1))  # [B, 49, 2048]
        return features