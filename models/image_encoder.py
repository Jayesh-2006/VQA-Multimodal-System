from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove last classification layer [2048 --> 1000]
        # Replace it with linear layer [2048-->512]
        self.resnet.fc = nn.Linear(2048,512)

        #freeze all resnet parameters excpet last fc layer and layer 4
        for name,param in self.resnet.named_parameters():
            if (not name.startswith("fc") and not name.startswith("layer4")):
                param.requires_grad = False
       
    def forward(self,x):
        return self.resnet(x)  # [B,512]

