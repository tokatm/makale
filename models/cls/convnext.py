import torch
import torch.nn as nn
from torchvision.models import convnext_tiny 


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ConvNeXt, self).__init__()
        self.model = convnext_tiny(pretrained=pretrained)
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    model = ConvNeXt(num_classes=2)
    x = torch.randn(1, 3, 384, 640)
    y = model(x)
    print("Output Shape: ",y.shape)