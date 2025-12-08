import torch 
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3_small(pretrained=pretrained)
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
    model = MobileNetV3(num_classes=2)
    x = torch.randn(1, 3, 384, 640)
    y = model(x)
    print("Output Shape: ",y.shape)  # Expected output: torch.Size([1, 2])