import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


class FPNLight(nn.Module):
    """
    Hafif Feature Pyramid Network (MobileNet için)
    3 ayrı feature seviyesini birleştirir.
    """
    def __init__(self, C3, C4, C5, out_channels=128):
        super().__init__()

        self.lateral3 = nn.Conv2d(C3, out_channels, 1)
        self.lateral4 = nn.Conv2d(C4, out_channels, 1)
        self.lateral5 = nn.Conv2d(C5, out_channels, 1)

        self.smooth3 = ConvBlock(out_channels, out_channels)
        self.smooth4 = ConvBlock(out_channels, out_channels)
        self.smooth5 = ConvBlock(out_channels, out_channels)

    def forward(self, f3, f4, f5):
        # Top-down pathway
        p5 = self.lateral5(f5)
        p4 = self.lateral4(f4) + F.interpolate(p5, size=f4.shape[-2:], mode="nearest")
        p3 = self.lateral3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="nearest")

        # Smooth
        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        # output feature levels
        return [p3, p4, p5]


class DetectionHead(nn.Module):
    """
    Tek sınıf (pothole) için:
    - Class logits: [B, N, 1]
    - BBox regression: [B, N, 4] - ABSOLUTE PIXEL COORDINATES
    """
    def __init__(self, in_c=128, num_levels=3, img_size=640):
        super().__init__()
        
        self.img_size = img_size
        
        self.cls_conv = nn.Sequential(
            ConvBlock(in_c, in_c),
            ConvBlock(in_c, in_c),
            nn.Conv2d(in_c, 1, 1)  # binary cls output
        )
        
        self.box_conv = nn.Sequential(
            ConvBlock(in_c, in_c),
            ConvBlock(in_c, in_c),
            nn.Conv2d(in_c, 4, 1)  # bbox coordinates
        )

        self.num_levels = num_levels

    def forward(self, features):
        """
        Args:
            features: list of [B, C, H, W] feature maps
        
        Returns:
            cls_outs: [B, N, 1] classification logits
            box_outs: [B, N, 4] boxes in [xmin, ymin, xmax, ymax] format (pixel coords)
        """
        cls_outs = []
        box_outs = []

        for feat in features:   # p3, p4, p5
            B, C, H, W = feat.shape
            
            # Classification
            cls_map = self.cls_conv(feat)  # [B, 1, H, W]
            cls_outs.append(cls_map.permute(0, 2, 3, 1).reshape(B, -1, 1))
            
            # Bounding box regression
            box_map = self.box_conv(feat)  # [B, 4, H, W]
            
            # Generate grid centers for this feature level
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=feat.device, dtype=torch.float32),
                torch.arange(W, device=feat.device, dtype=torch.float32),
                indexing='ij'
            )
            
            # Calculate stride (how much each feature map cell corresponds to in pixels)
            stride = self.img_size / H
            
            # Grid centers in pixel coordinates
            centers_x = (grid_x + 0.5) * stride  # [H, W]
            centers_y = (grid_y + 0.5) * stride  # [H, W]
            
            centers_x = centers_x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            centers_y = centers_y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Decode box offsets to absolute coordinates
            # box_map channels: [dx1, dy1, dx2, dy2] - offsets from center
            box_map = box_map.permute(0, 2, 3, 1)  # [B, H, W, 4]
            
            # Apply sigmoid to make offsets positive, then scale
            box_offsets = torch.sigmoid(box_map) * (self.img_size / 2)
            
            # Calculate absolute coordinates
            # xmin = center_x - dx1
            # ymin = center_y - dy1  
            # xmax = center_x + dx2
            # ymax = center_y + dy2
            centers_x_flat = centers_x.expand(B, 1, H, W).permute(0, 2, 3, 1)  # [B, H, W, 1]
            centers_y_flat = centers_y.expand(B, 1, H, W).permute(0, 2, 3, 1)  # [B, H, W, 1]
            
            xmin = centers_x_flat - box_offsets[..., 0:1]
            ymin = centers_y_flat - box_offsets[..., 1:2]
            xmax = centers_x_flat + box_offsets[..., 2:3]
            ymax = centers_y_flat + box_offsets[..., 3:4]
            
            # Stack to [B, H, W, 4]
            boxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
            
            # Clamp to image boundaries
            boxes = torch.clamp(boxes, min=0, max=self.img_size)
            
            # Reshape to [B, H*W, 4]
            box_outs.append(boxes.reshape(B, -1, 4))

        cls_outs = torch.cat(cls_outs, dim=1)   # [B, N, 1]
        box_outs = torch.cat(box_outs, dim=1)   # [B, N, 4]

        return cls_outs, box_outs


class MobileNetDetector(nn.Module):
    def __init__(self, pretrained=True, out_channels=128, img_size=640):
        super().__init__()

        # Load MobileNetV3-Large backbone
        m = mobilenet_v3_large(weights="DEFAULT" if pretrained else None)

        # backbone feature layers
        self.stage3 = nn.Sequential(*m.features[:5])   # low-level
        self.stage4 = nn.Sequential(*m.features[5:8])  # mid-level
        self.stage5 = nn.Sequential(*m.features[8:13]) # high-level

        # Channel sizes of the extracted feature maps
        C3 = 40  
        C4 = 80 
        C5 = 112 

        # FPN
        self.fpn = FPNLight(C3, C4, C5, out_channels)

        # Detection head
        self.head = DetectionHead(in_c=out_channels, img_size=img_size)

    def forward(self, x):
        # Extract feature maps
        f3 = self.stage3(x)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        # Build pyramid
        features = self.fpn(f3, f4, f5)

        # Detection head
        cls_logits, bbox_reg = self.head(features)

        return {
            "cls_logits": cls_logits,   # [B, N, 1]
            "bbox_reg": bbox_reg        # [B, N, 4] in pixel coordinates [xmin, ymin, xmax, ymax]
        }