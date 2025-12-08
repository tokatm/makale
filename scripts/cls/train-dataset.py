import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        if img_name.startswith('positive'):
            label = 1
        elif img_name.startswith('negative'):
            label = 0
        else:
            raise ValueError(f"Etiket adi siniflandirilamadi: {img_name}")
        
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32)
        
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

config_path = "configs/mobilenetv3.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

img_size = tuple(cfg['data']['img_size'])

aug_cfg = cfg['augment']
transform_list = []

# Resize
transform_list.append(A.Resize(height=img_size[0], width=img_size[1]))

# Rotation
if 'rotation' in aug_cfg:
    rot_cfg = aug_cfg['rotation']
    transform_list.append(A.Rotate(
        limit=(rot_cfg['angle_min'], rot_cfg['angle_max']),
        border_mode=cv2.BORDER_CONSTANT,
        p=rot_cfg['p']
    ))

# Random Resize (Albumentations'da RandomScale)
if 'random_resize' in aug_cfg:
    resize_cfg = aug_cfg['random_resize']
    transform_list.append(A.RandomScale(
        scale_limit=(resize_cfg['scale_min']-1, resize_cfg['scale_max']-1),
        p=resize_cfg['p']
    ))

# Motion Blur
if 'motion_blur' in aug_cfg:
    blur_cfg = aug_cfg['motion_blur']
    transform_list.append(A.MotionBlur(
        blur_limit= tuple(blur_cfg['kernel_size']),
        p=blur_cfg['p']
    ))

# Gaussian Blur
if 'gaussian_blur' in aug_cfg:
    blur_cfg = aug_cfg['gaussian_blur']
    transform_list.append(A.GaussianBlur(
        sigma_limit=(blur_cfg['sigma_limit'][0], blur_cfg['sigma_limit'][1]),
        blur_limit=blur_cfg['blur_limit'],
        p=blur_cfg['p']
    ))

# Gamma
if 'gamma' in aug_cfg:
    gamma_cfg = aug_cfg['gamma']
    transform_list.append(A.RandomGamma(
        gamma_limit=(80, 120),
        p=gamma_cfg['p']
    ))

transform_list.append(A.Resize(height=img_size[0], width=img_size[1], p=1.0))
transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), 
                                  std=(0.229, 0.224, 0.225)))
transform_list.append(ToTensorV2())
transform = A.Compose(transform_list)

#------------------------Dataset Loading------------------------#
dataset = cfg['data']['root']
train_dir = os.path.join(dataset, 'train')
val_dir = os.path.join(dataset, 'val')  
test_dir = os.path.join(dataset, 'test')

train_dataset = PotholeDataset(root_dir=train_dir, transform=transform)
val_transform = A.Compose([
    A.Resize(height=img_size[0], width=img_size[1]),
    ToTensorV2()
])
val_dataset = PotholeDataset(root_dir=val_dir, transform=val_transform)

test_dataset = PotholeDataset(root_dir=test_dir, transform=val_transform)

#------------------------DataLoader------------------------#

train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], 
                        shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=False, persistent_workers=False)
val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], 
                        shuffle=False, num_workers=cfg['data']['num_workers'],pin_memory=False, persistent_workers=False)
test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'],
                        shuffle=False, num_workers=cfg['data']['num_workers'],pin_memory=False, persistent_workers=False)    


#------------------------Model Definition------------------------#
from models.mobilenetv3 import MobileNetV3
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = MobileNetV3(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_cfg = cfg['train']['optimizer']
if optimizer_cfg['name'] == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])
    print("Using AdamW optimizer")
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_cfg['name']}")

scheduler = cfg['train']['scheduler']


#------------------------Training Loop------------------------#
num_epochs = cfg['train']['epochs']
best_val_loss = float('inf')
patience = cfg['train']['early_stop']['patience']
min_delta = cfg['train']['early_stop']['min_delta']
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _ , preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total * 100
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation step can be added here
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total * 100 
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")    
    
    if scheduler:
        scheduler.step()
    
    chechpoint_dir = 'checkpoints'
    if val_loss + min_delta < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        save_path = os.path.join(chechpoint_dir, f"model_name{model}_bestModel_valLoss_{best_val_loss:.4f}.pth")
        torch.save(model.state_dict(), save_path)
        print("Model saved.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break 
