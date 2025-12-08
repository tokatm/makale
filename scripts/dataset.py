import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


class YOLODataset(Dataset):
    """
    YOLO formatted dataset with separate image and label directories:
    
    img_dir = ".../images/train"
    label_dir = ".../labels/train"

    Label format:
        class x_center y_center w h  (normalized)
    """

    def __init__(self, img_dir, label_dir, img_size=(640, 640), transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.img_size = img_size

        # Load image list
        self.img_files = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
            + glob.glob(os.path.join(self.img_dir, "*.jpeg"))
        )

    def __len__(self):
        return len(self.img_files)

    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_labels(self, label_path, W, H):
        """
        Convert YOLO normalized format -> pascal voc [xmin, ymin, xmax, ymax]
        """

        if not os.path.exists(label_path):
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                c, xc, yc, w, h = map(float, line.strip().split())
                # clamp YOLO coords before converting to VOC
                xc = max(0.0, min(xc, 1.0))
                yc = max(0.0, min(yc, 1.0))
                w  = max(0.0, min(w, 1.0))
                h  = max(0.0, min(h, 1.0))

                xmin = (xc - w / 2) * W
                ymin = (yc - h / 2) * H
                xmax = (xc + w / 2) * W
                ymax = (yc + h / 2) * H

                xmin = max(0, min(xmin, W - 1))
                ymin = max(0, min(ymin, H - 1))
                xmax = max(0, min(xmax, W - 1))
                ymax = max(0, min(ymax, H - 1))

                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(c))

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{name}.txt")

        img = self.load_image(img_path)
        H, W, _ = img.shape

        boxes, labels = self.load_labels(label_path, W, H)

        # Albumentations transform
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes.numpy().tolist(),
                labels=labels.numpy().tolist(),
            )
            img = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.long)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.

        target = {
            "boxes": boxes,
            "labels": labels
        }

        print(img, target)
        return img, target
        

img_dir = "dataset-2/images/test"
label_dir = "dataset-2/labels/test"

dataset = YOLODataset(img_dir, label_dir, transforms=None, img_size=(640, 640))
print(f"Dataset size: {len(dataset)}")
img, target = dataset[0]
print(f"Image shape: {img.shape}")
print("Boxes:", target["boxes"])
print("Labels:", target["labels"])
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
for imgs, targets in loader:
    print(f"Batch size: {len(imgs)}")
    print("image tensor shape:", imgs[0].shape)
    print("first target boxes:", targets[0]["boxes"])
    break
