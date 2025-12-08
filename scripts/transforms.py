import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_inference_transforms(cfg):
    img_h, img_w = cfg["data"]["img_size"]

    return A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])



def get_train_transforms(cfg):
    """
    Create albumentations transform pipeline based on YAML config.
    cfg: config['augment']
    """
    img_h, img_w = cfg["data"]["img_size"]

    transforms_list = [
        A.Resize(height=img_h, width=img_w)
    ]

    # Horizontal Flip
    if cfg["augment"].get("horizontal_flip", False):
        transforms_list.append(
            A.HorizontalFlip(p=0.5)
        )

    # Color Jitter
    if cfg["augment"].get("color_jitter", False):
        transforms_list.append(
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.4
            )
        )

    # Blur
    if cfg["augment"].get("blur", False):
        transforms_list.append(
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.3
            )
        )

    # Random Rotate
    if cfg["augment"].get("rotation", False):
        transforms_list.append(
            A.Rotate(limit=10, p=0.4)
        )

    # Normalize
    transforms_list.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    )

    transforms_list.append(ToTensorV2())

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=4,        # çok küçük kutuları siler
            min_visibility=0.1 # görünürlük eşiği
        )
    )


def get_val_transforms(cfg):
    img_h, img_w = cfg["data"]["img_size"]

    return A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        )
    )
