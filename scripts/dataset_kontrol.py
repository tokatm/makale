import os
from PIL import Image

# ===============================
#  CONFIG
# ===============================
DATASET_ROOT = "/Users/mustafa/Library/Mobile Documents/com~apple~CloudDocs/Computer_Science/09-Makalelerim/4- AnomalyDetection/VIT_LSTM/yolo_dataset"   # Root klasör
MAX_WIDTH = 1280                                   # Resize genişliği
THRESHOLD_KB = 900                                 # Büyük dosya limiti
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


# ===============================
#  RESIZE FUNCTION
# ===============================
def reduce_image_size(image_path, max_width=1280):
    img = Image.open(image_path)
    w, h = img.size

    if w > max_width:
        ratio = max_width / w
        new_size = (max_width, int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
        print(f"🔧 Resized → {image_path}")
        return True
    return False


# ===============================
#  IMAGE-LABEL PAIR CHECK
# ===============================
def check_pairs(images_dir, labels_dir):
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(tuple(IMAGE_EXTS))])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".txt")])

    img_no_label = []
    label_no_img = []

    for img in image_files:
        b = os.path.splitext(img)[0]
        lbl = b + ".txt"
        if lbl not in label_files:
            img_no_label.append(img)

    for lbl in label_files:
        b = os.path.splitext(lbl)[0]
        expected_imgs = [b + ext for ext in IMAGE_EXTS]
        if not any(x in image_files for x in expected_imgs):
            label_no_img.append(lbl)

    return img_no_label, label_no_img, image_files, label_files


# ===============================
#  COUNT POSITIVE / NEGATIVE
# ===============================
def count_classes(labels_dir):
    pos = 0
    neg = 0

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(labels_dir, fname)
        lines = open(fpath).read().strip().splitlines()

        if len(lines) == 0:
            neg += 1
        else:
            pos += 1

    return pos, neg




# ===============================
#  MAIN CLEANER PIPELINE
# ===============================
total_pos = 0
total_neg = 0

print("\n=================================================")
print("🚀 FULL AUTOMATIC DATASET CLEANER STARTED")
print("=================================================\n")

for split in SPLITS:
    print(f"\n🔹 Checking split: {split.upper()}")
    images_dir = os.path.join(DATASET_ROOT, "images", split)
    labels_dir = os.path.join(DATASET_ROOT, "labels", split)

    # 1) Eşleşmeleri kontrol et
    img_no_label, label_no_img, image_files, label_files = check_pairs(images_dir, labels_dir)

    print(f"  Total images: {len(image_files)}")
    print(f"  Total labels: {len(label_files)}")

    if img_no_label:
        print("  ⚠️ Images WITHOUT labels:")
        for x in img_no_label:
            print("    -", x)

    if label_no_img:
        print("  ⚠️ Labels WITHOUT images:")
        for x in label_no_img:
            print("    -", x)

    if not img_no_label and not label_no_img:
        print("  ✔ All image-label pairs are valid")

    # 2) Sınıf sayıları
    pos, neg = count_classes(labels_dir)
    total_pos += pos
    total_neg += neg

    print(f"  ✔ POSITIVE: {pos}  |  NEGATIVE: {neg}")

   


print("\n=================================================")
print("📊 DATASET SUMMARY")
print("=================================================")
print(f"Total POSITIVE (Pothole): {total_pos}")
print(f"Total NEGATIVE (No pothole): {total_neg}")
print("\n🚀 Dataset cleaning completed!")
print("=================================================\n")
