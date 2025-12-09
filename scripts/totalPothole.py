import os
from pathlib import Path

def count_total_potholes_simple(train_label_dir, val_label_dir):
    """Sadece toplam çukur sayısını hesaplar"""
    
    def count_in_folder(label_dir):
        total = 0
        for txt_file in Path(label_dir).glob("*.txt"):
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and line.split()[0] == '0':
                            total += 1
            except:
                continue
        return total
    
    train_count = count_in_folder(train_label_dir)
    val_count = count_in_folder(val_label_dir)
    total_count = train_count + val_count
    
    print("="*50)
    print("TOPLAM ÇUKUR SAYISI")
    print("="*50)
    print(f"Train setindeki çukur sayısı: {train_count}")
    print(f"Validation setindeki çukur sayısı: {val_count}")
    print(f"TOPLAM ÇUKUR SAYISI: {total_count}")
    print("="*50)
    
    # Resim başına ortalama
    train_images = len(list(Path(train_label_dir).parent.parent.glob("images/train/*.jpg")))
    val_images = len(list(Path(val_label_dir).parent.parent.glob("images/val/*.jpg")))
    
    if train_images > 0 and val_images > 0:
        print(f"\nResim başına ortalama çukur:")
        print(f"Train: {train_count/train_images:.2f} çukur/resim")
        print(f"Validation: {val_count/val_images:.2f} çukur/resim")
        print(f"Toplam: {total_count/(train_images+val_images):.2f} çukur/resim")
    
    return total_count

# Kullanım
train_label_dir = "dataset-2/labels/train"
val_label_dir = "dataset-2/labels/val"

if os.path.exists(train_label_dir):
    total_potholes = count_total_potholes_simple(train_label_dir, val_label_dir)
else:
    print("Klasör bulunamadı!")