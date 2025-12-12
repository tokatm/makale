import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_bbox_sizes(label_dir):
    widths = []
    heights = []
    aspect_ratios = []
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO formatı: class_id, x_center, y_center, width, height (normalize)
                    _, _, _, w, h = map(float, parts)
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w/h)
    
    return widths, heights, aspect_ratios

# Kullanım:
label_dir = 'dataset-2/labels/train'  # veya val
widths, heights, aspect_ratios = analyze_bbox_sizes(label_dir)

# İstatistikleri yazdır
print(f"Toplam çukur sayısı: {len(widths)}")
print(f"Genişlik ortalaması: {np.mean(widths):.4f}, std: {np.std(widths):.4f}")
print(f"Yükseklik ortalaması: {np.mean(heights):.4f}, std: {np.std(heights):.4f}")
print(f"En-boy oranı ortalaması: {np.mean(aspect_ratios):.4f}, std: {np.std(aspect_ratios):.4f}")

# Histogramları çiz
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(widths, bins=50, edgecolor='black')
axes[0].set_xlabel('Normalize Genişlik')
axes[0].set_ylabel('Frekans')
axes[0].set_title('Genişlik Dağılımı')

axes[1].hist(heights, bins=50, edgecolor='black')
axes[1].set_xlabel('Normalize Yükseklik')
axes[1].set_ylabel('Frekans')
axes[1].set_title('Yükseklik Dağılımı')

axes[2].hist(aspect_ratios, bins=50, edgecolor='black')
axes[2].set_xlabel('En-Boy Oranı (genişlik/yükseklik)')
axes[2].set_ylabel('Frekans')
axes[2].set_title('En-Boy Oranı Dağılımı')

plt.tight_layout()
plt.show()

# 5. ve 95. percentilleri göster
print(f"\nGenişlik 5. percentile: {np.percentile(widths, 5):.4f}, 95. percentile: {np.percentile(widths, 95):.4f}")
print(f"Yükseklik 5. percentile: {np.percentile(heights, 5):.4f}, 95. percentile: {np.percentile(heights, 95):.4f}")
print(f"En-boy oranı 5. percentile: {np.percentile(aspect_ratios, 5):.4f}, 95. percentile: {np.percentile(aspect_ratios, 95):.4f}")