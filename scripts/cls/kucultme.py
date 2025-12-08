from PIL import Image
import os

input_dir = "/Users/mustafa/Library/Mobile Documents/com~apple~CloudDocs/Computer_Science/09-Makalelerim/6-ADAS/dataset/test"
output_dir = "/Users/mustafa/Library/Mobile Documents/com~apple~CloudDocs/Computer_Science/09-Makalelerim/6-ADAS/dataset/test_comp"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(input_dir, filename))

        # JPEG'e dönüştürüp kaliteyi düşür
        rgb_img = img.convert("RGB")
        rgb_img.save(
            os.path.join(output_dir, filename.replace(".png", ".jpg")),
            "JPEG",
            quality=80,   # kalite: 100 (yüksek) → 60 (küçük boyut)
            optimize=True
        )
