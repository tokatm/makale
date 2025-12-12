def debug_batch_samples(dataset, output_dir='debug_samples', num_positive=10, num_negative=10):
    """Pozitif ve negatif örnekleri kaydet"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    positive_samples = []
    negative_samples = []
    
    # Örnekleri topla
    for idx in range(len(dataset)):
        image, target = dataset[idx]
        img_name = dataset.images[idx]
        num_boxes = len(target['boxes'])
        
        if num_boxes > 0 and len(positive_samples) < num_positive:
            positive_samples.append((idx, img_name, image, target))
        elif num_boxes == 0 and len(negative_samples) < num_negative:
            negative_samples.append((idx, img_name, image, target))
        
        if len(positive_samples) >= num_positive and len(negative_samples) >= num_negative:
            break
    
    # Kaydet
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    for idx, img_name, image, target in all_samples:
        if torch.is_tensor(image):
            image_np = image.numpy().transpose(1, 2, 0)
            image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = image
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        
        boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'Pothole', color='white', fontsize=10, 
                    bbox=dict(facecolor='red', alpha=0.8))
        
        sample_type = "POS" if len(boxes) > 0 else "NEG"
        ax.set_title(f'{sample_type} - {img_name} - Boxes: {len(boxes)}')
        ax.axis('off')
        
        save_name = f"{sample_type}_{os.path.splitext(img_name)[0]}.png"
        plt.savefig(os.path.join(output_dir, save_name), bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved: {save_name}")
    
    print(f"\n✓ {len(positive_samples)} pozitif, {len(negative_samples)} negatif örnek kaydedildi: {output_dir}/")

# Kullanım:
debug_batch_samples(train_dataset, output_dir='debug_samples', num_positive=10, num_negative=10)