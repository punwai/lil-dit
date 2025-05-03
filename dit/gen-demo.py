import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Create a 6x6 grid to display images
fig, axes = plt.subplots(6, 6, figsize=(15, 15))
fig.suptitle('Image Classification Demo', fontsize=16)

# Load and display images
for i in range(36):  # 6 classes * 6 images per class
    row = i // 6
    col = i % 6
    
    # Load image
    img_path = f'images/sample_{i}.png'
    if os.path.exists(img_path):
        img = Image.open(img_path)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Class {i%10}')
        axes[row, col].axis('off')
    else:
        axes[row, col].axis('off')
        axes[row, col].set_title('Image not found')

plt.tight_layout()
plt.savefig('demo_grid.png')
plt.close()
