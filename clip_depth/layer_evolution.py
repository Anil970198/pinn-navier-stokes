"""
Layer-Wise Depth Evolution
==========================

Generates a sequence of images showing the depth map evolving
layer by layer inside the Vision Transformer.
This provides structural proof that the semantic concept of "depth"
gradually crystallizes deeper in the network.
"""

import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from extractor import CLIPGradientExtractor

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def main():
    parser = argparse.ArgumentParser(description="Visualize layer-wise depth evolution in CLIP")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading extractor...")
    extractor = CLIPGradientExtractor()

    close_prompt = "a close up photograph"
    far_prompt = "a photograph taken from far away"

    print(f"Extracting layer-wise features for: {args.image}")
    close_layers = extractor.compute_layerwise_similarities(args.image, close_prompt)
    far_layers = extractor.compute_layerwise_similarities(args.image, far_prompt)

    num_layers = len(close_layers)
    
    # We will plot original image + num_layers states
    cols = 5
    rows = (num_layers + 1 + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    # Plot original
    orig = Image.open(args.image).convert('RGB').resize((224, 224))
    axes[0].imshow(orig)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    print("Generating layer-wise depth maps...")
    for i in range(num_layers):
        depth_patch = close_layers[i] - far_layers[i]
        
        # Upsample to 224x224
        depth_up = cv2.resize(
            depth_patch.astype(np.float32), 
            (224, 224), 
            interpolation=cv2.INTER_CUBIC
        )
        depth_norm = normalize(depth_up)
        
        ax = axes[i + 1]
        im = ax.imshow(depth_norm, cmap='inferno')
        
        if i == 0:
            title = "Patch Embedding (Layer 0)"
        else:
            title = f"Transformer Layer {i}"
            
        ax.set_title(title, fontsize=11)
        ax.axis('off')
        
    # Hide unused subplots
    for j in range(num_layers + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Layer-Wise Evolution of Depth Geometry in ViT", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, "06_layer_evolution.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved evolution sequence to: {save_path}")

if __name__ == "__main__":
    main()
