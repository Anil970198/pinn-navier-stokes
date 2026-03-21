"""
Depth from CLIP Text-Image Gradient Alignment
==============================================

Extracts monocular depth from a frozen CLIP model by computing the gradient
of text-image similarity w.r.t. input pixels. CLIP was never trained for any
spatial task — the depth signal is an emergent artifact of its contrastive
language-vision training.

Usage:
    python main.py <image_path> [--output_dir results] [--sigma 3.0]

Example:
    python main.py ../dinov2_affordance/photo.jpeg --output_dir results
"""

import argparse
import sys
from PIL import Image

from extractor import CLIPGradientExtractor
from analysis import DepthAnalyzer
from visualize import DepthVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract monocular depth from CLIP text-image gradients"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory for output visualizations (default: results)"
    )
    parser.add_argument(
        "--sigma", type=float, default=3.0,
        help="Gaussian smoothing sigma for gradient depth (default: 3.0)"
    )
    args = parser.parse_args()

    # --- Step 1: Load CLIP ---
    print("=" * 60)
    print("  CLIP Gradient Depth Estimation")
    print("=" * 60)
    extractor = CLIPGradientExtractor()

    # --- Step 2: Load image ---
    print(f"\nInput image: {args.image}")
    original = Image.open(args.image).convert('RGB')
    print(f"  Size: {original.size}")

    # --- Step 3: Run depth analysis ---
    print("\nComputing depth from text-image gradients...")
    analyzer = DepthAnalyzer(extractor)
    result = analyzer.multi_prompt_depth(args.image, sigma=args.sigma)

    # Print similarity scores for each prompt pair
    print("\nSimilarity scores per prompt pair:")
    for i, pair in enumerate(result['per_pair']):
        print(f"  Pair {i+1}:")
        print(f"    \"{pair['close_prompt']}\": {pair['gradient']['sim_close']:.4f}")
        print(f"    \"{pair['far_prompt']}\":  {pair['gradient']['sim_far']:.4f}")

    # --- Step 4: Generate deliverables ---
    viz = DepthVisualizer(output_dir=args.output_dir)
    viz.generate_all(original, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
