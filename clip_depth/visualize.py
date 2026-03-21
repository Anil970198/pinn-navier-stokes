import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


class DepthVisualizer:
    """Generates all deliverable visualizations for the CLIP depth project."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_main_strip(self, original_image, result, pair_idx=0, save_name="01_main_strip.png"):
        """
        Main deliverable:
        [Original] → [Close gradient] → [Far gradient] → [Depth map]
        """
        pair = result['per_pair'][pair_idx]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        orig = original_image.resize((224, 224))

        axes[0].imshow(orig)
        axes[0].set_title("Original Image", fontsize=11)

        im1 = axes[1].imshow(pair['gradient']['grad_close'], cmap='hot')
        axes[1].set_title(f'∂S/∂x for\n"{pair["close_prompt"][:35]}"', fontsize=9)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(pair['gradient']['grad_far'], cmap='hot')
        axes[2].set_title(f'∂S/∂x for\n"{pair["far_prompt"][:35]}"', fontsize=9)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        im3 = axes[3].imshow(result['gradient_depth'], cmap='inferno')
        axes[3].set_title("Depth (gradient method)", fontsize=11)
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis('off')

        plt.suptitle("CLIP Text-Image Gradient → Monocular Depth", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_method_comparison(self, original_image, result, save_name="02_method_comparison.png"):
        """Gradient depth vs patch-token depth vs combined."""
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        axes[0].imshow(original_image.resize((224, 224)))
        axes[0].set_title("Original", fontsize=11)

        for i, (key, title) in enumerate([
            ('gradient_depth', 'Gradient Method (224×224)'),
            ('patch_depth', 'Patch-Token Method (7×7 upsampled)'),
            ('combined_depth', 'Combined'),
        ]):
            im = axes[i+1].imshow(result[key], cmap='inferno')
            axes[i+1].set_title(title, fontsize=11)
            plt.colorbar(im, ax=axes[i+1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis('off')

        plt.suptitle("Two Independent Methods for Depth from CLIP", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_prompt_analysis(self, original_image, result, save_name="03_prompt_analysis.png"):
        """How each prompt pair produces a different depth signal."""
        n = len(result['per_pair'])
        fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))

        for i, pair in enumerate(result['per_pair']):
            # Top row: gradient depth per pair
            gd = pair['gradient']['depth_raw']
            gd_n = (gd - gd.min()) / (gd.max() - gd.min() + 1e-8)
            axes[0, i].imshow(gd_n, cmap='inferno')
            axes[0, i].set_title(
                f'Pair {i+1}\n"{pair["close_prompt"][:22]}..."\nvs\n"{pair["far_prompt"][:22]}..."',
                fontsize=7
            )
            axes[0, i].axis('off')

            # Bottom row: patch depth per pair
            pd = pair['patch']['depth_upsampled']
            pd_n = (pd - pd.min()) / (pd.max() - pd.min() + 1e-8)
            axes[1, i].imshow(pd_n, cmap='inferno')
            axes[1, i].axis('off')

        # Average column
        axes[0, n].imshow(result['gradient_depth'], cmap='inferno')
        axes[0, n].set_title("Average\n(gradient)", fontsize=10, fontweight='bold')
        axes[0, n].axis('off')

        axes[1, n].imshow(result['patch_depth'], cmap='inferno')
        axes[1, n].set_title("Average\n(patch-token)", fontsize=10, fontweight='bold')
        axes[1, n].axis('off')

        axes[0, 0].set_ylabel("Gradient Method", fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel("Patch-Token Method", fontsize=11, fontweight='bold')

        plt.suptitle("Depth Signal Across Different Prompt Pairs", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_depth_overlay(self, original_image, result, save_name="04_depth_overlay.png"):
        """Depth map overlaid on the original image with transparency."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        orig_arr = np.array(original_image.resize((224, 224))) / 255.0
        depth = result['combined_depth']
        depth_rgb = plt.cm.inferno(depth)[:, :, :3]

        blend = np.clip(0.6 * orig_arr + 0.4 * depth_rgb, 0, 1)

        axes[0].imshow(orig_arr)
        axes[0].set_title("Original", fontsize=11)

        axes[1].imshow(depth, cmap='inferno')
        axes[1].set_title("Depth Map (combined)", fontsize=11)

        axes[2].imshow(blend)
        axes[2].set_title("Depth Overlaid on Original", fontsize=11)

        for ax in axes:
            ax.axis('off')

        plt.suptitle("Depth Overlay Visualization", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    def plot_patch_grid(self, original_image, result, pair_idx=0, save_name="05_patch_grid.png"):
        """Raw 7×7 patch-level similarity maps with grid lines."""
        pair = result['per_pair'][pair_idx]
        sc = pair['patch']['sim_close_patch']
        sf = pair['patch']['sim_far_patch']
        dp = pair['patch']['depth_patch_raw']

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(original_image.resize((224, 224)))
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis('off')

        for ax, data, cmap, title in [
            (axes[1], sc, 'Reds', 'Patch sim: "close"'),
            (axes[2], sf, 'Blues', 'Patch sim: "far"'),
            (axes[3], dp, 'inferno', 'Depth (close − far)'),
        ]:
            im = ax.imshow(data, cmap=cmap, interpolation='nearest')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            # Grid lines showing patch boundaries
            for j in range(data.shape[0]):
                ax.axhline(j - 0.5, color='gray', linewidth=0.5)
                ax.axvline(j - 0.5, color='gray', linewidth=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Patch-Level Similarity Analysis (raw 7×7 grid)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, save_name)

    def generate_all(self, original_image, result):
        """Generate all deliverables."""
        print("\nGenerating deliverables...")
        paths = [
            self.plot_main_strip(original_image, result),
            self.plot_method_comparison(original_image, result),
            self.plot_prompt_analysis(original_image, result),
            self.plot_depth_overlay(original_image, result),
            self.plot_patch_grid(original_image, result),
        ]
        print(f"\nAll deliverables saved to: {self.output_dir}/")
        return paths

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")
        return path
