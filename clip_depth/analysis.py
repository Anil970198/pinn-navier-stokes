import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


# Prompt pairs: (close, far) — averaged for robustness
DEFAULT_PROMPT_PAIRS = [
    ("a close up photograph", "a photograph taken from far away"),
    ("a macro shot near the camera", "a wide angle landscape in the distance"),
    ("an object in the foreground very close", "a scene stretching into the far background"),
]


class DepthAnalyzer:
    """
    Processes CLIP gradient maps and patch similarities into depth estimates.

    Two independent methods:
    1. Gradient method: |∂S_close/∂x| - |∂S_far/∂x| at pixel level (224x224)
    2. Patch-token method: sim(patch, "close") - sim(patch, "far") at patch level (7x7, upsampled)

    Both are averaged across multiple prompt pairs for robustness.
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def gradient_depth_single(self, image_path, close_prompt, far_prompt):
        """Depth from a single (close, far) prompt pair using pixel gradients."""
        grad_close, sim_close = self.extractor.compute_gradient_map(image_path, close_prompt)
        grad_far, sim_far = self.extractor.compute_gradient_map(image_path, far_prompt)

        depth_raw = grad_close - grad_far

        return {
            'depth_raw': depth_raw,
            'grad_close': grad_close,
            'grad_far': grad_far,
            'sim_close': sim_close,
            'sim_far': sim_far,
        }

    def patch_depth_single(self, image_path, close_prompt, far_prompt, target_size=224):
        """Depth from a single (close, far) prompt pair using patch-token similarities."""
        sim_close = self.extractor.compute_patch_similarities(image_path, close_prompt)
        sim_far = self.extractor.compute_patch_similarities(image_path, far_prompt)

        depth_patch = sim_close - sim_far

        # Upsample from 7x7 to target_size via bicubic interpolation
        depth_upsampled = cv2.resize(
            depth_patch.astype(np.float32),
            (target_size, target_size),
            interpolation=cv2.INTER_CUBIC
        )

        return {
            'depth_patch_raw': depth_patch,
            'depth_upsampled': depth_upsampled,
            'sim_close_patch': sim_close,
            'sim_far_patch': sim_far,
        }

    def multi_prompt_depth(self, image_path, prompt_pairs=None, sigma=3.0):
        """
        Average depth maps across multiple prompt pairs.

        Args:
            image_path: path to input image
            prompt_pairs: list of (close, far) tuples, or None for defaults
            sigma: Gaussian smoothing for the gradient-based depth map

        Returns:
            dict with all depth maps and per-pair intermediate results
        """
        if prompt_pairs is None:
            prompt_pairs = DEFAULT_PROMPT_PAIRS

        gradient_depths = []
        patch_depths = []
        per_pair_results = []

        for i, (close_p, far_p) in enumerate(prompt_pairs):
            print(f"  Prompt pair {i+1}/{len(prompt_pairs)}...")

            grad_result = self.gradient_depth_single(image_path, close_p, far_p)
            gradient_depths.append(grad_result['depth_raw'])

            patch_result = self.patch_depth_single(image_path, close_p, far_p)
            patch_depths.append(patch_result['depth_upsampled'])

            per_pair_results.append({
                'close_prompt': close_p,
                'far_prompt': far_p,
                'gradient': grad_result,
                'patch': patch_result,
            })

        # Average across pairs
        avg_gradient = np.mean(gradient_depths, axis=0)
        avg_patch = np.mean(patch_depths, axis=0)

        # Smooth the noisy pixel-level gradient map
        smooth_gradient = gaussian_filter(avg_gradient, sigma=sigma)

        # Normalize to [0, 1]
        norm_gradient = _normalize(smooth_gradient)
        norm_patch = _normalize(avg_patch)

        # Combined: average of both normalized approaches
        combined = _normalize(0.5 * norm_gradient + 0.5 * norm_patch)

        return {
            'gradient_depth': norm_gradient,
            'gradient_depth_raw': avg_gradient,
            'patch_depth': norm_patch,
            'patch_depth_raw': avg_patch,
            'combined_depth': combined,
            'per_pair': per_pair_results,
            'prompt_pairs': prompt_pairs,
        }


def _normalize(arr):
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
