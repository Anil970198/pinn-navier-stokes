import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPGradientExtractor:
    """
    Loads a frozen CLIP model and extracts:
    1. Pixel-level gradient maps of text-image similarity (224x224 resolution)
    2. Patch-token level text similarities (7x7 resolution for ViT-B/32)

    The key insight: CLIP was trained for global image-text matching, never for
    any spatial task. Yet the gradient of similarity w.r.t. input pixels reveals
    which spatial regions drive the alignment — an emergent spatial signal from
    a model with no spatial supervision.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: {model_name}")

        # Use CPU for gradient computation — most reliable for autograd
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Freeze all parameters — we only want gradients w.r.t. input pixels
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.patch_size = self.model.config.vision_config.patch_size
        self.image_size = self.model.config.vision_config.image_size
        self.num_patches_side = self.image_size // self.patch_size

        print(f"  Patch size: {self.patch_size}")
        print(f"  Spatial grid: {self.num_patches_side}x{self.num_patches_side}")

    def _preprocess(self, image_path):
        """Load image and return preprocessed pixel tensor."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        return image, pixel_values

    def encode_text(self, prompts):
        """Encode text prompt(s) into normalized CLIP embeddings."""
        if isinstance(prompts, str):
            prompts = [prompts]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_gradient_map(self, image_path, text_prompt):
        """
        Gradient of CLIP similarity(image, text) w.r.t. patch embeddings.

        We compute gradients AFTER the patch embedding convolution (Conv2d with
        kernel=32, stride=32), not at the raw pixel level. This eliminates the
        grid artifact caused by the stride-32 boundary, and gives one gradient
        vector per patch — which is the natural spatial unit of the ViT.

        The per-patch gradient magnitude is then upsampled to 224x224 for
        visualization.

        Returns:
            gradient_map: (224, 224) numpy array — upsampled gradient magnitude
            similarity: float — the cosine similarity score
        """
        import cv2

        _, pixel_values = self._preprocess(image_path)
        pixel_values.requires_grad_(True)

        text_emb = self.encode_text(text_prompt)

        # --- Hook into the patch embedding output ---
        # CLIP's vision encoder: pixel_values → patch_embedding (Conv2d) → transformer
        # We intercept right after the Conv2d to get gradients at patch level.
        patch_embed_output = {}

        def hook_fn(module, input, output):
            # output shape: (1, hidden_dim, grid_h, grid_w) for Conv2d
            # or (1, num_patches, hidden_dim) depending on architecture
            patch_embed_output['value'] = output
            output.retain_grad()  # keep gradient for this non-leaf tensor

        # Register hook on the patch embedding layer
        embeddings = self.model.vision_model.embeddings
        handle = embeddings.patch_embedding.register_forward_hook(hook_fn)

        try:
            # Forward pass
            image_features = self.model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = (image_features * text_emb).sum()
            similarity.backward()

            # Get gradient w.r.t. patch embeddings
            embed_out = patch_embed_output['value']
            grad = embed_out.grad.detach()  # (1, hidden_dim, grid_h, grid_w)

            # L2 norm across the hidden dimension → (grid_h, grid_w)
            grad_map = grad.squeeze(0).norm(dim=0).numpy()

            # Upsample from 7x7 to 224x224 with bicubic interpolation
            gradient_map = cv2.resize(
                grad_map.astype('float32'),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_CUBIC
            )
        finally:
            handle.remove()

        return gradient_map, similarity.item()

    def compute_patch_similarities(self, image_path, text_prompt):
        """
        Per-patch-token similarity with a text prompt.

        Each of the 7x7 patch tokens from the ViT encoder is projected into
        CLIP's shared embedding space and compared with the text embedding.
        This gives a coarse but clean spatial similarity map.

        Returns:
            sim_map: (num_patches_side, num_patches_side) numpy array
        """
        _, pixel_values = self._preprocess(image_path)
        text_emb = self.encode_text(text_prompt)

        with torch.no_grad():
            vision_out = self.model.vision_model(pixel_values=pixel_values)

            # last_hidden_state: (1, num_patches+1, hidden_dim)
            # Position 0 = CLS token, positions 1: = patch tokens
            patch_tokens = vision_out.last_hidden_state[:, 1:, :]

            # Project each patch token to CLIP's shared text-image space
            patch_features = self.model.visual_projection(patch_tokens)
            patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

            # Cosine similarity: each patch vs text embedding
            sims = (patch_features * text_emb.unsqueeze(1)).sum(dim=-1)  # (1, N)

            sim_map = sims.squeeze(0).reshape(
                self.num_patches_side, self.num_patches_side
            ).numpy()

        return sim_map
    def compute_layerwise_similarities(self, image_path, text_prompt):
        """
        Extracts patch similarities at EVERY layer of the ViT (0 to 12).
        This provides proof of how the spatial/depth representation evolves.
        
        Returns:
            list of (num_patches_side, num_patches_side) numpy arrays
        """
        _, pixel_values = self._preprocess(image_path)
        text_emb = self.encode_text(text_prompt)

        with torch.no_grad():
            vision_out = self.model.vision_model(
                pixel_values=pixel_values, 
                output_hidden_states=True
            )

            # hidden_states is a tuple of length 13 (embedding + 12 layers)
            layer_maps = []
            for h in vision_out.hidden_states:
                patch_tokens = h[:, 1:, :] # Drop CLS
                
                # Project intermediate features using the final visual projection
                patch_features = self.model.visual_projection(patch_tokens)
                patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

                # Cosine similarity
                sims = (patch_features * text_emb.unsqueeze(1)).sum(dim=-1)
                
                sim_map = sims.squeeze(0).reshape(
                    self.num_patches_side, self.num_patches_side
                ).numpy()
                
                layer_maps.append(sim_map)

        return layer_maps


if __name__ == "__main__":
    extractor = CLIPGradientExtractor()
    print("\nCLIP gradient extractor initialized successfully.")
