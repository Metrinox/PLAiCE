import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np

class ViTFeatureExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTModel.from_pretrained(
            "google/vit-base-patch16-224"
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> torch.Tensor:
        """
        image: numpy array (H, W, 3), uint8
        returns: (num_patches, hidden_dim)
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # shape: (1, num_tokens, hidden_dim)
        return outputs.last_hidden_state.squeeze(0)
