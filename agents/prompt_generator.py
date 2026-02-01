"""
Generate agent prompts directly from image classification.

Uses the ViT image classifier to identify the dominant visual content
and uses the top prediction label as the prompt.
"""

import numpy as np
from PIL import Image
import torch


class PromptGenerator:
    """Generate prompts from image classification using ViT."""

    def __init__(self, device: str = "cpu"):
        """Initialize the image classifier."""
        from transformers import ViTImageProcessor, ViTForImageClassification

        self.device = device
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = self.model.to(device)
        self.model.eval()

    def generate_prompt_from_image(self, image: Image.Image) -> str:
        """
        Classify an image using ViT and return the top predicted label as prompt.

        Args:
            image: PIL.Image to classify

        Returns:
            str: The top predicted class label (used as prompt)
        """
        # Convert to RGB if needed
        image = image.convert("RGB")

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Get the predicted class label
        predicted_class = self.model.config.id2label[predicted_class_idx]

        return predicted_class


