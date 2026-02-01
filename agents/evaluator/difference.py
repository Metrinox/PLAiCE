import numpy as np
import torch
from agents.classifier.vit_extractor import ViTFeatureExtractor

class ImageDifference:
    def __init__(self, device="cpu"):
        self.extractor = ViTFeatureExtractor(device=device)

    def compute_patch_difference(self, img_a: np.ndarray, img_b: np.ndarray):
        """
        Returns per-patch difference scores using ViT features
        """
        feat_a, attn_a = self.extractor.extract_with_attention(img_a)
        feat_b, attn_b = self.extractor.extract_with_attention(img_b)

        feat_a = feat_a[1:]  # remove CLS
        feat_b = feat_b[1:]

        # cosine distance
        diff = 1.0 - torch.nn.functional.cosine_similarity(
            feat_a, feat_b, dim=1
        )

        # combine CLS attention from both images as weighting
        attn = 0.5 * (attn_a + attn_b)
        weighted = diff * (attn + 1e-6)

        return weighted.cpu().numpy()
