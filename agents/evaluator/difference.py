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
        feat_a = self.extractor.extract(img_a)[1:]  # remove CLS
        feat_b = self.extractor.extract(img_b)[1:]

        # cosine distance
        diff = 1.0 - torch.nn.functional.cosine_similarity(
            feat_a, feat_b, dim=1
        )

        return diff.cpu().numpy()
