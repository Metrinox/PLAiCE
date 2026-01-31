import numpy as np
from agents.classifier.vit_extractor import ViTFeatureExtractor
from agents.classifier.similarity import patchwise_similarity
from agents.classifier.saliency import patch_importance, expand_to_pixel_map

class LocalEvaluator:
    def __init__(self, device="cpu"):
        self.extractor = ViTFeatureExtractor(device=device)

    def evaluate(self, current_fov: np.ndarray, generated_fov: np.ndarray):
        """
        Returns pixel importance heatmap.
        """
        feat_current = self.extractor.extract(current_fov)
        feat_generated = self.extractor.extract(generated_fov)

        # Ignore CLS token (index 0)
        feat_current = feat_current[1:]
        feat_generated = feat_generated[1:]

        similarities = patchwise_similarity(feat_current, feat_generated)
        importance = patch_importance(similarities)

        heatmap = expand_to_pixel_map(
            importance,
            image_shape=current_fov.shape
        )

        return heatmap
