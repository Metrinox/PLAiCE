import torch
import torch.nn.functional as F

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a, b: (hidden_dim,)
    """
    return F.cosine_similarity(a, b, dim=0).item()


def patchwise_similarity(features_a, features_b):
    """
    Compare two ViT feature maps patch-by-patch.
    Returns similarity per patch.
    """
    sims = []
    for fa, fb in zip(features_a, features_b):
        sims.append(cosine_similarity(fa, fb))
    return sims
