"""
Slow integration test using real models (ViT + aMUSEd).

Set RUN_REAL_MODELS=1 to execute.
"""

import os
from PIL import Image
import numpy as np

from agents.pipeline import PipelineConfig, LocalRegionPipeline


def create_test_canvas_region(width: int = 128, height: int = 128) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2] = (255, 0, 0)
    arr[:, width // 2 :] = (0, 0, 255)
    return Image.fromarray(arr)


def test_real_models_pipeline():
    if os.environ.get("RUN_REAL_MODELS") != "1":
        print("Skipping real model test. Set RUN_REAL_MODELS=1 to run.")
        return

    config = PipelineConfig(image_size=128, top_x_proposals=5)
    pipeline = LocalRegionPipeline(config)
    canvas_region = create_test_canvas_region()

    prompt, generated_img, proposals = pipeline.process_canvas_region_to_proposals(
        canvas_region
    )

    assert isinstance(prompt, str) and len(prompt) > 0
    assert generated_img.size == (128, 128)
    assert isinstance(proposals, list)

    print("Prompt:", prompt)
    print("Proposals:", proposals[:5])


if __name__ == "__main__":
    test_real_models_pipeline()
