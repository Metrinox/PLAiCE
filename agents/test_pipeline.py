"""
Integration tests for the pipeline module.

These tests use lightweight fakes to validate the data flow and
composition without loading large models.
"""

import numpy as np
from PIL import Image
from agents import pipeline as pipeline_module
from agents import prompt_generator as prompt_module
from agents.pipeline import PipelineConfig, LocalRegionPipeline, generate_and_evaluate
from contextlib import contextmanager


def create_test_canvas_region(
    width: int = 128, height: int = 128, color: tuple = (128, 128, 128)
) -> Image.Image:
    """Create a solid-color test image."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


class FakePromptGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.last_image_size = None

    def generate_prompt_from_image(self, image: Image.Image) -> str:
        self.last_image_size = image.size
        return "cat"


class FakeDiffuser:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_prompt = None

    def generate(self, prompt: str) -> Image.Image:
        self.last_prompt = prompt
        arr = np.full(
            (self.config.image_size, self.config.image_size, 3),
            (10, 20, 30),
            dtype=np.uint8,
        )
        return Image.fromarray(arr)


class FakeEvaluationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.last_current = None
        self.last_generated = None

    def propose_pixels(self, current_canvas_img: Image.Image, generated_img: Image.Image):
        self.last_current = current_canvas_img
        self.last_generated = generated_img
        return [(1, 2, (3, 4, 5)), (6, 7, (8, 9, 10))]


@contextmanager
def patched_pipeline():
    old_prompt = prompt_module.PromptGenerator
    old_diffuser = pipeline_module.DiffusionPromptPipeline
    old_eval = pipeline_module.EvaluationPipeline

    prompt_module.PromptGenerator = FakePromptGenerator
    pipeline_module.DiffusionPromptPipeline = FakeDiffuser
    pipeline_module.EvaluationPipeline = FakeEvaluationPipeline
    try:
        yield
    finally:
        prompt_module.PromptGenerator = old_prompt
        pipeline_module.DiffusionPromptPipeline = old_diffuser
        pipeline_module.EvaluationPipeline = old_eval


def test_local_region_pipeline():
    """Validate full pipeline: classifier → diffuser → evaluator."""
    print("Test 1: LocalRegionPipeline data flow")
    print("-" * 50)

    config = PipelineConfig(image_size=32, top_x_proposals=2)
    canvas_region = create_test_canvas_region(width=32, height=32, color=(0, 0, 0))

    with patched_pipeline():
        pipeline = LocalRegionPipeline(config)
        prompt, generated_img, proposals = pipeline.process_canvas_region_to_proposals(
            canvas_region
        )

    assert prompt == "cat", "Classifier should return prompt"
    assert generated_img.size == (32, 32), "Generated image should match config size"
    assert proposals == [(1, 2, (3, 4, 5)), (6, 7, (8, 9, 10))], "Evaluator proposals mismatch"

    print("  ✓ Test passed!\n")


def test_prompt_against_canvas():
    """Validate prompt → diffuser → evaluator path."""
    print("Test 2: process_prompt_against_canvas")
    print("-" * 50)

    config = PipelineConfig(image_size=16, top_x_proposals=2)
    canvas_region = create_test_canvas_region(width=16, height=16, color=(128, 128, 128))

    with patched_pipeline():
        pipeline = LocalRegionPipeline(config)
        proposals = pipeline.process_prompt_against_canvas("a blue cube", canvas_region)

    assert proposals, "Should return proposals"
    assert proposals[0] == (1, 2, (3, 4, 5)), "Proposal format mismatch"

    print("  ✓ Test passed!\n")


def test_convenience_function():
    """Validate the one-shot convenience function."""
    print("Test 3: generate_and_evaluate")
    print("-" * 50)

    canvas_region = create_test_canvas_region(width=8, height=8, color=(10, 10, 10))
    with patched_pipeline():
        proposals = generate_and_evaluate(
            "a blue gradient",
            canvas_region,
            image_size=8,
            evaluator_device="cpu",
            top_x_proposals=2,
        )

    assert proposals == [(1, 2, (3, 4, 5)), (6, 7, (8, 9, 10))]

    print("  ✓ Test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Pipeline Integration Tests")
    print("=" * 50 + "\n")

    try:
        test_local_region_pipeline()
        test_prompt_against_canvas()
        test_convenience_function()

        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error:\n{e}")
        import traceback

        traceback.print_exc()
