"""
Integration tests for the pipeline module.

Tests the composition of Diffuser → Evaluator → Proposals.
"""

import numpy as np
from PIL import Image
from agents.pipeline import (
    PipelineConfig,
    DiffusionPromptPipeline,
    EvaluationPipeline,
    LocalRegionPipeline,
    generate_and_evaluate,
)


def create_test_canvas_region(
    width: int = 128, height: int = 128, color: tuple = (128, 128, 128)
) -> Image.Image:
    """Create a solid-color test image."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def test_diffusion_pipeline():
    """Test that diffuser generates images of correct size."""
    print("Test 1: DiffusionPromptPipeline")
    print("-" * 50)

    config = PipelineConfig(image_size=128)
    diffuser = DiffusionPromptPipeline(config)

    prompt = "a simple red square"
    image = diffuser.generate(prompt)

    print(f"  Prompt: '{prompt}'")
    print(f"  Generated image size: {image.size}")
    print(f"  Image mode: {image.mode}")

    assert image.size == (128, 128), f"Expected (128, 128), got {image.size}"
    assert image.mode == "RGB", f"Expected RGB, got {image.mode}"

    print("  ✓ Test passed!\n")
    return image


def test_evaluation_pipeline(generated_img: Image.Image):
    """Test that evaluator produces proposals from two images."""
    print("Test 2: EvaluationPipeline")
    print("-" * 50)

    config = PipelineConfig(image_size=128, top_x_proposals=5)
    evaluator = EvaluationPipeline(config)

    # Create a contrasting canvas region
    canvas_region = create_test_canvas_region(width=128, height=128, color=(50, 50, 50))

    proposals = evaluator.propose_pixels(canvas_region, generated_img)

    print(f"  Canvas region: solid gray (50, 50, 50)")
    print(f"  Generated image: from diffuser")
    print(f"  Number of proposals: {len(proposals)}")
    print(f"  Proposal format: (x, y, (r, g, b))")
    if proposals:
        print(f"  First 3 proposals:")
        for x, y, rgb in proposals[:3]:
            print(f"    ({x}, {y}) → RGB{rgb}")

    assert len(proposals) <= 5, f"Expected ≤5 proposals, got {len(proposals)}"
    assert all(len(p) == 3 for p in proposals), "Each proposal should be (x, y, rgb)"
    assert all(
        len(rgb) == 3 for _, _, rgb in proposals
    ), "RGB should be (r, g, b)"

    print("  ✓ Test passed!\n")
    return proposals


def test_local_region_pipeline():
    """Test end-to-end pipeline: prompt → image → proposals."""
    print("Test 3: LocalRegionPipeline (end-to-end)")
    print("-" * 50)

    config = PipelineConfig(image_size=128, top_x_proposals=5)
    pipeline = LocalRegionPipeline(config)

    prompt = "a bright yellow circle"
    canvas_region = create_test_canvas_region(width=128, height=128, color=(0, 0, 0))

    proposals = pipeline.process_prompt_against_canvas(prompt, canvas_region)

    print(f"  Prompt: '{prompt}'")
    print(f"  Canvas: solid black (0, 0, 0)")
    print(f"  Generated proposals: {len(proposals)}")

    assert isinstance(proposals, list), "Should return a list of proposals"
    assert len(proposals) > 0, "Should generate at least one proposal"

    print("  ✓ Test passed!\n")


def test_convenience_function():
    """Test the one-shot convenience function."""
    print("Test 4: Convenience function (generate_and_evaluate)")
    print("-" * 50)

    prompt = "a blue gradient"
    canvas_region = create_test_canvas_region(width=128, height=128, color=(200, 200, 200))

    proposals = generate_and_evaluate(
        prompt,
        canvas_region,
        image_size=128,
        evaluator_device="cpu",
        top_x_proposals=3,
    )

    print(f"  Prompt: '{prompt}'")
    print(f"  Proposals returned: {len(proposals)}")

    assert isinstance(proposals, list), "Should return a list"
    assert len(proposals) <= 3, "Should respect top_x_proposals"

    print("  ✓ Test passed!\n")


def test_data_flow():
    """Test that data flows correctly through the pipeline."""
    print("Test 5: Data flow and composition")
    print("-" * 50)

    # Create a pipeline with specific config
    config = PipelineConfig(image_size=128, top_x_proposals=5)
    pipeline = LocalRegionPipeline(config)

    # Test prompt-only
    prompt = "test image"
    img = pipeline.process_prompt_to_image(prompt)
    assert img.size == (128, 128), "Image size should match config"
    print(f"  ✓ Diffuser output size correct: {img.size}")

    # Test full pipeline
    canvas = create_test_canvas_region()
    proposals = pipeline.process_prompt_against_canvas(prompt, canvas)
    assert isinstance(proposals, list), "Should return proposals"
    print(f"  ✓ Evaluator output format correct: {len(proposals)} proposals")

    # Verify proposal format
    if proposals:
        x, y, rgb = proposals[0]
        assert isinstance(x, (int, np.integer)), "x should be int"
        assert isinstance(y, (int, np.integer)), "y should be int"
        assert isinstance(rgb, tuple) and len(rgb) == 3, "rgb should be 3-tuple"
        print(f"  ✓ Proposal format correct: ({x}, {y}, {rgb})")

    print("  ✓ Test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Pipeline Integration Tests")
    print("=" * 50 + "\n")

    try:
        gen_img = test_diffusion_pipeline()
        test_evaluation_pipeline(gen_img)
        test_local_region_pipeline()
        test_convenience_function()
        test_data_flow()

        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error:\n{e}")
        import traceback

        traceback.print_exc()
