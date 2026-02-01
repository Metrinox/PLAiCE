"""
End-to-end integration test using real output.png canvas image.

This test demonstrates the full pipeline:
1. Load a real canvas image (output.png)
2. Generate an agent prompt
3. Run diffuser to create a local image
4. Evaluate differences with the canvas
5. Get pixel-level proposals
6. Visualize results
"""

import numpy as np
from PIL import Image
from agents.pipeline import LocalRegionPipeline, PipelineConfig


def test_e2e_with_real_canvas():
    """
    Full end-to-end test using output.png as the canvas.
    
    Uses the classifier to analyze the canvas and generate a context-aware prompt.
    """
    print("\n" + "=" * 70)
    print("End-to-End Integration Test with Real Canvas (output.png)")
    print("=" * 70 + "\n")

    # Load the real canvas image
    print("Step 1: Load canvas image")
    print("-" * 70)
    try:
        canvas_img = Image.open("output.png")
        canvas_width, canvas_height = canvas_img.size
        print(f"✓ Loaded output.png")
        print(f"  Size: {canvas_width}×{canvas_height}")
        print(f"  Mode: {canvas_img.mode}")
    except FileNotFoundError:
        print("✗ output.png not found. Please ensure it exists in the workspace root.")
        return

    # Define a region to work with (e.g., center 128×128 region)
    print("\nStep 2: Extract a local region from canvas")
    print("-" * 70)
    region_size = 128
    start_x = max(0, (canvas_width - region_size) // 2)
    start_y = max(0, (canvas_height - region_size) // 2)
    end_x = min(canvas_width, start_x + region_size)
    end_y = min(canvas_height, start_y + region_size)

    # Extract the region (and pad if needed)
    canvas_region = canvas_img.crop((start_x, start_y, end_x, end_y))
    if canvas_region.size != (region_size, region_size):
        canvas_region = canvas_region.resize((region_size, region_size), Image.LANCZOS)

    print(f"✓ Extracted region from canvas")
    print(f"  Region bounds: ({start_x}, {start_y}) to ({end_x}, {end_y})")
    print(f"  Region size: {canvas_region.size}")

    # Save the canvas region for reference
    canvas_region.save("e2e_canvas_region.png")
    print(f"  Saved to: e2e_canvas_region.png")

    # Create pipeline
    print("\nStep 3: Initialize pipeline")
    print("-" * 70)
    config = PipelineConfig(
        image_size=128,
        evaluator_device="cpu",
        top_x_proposals=10,
    )
    pipeline = LocalRegionPipeline(config)
    print("✓ Pipeline initialized")
    print(f"  Image size: {config.image_size}×{config.image_size}")
    print(f"  Top proposals: {config.top_x_proposals}")

    # Generate agent prompt (in a real scenario, the agent would generate this)
    print("\nStep 4: Analyze canvas with classifier and generate prompt")
    print("-" * 70)
    print(f"  Analyzing canvas region using ViT classifier...")
    prompt, generated_img, proposals = pipeline.process_canvas_region_to_proposals(
        canvas_region
    )
    print(f"✓ Classifier-generated prompt:")
    print(f"  '{prompt}'")

    # Run diffuser to generate local image
    print("\nStep 5: Diffuser - Generate image from prompt")
    print("-" * 70)
    print(f"  Image already generated in Step 4")
    print(f"✓ Generated image details:")
    print(f"  Size: {generated_img.size}")
    print(f"  Mode: {generated_img.mode}")

    # Save generated image
    generated_img.save("e2e_generated_image.png")
    print(f"  Saved to: e2e_generated_image.png")

    # Evaluate differences
    print("\nStep 6: Evaluator - Compare canvas region with generated image")
    print("-" * 70)
    print(f"  Evaluation already complete in Step 4")
    print(f"✓ Evaluation summary:")
    print(f"  Proposals generated: {len(proposals)}")

    # Display proposals
    print("\nStep 7: Pixel-level proposals")
    print("-" * 70)
    if proposals:
        print(f"  Top proposals (x, y, RGB):")
        for i, (x, y, rgb) in enumerate(proposals[:5], 1):
            r, g, b = rgb
            print(f"    {i}. ({x:3d}, {y:3d}) → RGB({r:3d}, {g:3d}, {b:3d})")
        if len(proposals) > 5:
            print(f"    ... and {len(proposals) - 5} more")
    else:
        print(f"  No proposals generated")

    # Create visualization: highlight proposal locations on canvas
    print("\nStep 8: Create visualization")
    print("-" * 70)
    vis_img = canvas_region.copy()
    vis_array = np.array(vis_img)

    # Mark proposal locations with a bright accent color
    for x, y, (r, g, b) in proposals[:10]:  # Limit to first 10 for visibility
        # Draw a small circle around the proposal location
        x, y = int(x), int(y)
        radius = 3
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < region_size and 0 <= ny < region_size:
                        # Use cyan as the highlight color
                        vis_array[ny, nx] = [0, 255, 255]

    vis_img = Image.fromarray(vis_array)
    vis_img.save("e2e_proposals_visualization.png")
    print(f"✓ Visualization created")
    print(f"  Saved to: e2e_proposals_visualization.png")
    print(f"  (Cyan circles mark proposed pixel locations)")

    # Summary
    print("\nStep 9: Summary")
    print("-" * 70)
    print(f"✓ End-to-end pipeline execution complete!")
    print(f"\n  Classifier-based prompt generation:")
    print(f"    Input: Canvas region (128×128 image)")
    print(f"    Process: ViT feature extraction + analysis")
    print(f"    Output: Contextual prompt for diffuser")
    print(f"\n  Generated files:")
    print(f"    - e2e_canvas_region.png (extracted region from canvas)")
    print(f"    - e2e_generated_image.png (output from diffuser)")
    print(f"    - e2e_proposals_visualization.png (proposals marked on canvas)")
    print(f"\n  Pipeline stats:")
    print(f"    - Canvas region size: 128×128")
    print(f"    - Proposals generated: {len(proposals)}")
    print(f"    - Evaluator device: {config.evaluator_device}")
    print(f"\n  Generated prompt:")
    print(f"    '{prompt}'")

    print("\n" + "=" * 70)
    print("✓ All steps completed successfully!")
    print("=" * 70 + "\n")

    return {
        "canvas_region": canvas_region,
        "generated_img": generated_img,
        "proposals": proposals,
        "visualization": vis_img,
    }


if __name__ == "__main__":
    try:
        result = test_e2e_with_real_canvas()
        if result:
            print("Test completed. Check the generated PNG files in the workspace root.")
    except Exception as e:
        print(f"\n✗ Test failed with error:\n{e}")
        import traceback

        traceback.print_exc()
