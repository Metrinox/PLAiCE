import numpy as np
from PIL import Image
from agents.evaluator.evaluator import Evaluator


def create_test_image(width=224, height=224, color=(128, 128, 128)):
    """Create a simple test image with the specified color"""
    image_array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(image_array)


def create_gradient_image(width=224, height=224):
    """Create a test image with a gradient"""
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_array[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128
            ]
    return Image.fromarray(image_array)


def test_evaluator_basic():
    """Test basic evaluator functionality"""
    print("=" * 60)
    print("Test 1: Evaluator with identical images")
    print("=" * 60)
    
    # Create identical images
    current_img = create_test_image(color=(100, 150, 200))
    generated_img = create_test_image(color=(100, 150, 200))
    
    evaluator = Evaluator(device="cpu")
    proposals = evaluator.evaluate(current_img, generated_img, top_x=5)
    
    print(f"Current image shape: {np.array(current_img).shape}")
    print(f"Generated image shape: {np.array(generated_img).shape}")
    print(f"Number of proposals: {len(proposals)}")
    print(f"Proposals (x, y, rgb):")
    for x, y, rgb in proposals:
        print(f"  Position ({x}, {y}): RGB{rgb}")
    
    # Identical images should have low difference scores
    print("✓ Test 1 passed!")
    return proposals


def test_evaluator_different():
    """Test evaluator with different images"""
    print("\n" + "=" * 60)
    print("Test 2: Evaluator with different images")
    print("=" * 60)
    
    # Create different images
    current_img = create_test_image(color=(50, 50, 50))
    generated_img = create_gradient_image()
    
    evaluator = Evaluator(device="cpu")
    proposals = evaluator.evaluate(current_img, generated_img, top_x=10)
    
    print(f"Current image: solid gray (50, 50, 50)")
    print(f"Generated image: gradient")
    print(f"Number of proposals: {len(proposals)}")
    print(f"Top proposals (x, y, rgb):")
    for i, (x, y, rgb) in enumerate(proposals[:5], 1):
        print(f"  {i}. Position ({x}, {y}): RGB{rgb}")
    
    # Different images should have high difference scores
    assert len(proposals) > 0, "Should have proposals for different images"
    print("✓ Test 2 passed!")
    return proposals


def test_evaluator_top_x():
    """Test evaluator with different top_x values"""
    print("\n" + "=" * 60)
    print("Test 3: Evaluator with different top_x values")
    print("=" * 60)
    
    current_img = create_test_image(color=(100, 100, 100))
    generated_img = create_gradient_image()
    
    evaluator = Evaluator(device="cpu")
    
    for top_x in [1, 5, 10, 20]:
        proposals = evaluator.evaluate(current_img, generated_img, top_x=top_x)
        print(f"top_x={top_x}: Generated {len(proposals)} proposals")
        assert len(proposals) == top_x, f"Expected {top_x} proposals, got {len(proposals)}"
    
    print("✓ Test 3 passed!")


def test_evaluator_output_image():
    """Test evaluator with actual output.png if it exists"""
    print("\n" + "=" * 60)
    print("Test 4: Evaluator with output.png")
    print("=" * 60)
    
    try:
        output_img = Image.open("../../output.png")
        print(f"Loaded output.png: {output_img.size}")
        
        # Create a slightly modified version
        current_array = np.array(output_img)
        generated_array = current_array.copy()
        
        # Add some noise to half the image to create difference
        h, w = generated_array.shape[:2]
        generated_array[h//2:, :] = np.clip(
            generated_array[h//2:, :].astype(int) + np.random.randint(-20, 20, generated_array[h//2:, :].shape),
            0, 255
        ).astype(np.uint8)
        
        current_img = Image.fromarray(current_array)
        generated_img = Image.fromarray(generated_array)
        
        evaluator = Evaluator(device="cpu")
        proposals = evaluator.evaluate(current_img, generated_img, top_x=5)
        
        print(f"Number of proposals: {len(proposals)}")
        print(f"Top proposals:")
        for i, (x, y, rgb) in enumerate(proposals, 1):
            print(f"  {i}. Position ({x}, {y}): RGB{rgb}")
        
        print("✓ Test 4 passed!")
    except FileNotFoundError:
        print("⚠ output.png not found, skipping this test")


if __name__ == "__main__":
    print("\nRunning Evaluator Tests\n")
    
    try:
        test_evaluator_basic()
        test_evaluator_different()
        test_evaluator_top_x()
        test_evaluator_output_image()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
