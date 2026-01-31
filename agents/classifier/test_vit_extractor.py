import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification


def test_vit_classification():
    """Test whether the ViT classifier can classify the output.png image"""
    
    # Load the image
    image_path = "../../output.png"
    image = Image.open(image_path)
    image_array = np.array(image)
    
    print(f"Image shape: {image_array.shape}")
    print(f"Image dtype: {image_array.dtype}")
    
    # Convert to RGB if needed (in case the image has alpha channel)
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
        image = Image.fromarray(image_array[:, :, :3])
    
    # Initialize the processor and classification model
    device = "cpu"  # Change to "cuda" if you have GPU
    print(f"\nInitializing ViT classification model on device: {device}")
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = model.to(device)
    model.eval()
    
    # Process the image
    print("\nProcessing image...")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    print("Classifying...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Get the predicted class label
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=-1).max().item()
    
    print(f"\nClassification Results:")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Class index: {predicted_class_idx}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Print top 5 predictions
    print(f"\nTop 5 predictions:")
    top_5_probs, top_5_indices = torch.softmax(logits, dim=-1).topk(5)
    for i, (prob, idx) in enumerate(zip(top_5_probs[0], top_5_indices[0]), 1):
        print(f"  {i}. {model.config.id2label[idx.item()]}: {prob.item():.4f}")
    
    return predicted_class, confidence


if __name__ == "__main__":
    predicted_class, confidence = test_vit_classification()
    print(f"\n✓ Classification test passed!")
    print(f"The image is classified as: {predicted_class} (confidence: {confidence:.2%})")
