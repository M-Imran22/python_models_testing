from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/exp13/weights/best.pt')

# Set the path to your test images
test_images = './test_images/'  # Replace with your image or folder path

# Run predictions
results = model.predict(source=test_images, save=True, save_txt=True)

# Print the results (optional)
for result in results:
    print(f"Image: {result.path}")
    print(f"Predicted Class: {result.probs.top1}")
    print(f"Confidence: {result.probs.top1conf:.2f}")
