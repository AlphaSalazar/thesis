import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import time

# Define classification mapping
class_labels = ['bruised', 'unbruised', 'green', 'yellow_green', 'yellow']
ripeness_scores = {'yellow': 3, 'yellow_green': 2, 'green': 1}
bruiseness_scores = {'bruised': 1, 'unbruised': 2}

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_labels))
model.load_state_dict(torch.load("carabao_mango_cnn.pth", map_location=device))
model.eval()
model.to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_image(image):
    """Classifies a given image and returns the predicted class."""
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return class_labels[predicted.item()]

def capture_image():
    """Captures an image from the webcam."""
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow camera to adjust
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image")
        return None
    
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Capture and classify top part
print("Capturing top part of the mango...")
top_image = capture_image()
if top_image:
    top_class = classify_image(top_image)
    print(f"Top classification: {top_class}")

# Wait for 10 seconds
time.sleep(4)

# Capture and classify bottom part
print("Capturing bottom part of the mango...")
bottom_image = capture_image()
if bottom_image:
    bottom_class = classify_image(bottom_image)
    print(f"Bottom classification: {bottom_class}")

# Compute final mango score
if top_image and bottom_image:
    ripeness_score = (ripeness_scores.get(top_class, 0) + ripeness_scores.get(bottom_class, 0)) / 2
    bruiseness_score = (bruiseness_scores.get(top_class, 0) + bruiseness_scores.get(bottom_class, 0)) / 2
    print(f"Final Ripeness Score: {ripeness_score:.1f}")
    print(f"Final Bruiseness Score: {bruiseness_score:.1f}")
