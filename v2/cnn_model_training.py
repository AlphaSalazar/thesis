import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define dataset directory
data_dir = "carabao_mango_dataset"

def create_dataloaders():
    """Creates train and test dataloaders."""
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes

def train_model():
    """Train the CNN model and evaluate performance."""
    train_loader, test_loader, class_names = create_dataloaders()

    # Define the model (EfficientNet-B0)
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(class_names))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Wrap the DataLoader with tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), "carabao_mango_cnn.pth")
    print("Model saved successfully.")

    # Evaluate the model
    evaluate_model(model, test_loader, class_names)

def evaluate_model(model, test_loader, class_names):
    """Evaluate the trained model and display performance metrics."""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute precision, recall, F1-score
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Ensure safe multiprocessing in Windows
if __name__ == "__main__":
    train_model()
