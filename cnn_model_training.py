import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # Windows fix: set num_workers=0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes

def train_model():
    """Train the CNN model."""
    train_loader, test_loader, num_classes = create_dataloaders()

    # Define the model (EfficientNet-B0)
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=len(num_classes))
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), "carabao_mango_cnn.pth")
    print("Model saved successfully.")

# Ensure safe multiprocessing in Windows
if __name__ == "__main__":
    train_model()
