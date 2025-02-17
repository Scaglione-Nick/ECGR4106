import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models

# 1. **Download and Preprocess CIFAR-10 and CIFAR-100 Data**

# Define the transform (normalization specific to CIFAR datasets)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR normalization
])

# Load CIFAR-10 dataset
trainset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=64, shuffle=True)

testset_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader_cifar10 = torch.utils.data.DataLoader(testset_cifar10, batch_size=64, shuffle=False)

# Load CIFAR-100 dataset (same transform)
trainset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader_cifar100 = torch.utils.data.DataLoader(trainset_cifar100, batch_size=64, shuffle=True)

testset_cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader_cifar100 = torch.utils.data.DataLoader(testset_cifar100, batch_size=64, shuffle=False)

# 2. **Modify ResNet-18 for CIFAR-10/100**

# Load pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Modify the first convolution layer (CIFAR-10 images are 32x32, RGB)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

# Modify the fully connected layer (for CIFAR-10, we have 10 classes; CIFAR-100 has 100 classes)
model.fc = nn.Linear(model.fc.in_features, 10)  # For CIFAR-10, change 10 to 100 for CIFAR-100

# 3. **Setup Device, Optimizer, Loss Function**

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. **Training and Evaluation Functions**

# Function to train the model
def train(model, trainloader, criterion, optimizer, num_epochs=5,num_classes=10):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        train_losses.append(running_loss / len(trainloader))
        val_loss, val_accuracy = evaluate(model, testloader_cifar10 if num_classes == 10 else testloader_cifar100)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_losses[-1]} - Validation Loss: {val_loss} - Validation Accuracy: {val_accuracy}%")
        
    return train_losses, val_losses, val_accuracies

# Function to evaluate the model
def evaluate(model, testloader):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(testloader)
    accuracy = 100 * correct / total
    return loss, accuracy

# 5. **Train the Model**

# Train on CIFAR-10
train_losses_cifar10, val_losses_cifar10, val_accuracies_cifar10 = train(model, trainloader_cifar10, criterion, optimizer, num_epochs=5,num_classes=10)

# 6. **Plot Training Loss, Validation Loss, and Validation Accuracy**

# Plot for CIFAR-10
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses_cifar10, label='Training Loss')
plt.plot(val_losses_cifar10, label='Validation Loss')
plt.title("Loss vs Epochs (CIFAR-10)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies_cifar10, label='Validation Accuracy')
plt.title("Validation Accuracy vs Epochs (CIFAR-10)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()

# Train on CIFAR-100 (same process as CIFAR-10)
model.fc = nn.Linear(model.fc.in_features, 100)  # Change the output layer for CIFAR-100
train_losses_cifar100, val_losses_cifar100, val_accuracies_cifar100 = train(model, trainloader_cifar100, criterion, optimizer, num_epochs=5, num_classes=100)

# Plot for CIFAR-100
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses_cifar100, label='Training Loss')
plt.plot(val_losses_cifar100, label='Validation Loss')
plt.title("Loss vs Epochs (CIFAR-100)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies_cifar100, label='Validation Accuracy')
plt.title("Validation Accuracy vs Epochs (CIFAR-100)")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.show()
