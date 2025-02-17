from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Conv2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Conv3
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Conv4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Conv5
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv6
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Conv7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Conv8
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv9
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv11
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv12
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Conv13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Pooling
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout layer for regularization
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x


# Define the transformations for CIFAR-10 or CIFAR-100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing CIFAR-10 data
])

# Choose dataset (CIFAR-10 or CIFAR-100)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
model = VGGNet(num_classes=10)  # Change num_classes to 100 for CIFAR-100
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
# Training loop
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(test_loader)
        val_accuracy = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")
# Call the function to train and evaluate
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=5)
