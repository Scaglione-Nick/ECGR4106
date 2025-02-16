import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Modified AlexNet model with dropout
class ModifiedAlexNetWithDropout(nn.Module):
    def __init__(self, num_classes, use_dropout=True):
        super(ModifiedAlexNetWithDropout, self).__init__()
        self.use_dropout = use_dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # First conv layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pool layer
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # Second conv layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 256, kernel_size=3, padding=1),  # Third conv layer
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Final pool layer
        )
        
        # Update the first Linear layer input size to match the flattened tensor
        self.classifier = nn.Sequential(
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Linear(256 * 4 * 4, 4096),  # Correct input size to 4096 (256 * 4 * 4)
            nn.ReLU(inplace=True),
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        #print(f"Shape before flattening: {x.shape}")  # Shape should be [64, 256, 4, 4]  #used for debugging
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x





# Calculate the total number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train model and track losses/accuracy
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    training_loss = []
    validation_loss = []
    validation_accuracy = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # Track training loss and accuracy
        training_loss.append(running_train_loss / len(train_loader))
        train_accuracy = 100. * correct_train / total_train

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        # Track validation loss and accuracy
        validation_loss.append(running_val_loss / len(test_loader))
        val_accuracy = 100. * correct_val / total_val
        validation_accuracy.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {running_train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {running_val_loss / len(test_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return training_loss, validation_loss, validation_accuracy


# Load CIFAR-10 or CIFAR-100 dataset
def load_data(dataset='CIFAR-10'):
    if dataset == 'CIFAR-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset == 'CIFAR-100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError("Dataset should be either 'CIFAR-10' or 'CIFAR-100'")
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    return train_loader, test_loader, num_classes


# Example: CIFAR-10 with dropout
train_loader, test_loader, num_classes = load_data('CIFAR-10')
model_with_dropout = ModifiedAlexNetWithDropout(num_classes=num_classes, use_dropout=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_with_dropout.parameters(), lr=0.001)

# Train the model with dropout
training_loss, validation_loss, validation_accuracy = train_and_evaluate(
    model_with_dropout, train_loader, test_loader, criterion, optimizer, epochs=5
)

# Model parameters count
params_count_with_dropout = count_parameters(model_with_dropout)
print(f"Number of parameters (with dropout): {params_count_with_dropout}")

# You can repeat the same for a model without dropout by setting `use_dropout=False`:
model_without_dropout = ModifiedAlexNetWithDropout(num_classes=num_classes, use_dropout=False)
optimizer = optim.Adam(model_without_dropout.parameters(), lr=0.001)

# Train the model without dropout
training_loss_no_dropout, validation_loss_no_dropout, validation_accuracy_no_dropout = train_and_evaluate(
    model_without_dropout, train_loader, test_loader, criterion, optimizer, epochs=5
)

# Model parameters count without dropout
params_count_no_dropout = count_parameters(model_without_dropout)
print(f"Number of parameters (without dropout): {params_count_no_dropout}")
