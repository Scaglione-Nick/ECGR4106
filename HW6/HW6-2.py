from transformers import SwinForImageClassification, SwinConfig
from transformers import TrainingArguments, Trainer
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Constants
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
NUM_CLASSES = 100

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Utility to freeze backbone
def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def evaluate(model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc * 100

# 🏋️ Training function
def train(model, name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    print(f"\n🟢 Training {name}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        acc = evaluate(model)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss:.4f} | Time: {epoch_time:.2f}s | Test Acc: {acc:.2f}%")
        results[name]["times"].append(epoch_time)
        results[name]["accuracies"].append(acc)

results = {
    "Swin-Tiny Pretrained": {"times": [], "accuracies": []},
    "Swin-Small Pretrained": {"times": [], "accuracies": []},
    "Swin-Tiny Scratch": {"times": [], "accuracies": []},
}

# Load pretrained Swin Tiny
model_tiny = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224",
    num_labels=100,
    ignore_mismatched_sizes=True
)

freeze_backbone(model_tiny)
train(model_tiny, "Swin-Tiny Pretrained")

# Load pretrained Swin Small
model_small = SwinForImageClassification.from_pretrained(
    "microsoft/swin-small-patch4-window7-224",
    num_labels=100,
    ignore_mismatched_sizes=True
)

freeze_backbone(model_small)
train(model_small, "Swin-Small Pretrained")

from transformers import SwinConfig
model_scratch = SwinForImageClassification(SwinConfig(num_labels=NUM_CLASSES))
train(model_scratch, "Swin-Tiny Scratch")

# 📊 Summary Table
print("\n🧾 Final Results Table:\n")
print(f"{'Model':<25} {'Time/Epoch (s)':<15} {'Final Acc (%)':<15}")
for model_name, data in results.items():
    avg_time = sum(data["times"]) / len(data["times"])
    final_acc = data["accuracies"][-1]
    print(f"{model_name:<25} {avg_time:<15.2f} {final_acc:<15.2f}")