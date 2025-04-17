import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from ptflops import get_model_complexity_info
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ViT(nn.Module):
    def __init__(self, patch_size, emb_dim, num_layers, num_heads, mlp_dim, num_classes=100, img_size=32, in_channels=3):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Encoding
        self.position_embeddings = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, emb_dim))
        
        # Transformer Encoder Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        
        # Classification Head
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # Shape: (batch_size, emb_dim, num_patches, num_patches)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, emb_dim)
        
        # Add Positional Encoding
        x = x + self.position_embeddings
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Classification Head (Use the representation of the first token)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.classifier(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim)
        )

    def forward(self, x):
        # Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x


# Hyperparameters
patch_size = 4
emb_dim = 256
num_layers = 4
num_heads = 2
mlp_dim = emb_dim * 4  # MLP hidden dimension
batch_size = 64
epochs = 20
lr = 0.001

# CIFAR-100 Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Initialization
model = ViT(patch_size=patch_size, emb_dim=emb_dim, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

# Training Loop


for epoch in range(epochs):
    model.train()
    start_time = time.time()
    
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {100 * correct/total:.2f}% | Time: {epoch_time:.2f}s")

    # Only evaluate after epoch 10
    if epoch + 1 == 10:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"[Evaluation after 10 epochs] Test Accuracy: {100 * correct / total:.2f}%")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")


# Wrap your model for FLOPs calculation
with torch.cuda.device(0 if torch.cuda.is_available() else -1):
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print(f"FLOPs (MACs): {macs}")

