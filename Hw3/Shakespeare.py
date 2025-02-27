import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define your custom Dataset class (if needed)
class ShakespeareDataset(Dataset):
    def __init__(self, text, sequence_length):
        self.text = text
        self.sequence_length = sequence_length
        self.chars = sorted(set(text))  # All unique characters
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}  # Mapping chars to indices
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}  # Reverse mapping

    def __len__(self):
        return len(self.text) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = self.text[idx: idx + self.sequence_length]
        target_seq = self.text[idx + 1: idx + self.sequence_length + 1]
        
        input_tensor = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)
        
        return input_tensor, target_tensor

# Example usage: load text data (assuming you've already loaded the Shakespeare text)
with open('tiny-shakespeare.txt', 'r') as f:
    text = f.read()

sequence_length = 50
train_dataset = ShakespeareDataset(text, sequence_length)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert input to embedding
        out, _ = self.lstm(x)  # Output from LSTM
        out = out.contiguous().view(-1, out.size(2))  # Flatten to (batch_size * sequence_length, hidden_size)
        out = self.fc(out)  # Fully connected layer to output_size
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Convert input to embedding
        out, _ = self.gru(x)  # Output from GRU
        out = out.contiguous().view(-1, out.size(2))  # Flatten to (batch_size * sequence_length, hidden_size)
        out = self.fc(out)  # Fully connected layer to output_size
        return out

def train_and_evaluate(model, train_loader, epochs, optimizer, criterion, device):
    model.to(device)  # Move model to the appropriate device (CPU or GPU)
    model.train()  # Set the model to training mode
    total_loss = 0
    start_time = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_x)  # Get model outputs
            # Reshape the outputs to (batch_size * sequence_length, num_classes)
            outputs = outputs.view(-1, outputs.size(1))  # Flatten to (batch_size * sequence_length, num_classes)
            batch_y = batch_y.view(-1)  # Flatten the target tensor to (batch_size * sequence_length)
            # Calculate the loss
            loss = criterion(outputs, batch_y)
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Compute accuracy
            _, predicted = outputs.max(1)  # Get the predicted classes (the index of max probability)
            total_correct += (predicted == batch_y).sum().item()
            total_samples += batch_y.size(0)

        # Compute the epoch loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * total_correct / total_samples
        total_loss += epoch_loss

        # Print results for the current epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Calculate the total training time
    end_time = time.time()
    training_time = end_time - start_time
    return total_loss / epochs, training_time  # Return average loss and training time
# Hyperparameters for LSTM and GRU
input_size = len(train_dataset.chars)
output_size = len(train_dataset.chars)
hidden_size = 64
num_layers = 4
epochs = 10
batch_size = 64

# Choose models for comparison
lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
gru_model = GRUModel(input_size, hidden_size, num_layers, output_size)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()

lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)

# Train models
print("Training LSTM model...")
lstm_loss, lstm_time = train_and_evaluate(lstm_model, train_loader, epochs, lstm_optimizer, criterion, device)

print("\nTraining GRU model...")
gru_loss, gru_time = train_and_evaluate(gru_model, train_loader, epochs, gru_optimizer, criterion, device)

# Print results
print(f"LSTM Model Loss: {lstm_loss}, Training Time: {lstm_time} seconds")
print(f"GRU Model Loss: {gru_loss}, Training Time: {gru_time} seconds")
def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params} parameters")

print("LSTM Model size:")
print_model_size(lstm_model)

print("\nGRU Model size:")
print_model_size(gru_model)
