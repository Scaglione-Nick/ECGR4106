import numpy as np
import torch
import torch.nn as nn
import time
# Text sequence
text = """
Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology.
"""

# Prepare a set of unique characters from the text
chars = sorted(set(text))
vocab_size = len(chars)
char_to_index = {ch: i for i, ch in enumerate(chars)}
index_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text as integer indices
encoded_text = [char_to_index[ch] for ch in text]

# Define a function to prepare sequences of a given length
def one_hot_encode(sequence, vocab_size):
    # Initialize a zeros array for the one-hot encoded vectors
    one_hot = np.zeros((len(sequence), vocab_size), dtype=np.float32)
    for i, idx in enumerate(sequence):
        one_hot[i, idx] = 1
    return one_hot
def prepare_sequences(sequence_length, encoded_text, vocab_size):
    x_data, y_data = [], []
    for i in range(len(encoded_text) - sequence_length):
        x_data.append(encoded_text[i:i+sequence_length])
        y_data.append(encoded_text[i+sequence_length])
    # Convert to one-hot encoding
    x_data_one_hot = [one_hot_encode(seq, vocab_size) for seq in x_data]
    return np.array(x_data_one_hot), np.array(y_data)
# Example of sequence length and vocab size
sequence_lengths = [10, 20, 30]
vocab_size = 45  # assuming this is your vocabulary size
data = {}
for seq_len in sequence_lengths:
    x, y = prepare_sequences(seq_len, encoded_text, vocab_size)
    data[seq_len] = (x, y)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, embedding_dim=64):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use only the last time step's output
        return out
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use only the last time step's output
        return out
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use only the last time step's output
        return out
def train_and_evaluate(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    train_losses = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()  # Define the loss function here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Iterate through batches
        for i in range(0, len(x_train), batch_size):
            batch_x = torch.tensor(x_train[i:i + batch_size])
            batch_y = batch_y = torch.tensor(y_train[i:i + batch_size], dtype=torch.long)

            # Zero gradients
            model.zero_grad()

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            running_loss += loss.item()

            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == batch_y).sum().item()
            total_predictions += batch_y.size(0)

        # Calculate training loss and accuracy
        avg_train_loss = running_loss / len(x_train)
        train_losses.append(avg_train_loss)

        train_accuracy = correct_predictions / total_predictions
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(x_val))
            _, val_pred = torch.max(val_outputs, 1)
            correct = (val_pred == torch.tensor(y_val)).sum().item()
            val_accuracy = correct / len(y_val)
            val_accuracies.append(val_accuracy)
            if (epoch + 1) % 10 == 0:
                print(f"Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, val_accuracies
# Prepare data for training and validation
x_train, y_train = data[10]
x_val, y_val = data[10]  # Example for sequence length 10

# Hyperparameters
input_size = vocab_size
hidden_size = 128
output_size = vocab_size
epochs = 100
batch_size = 32

# Training and evaluation for RNN
# Example training and evaluation for RNN with batch size of 32 and using embeddings
start_time_rnn = time.time()
rnn_model = RNNModel(input_size=vocab_size, hidden_size=128, output_size=vocab_size)
train_loss_rnn, val_accuracy_rnn = train_and_evaluate(rnn_model, x_train, y_train, x_val, y_val, epochs, batch_size=32)
end_time_rnn = time.time()
execution_time1 = end_time_rnn - start_time_rnn
print(f"RNN Training Time: {execution_time1:.2f} seconds")
print("RNN Model Parameters: ")
print(sum(p.numel() for p in rnn_model.parameters() if p.requires_grad))
start_time_lstm = time.time()
# Example training and evaluation for LSTM with batch size of 32 and using embeddings
lstm_model = LSTMModel(input_size=vocab_size, hidden_size=128, output_size=vocab_size)
train_loss_lstm, val_accuracy_lstm = train_and_evaluate(lstm_model, x_train, y_train, x_val, y_val, epochs, batch_size=32)
end_time_lstm = time.time()
execution_time2 = end_time_lstm - start_time_lstm
print(f"LSTM Training Time: {execution_time2:.2f} seconds")
print("LSTM Model Parameters: ")
print(sum(p.numel() for p in lstm_model.parameters() if p.requires_grad))
start_time_gru = time.time()
# Example training and evaluation for GRU with batch size of 32 and using embeddings
gru_model = GRUModel(input_size=vocab_size, hidden_size=128, output_size=vocab_size)
train_loss_gru, val_accuracy_gru = train_and_evaluate(gru_model, x_train, y_train, x_val, y_val, epochs, batch_size=32)
end_time_gru = time.time()
execution_time3 = end_time_gru - start_time_gru
print(f"GRU Training Time: {execution_time3:.2f} seconds")
print("GRU Model Parameters: ")
print(sum(p.numel() for p in gru_model.parameters() if p.requires_grad))

