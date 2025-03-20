import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Data Loading Function
def load_data_from_txt(file_path):
    english_sentences = []
    french_sentences = []

    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
        
        # Join all lines into one string (to handle multiline tuples)
        content = ''.join(lines)
        
        # Regular expression to find all English-French pairs
        pattern = r'\("([^"]+)", "([^"]+)"\)'
        
        # Find all matching pairs (English, French)
        matches = re.findall(pattern, content)

        for match in matches:
            english_sentences.append(match[0])  # English sentence
            french_sentences.append(match[1])   # French sentence

    return english_sentences, french_sentences

english_sentences, french_sentences = load_data_from_txt('E2F_dataset.txt')

# Tokenizing the text
eng_tokenizer = Tokenizer()
fr_tokenizer = Tokenizer()

eng_tokenizer.fit_on_texts(english_sentences)
fr_tokenizer.fit_on_texts(french_sentences)
fr_tokenizer.word_index['<start>'] = len(fr_tokenizer.word_index) + 1
fr_tokenizer.word_index['<end>'] = len(fr_tokenizer.word_index) + 2

# Create input sequences and output sequences
encoder_input_sequences = eng_tokenizer.texts_to_sequences(english_sentences)
decoder_input_sequences = fr_tokenizer.texts_to_sequences(french_sentences)

# Pad sequences to ensure equal length
encoder_input_sequences = pad_sequences(encoder_input_sequences, padding='post')
decoder_input_sequences = pad_sequences(decoder_input_sequences, padding='post')

# Get the maximum lengths and vocabulary sizes
max_encoder_seq_length = max([len(seq) for seq in encoder_input_sequences])
max_decoder_seq_length = max([len(seq) for seq in decoder_input_sequences])

num_encoder_tokens = len(eng_tokenizer.word_index) + 1
num_decoder_tokens = len(fr_tokenizer.word_index) + 1

# Model Architecture
from tensorflow.keras import layers, models

latent_dim = 256  # Latent space dimension

# Encoder
encoder_inputs = layers.Input(shape=(max_encoder_seq_length,))
encoder_embedding = layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_gru = layers.GRU(latent_dim, return_state=True)
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)


# Decoder
decoder_inputs = layers.Input(shape=(max_decoder_seq_length,))
decoder_embedding = layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_gru = layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)

# Dense layer for output prediction
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)



# Full Model
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
decoder_output_sequences = np.zeros_like(decoder_input_sequences)
decoder_output_sequences[:, :-1] = decoder_input_sequences[:, 1:]

X_train = encoder_input_sequences
y_train = decoder_output_sequences
X_val = encoder_input_sequences
y_val = decoder_output_sequences

history = model.fit([X_train, y_train], y_train, validation_data=([X_val, y_val], y_val), epochs=100, batch_size=64)

# Report loss and accuracy
print("Training loss:", history.history['loss'][-1])
print("Validation loss:", history.history['val_loss'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])



encoder_model = models.Model(encoder_inputs, encoder_state)
# Decoder Model for Inference
decoder_state_input = layers.Input(shape=(latent_dim,))
decoder_input = layers.Input(shape=(1,))  # decoder input token at time t
decoder_emb = layers.Embedding(num_decoder_tokens, latent_dim)(decoder_input)
decoder_output, decoder_state = decoder_gru(decoder_emb, initial_state=decoder_state_input)
decoder_output = decoder_dense(decoder_output)
decoder_model = models.Model([decoder_input, decoder_state_input], [decoder_output, decoder_state])
# Evaluation - Decoding Sequence
def decode_sequence(input_seq):
    # Ensure input_seq has the correct shape (1, max_encoder_seq_length)
    input_seq = np.reshape(input_seq, (1, max_encoder_seq_length))
    
    # Encode the input as state vectors using the encoder model
    states_value = encoder_model.predict(input_seq)

    # Start the decoder with the start token
    target_seq = np.zeros((1, 1))  # Initialize target sequence with zeros
    start_token = fr_tokenizer.word_index.get('<start>', 1)  # Ensure the token is in the word index
    target_seq[0, 0] = start_token  # Set the start token
    
    # Initialize the stop condition and decoded sentence
    stop_condition = False
    decoded_sentence = ''
    
    # Loop to generate words until we get the <end> token or max length is reached
    while not stop_condition:
        # Pass both the target sequence and the decoder state to the decoder model
        output_tokens, decoder_state = decoder_model.predict([target_seq] + [states_value])
        
        # Get the most probable next word (token)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = fr_tokenizer.index_word.get(sampled_token_index, '')
        decoded_sentence += ' ' + sampled_word
        
        # Stop if we reach the <end> token or exceed max length
        if sampled_word == '<end>' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence for the next time step
        target_seq[0, 0] = sampled_token_index
        states_value = [decoder_state]  # Update the decoder state for the next step

    return decoded_sentence.strip()
# Test some samples
for seq_index in range(1):
    input_seq = X_val[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print(f"English: {english_sentences[seq_index]}")
    print(f"French (predicted): {decoded_sentence}")
    print()
