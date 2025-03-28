import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import os

# Generator to load data in chunks
def password_generator(file_path, chunk_size=10000):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        chunk = []
        for line in f:
            pw = line.strip()
            if pw:
                chunk.append(pw)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

# Preprocess a chunk
def preprocess_chunk(chunk, char_to_idx, seq_length, pad_char='<PAD>'):
    encoded_passwords = []
    for pw in chunk:
        encoded = [char_to_idx[c] for c in pw if c in char_to_idx]
        if len(encoded) < seq_length:
            encoded += [char_to_idx[pad_char]] * (seq_length - len(encoded))
        else:
            encoded = encoded[:seq_length]
        encoded_passwords.append(encoded)
    return encoded_passwords

# Custom Dataset class
class PasswordDataset(Dataset):
    def __init__(self, encoded_passwords):
        self.inputs = torch.tensor([pw[:-1] for pw in encoded_passwords], dtype=torch.long)
        self.targets = torch.tensor([pw[1:] for pw in encoded_passwords], dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Define model
class PasswordLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# Load or create vocabulary
def load_or_create_vocab(file_path, seq_length=30):
    vocab_file = 'data/vocab.pth'
    if os.path.exists(vocab_file):
        vocab_data = torch.load(vocab_file)
        return vocab_data['char_to_idx'], vocab_data['idx_to_char'], vocab_data['vocab_size']
    all_chars = set()
    pad_char = '<PAD>'
    all_chars.add(pad_char)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pw = line.strip()
            if pw:
                all_chars.update(pw)
    char_to_idx = {c: i for i, c in enumerate(all_chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(all_chars)
    torch.save({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char, 'vocab_size': vocab_size}, vocab_file)
    return char_to_idx, idx_to_char, vocab_size

# Training function
def train_chunk(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# Main execution
file_path = 'data/rockyou.txt'
seq_length = 30
chunk_size = 10000  # Adjust based on your Mac's capacity
batch_size = 32     # Smaller batch size to reduce memory usage
num_epochs_per_chunk = 1  # Train 1 epoch per chunk, repeat for all chunks
embedding_dim = 128
hidden_dim = 256

# Load vocabulary
char_to_idx, idx_to_char, vocab_size = load_or_create_vocab(file_path)
print(f"Vocabulary size: {vocab_size}")

# Initialize model
model = PasswordLM(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load checkpoint if exists
checkpoint_path = 'models/password_lm_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_chunk = checkpoint['chunk_idx'] + 1
    print(f"Resuming from chunk {start_chunk}")
else:
    start_chunk = 0

# Process and train on chunks
gen = password_generator(file_path, chunk_size)
for chunk_idx, chunk in enumerate(gen):
    if chunk_idx < start_chunk:
        continue
    print(f"Processing chunk {chunk_idx + 1} with {len(chunk)} passwords")
    encoded_passwords = preprocess_chunk(chunk, char_to_idx, seq_length)
    dataset = PasswordDataset(encoded_passwords)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train on this chunk
    for epoch in range(num_epochs_per_chunk):
        train_loss = train_chunk(model, train_loader, criterion, optimizer)
        print(f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'chunk_idx': chunk_idx
    }, checkpoint_path)

# Save final model
torch.save(model.state_dict(), 'models/password_lm.pth')
print("Final model saved to models/password_lm.pth")