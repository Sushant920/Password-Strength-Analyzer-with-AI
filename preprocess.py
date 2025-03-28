import torch

# Read and clean the dataset
with open('data/rockyou.txt', 'r', encoding='utf-8', errors='ignore') as f:
    passwords = [line.strip() for line in f if line.strip()]

# Define sequence length
seq_length = 30

# Create vocabulary
all_chars = set(''.join(passwords))
pad_char = '<PAD>'
all_chars.add(pad_char)
vocab_size = len(all_chars)
char_to_idx = {c: i for i, c in enumerate(all_chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Encode passwords
encoded_passwords = []
for pw in passwords:
    encoded = [char_to_idx[c] for c in pw]
    if len(encoded) < seq_length:
        encoded += [char_to_idx[pad_char]] * (seq_length - len(encoded))
    else:
        encoded = encoded[:seq_length]
    encoded_passwords.append(encoded)

# Save encoded data (optional)
torch.save({
    'encoded_passwords': encoded_passwords,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'seq_length': seq_length,
    'vocab_size': vocab_size
}, 'data/processed_data.pth')

print(f"Processed {len(passwords)} passwords. Vocab size: {vocab_size}")