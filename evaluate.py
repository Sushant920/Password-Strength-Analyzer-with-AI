import torch
import torch.nn as nn
import math
import random

# Load vocabulary
vocab_data = torch.load('data/vocab.pth')
char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
vocab_size = vocab_data['vocab_size']
seq_length = 30

# Define model class
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

# Load trained model
embedding_dim = 128
hidden_dim = 256
model = PasswordLM(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('models/password_lm.pth'))
model.eval()

def compute_log_prob(password):
    encoded = [char_to_idx.get(c, char_to_idx['<PAD>']) for c in password]
    actual_length = len(encoded)
    if actual_length == 0:
        return -float('inf')
    if actual_length < seq_length:
        encoded += [char_to_idx['<PAD>']] * (seq_length - actual_length)
    else:
        encoded = encoded[:seq_length]
        actual_length = seq_length
    input_seq = encoded[:-1]
    target_seq = encoded[1:]
    input_tensor = torch.tensor([input_seq], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        log_probs = nn.functional.log_softmax(outputs, dim=2)
        log_prob = 0
        for t in range(actual_length - 1):
            log_prob += log_probs[0, t, target_seq[t]].item()
    return log_prob

def categorize_strength(log_prob):
    if log_prob > -10:
        return "Weak"
    elif log_prob > -20:
        return "Medium"
    else:
        return "Strong"

def estimate_time_to_crack(log_prob):
    entropy = -log_prob / math.log(2)
    cracking_speed = 1e9  # 1 billion guesses/sec
    time_seconds = 2 ** entropy / cracking_speed
    if time_seconds < 60:
        return f"{time_seconds:.2f} seconds"
    elif time_seconds < 3600:
        return f"{time_seconds / 60:.2f} minutes"
    elif time_seconds < 86400:
        return f"{time_seconds / 3600:.2f} hours"
    else:
        return f"{time_seconds / 86400:.2f} days"

def generate_suggestions(password, threshold=0.5):
    encoded = [char_to_idx.get(c, char_to_idx['<PAD>']) for c in password]
    if not encoded:
        return ["Enter a valid password."]
    if len(encoded) < seq_length:
        encoded += [char_to_idx['<PAD>']] * (seq_length - len(encoded))
    else:
        encoded = encoded[:seq_length]
    input_seq = encoded[:-1]
    input_tensor = torch.tensor([input_seq], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = nn.functional.softmax(outputs, dim=2)
        suggestions = []
        for t in range(len(password) - 1):
            if t + 1 < len(password):
                actual_idx = char_to_idx.get(password[t + 1], char_to_idx['<PAD>'])
                prob = probs[0, t, actual_idx].item()
                if prob > threshold:
                    low_prob_chars = [idx for idx, p in enumerate(probs[0, t]) if p < 0.1 and idx != char_to_idx['<PAD>']]
                    if low_prob_chars:
                        new_idx = random.choice(low_prob_chars)
                        new_char = idx_to_char[new_idx]
                        suggestions.append(f"Replace '{password[t + 1]}' with '{new_char}' at position {t + 2}")
    if len(password) < 8:
        suggestions.append("Increase length to at least 8 characters.")
    if not any(c.isdigit() for c in password):
        suggestions.append("Add at least one digit.")
    if not any(c in "!@#$%^&*" for c in password):
        suggestions.append("Add a special character (e.g., !, @, #).")
    return suggestions if suggestions else ["Password is strong, but consider adding more complexity."]