from flask import Flask, render_template, request
import torch
from evaluate import PasswordLM, compute_log_prob, categorize_strength, estimate_time_to_crack, generate_suggestions

app = Flask(__name__)

# Load vocabulary and model
vocab_data = torch.load('data/vocab.pth')
char_to_idx = vocab_data['char_to_idx']
idx_to_char = vocab_data['idx_to_char']
vocab_size = vocab_data['vocab_size']

embedding_dim = 128
hidden_dim = 256
model = PasswordLM(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('models/password_lm.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    password = request.form['password']
    log_prob = compute_log_prob(password)
    strength = categorize_strength(log_prob)
    time_to_crack = estimate_time_to_crack(log_prob)
    suggestions = generate_suggestions(password)
    reasoning = f"This password is {strength.lower()} because it {'is highly predictable' if strength == 'Weak' else 'has some predictable patterns' if strength == 'Medium' else 'is highly unpredictable'} based on patterns in leaked passwords."
    attack_type = "Dictionary attack" if strength == "Weak" else "Hybrid attack" if strength == "Medium" else "Brute-force attack"
    return render_template('result.html', password=password, strength=strength, time_to_crack=time_to_crack, suggestions=suggestions, reasoning=reasoning, attack_type=attack_type)

if __name__ == '__main__':
    app.run(debug=True)