# Password Strength Analyzer with AI

An intelligent tool built with Generative AI (GenAI) and Machine Learning (ML) to analyze password strength, estimate "time-to-crack," and provide improvement suggestions. This project uses an LSTM model trained on password patterns to predict vulnerability and offers real-time feedback via a Flask web interface.

## Features
- **Password Analysis**: Evaluates strength beyond basic metrics using GenAI.
- **Time-to-Crack**: Estimates cracking time based on entropy (assuming 1B guesses/sec).
- **Suggestions**: Provides interpretable suggestions to improve weak passwords.
- **Feedback**: Explains why a password is weak/strong and likely attack types (e.g., brute-force).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Sushant920/Password-Strength-Analyzer-with-AI.git
cd Password-Strength-Analyzer-with-AI
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install flask torch scikit-learn
```

### 4. Obtain the RockYou Dataset
The project requires the `rockyou.txt` file for training. Due to its size (133 MB), it’s not included in the repo.

**Where to Get It:**
- Search for "RockYou password list" on GitHub, Kaggle, or security research sites.
- Example: SecLists on GitHub (look for `rockyou.txt.tar.gz`).

**Legal Note**: Use this dataset for educational purposes only, respecting local laws and ethical guidelines.

### 5. Generate Data and Model Files
The `data/processed_data.pth`, `data/vocab.pth`, and `models/password_lm.pth` files are excluded from the repo. Generate them by training the model:

**Train the Model:**
```bash
python train_model.py
```
This processes `rockyou.txt` in chunks, creates the vocabulary (`data/vocab.pth`), and saves the trained LSTM model (`models/password_lm.pth`).

**Notes:**
- Training may take time depending on your hardware and dataset size.
- Checkpoints (`models/password_lm_checkpoint.pth`) are saved during training to resume if interrupted.

### 6. Run the Application
```bash
python app.py
```
Open your browser and visit `http://127.0.0.1:5000`.

Enter a password to analyze its strength, see time-to-crack, and get suggestions.

## Project Structure
```
Password-Strength-Analyzer-with-AI/
├── app.py                  # Flask application
├── evaluate.py             # Password evaluation logic
├── train_model.py          # Model training script
├── data/                   # Dataset storage (not in repo)
│   ├── rockyou.txt         # (Add manually)
│   ├── processed_data.pth  # (Generated)
│   └── vocab.pth           # (Generated)
├── models/                 # Model storage (not in repo)
│   ├── password_lm.pth     # (Generated)
│   └── password_lm_checkpoint.pth  # (Generated)
├── templates/              # HTML templates
│   ├── index.html
│   └── result.html
├── static/                 # CSS (optional)
│   └── style.css
└── .gitignore              # Excludes large files
```

## Example Usage
### Input: `Summer2024`
**Output:**
- **Strength:** Medium/Weak
- **Time-to-Crack:** e.g., "hours"
- **Suggestions:** e.g., "Replace 'm' with '#', add a digit"
- **Reasoning:** "Has predictable patterns from leaked passwords"
- **Attack Type:** "Dictionary attack"

### Input: `5uMM3r#2024*Q`
**Output:**
- **Strength:** Strong
- **Time-to-Crack:** e.g., "billions of days"
- **Suggestions:** "Consider adding more complexity"

## Troubleshooting
- **Memory Issues:** Reduce `chunk_size` or `batch_size` in `train_model.py`.
- **Missing Files:** Ensure `rockyou.txt` is in `data/` and run `train_model.py`.
- **Port Conflict:** Change the Flask port in `app.py` (e.g., `app.run(port=5001)`).
