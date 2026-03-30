# 🤖 GPT-2 From Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Parameters](https://img.shields.io/badge/Parameters-13M-purple?style=flat-square)

A complete, from-scratch implementation of a **GPT-2 style decoder-only Transformer Language Model** built using pure PyTorch — no Hugging Face, no pre-built model layers.

Every fundamental component — self-attention, causal masking, positional embeddings, and autoregressive sampling — is implemented manually to provide a clear, educational view into how modern LLMs work.

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)

---

## 🏗️ Architecture

| Hyperparameter | Value |
|---|---|
| Model Type | Decoder-only Transformer (GPT-2 style) |
| Vocabulary Size | 50,257 |
| Embedding Dimension | 256 |
| Context Window | 256 tokens |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| Total Parameters | ~13 Million |

---

## 📁 Project Structure

```
gpt-from-scratch/
│
├── src/
│   ├── model.py          # GPT architecture (SelfAttention, TransformerBlock, GPT)
│   ├── train.py          # Training loop with AdamW optimizer
│   ├── generate.py       # Autoregressive text generation with top-k sampling
│   └── utils.py          # Helper functions (model size, checkpointing)
│
├── data/
│   └── input.txt         # Raw training text
│
├── run_generation.py     # Entry point for generating text
├── model.pth             # Saved model weights (after training)
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/gpt-from-scratch.git
cd gpt-from-scratch
```

**2. Install dependencies**
```bash
pip install torch tiktoken
```

---

## 🚀 Usage

### Training

Add your text data to `data/input.txt`, then run:

```bash
python src/train.py
```

This will train for 5,000 steps and save the weights to `model.pth`.

### Generating Text

Once training is complete:

```bash
python run_generation.py
```

The model will start from the seed text `"The "` and generate new text using top-k sampling.

---

## 🔍 How It Works

### 1. Tokenization
Raw text from `data/input.txt` is encoded using OpenAI's `tiktoken` Byte-Pair Encoding (BPE) — the same tokenizer used by GPT-2.

### 2. Self-Attention with Causal Masking
The `SelfAttention` module computes Queries, Keys, and Values. A lower-triangular mask (`torch.tril`) ensures the model can only attend to **past and present** tokens — never future ones.

```
Token 1 → can see: [Token 1]
Token 2 → can see: [Token 1, Token 2]
Token 3 → can see: [Token 1, Token 2, Token 3]
```

### 3. Transformer Block
Each block combines:
- Multi-Head Self-Attention
- Feed-Forward Network
- LayerNorm + Residual Connections (`x = x + attention(x)`)

### 4. Training
- **Optimizer:** AdamW
- **Loss:** CrossEntropyLoss (predicting the next token)
- **Gradient Clipping:** `clip_grad_norm_` to prevent exploding gradients
- **Steps:** 5,000

### 5. Text Generation
Autoregressive generation with:
- **Temperature Scaling (0.8):** Makes the model more confident in top predictions
- **Top-K Sampling (k=50):** Restricts sampling to the 50 most likely next tokens

---

## 🛠️ Configuration

Key parameters can be adjusted in `src/train.py` and `src/model.py`:

```python
# Model config
vocab_size   = 50257
embed_dim    = 256
block_size   = 256   # context window
n_heads      = 8
n_layers     = 6

# Training config
batch_size   = 32
max_steps    = 5000
learning_rate = 3e-4

# Generation config
temperature  = 0.8
top_k        = 50
```

---

## ✨ Key Features

- **Weight Tying** — Token embedding weights are shared with the output layer, saving memory and stabilizing training
- **Causal Masking** — Prevents the model from attending to future tokens during training
- **BPE Tokenization** — Uses `tiktoken` for the standard GPT-2 vocabulary of 50,257 tokens
- **Gradient Clipping** — Prevents exploding gradients during training
- **Top-K Sampling** — Produces coherent text by limiting generation to the most probable tokens

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model architecture & training |
| `tiktoken` | Byte-Pair Encoding tokenization |

---

## 📜 License

This project is licensed under the MIT License.
