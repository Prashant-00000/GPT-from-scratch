"""Interface for generating text from a trained GPT model."""

import os
import torch
import sys
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.model import GPT
from src.generate import generate


def main():
    """Load trained model and generate text."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    # Initialize model - must match training architecture exactly
    model = GPT(
        vocab_size,
        embed_size=256,
        block_size=256,
        num_heads=8,
        num_layers=6,
        dropout=0.2
    )
    
    # Load trained weights
    model_path = os.path.join(project_root, "model.pth")
    if not os.path.exists(model_path):
        print("❌ Model file 'model.pth' not found. Please train first using: python src/train.py")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"✓ Model loaded from {model_path}")
    
    # Generate text
    prompt = "The "
    start_tokens = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    
    print("\n🔄 Generating text (100 tokens)...\n")
    generated_tokens = generate(
        model,
        start_tokens,
        max_length=100,
        temperature=0.8,
        top_k=50
    )
    generated_text = enc.decode(generated_tokens[0].tolist())
    
    print("=" * 50)
    print(generated_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
