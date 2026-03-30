"""Training script for GPT language model."""

import torch
import torch.nn as nn
from model import GPT
import os
import tiktoken
from typing import Tuple


def get_batch(
    data: torch.Tensor,
    batch_size: int = 32,
    block_size: int = 256,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of training data.
    
    Args:
        data: Encoded text data
        batch_size: Number of sequences in batch
        block_size: Length of sequences
        device: Device to place tensors on
        
    Returns:
        Tuple of (input_tokens, target_tokens)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def main():
    """Train GPT model on text data."""
    current_dir = os.path.dirname(__file__)  # src/
    project_root = os.path.abspath(os.path.join(current_dir, ".."))  # project folder

    file_path = os.path.join(project_root, "data", "input.txt")

    with open(file_path, "r") as f:
        text = f.read()

    # Encode using OpenAI's tiktoken (Byte-Pair Encoding)
    print("Encoding dataset with tiktoken (this may take a few seconds)...")
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    vocab_size = enc.n_vocab

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    block_size = 256
    learning_rate = 3e-4
    num_steps = 5000
    eval_interval = 250

    # Create model
    model = GPT(
        vocab_size,
        embed_size=256,
        block_size=block_size,
        num_heads=8,
        num_layers=6,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Starting training...")

    for step in range(num_steps):
        x, y = get_batch(data, batch_size, block_size, device)

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # Save model
    model_path = os.path.join(project_root, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()