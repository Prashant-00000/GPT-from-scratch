"""Text generation utilities for GPT model."""

import torch
from typing import Optional
from model import GPT


def generate(
    model: GPT,
    start_tokens: torch.Tensor,
    max_length: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None
) -> torch.Tensor:
    """Generate text using the trained model.
    
    Args:
        model: Trained GPT model
        start_tokens: Initial token ids [batch_size, seq_length]
        max_length: Maximum number of new tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: If set, only sample from top-k most likely tokens
        
    Returns:
        Generated token sequence [batch_size, seq_length + max_length]
    """
    model.eval()
    tokens = start_tokens

    # Use the model's block_size to crop the context window
    block_size = model.position_embedding.weight.shape[0]

    with torch.no_grad():
        for _ in range(max_length):
            cond_tokens = tokens[:, -block_size:]  # Crop context to block size
            logits = model(cond_tokens)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]
            
            # Scale by desired temperature
            logits = logits / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens