"""GPT model architecture with multi-head self-attention and transformer blocks."""

import torch
import torch.nn as nn
from typing import Optional


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism.
    
    Enables the model to attend to different parts of the input sequence
    in parallel, capturing various types of relationships.
    """
    
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.2):
        """Initialize self-attention layer.
        
        Args:
            embed_size: Dimensionality of embeddings
            heads: Number of parallel attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_size]
            
        Returns:
            Attention output [batch_size, seq_length, embed_size]
        """
        N, seq_length, embed_size = x.shape

        # [N, seq_length, heads, head_dim] -> [N, heads, seq_length, head_dim]
        values = self.values(x).view(N, seq_length, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(x).view(N, seq_length, self.heads, self.head_dim).transpose(1, 2)
        queries = self.queries(x).view(N, seq_length, self.heads, self.head_dim).transpose(1, 2)

        # energy: [N, heads, seq_length, seq_length]
        energy = queries @ keys.transpose(-2, -1) / (self.head_dim ** 0.5)

        # Causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).view(1, 1, seq_length, seq_length)
        energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # out: [N, heads, seq_length, head_dim]
        out = attention @ values
        
        # reshape back to [N, seq_length, embed_size]
        out = out.transpose(1, 2).contiguous().view(N, seq_length, embed_size)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    """Transformer block combining multi-head self-attention and feed-forward network.
    
    Uses pre-normalization architecture (GPT-2 style) with residual connections.
    """
    
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.2):
        """Initialize transformer block.
        
        Args:
            embed_size: Dimensionality of embeddings
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.attention = SelfAttention(embed_size, heads, dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_length, embed_size]
            
        Returns:
            Output tensor [batch_size, seq_length, embed_size]
        """
        # Pre-Norm Architecture (GPT-2 style)
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(nn.Module):
    """GPT-style transformer language model.
    
    Multi-layer transformer architecture with token and positional embeddings,
    multi-head self-attention, and feed-forward networks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        block_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.2
    ):
        """Initialize GPT model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_size: Dimensionality of embeddings
            block_size: Maximum sequence length
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout probability
        """
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)]
        )
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size, bias=False)

        # Weight tying: share weights between input embeddings and output linear layer
        self.token_embedding.weight = self.fc_out.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights using normal distribution.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate logits for input tokens.
        
        Args:
            x: Input token indices [batch_size, seq_length]
            
        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = x.shape
        
        # Embed tokens and positions
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq_length, device=x.device))
        x = self.drop(token_emb + pos_emb)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        return logits

    def forward(self, x):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(N, seq_length)

        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.fc_out(x)

        return logits