"""
Transformer from Scratch - Clean, Production-Ready Implementation

This is the complete transformer architecture explained in the video.
No magic. Just math, code, and logic.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Add positional information to embeddings using sine and cosine waves."""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional_encoding: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class AttentionHead(nn.Module):
    """Single attention head: Compute Q, K, V and attention weights."""
    
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            attention_output: (batch_size, seq_len, d_head)
        """
        # Project to Q, K, V
        Q = self.query(x)  # (batch, seq_len, d_head)
        K = self.key(x)    # (batch, seq_len, d_head)
        V = self.value(x)  # (batch, seq_len, d_head)
        
        # Compute attention scores: Q @ K^T / sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel."""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.d_head) 
            for _ in range(num_heads)
        ])
        
        # Output projection (combine all heads)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Run all heads in parallel
        head_outputs = [head(x) for head in self.heads]
        
        # Concatenate results (each head contributes d_head dimensions)
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Project back to d_model dimensions
        output = self.output_proj(concatenated)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network: Dense -> ReLU -> Dense."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Expand to d_ff dimensions
        x = self.dense1(x)
        
        # Apply ReLU non-linearity
        x = self.relu(x)
        
        # Project back to d_model dimensions
        x = self.dense2(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: Attention -> Residual -> Norm -> FFN -> Residual -> Norm."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Attention with residual connection and normalization
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class Transformer(nn.Module):
    """Complete transformer model."""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=5000):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output layer (project to vocabulary size for next-token prediction)
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: token IDs (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Embed tokens
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through all transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Project to vocabulary size
        logits = self.output_head(x)
        
        return logits
    
    def generate(self, prompt, max_tokens=100, temperature=1.0):
        """
        Generate text autoregressively.
        
        Args:
            prompt: starting token IDs (batch_size, prompt_len)
            max_tokens: number of tokens to generate
            temperature: controls randomness (1.0 = normal, higher = more random)
        Returns:
            generated_ids: (batch_size, prompt_len + max_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get predictions
                logits = self(prompt)
                
                # Get last token logits
                next_logits = logits[:, -1, :]
                
                # Apply temperature
                next_logits = next_logits / temperature
                
                # Get probabilities
                probs = torch.softmax(next_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to prompt
                prompt = torch.cat([prompt, next_token], dim=1)
        
        return prompt


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    # Random input tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate example
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_tokens=10)
    print(f"Generated sequence: {generated.shape}")
