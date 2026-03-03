# Transformer from Scratch

A clean, from-scratch implementation of the transformer architecture from "Attention Is All You Need" in PyTorch. No magic — just math, code, and logic.

## Architecture

| Component | Description |
|-----------|-------------|
| `PositionalEncoding` | Sine/cosine positional embeddings |
| `AttentionHead` | Single scaled dot-product attention (Q, K, V) |
| `MultiHeadAttention` | 8 parallel attention heads with output projection |
| `FeedForward` | Dense → ReLU → Dense (4x expansion) |
| `TransformerBlock` | Attention + residual + LayerNorm + FFN + residual + LayerNorm |
| `Transformer` | Full model: embedding → positional encoding → N blocks → output head |

## Model Specs

- **Parameters:** ~19.9M
- **Model dimension:** 512
- **Attention heads:** 8 (64 dims each)
- **Feed-forward dimension:** 2,048
- **Layers:** 6
- **Max sequence length:** 5,000

## Quick Start

```bash
pip install -r requirements.txt
python transformer.py
```

Output:
```
Input shape: torch.Size([2, 10])
Output shape: torch.Size([2, 10, 1000])
Total parameters: 19,939,304
Generated sequence: torch.Size([1, 15])
```

## Run Tests

```bash
python test_transformer.py
```

All 8 test suites verify shape correctness, numerical stability, gradient flow, and autoregressive generation.

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## License

MIT
