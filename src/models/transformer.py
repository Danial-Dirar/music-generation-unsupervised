from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Input/output shape:
        [B, T, D]
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Causal mask for autoregressive attention.
    Shape: [T, T]
    True => masked position
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask


class MusicTransformer(nn.Module):
    """
    Decoder-style autoregressive transformer for piano-roll sequences.

    Input:
        x: [B, T, F]  (F = 88 usually)

    Output:
        logits: [B, T, F]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]

        Returns:
            logits: [B, T, F]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, F], got {x.shape}")

        h = self.input_projection(x) * math.sqrt(self.d_model)
        h = self.positional_encoding(h)

        seq_len = x.size(1)
        causal_mask = generate_causal_mask(seq_len=seq_len, device=x.device)

        h = self.transformer(h, mask=causal_mask)
        logits = self.output_projection(h)

        return logits

    @torch.no_grad()
    def generate(
        self,
        seed: torch.Tensor,
        steps: int = 128,
        temperature: float = 1.0,
        threshold: float = 0.35,
        context_len: int = 128,
        max_active_notes: int = 6,
        sample_probs: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressively generate future frames.

        Args:
            seed: [1, T0, F]
            steps: number of new frames to append
            temperature: logit scaling
            threshold: threshold used when sample_probs=False
            context_len: only most recent frames are used as transformer context
            max_active_notes: cap number of active notes per frame
            sample_probs: if True, sample Bernoulli from probabilities

        Returns:
            generated: [1, T0 + steps, F]
        """
        self.eval()

        if seed.ndim != 3 or seed.size(0) != 1:
            raise ValueError("seed must have shape [1, T, F]")

        x = seed.clone()

        for _ in range(steps):
            context = x[:, -context_len:, :]
            logits = self.forward(context)          # [1, Tc, F]
            next_logits = logits[:, -1, :]         # [1, F]

            probs = torch.sigmoid(next_logits / max(temperature, 1e-6))
            probs = torch.clamp(probs, 0.0, 0.98)

            if max_active_notes is not None and max_active_notes < probs.shape[-1]:
                top_vals, top_idx = torch.topk(
                    probs,
                    k=max_active_notes,
                    dim=-1,
                )
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(1, top_idx, top_vals)
                probs = filtered_probs

            if sample_probs:
                next_frame = torch.bernoulli(probs)
            else:
                next_frame = (probs >= threshold).float()

            if next_frame.sum() == 0:
                top_idx = torch.argmax(probs, dim=-1, keepdim=True)
                next_frame = torch.zeros_like(probs)
                next_frame.scatter_(1, top_idx, 1.0)

            next_frame = next_frame.unsqueeze(1)   # [1, 1, F]
            x = torch.cat([x, next_frame], dim=1)

        return x


def transformer_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    BCE-with-logits loss for piano-roll next-step prediction.
    """
    return F.binary_cross_entropy_with_logits(logits, target)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 128
    input_dim = 88

    x = torch.rand(batch_size, seq_len, input_dim)

    model = MusicTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=2048,
    )

    logits = model(x)
    print("Input shape: ", x.shape)
    print("Logits shape:", logits.shape)

    loss = transformer_loss(logits, x)
    print("Sample loss: ", float(loss.item()))