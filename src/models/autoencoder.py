from __future__ import annotations

import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    """
    Encodes an input piano-roll sequence into a latent vector.

    Input shape:
        x: [batch_size, seq_len, input_dim]

    Output shape:
        z: [batch_size, latent_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F]

        Returns:
            z: [B, latent_dim]
        """
        _, (hidden_n, _) = self.lstm(x)

        # hidden_n shape: [num_layers, B, hidden_dim]
        last_hidden = hidden_n[-1]  # [B, hidden_dim]
        z = self.to_latent(last_hidden)  # [B, latent_dim]
        return z


class DecoderLSTM(nn.Module):
    """
    Decodes a latent vector back into a piano-roll sequence.

    Strategy:
    - project latent vector to hidden state space
    - repeat across seq_len
    - decode with LSTM
    - map to note probabilities

    Output is squashed with sigmoid so values stay in [0, 1].
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.latent_to_input = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim]

        Returns:
            x_hat: [B, seq_len, output_dim]
        """
        repeated = self.latent_to_input(z)              # [B, hidden_dim]
        repeated = repeated.unsqueeze(1)                # [B, 1, hidden_dim]
        repeated = repeated.repeat(1, self.seq_len, 1)  # [B, T, hidden_dim]

        decoded_seq, _ = self.lstm(repeated)            # [B, T, hidden_dim]
        x_hat = self.output_layer(decoded_seq)          # [B, T, output_dim]
        x_hat = self.output_activation(x_hat)           # [B, T, output_dim]
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for piano-roll reconstruction.

    Example:
        model = LSTMAutoencoder(
            input_dim=88,
            hidden_dim=256,
            latent_dim=64,
            seq_len=128,
            num_layers=2,
        )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        seq_len: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.encoder = EncoderLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = DecoderLSTM(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, F]

        Returns:
            x_hat: reconstructed sequence [B, T, F]
            z: latent vector [B, latent_dim]
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    loss_type: str = "bce",
) -> torch.Tensor:
    """
    Reconstruction loss for piano-roll data.

    Args:
        x_hat: predicted tensor [B, T, F]
        x: target tensor [B, T, F]
        loss_type: "bce" or "mse"

    Returns:
        scalar loss
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "bce":
        return nn.functional.binary_cross_entropy(x_hat, x)

    if loss_type == "mse":
        return nn.functional.mse_loss(x_hat, x)

    raise ValueError("loss_type must be either 'bce' or 'mse'")


if __name__ == "__main__":
    # Quick sanity check
    batch_size = 4
    seq_len = 128
    input_dim = 88

    x = torch.rand(batch_size, seq_len, input_dim)

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=64,
        seq_len=seq_len,
        num_layers=2,
    )

    x_hat, z = model(x)

    print("Input shape: ", x.shape)
    print("Latent shape:", z.shape)
    print("Recon shape: ", x_hat.shape)

    loss = reconstruction_loss(x_hat, x, loss_type="bce")
    print("Sample loss: ", float(loss.item()))