from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    LSTM encoder for piano-roll sequences.

    Input:
        x -> [B, T, F]

    Outputs:
        mu     -> [B, latent_dim]
        logvar -> [B, latent_dim]
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

        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden_n, _) = self.lstm(x)
        last_hidden = hidden_n[-1]  # [B, hidden_dim]

        mu = self.to_mu(last_hidden)
        logvar = self.to_logvar(last_hidden)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    LSTM decoder for reconstructing piano-roll sequences from latent z.

    Input:
        z -> [B, latent_dim]

    Output:
        x_hat -> [B, T, F]
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
        repeated = self.latent_to_input(z)              # [B, H]
        repeated = repeated.unsqueeze(1)                # [B, 1, H]
        repeated = repeated.repeat(1, self.seq_len, 1)  # [B, T, H]

        decoded_seq, _ = self.lstm(repeated)            # [B, T, H]
        x_hat = self.output_layer(decoded_seq)          # [B, T, F]
        x_hat = self.output_activation(x_hat)           # [B, T, F]

        return x_hat


class MusicVAE(nn.Module):
    """
    Full VAE model for piano-roll music sequences.
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

        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick:
            z = mu + sigma * eps
            where sigma = exp(0.5 * logvar), eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat  -> reconstruction [B, T, F]
            mu     -> latent mean [B, latent_dim]
            logvar -> latent log variance [B, latent_dim]
            z      -> sampled latent [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    loss_type: str = "bce",
) -> torch.Tensor:
    """
    Reconstruction loss for piano-roll data.
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "bce":
        return F.binary_cross_entropy(x_hat, x, reduction="mean")

    if loss_type == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")

    raise ValueError("loss_type must be either 'bce' or 'mse'")


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0, I)

    Formula:
        D_KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    We normalize by batch size.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / mu.size(0)
    return kl


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_type: str = "bce",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total VAE loss:
        total = recon + beta * kl

    Returns:
        total_loss, recon_loss, kl_loss
    """
    recon = reconstruction_loss(x_hat, x, loss_type=recon_loss_type)
    kl = kl_divergence_loss(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl


if __name__ == "__main__":
    # Quick sanity check
    batch_size = 4
    seq_len = 128
    input_dim = 88

    x = torch.rand(batch_size, seq_len, input_dim)

    model = MusicVAE(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=64,
        seq_len=seq_len,
        num_layers=2,
    )

    x_hat, mu, logvar, z = model(x)

    print("Input shape: ", x.shape)
    print("Recon shape: ", x_hat.shape)
    print("Mu shape:    ", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("Z shape:     ", z.shape)

    total, recon, kl = vae_loss(
        x_hat=x_hat,
        x=x,
        mu=mu,
        logvar=logvar,
        beta=1.0,
        recon_loss_type="bce",
    )

    print("Total loss:  ", float(total.item()))
    print("Recon loss:  ", float(recon.item()))
    print("KL loss:     ", float(kl.item()))