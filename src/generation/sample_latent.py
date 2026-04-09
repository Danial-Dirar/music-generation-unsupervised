from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from src.config import (
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    OUTPUT_MIDI_DIR,
    PROJECT_ROOT,
    SEQ_LEN,
)
from src.generation.midi_export import save_midi
from src.models.vae import MusicVAE


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vae_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[MusicVAE, Dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    seq_len = int(checkpoint.get("seq_len", SEQ_LEN))
    hidden_dim = int(checkpoint.get("hidden_dim", HIDDEN_DIM))
    latent_dim = int(checkpoint.get("latent_dim", LATENT_DIM))
    num_layers = int(checkpoint.get("num_layers", NUM_LAYERS))

    model = MusicVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def sample_latent_vectors(
    num_samples: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample z ~ N(0, I)
    """
    return torch.randn(num_samples, latent_dim, device=device)


@torch.no_grad()
def decode_latents(
    model: MusicVAE,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    Decode latent vectors into piano-roll probabilities.

    Returns:
        [N, T, F]
    """
    x_hat = model.decode(z)
    return x_hat


def save_generated_samples(
    generated_rolls: torch.Tensor,
    output_dir: str | Path,
    prefix: str = "vae_sample",
    threshold: float = 0.3,
) -> list[Path]:
    """
    Save generated piano-roll samples as MIDI files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_rolls = generated_rolls.detach().cpu().numpy()
    saved_paths: list[Path] = []

    for i in range(generated_rolls.shape[0]):
        sample_roll = generated_rolls[i]

        print(
            f"Sample {i} stats -> "
            f"min: {sample_roll.min():.6f}, "
            f"max: {sample_roll.max():.6f}, "
            f"mean: {sample_roll.mean():.6f}"
        )

        out_path = output_dir / f"{prefix}_{i:03d}.mid"
        save_midi(
            piano_roll=sample_roll,
            output_path=out_path,
            threshold=threshold,
            debug=False,
        )
        saved_paths.append(out_path)

        print(f"Saved VAE sample {i} -> {out_path}")

    return saved_paths


def generate_from_vae(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 8,
    threshold: float = 0.3,
) -> list[Path]:
    device = get_device()
    print(f"Using device: {device}")

    model, checkpoint = load_vae_checkpoint(checkpoint_path, device)
    print(f"Loaded VAE checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    latent_dim = int(checkpoint.get("latent_dim", LATENT_DIM))
    z = sample_latent_vectors(
        num_samples=num_samples,
        latent_dim=latent_dim,
        device=device,
    )

    generated_rolls = decode_latents(model, z)

    saved_paths = save_generated_samples(
        generated_rolls=generated_rolls,
        output_dir=output_dir,
        prefix="vae_sample",
        threshold=threshold,
    )

    return saved_paths


def main() -> None:
    checkpoint_path = Path(PROJECT_ROOT) / "outputs" / "checkpoints" / "vae_best.pt"
    output_dir = Path(OUTPUT_MIDI_DIR) / "vae_generated"

    generate_from_vae(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        num_samples=8,
        threshold=0.08,
    )


if __name__ == "__main__":
    main()