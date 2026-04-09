from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.config import (
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    OUTPUT_MIDI_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SEQ_LEN,
)
from src.generation.midi_export import save_midi
from src.models.autoencoder import LSTMAutoencoder


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sequences(npy_path: str | Path) -> np.ndarray:
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {npy_path}")

    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, T, F], got {arr.shape}")
    return arr


def load_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[LSTMAutoencoder, Dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    seq_len = int(checkpoint.get("seq_len", SEQ_LEN))

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        seq_len=seq_len,
        num_layers=NUM_LAYERS,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def reconstruct_sequence(
    model: LSTMAutoencoder,
    sequence: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        sequence: [T, F]

    Returns:
        original_seq: [T, F]
        reconstructed_seq: [T, F]
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected [T, F], got {sequence.shape}")

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, F]
    x_hat, _ = model(x)

    original_seq = x.squeeze(0).cpu().numpy()
    reconstructed_seq = x_hat.squeeze(0).cpu().numpy()

    return original_seq, reconstructed_seq


def save_original_and_reconstruction(
    original_seq: np.ndarray,
    reconstructed_seq: np.ndarray,
    output_dir: str | Path,
    sample_name: str,
    threshold: float = 0.1,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / f"{sample_name}_original.mid"
    recon_path = output_dir / f"{sample_name}_reconstructed.mid"

    save_midi(original_seq, original_path, threshold=threshold)
    save_midi(reconstructed_seq, recon_path, threshold=threshold)

    return {
        "original": original_path,
        "reconstructed": recon_path,
    }


def generate_reconstructions(
    checkpoint_path: str | Path,
    test_sequences_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 5,
    start_index: int = 0,
    threshold: float = 0.1,
) -> list[dict[str, Path]]:
    device = get_device()
    print(f"Using device: {device}")

    model, checkpoint = load_autoencoder_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    sequences = load_sequences(test_sequences_path)
    total_samples = sequences.shape[0]

    if total_samples == 0:
        raise ValueError("Test sequence file is empty.")

    end_index = min(start_index + num_samples, total_samples)
    if start_index >= end_index:
        raise ValueError("Invalid start_index / num_samples combination.")

    saved_items: list[dict[str, Path]] = []

    for idx in range(start_index, end_index):
        sequence = np.array(sequences[idx], dtype=np.float32)

        original_seq, reconstructed_seq = reconstruct_sequence(
            model=model,
            sequence=sequence,
            device=device,
        )

        print(
            f"Sample {idx} stats -> "
            f"orig mean: {original_seq.mean():.6f}, "
            f"recon min: {reconstructed_seq.min():.6f}, "
            f"recon max: {reconstructed_seq.max():.6f}, "
            f"recon mean: {reconstructed_seq.mean():.6f}"
        )

        sample_name = f"ae_sample_{idx:03d}"
        saved = save_original_and_reconstruction(
            original_seq=original_seq,
            reconstructed_seq=reconstructed_seq,
            output_dir=output_dir,
            sample_name=sample_name,
            threshold=threshold,
        )
        saved_items.append(saved)

        print(f"Saved sample {idx}:")
        print(f"  original      -> {saved['original']}")
        print(f"  reconstructed -> {saved['reconstructed']}")

    return saved_items


def main() -> None:
    checkpoint_path = Path(PROJECT_ROOT) / "outputs" / "checkpoints" / "ae_best.pt"
    test_sequences_path = Path(PROCESSED_DIR) / "test_sequences.npy"
    output_dir = Path(OUTPUT_MIDI_DIR) / "ae_reconstructions"

    generate_reconstructions(
        checkpoint_path=checkpoint_path,
        test_sequences_path=test_sequences_path,
        output_dir=output_dir,
        num_samples=5,
        start_index=0,
        threshold=0.1,
    )


if __name__ == "__main__":
    main()