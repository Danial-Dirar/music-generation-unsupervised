from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.config import OUTPUT_MIDI_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.generation.midi_export import save_midi
from src.models.transformer import MusicTransformer


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


def load_transformer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[MusicTransformer, Dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_dim = int(checkpoint["input_dim"])

    # Larger max_len for longer generation
    model = MusicTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_len=2048,
    ).to(device)

    state_dict = checkpoint["model_state_dict"].copy()

    # Keep the new larger positional encoding
    if "positional_encoding.pe" in state_dict:
        state_dict.pop("positional_encoding.pe")

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def generate_from_seed(
    model: MusicTransformer,
    seed_sequence: np.ndarray,
    device: torch.device,
    seed_len: int = 32,
    generate_steps: int = 688,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    if seed_sequence.ndim != 2:
        raise ValueError(f"Expected [T, F], got {seed_sequence.shape}")

    if seed_len >= seed_sequence.shape[0]:
        raise ValueError("seed_len must be smaller than sequence length")

    seed = seed_sequence[:seed_len]
    seed_tensor = torch.tensor(seed, dtype=torch.float32).unsqueeze(0).to(device)

    generated = model.generate(
    seed=seed_tensor,
    steps=generate_steps,
    temperature=1.15,
    threshold=0.35,
    context_len=128,
    max_active_notes=6,
    sample_probs=True,
)

    generated = generated.squeeze(0).detach().cpu().numpy()
    reference = seed_sequence[: min(seed_len + generate_steps, seed_sequence.shape[0])]

    return reference, generated


def save_seed_and_generated(
    reference: np.ndarray,
    generated: np.ndarray,
    output_dir: str | Path,
    sample_name: str,
    threshold: float = 0.5,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = output_dir / f"{sample_name}_reference.mid"
    generated_path = output_dir / f"{sample_name}_generated.mid"

    save_midi(reference, reference_path, threshold=threshold, debug=False)
    save_midi(generated, generated_path, threshold=threshold, debug=False)

    return {
        "reference": reference_path,
        "generated": generated_path,
    }


def generate_transformer_samples(
    checkpoint_path: str | Path,
    test_sequences_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 5,
    start_index: int = 0,
    seed_len: int = 32,
    generate_steps: int = 688,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> list[dict[str, Path]]:
    device = get_device()
    print(f"Using device: {device}")

    model, checkpoint = load_transformer_checkpoint(checkpoint_path, device)
    print(f"Loaded transformer checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    sequences = load_sequences(test_sequences_path)
    total_samples = sequences.shape[0]

    if total_samples == 0:
        raise ValueError("Test sequence file is empty.")

    end_index = min(start_index + num_samples, total_samples)
    saved_items: list[dict[str, Path]] = []

    for idx in range(start_index, end_index):
        sequence = np.array(sequences[idx], dtype=np.float32)

        reference, generated = generate_from_seed(
            model=model,
            seed_sequence=sequence,
            device=device,
            seed_len=seed_len,
            generate_steps=generate_steps,
            temperature=temperature,
            threshold=threshold,
        )

        print(
            f"Sample {idx} stats -> "
            f"ref_mean: {reference.mean():.6f}, "
            f"gen_min: {generated.min():.6f}, "
            f"gen_max: {generated.max():.6f}, "
            f"gen_mean: {generated.mean():.6f}"
        )

        sample_name = f"transformer_sample_{idx:03d}"
        saved = save_seed_and_generated(
            reference=reference,
            generated=generated,
            output_dir=output_dir,
            sample_name=sample_name,
            threshold=threshold,
        )
        saved_items.append(saved)

        print(f"Saved sample {idx}:")
        print(f"  reference -> {saved['reference']}")
        print(f"  generated -> {saved['generated']}")

    return saved_items


def main() -> None:
    checkpoint_path = Path(PROJECT_ROOT) / "outputs" / "checkpoints" / "transformer_best.pt"
    test_sequences_path = Path(PROCESSED_DIR) / "test_sequences.npy"
    output_dir = Path(OUTPUT_MIDI_DIR) / "transformer_generated_45s_v2"

    generate_transformer_samples(
    checkpoint_path=checkpoint_path,
    test_sequences_path=test_sequences_path,
    output_dir=output_dir,
    num_samples=5,
    start_index=0,
    seed_len=96,
    generate_steps=624,   # total = 96 + 624 = 720 steps ≈ 45 sec at FS=16
    temperature=1.15,
    threshold=0.35,
)


if __name__ == "__main__":
    main()