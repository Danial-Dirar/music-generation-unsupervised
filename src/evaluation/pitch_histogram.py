from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


def load_midi(midi_path: str | Path) -> pretty_midi.PrettyMIDI:
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return pretty_midi.PrettyMIDI(str(midi_path))


def extract_pitch_classes(midi_data: pretty_midi.PrettyMIDI) -> list[int]:
    """
    Extract pitch classes (0-11) from all non-drum notes in a MIDI file.
    """
    pitch_classes: list[int] = []

    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch_classes.append(int(note.pitch) % 12)

    return pitch_classes


def compute_pitch_histogram_from_midi(midi_path: str | Path) -> np.ndarray:
    """
    Compute a normalized 12-bin pitch-class histogram from a MIDI file.
    """
    midi_data = load_midi(midi_path)
    pitch_classes = extract_pitch_classes(midi_data)

    hist = np.zeros(12, dtype=np.float32)

    if not pitch_classes:
        return hist

    for pc in pitch_classes:
        hist[pc] += 1.0

    hist /= hist.sum()
    return hist


def compute_pitch_histogram_from_many_midis(
    midi_paths: Iterable[str | Path],
) -> np.ndarray:
    """
    Aggregate pitch histogram across multiple MIDI files, then normalize.
    """
    hist = np.zeros(12, dtype=np.float32)

    for midi_path in midi_paths:
        midi_data = load_midi(midi_path)
        pitch_classes = extract_pitch_classes(midi_data)

        for pc in pitch_classes:
            hist[pc] += 1.0

    if hist.sum() > 0:
        hist /= hist.sum()

    return hist


def pitch_histogram_similarity(
    hist_a: np.ndarray,
    hist_b: np.ndarray,
) -> float:
    """
    Handout-style distance:
        H(p, q) = sum_i |p_i - q_i|

    Smaller is better.
    """
    hist_a = np.asarray(hist_a, dtype=np.float32)
    hist_b = np.asarray(hist_b, dtype=np.float32)

    if hist_a.shape != (12,) or hist_b.shape != (12,):
        raise ValueError("Both histograms must have shape (12,)")

    return float(np.sum(np.abs(hist_a - hist_b)))


def compare_two_midis(
    midi_path_a: str | Path,
    midi_path_b: str | Path,
) -> dict[str, object]:
    """
    Compare pitch histograms of two MIDI files.
    """
    hist_a = compute_pitch_histogram_from_midi(midi_path_a)
    hist_b = compute_pitch_histogram_from_midi(midi_path_b)
    distance = pitch_histogram_similarity(hist_a, hist_b)

    return {
        "midi_a": str(midi_path_a),
        "midi_b": str(midi_path_b),
        "hist_a": hist_a,
        "hist_b": hist_b,
        "distance": distance,
    }


def compare_directories(
    reference_dir: str | Path,
    generated_dir: str | Path,
    pattern: str = "*.mid",
) -> dict[str, object]:
    """
    Compare aggregate pitch histogram between two directories of MIDI files.
    """
    reference_dir = Path(reference_dir)
    generated_dir = Path(generated_dir)

    reference_midis = sorted(reference_dir.glob(pattern))
    generated_midis = sorted(generated_dir.glob(pattern))

    if not reference_midis:
        raise ValueError(f"No MIDI files found in reference_dir: {reference_dir}")
    if not generated_midis:
        raise ValueError(f"No MIDI files found in generated_dir: {generated_dir}")

    ref_hist = compute_pitch_histogram_from_many_midis(reference_midis)
    gen_hist = compute_pitch_histogram_from_many_midis(generated_midis)
    distance = pitch_histogram_similarity(ref_hist, gen_hist)

    return {
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "num_reference_midis": len(reference_midis),
        "num_generated_midis": len(generated_midis),
        "reference_hist": ref_hist,
        "generated_hist": gen_hist,
        "distance": distance,
    }


def main() -> None:
    """
    Quick test on one original/reconstructed pair.
    """
    project_root = Path(__file__).resolve().parents[2]
    recon_dir = project_root / "outputs" / "generated_midis" / "ae_reconstructions"

    midi_a = recon_dir / "ae_sample_000_original.mid"
    midi_b = recon_dir / "ae_sample_000_reconstructed.mid"

    result = compare_two_midis(midi_a, midi_b)

    print("Pitch histogram comparison:")
    print(f"  A: {result['midi_a']}")
    print(f"  B: {result['midi_b']}")
    print(f"  Distance: {result['distance']:.6f}")
    print(f"  Hist A: {result['hist_a']}")
    print(f"  Hist B: {result['hist_b']}")


if __name__ == "__main__":
    main()