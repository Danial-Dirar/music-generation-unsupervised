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


def extract_note_durations(
    midi_path: str | Path,
    round_digits: int = 3,
    ignore_drums: bool = True,
) -> list[float]:
    """
    Extract note durations from a MIDI file.

    Args:
        midi_path: path to .mid/.midi file
        round_digits: round durations to reduce tiny floating-point differences
        ignore_drums: skip drum tracks if present

    Returns:
        list of rounded note durations
    """
    midi_data = load_midi(midi_path)
    durations: list[float] = []

    for instrument in midi_data.instruments:
        if ignore_drums and instrument.is_drum:
            continue

        for note in instrument.notes:
            duration = max(0.0, float(note.end - note.start))
            duration = round(duration, round_digits)
            if duration > 0:
                durations.append(duration)

    return durations


def rhythm_diversity_score(
    durations: list[float],
) -> float:
    """
    Rhythm Diversity Score:
        #unique durations / #total notes

    Returns 0.0 if no durations are found.
    """
    if not durations:
        return 0.0

    unique_durations = len(set(durations))
    total_notes = len(durations)
    return float(unique_durations / total_notes)


def repetition_ratio(
    durations: list[float],
) -> float:
    """
    Simple repetition ratio based on repeated duration values:
        (#total durations - #unique durations) / #total durations

    Returns 0.0 if no durations are found.
    """
    if not durations:
        return 0.0

    unique_durations = len(set(durations))
    total_notes = len(durations)
    repeated = total_notes - unique_durations
    return float(repeated / total_notes)


def compare_two_midis(
    midi_path_a: str | Path,
    midi_path_b: str | Path,
    round_digits: int = 3,
) -> dict[str, object]:
    """
    Compare rhythm diversity and repetition ratio of two MIDI files.
    """
    durations_a = extract_note_durations(midi_path_a, round_digits=round_digits)
    durations_b = extract_note_durations(midi_path_b, round_digits=round_digits)

    result = {
        "midi_a": str(midi_path_a),
        "midi_b": str(midi_path_b),
        "num_notes_a": len(durations_a),
        "num_notes_b": len(durations_b),
        "unique_durations_a": len(set(durations_a)),
        "unique_durations_b": len(set(durations_b)),
        "rhythm_diversity_a": rhythm_diversity_score(durations_a),
        "rhythm_diversity_b": rhythm_diversity_score(durations_b),
        "repetition_ratio_a": repetition_ratio(durations_a),
        "repetition_ratio_b": repetition_ratio(durations_b),
    }
    return result


def aggregate_durations_from_many_midis(
    midi_paths: Iterable[str | Path],
    round_digits: int = 3,
) -> list[float]:
    """
    Collect durations from multiple MIDI files into one list.
    """
    all_durations: list[float] = []

    for midi_path in midi_paths:
        durations = extract_note_durations(midi_path, round_digits=round_digits)
        all_durations.extend(durations)

    return all_durations


def compare_directories(
    reference_dir: str | Path,
    generated_dir: str | Path,
    pattern: str = "*.mid",
    round_digits: int = 3,
) -> dict[str, object]:
    """
    Compare aggregate rhythm metrics between two directories of MIDI files.
    """
    reference_dir = Path(reference_dir)
    generated_dir = Path(generated_dir)

    reference_midis = sorted(reference_dir.glob(pattern))
    generated_midis = sorted(generated_dir.glob(pattern))

    if not reference_midis:
        raise ValueError(f"No MIDI files found in reference_dir: {reference_dir}")
    if not generated_midis:
        raise ValueError(f"No MIDI files found in generated_dir: {generated_dir}")

    ref_durations = aggregate_durations_from_many_midis(
        reference_midis, round_digits=round_digits
    )
    gen_durations = aggregate_durations_from_many_midis(
        generated_midis, round_digits=round_digits
    )

    return {
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "num_reference_midis": len(reference_midis),
        "num_generated_midis": len(generated_midis),
        "num_reference_notes": len(ref_durations),
        "num_generated_notes": len(gen_durations),
        "reference_unique_durations": len(set(ref_durations)),
        "generated_unique_durations": len(set(gen_durations)),
        "reference_rhythm_diversity": rhythm_diversity_score(ref_durations),
        "generated_rhythm_diversity": rhythm_diversity_score(gen_durations),
        "reference_repetition_ratio": repetition_ratio(ref_durations),
        "generated_repetition_ratio": repetition_ratio(gen_durations),
    }


def duration_histogram(
    durations: list[float],
) -> dict[float, int]:
    """
    Return a simple duration frequency dictionary.
    """
    hist: dict[float, int] = {}
    for d in durations:
        hist[d] = hist.get(d, 0) + 1
    return dict(sorted(hist.items(), key=lambda x: x[0]))


def main() -> None:
    """
    Quick test on one original/reconstructed pair.
    """
    project_root = Path(__file__).resolve().parents[2]
    recon_dir = project_root / "outputs" / "generated_midis" / "ae_reconstructions"

    midi_a = recon_dir / "ae_sample_000_original.mid"
    midi_b = recon_dir / "ae_sample_000_reconstructed.mid"

    result = compare_two_midis(midi_a, midi_b, round_digits=3)

    print("Rhythm comparison:")
    print(f"  A: {result['midi_a']}")
    print(f"  B: {result['midi_b']}")
    print(f"  num_notes_a: {result['num_notes_a']}")
    print(f"  num_notes_b: {result['num_notes_b']}")
    print(f"  unique_durations_a: {result['unique_durations_a']}")
    print(f"  unique_durations_b: {result['unique_durations_b']}")
    print(f"  rhythm_diversity_a: {result['rhythm_diversity_a']:.6f}")
    print(f"  rhythm_diversity_b: {result['rhythm_diversity_b']:.6f}")
    print(f"  repetition_ratio_a: {result['repetition_ratio_a']:.6f}")
    print(f"  repetition_ratio_b: {result['repetition_ratio_b']:.6f}")


if __name__ == "__main__":
    main()