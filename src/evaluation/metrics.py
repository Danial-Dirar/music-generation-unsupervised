from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.evaluation.pitch_histogram import compare_two_midis as compare_pitch_two_midis
from src.evaluation.pitch_histogram import compare_directories as compare_pitch_directories
from src.evaluation.rhythm_score import compare_two_midis as compare_rhythm_two_midis
from src.evaluation.rhythm_score import compare_directories as compare_rhythm_directories


def evaluate_two_midis(
    midi_a: str | Path,
    midi_b: str | Path,
) -> Dict[str, object]:
    """
    Evaluate two MIDI files using:
    - pitch histogram distance
    - rhythm diversity
    - repetition ratio
    """
    pitch_result = compare_pitch_two_midis(midi_a, midi_b)
    rhythm_result = compare_rhythm_two_midis(midi_a, midi_b)

    return {
        "midi_a": str(midi_a),
        "midi_b": str(midi_b),
        "pitch_histogram_distance": pitch_result["distance"],
        "pitch_histogram_a": pitch_result["hist_a"],
        "pitch_histogram_b": pitch_result["hist_b"],
        "num_notes_a": rhythm_result["num_notes_a"],
        "num_notes_b": rhythm_result["num_notes_b"],
        "unique_durations_a": rhythm_result["unique_durations_a"],
        "unique_durations_b": rhythm_result["unique_durations_b"],
        "rhythm_diversity_a": rhythm_result["rhythm_diversity_a"],
        "rhythm_diversity_b": rhythm_result["rhythm_diversity_b"],
        "repetition_ratio_a": rhythm_result["repetition_ratio_a"],
        "repetition_ratio_b": rhythm_result["repetition_ratio_b"],
    }


def evaluate_directories(
    reference_dir: str | Path,
    generated_dir: str | Path,
    pattern: str = "*.mid",
) -> Dict[str, object]:
    """
    Evaluate two directories of MIDI files in aggregate.
    """
    pitch_result = compare_pitch_directories(
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        pattern=pattern,
    )

    rhythm_result = compare_rhythm_directories(
        reference_dir=reference_dir,
        generated_dir=generated_dir,
        pattern=pattern,
    )

    return {
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "num_reference_midis": pitch_result["num_reference_midis"],
        "num_generated_midis": pitch_result["num_generated_midis"],
        "pitch_histogram_distance": pitch_result["distance"],
        "reference_pitch_histogram": pitch_result["reference_hist"],
        "generated_pitch_histogram": pitch_result["generated_hist"],
        "num_reference_notes": rhythm_result["num_reference_notes"],
        "num_generated_notes": rhythm_result["num_generated_notes"],
        "reference_unique_durations": rhythm_result["reference_unique_durations"],
        "generated_unique_durations": rhythm_result["generated_unique_durations"],
        "reference_rhythm_diversity": rhythm_result["reference_rhythm_diversity"],
        "generated_rhythm_diversity": rhythm_result["generated_rhythm_diversity"],
        "reference_repetition_ratio": rhythm_result["reference_repetition_ratio"],
        "generated_repetition_ratio": rhythm_result["generated_repetition_ratio"],
    }


def pretty_print_two_midi_results(results: Dict[str, object]) -> None:
    print("Evaluation results for two MIDI files:")
    print(f"  MIDI A: {results['midi_a']}")
    print(f"  MIDI B: {results['midi_b']}")
    print(f"  Pitch Histogram Distance: {results['pitch_histogram_distance']:.6f}")
    print(f"  Num Notes A: {results['num_notes_a']}")
    print(f"  Num Notes B: {results['num_notes_b']}")
    print(f"  Unique Durations A: {results['unique_durations_a']}")
    print(f"  Unique Durations B: {results['unique_durations_b']}")
    print(f"  Rhythm Diversity A: {results['rhythm_diversity_a']:.6f}")
    print(f"  Rhythm Diversity B: {results['rhythm_diversity_b']:.6f}")
    print(f"  Repetition Ratio A: {results['repetition_ratio_a']:.6f}")
    print(f"  Repetition Ratio B: {results['repetition_ratio_b']:.6f}")


def pretty_print_directory_results(results: Dict[str, object]) -> None:
    print("Aggregate directory evaluation results:")
    print(f"  Reference Dir: {results['reference_dir']}")
    print(f"  Generated Dir: {results['generated_dir']}")
    print(f"  Num Reference MIDIs: {results['num_reference_midis']}")
    print(f"  Num Generated MIDIs: {results['num_generated_midis']}")
    print(f"  Pitch Histogram Distance: {results['pitch_histogram_distance']:.6f}")
    print(f"  Num Reference Notes: {results['num_reference_notes']}")
    print(f"  Num Generated Notes: {results['num_generated_notes']}")
    print(f"  Reference Unique Durations: {results['reference_unique_durations']}")
    print(f"  Generated Unique Durations: {results['generated_unique_durations']}")
    print(
        f"  Reference Rhythm Diversity: "
        f"{results['reference_rhythm_diversity']:.6f}"
    )
    print(
        f"  Generated Rhythm Diversity: "
        f"{results['generated_rhythm_diversity']:.6f}"
    )
    print(
        f"  Reference Repetition Ratio: "
        f"{results['reference_repetition_ratio']:.6f}"
    )
    print(
        f"  Generated Repetition Ratio: "
        f"{results['generated_repetition_ratio']:.6f}"
    )


def main() -> None:
    """
    Quick test on one original/reconstructed pair.
    """
    project_root = Path(__file__).resolve().parents[2]
    recon_dir = project_root / "outputs" / "generated_midis" / "ae_reconstructions"

    midi_a = recon_dir / "ae_sample_000_original.mid"
    midi_b = recon_dir / "ae_sample_000_reconstructed.mid"

    results = evaluate_two_midis(midi_a, midi_b)
    pretty_print_two_midi_results(results)


if __name__ == "__main__":
    main()