from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from src.config import FS, MIN_PITCH


def ensure_2d_pianoroll(piano_roll: np.ndarray) -> np.ndarray:
    piano_roll = np.asarray(piano_roll, dtype=np.float32)

    if piano_roll.ndim != 2:
        raise ValueError(
            f"Expected piano_roll with shape [time_steps, pitch_dim], got {piano_roll.shape}"
        )

    return piano_roll


def binarize_pianoroll(
    piano_roll: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    piano_roll = ensure_2d_pianoroll(piano_roll)
    return (piano_roll >= threshold).astype(np.int32)


def cropped_to_full_midi_pitch(
    piano_roll: np.ndarray,
    min_pitch: int = MIN_PITCH,
    full_pitch_dim: int = 128,
) -> np.ndarray:
    piano_roll = ensure_2d_pianoroll(piano_roll)
    time_steps, pitch_dim = piano_roll.shape

    if min_pitch < 0 or min_pitch >= full_pitch_dim:
        raise ValueError("min_pitch must be within full MIDI pitch range [0, 127]")

    if min_pitch + pitch_dim > full_pitch_dim:
        raise ValueError(
            "Cropped pitch range exceeds full MIDI pitch dimension after expansion"
        )

    full_roll = np.zeros((time_steps, full_pitch_dim), dtype=piano_roll.dtype)
    full_roll[:, min_pitch : min_pitch + pitch_dim] = piano_roll
    return full_roll


def pianoroll_to_pretty_midi(
    piano_roll: np.ndarray,
    fs: int = FS,
    program: int = 0,
    velocity: int = 100,
    threshold: float = 0.1,
    min_pitch: int = MIN_PITCH,
    instrument_name: str = "Piano",
    binarize: bool = True,
    debug: bool = True,
) -> pretty_midi.PrettyMIDI:
    piano_roll = ensure_2d_pianoroll(piano_roll)

    if debug:
        print("pianoroll_to_pretty_midi input debug:")
        print("  input shape:", piano_roll.shape)
        print("  input min:", float(np.min(piano_roll)))
        print("  input max:", float(np.max(piano_roll)))
        print("  input mean:", float(np.mean(piano_roll)))
        print("  threshold:", threshold)

    if piano_roll.shape[1] != 128:
        piano_roll = cropped_to_full_midi_pitch(
            piano_roll,
            min_pitch=min_pitch,
            full_pitch_dim=128,
        )

    if binarize:
        piano_roll = binarize_pianoroll(piano_roll, threshold=threshold)

    active_after_threshold = int(np.sum(piano_roll))
    if debug:
        print("  after threshold shape:", piano_roll.shape)
        print("  active_after_threshold:", active_after_threshold)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name=instrument_name)

    time_steps, num_pitches = piano_roll.shape
    seconds_per_step = 1.0 / fs

    padded = np.pad(piano_roll, ((1, 1), (0, 0)), mode="constant")
    diff = np.diff(padded, axis=0)

    note_count = 0

    for pitch in range(num_pitches):
        note_on_steps = np.where(diff[:, pitch] > 0)[0]
        note_off_steps = np.where(diff[:, pitch] < 0)[0]

        if len(note_on_steps) != len(note_off_steps):
            if debug:
                print(
                    f"  warning: pitch {pitch} has mismatched on/off counts "
                    f"({len(note_on_steps)} vs {len(note_off_steps)})"
                )
            continue

        for start_step, end_step in zip(note_on_steps, note_off_steps):
            start_time = float(start_step * seconds_per_step)
            end_time = float(end_step * seconds_per_step)

            if end_time <= start_time:
                continue

            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(note)
            note_count += 1

    if debug:
        print("  notes_created:", note_count)

    if instrument.notes:
        midi.instruments.append(instrument)

    return midi


def save_midi(
    piano_roll: np.ndarray,
    output_path: str | Path,
    fs: int = FS,
    program: int = 0,
    velocity: int = 100,
    threshold: float = 0.1,
    min_pitch: int = MIN_PITCH,
    instrument_name: str = "Piano",
    binarize: bool = True,
    debug: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    piano_roll = ensure_2d_pianoroll(piano_roll)

    if debug:
        print("save_midi debug:")
        print("  output_path:", output_path)
        print("  shape:", piano_roll.shape)
        print("  min:", float(np.min(piano_roll)))
        print("  max:", float(np.max(piano_roll)))
        print("  mean:", float(np.mean(piano_roll)))
        print("  threshold:", threshold)
        print("  active_if_thresholded_here:", int((piano_roll >= threshold).sum()))

    midi = pianoroll_to_pretty_midi(
        piano_roll=piano_roll,
        fs=fs,
        program=program,
        velocity=velocity,
        threshold=threshold,
        min_pitch=min_pitch,
        instrument_name=instrument_name,
        binarize=binarize,
        debug=debug,
    )

    midi.write(str(output_path))
    return output_path


def save_batch_as_midis(
    batch_rolls: np.ndarray,
    output_dir: str | Path,
    prefix: str = "sample",
    fs: int = FS,
    threshold: float = 0.1,
    min_pitch: int = MIN_PITCH,
) -> list[Path]:
    batch_rolls = np.asarray(batch_rolls, dtype=np.float32)

    if batch_rolls.ndim != 3:
        raise ValueError(
            f"Expected batch_rolls with shape [N, T, F], got {batch_rolls.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for i in range(batch_rolls.shape[0]):
        out_path = output_dir / f"{prefix}_{i:03d}.mid"
        save_midi(
            piano_roll=batch_rolls[i],
            output_path=out_path,
            fs=fs,
            threshold=threshold,
            min_pitch=min_pitch,
        )
        saved_paths.append(out_path)

    return saved_paths


def load_npy_sequence(
    npy_path: str | Path,
    index: int = 0,
) -> np.ndarray:
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {npy_path}")

    arr = np.load(npy_path, mmap_mode="r")

    if arr.ndim != 3:
        raise ValueError(f"Expected [N, T, F], got {arr.shape}")

    if index < 0 or index >= arr.shape[0]:
        raise IndexError(f"Index out of range: {index}")

    return np.array(arr[index], dtype=np.float32)


def main() -> None:
    from src.config import OUTPUT_MIDI_DIR, PROCESSED_DIR

    train_npy = Path(PROCESSED_DIR) / "train_sequences.npy"
    sample_roll = load_npy_sequence(train_npy, index=0)

    out_path = Path(OUTPUT_MIDI_DIR) / "debug_original_sequence.mid"
    save_midi(sample_roll, out_path, threshold=0.1, debug=True)

    print(f"Saved debug MIDI to: {out_path}")


if __name__ == "__main__":
    main()