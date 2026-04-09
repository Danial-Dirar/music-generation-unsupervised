from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_midi

from src.config import (
    FS,
    MAX_PITCH,
    MIN_PITCH,
    PROCESSED_DIR,
    SEQ_LEN,
    SPLIT_DIR,
    STRIDE,
)


def midi_to_pianoroll(
    midi_path: str | Path,
    fs: int = FS,
    use_velocity: bool = False,
) -> np.ndarray:
    """
    Convert a MIDI file to piano-roll.

    Returns:
        np.ndarray of shape [time_steps, 128]
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))
    piano_roll = midi_data.get_piano_roll(fs=fs)  # shape: [128, T]

    if piano_roll.size == 0:
        return np.zeros((0, 128), dtype=np.float32)

    if use_velocity:
        piano_roll = piano_roll.astype(np.float32) / 127.0
    else:
        piano_roll = (piano_roll > 0).astype(np.float32)

    piano_roll = piano_roll.T  # [T, 128]
    return piano_roll


def crop_pitch_range(
    piano_roll: np.ndarray,
    min_pitch: int = MIN_PITCH,
    max_pitch: int = MAX_PITCH,
) -> np.ndarray:
    """
    Crop piano-roll from 128 pitches to the desired pitch range.
    Inclusive range: [min_pitch, max_pitch]
    """
    if piano_roll.ndim != 2:
        raise ValueError(
            f"Expected piano_roll with shape [time_steps, pitches], got {piano_roll.shape}"
        )

    if piano_roll.shape[1] != 128:
        raise ValueError(
            f"Expected 128 MIDI pitches before cropping, got {piano_roll.shape[1]}"
        )

    if not (0 <= min_pitch <= max_pitch <= 127):
        raise ValueError("Pitch range must satisfy 0 <= min_pitch <= max_pitch <= 127")

    return piano_roll[:, min_pitch : max_pitch + 1]


def segment_sequences(
    piano_roll: np.ndarray,
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
    min_active_notes: int = 1,
) -> np.ndarray:
    """
    Segment piano-roll into overlapping fixed-length windows.

    Args:
        piano_roll: np.ndarray [T, pitch_dim]
        seq_len: number of time steps per segment
        stride: shift between consecutive segments
        min_active_notes: skip segments with fewer active notes than this

    Returns:
        np.ndarray [num_segments, seq_len, pitch_dim]
    """
    if piano_roll.ndim != 2:
        raise ValueError(f"Expected 2D piano_roll, got shape {piano_roll.shape}")

    total_steps, pitch_dim = piano_roll.shape

    if total_steps < seq_len:
        return np.zeros((0, seq_len, pitch_dim), dtype=np.float32)

    segments: List[np.ndarray] = []

    for start in range(0, total_steps - seq_len + 1, stride):
        end = start + seq_len
        segment = piano_roll[start:end]

        active_count = int(segment.sum())
        if active_count < min_active_notes:
            continue

        segments.append(segment.astype(np.float32))

    if not segments:
        return np.zeros((0, seq_len, pitch_dim), dtype=np.float32)

    return np.stack(segments, axis=0)


def process_single_midi(
    midi_path: str | Path,
    fs: int = FS,
    min_pitch: int = MIN_PITCH,
    max_pitch: int = MAX_PITCH,
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
    use_velocity: bool = False,
) -> np.ndarray:
    """
    Full preprocessing for one MIDI:
    MIDI -> piano-roll -> cropped pitch range -> segmented windows

    Returns:
        np.ndarray [num_segments, seq_len, pitch_dim]
    """
    piano_roll = midi_to_pianoroll(
        midi_path=midi_path,
        fs=fs,
        use_velocity=use_velocity,
    )
    piano_roll = crop_pitch_range(
        piano_roll,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
    )
    segments = segment_sequences(
        piano_roll,
        seq_len=seq_len,
        stride=stride,
    )
    return segments


def load_split_csv(split_name: str) -> pd.DataFrame:
    """
    Load one of:
        train.csv / validation.csv / test.csv
    from data/train_test_split/
    """
    split_name = split_name.strip().lower()

    mapping = {
        "train": "train.csv",
        "validation": "validation.csv",
        "val": "validation.csv",
        "test": "test.csv",
    }

    if split_name not in mapping:
        raise ValueError("split_name must be one of: train, validation/val, test")

    csv_path = Path(SPLIT_DIR) / mapping[split_name]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {csv_path}. Run midi_parser.py first."
        )

    return pd.read_csv(csv_path)


def build_sequences_from_split(
    split_name: str,
    max_files: Optional[int] = None,
    fs: int = FS,
    min_pitch: int = MIN_PITCH,
    max_pitch: int = MAX_PITCH,
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE,
    use_velocity: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build sequence dataset from one split.

    Returns:
        sequences: np.ndarray [N, seq_len, pitch_dim]
        stats_df: per-file stats DataFrame
    """
    df = load_split_csv(split_name)

    if "midi_path" not in df.columns:
        raise ValueError(
            f"{split_name}.csv is missing 'midi_path' column. Re-run midi_parser.py."
        )

    if max_files is not None:
        df = df.head(max_files).copy()

    all_sequences: List[np.ndarray] = []
    stats_rows: List[dict] = []

    for idx, row in df.iterrows():
        midi_path = Path(row["midi_path"])

        try:
            segments = process_single_midi(
                midi_path=midi_path,
                fs=fs,
                min_pitch=min_pitch,
                max_pitch=max_pitch,
                seq_len=seq_len,
                stride=stride,
                use_velocity=use_velocity,
            )

            num_segments = len(segments)
            if num_segments > 0:
                all_sequences.append(segments)

            stats_rows.append(
                {
                    "index": int(idx),
                    "midi_path": str(midi_path),
                    "num_segments": int(num_segments),
                    "status": "ok",
                }
            )

            if verbose and (idx + 1) % 50 == 0:
                print(f"[{split_name}] processed {idx + 1}/{len(df)} files")

        except Exception as e:
            stats_rows.append(
                {
                    "index": int(idx),
                    "midi_path": str(midi_path),
                    "num_segments": 0,
                    "status": f"error: {e}",
                }
            )
            if verbose:
                print(f"[{split_name}] error processing {midi_path}: {e}")

    pitch_dim = max_pitch - min_pitch + 1

    if all_sequences:
        sequences = np.concatenate(all_sequences, axis=0).astype(np.float32)
    else:
        sequences = np.zeros((0, seq_len, pitch_dim), dtype=np.float32)

    stats_df = pd.DataFrame(stats_rows)
    return sequences, stats_df


def save_sequences(
    sequences: np.ndarray,
    output_path: str | Path,
) -> Path:
    """
    Save sequences as .npy
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequences)
    return output_path


def save_stats_csv(
    stats_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save per-file preprocessing stats CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)
    return output_path


def build_and_save_split(
    split_name: str,
    max_files: Optional[int] = None,
    use_velocity: bool = False,
) -> Dict[str, Path]:
    """
    Build sequences for one split and save:
        - sequences .npy
        - stats .csv
    """
    processed_dir = Path(PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sequences, stats_df = build_sequences_from_split(
        split_name=split_name,
        max_files=max_files,
        use_velocity=use_velocity,
    )

    split_tag = split_name.lower()
    if split_tag == "val":
        split_tag = "validation"

    seq_path = processed_dir / f"{split_tag}_sequences.npy"
    stats_path = processed_dir / f"{split_tag}_preprocessing_stats.csv"

    save_sequences(sequences, seq_path)
    save_stats_csv(stats_df, stats_path)

    return {
        "sequences_path": seq_path,
        "stats_path": stats_path,
    }


def inspect_saved_sequences(npy_path: str | Path) -> Dict[str, object]:
    """
    Quick inspection utility for saved .npy sequences
    """
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {npy_path}")

    arr = np.load(npy_path)

    info = {
        "path": str(npy_path),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size > 0 else None,
        "max": float(arr.max()) if arr.size > 0 else None,
        "mean": float(arr.mean()) if arr.size > 0 else None,
    }
    return info


def main() -> None:
    """
    Example usage:
        python -m src.preprocessing.piano_roll
    """
    print("Building piano-roll sequences for MAESTRO...")

    for split_name in ["train", "validation", "test"]:
        print(f"\nProcessing split: {split_name}")
        saved = build_and_save_split(split_name=split_name, max_files=None)

        info = inspect_saved_sequences(saved["sequences_path"])
        print(f"Saved sequences: {saved['sequences_path']}")
        print(f"Saved stats:     {saved['stats_path']}")
        print(f"Array info:      {info}")


if __name__ == "__main__":
    main()