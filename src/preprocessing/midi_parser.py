from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pretty_midi

from src.config import METADATA_CSV, RAW_MIDI_DIR, SPLIT_DIR


VALID_SPLITS = {"train", "validation", "test"}
REQUIRED_COLUMNS = {
    "canonical_composer",
    "canonical_title",
    "split",
    "year",
    "midi_filename",
    "audio_filename",
    "duration",
}


def _normalize_split_name(split_value: str) -> str:
    """
    Normalize split names into one of:
    train / validation / test
    """
    s = str(split_value).strip().lower()

    mapping = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
        "test": "test",
        "testing": "test",
    }

    return mapping.get(s, s)


def load_metadata(csv_path: str = METADATA_CSV) -> pd.DataFrame:
    """
    Load MAESTRO metadata CSV and standardize important columns.

    Returns:
        pd.DataFrame with extra columns:
            - split
            - midi_path
            - file_exists
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata CSV is missing required columns: {sorted(missing_cols)}"
        )

    df = df.copy()
    df["split"] = df["split"].apply(_normalize_split_name)

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Unexpected split names found: {sorted(invalid_splits)}")

    raw_midi_root = Path(RAW_MIDI_DIR)

    # Full path to each MIDI file
    df["midi_path"] = df["midi_filename"].apply(
        lambda x: str((raw_midi_root / str(x)).resolve())
    )

    # Whether the MIDI file exists locally
    df["file_exists"] = df["midi_path"].apply(lambda x: Path(x).exists())

    return df


def summarize_metadata(df: pd.DataFrame) -> Dict[str, object]:
    """
    Return a small summary dict for quick inspection.
    """
    summary = {
        "total_rows": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "num_existing_files": int(df["file_exists"].sum()),
        "num_missing_files": int((~df["file_exists"]).sum()),
        "num_unique_composers": int(df["canonical_composer"].nunique()),
        "num_unique_titles": int(df["canonical_title"].nunique()),
        "duration_hours": round(float(df["duration"].sum()) / 3600.0, 2),
    }
    return summary


def get_missing_files(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose MIDI files are missing locally.
    """
    return df.loc[~df["file_exists"]].copy()


def save_missing_file_report(
    df: pd.DataFrame, output_path: Optional[str] = None
) -> Optional[Path]:
    """
    Save a CSV report for missing files, if any.
    """
    missing_df = get_missing_files(df)
    if missing_df.empty:
        return None

    if output_path is None:
        output_path = str(Path(SPLIT_DIR) / "missing_files.csv")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(output_file, index=False)
    return output_file


def build_split_csvs(
    df: pd.DataFrame,
    out_dir: str = SPLIT_DIR,
    only_existing_files: bool = True,
) -> Dict[str, Path]:
    """
    Build train.csv / validation.csv / test.csv files from metadata.

    Args:
        df: metadata DataFrame
        out_dir: directory to save split CSVs
        only_existing_files: if True, only save rows whose MIDI exists locally

    Returns:
        dict: {"train": Path(...), "validation": Path(...), "test": Path(...)}
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()
    if only_existing_files:
        work_df = work_df.loc[work_df["file_exists"]].copy()

    split_to_filename = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }

    saved_paths: Dict[str, Path] = {}

    for split_name, filename in split_to_filename.items():
        split_df = work_df.loc[work_df["split"] == split_name].copy()
        save_path = out_path / filename
        split_df.to_csv(save_path, index=False)
        saved_paths[split_name] = save_path

    return saved_paths


def resolve_midi_path(relative_midi_filename: str) -> Path:
    """
    Convert a relative MIDI filename from metadata into a full local path.
    Example:
        2018/xxx.midi -> data/raw_midi/maestro-v3.0.0/2018/xxx.midi
    """
    return (Path(RAW_MIDI_DIR) / relative_midi_filename).resolve()


def get_midi_duration(midi_path: str) -> float:
    """
    Read a MIDI file and return its end time in seconds.
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    return float(midi_data.get_end_time())


def parse_midi_notes(midi_path: str) -> pd.DataFrame:
    """
    Parse notes from one MIDI file into a DataFrame.

    Returns columns:
        pitch, velocity, start, end, duration,
        instrument_program, instrument_name, is_drum
    """
    midi_file = Path(midi_path)
    if not midi_file.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file}")

    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    rows: List[dict] = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = float(note.start)
            end = float(note.end)
            duration = max(0.0, end - start)

            rows.append(
                {
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "instrument_program": int(instrument.program),
                    "instrument_name": instrument.name if instrument.name else "",
                    "is_drum": bool(instrument.is_drum),
                }
            )

    notes_df = pd.DataFrame(rows)

    if notes_df.empty:
        return pd.DataFrame(
            columns=[
                "pitch",
                "velocity",
                "start",
                "end",
                "duration",
                "instrument_program",
                "instrument_name",
                "is_drum",
            ]
        )

    notes_df = notes_df.sort_values(["start", "pitch"]).reset_index(drop=True)
    return notes_df


def inspect_sample_from_metadata(
    df: pd.DataFrame, split: str = "train", row_index: int = 0
) -> pd.DataFrame:
    """
    Pick one MIDI from the metadata DataFrame and parse its notes.
    Useful for quick testing.
    """
    split = _normalize_split_name(split)
    split_df = df.loc[df["split"] == split].copy()

    if split_df.empty:
        raise ValueError(f"No rows found for split='{split}'")

    split_df = split_df.reset_index(drop=True)

    if row_index < 0 or row_index >= len(split_df):
        raise IndexError(f"row_index out of range for split='{split}'")

    midi_path = split_df.loc[row_index, "midi_path"]
    return parse_midi_notes(midi_path)


def main() -> None:
    """
    Example usage:
        python -m src.preprocessing.midi_parser
    """
    print("Loading MAESTRO metadata...")
    df = load_metadata()

    summary = summarize_metadata(df)
    print("\nDataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    saved_paths = build_split_csvs(df, only_existing_files=True)

    print("\nSaved split CSV files:")
    for split_name, path in saved_paths.items():
        print(f"  {split_name}: {path}")

    missing_report = save_missing_file_report(df)
    if missing_report is not None:
        print(f"\nMissing files report saved to: {missing_report}")
    else:
        print("\nNo missing MIDI files found.")

    # Quick sample parse
    try:
        sample_notes = inspect_sample_from_metadata(df, split="train", row_index=0)
        print("\nSample parsed notes:")
        print(sample_notes.head())
        print(f"Total notes in sample file: {len(sample_notes)}")
    except Exception as e:
        print(f"\nCould not parse sample MIDI: {e}")


if __name__ == "__main__":
    main()