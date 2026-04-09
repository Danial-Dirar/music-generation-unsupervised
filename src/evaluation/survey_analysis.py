from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "participant_id",
    "sample_id",
    "model_type",
    "rating_musicality",
    "rating_coherence",
    "rating_rhythm",
    "rating_overall",
    "comment",
}


def load_survey(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Survey CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def clean_survey(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rating_cols = [
        "rating_musicality",
        "rating_coherence",
        "rating_rhythm",
        "rating_overall",
    ]

    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=rating_cols)
    return df


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("model_type")[
            [
                "rating_musicality",
                "rating_coherence",
                "rating_rhythm",
                "rating_overall",
            ]
        ]
        .mean()
        .reset_index()
    )

    grouped["human_score"] = grouped[
        [
            "rating_musicality",
            "rating_coherence",
            "rating_rhythm",
            "rating_overall",
        ]
    ].mean(axis=1)

    grouped = grouped.sort_values("human_score", ascending=False).reset_index(drop=True)
    return grouped


def compute_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["model_type", "sample_id"])[
            [
                "rating_musicality",
                "rating_coherence",
                "rating_rhythm",
                "rating_overall",
            ]
        ]
        .mean()
        .reset_index()
    )

    grouped["human_score"] = grouped[
        [
            "rating_musicality",
            "rating_coherence",
            "rating_rhythm",
            "rating_overall",
        ]
    ].mean(axis=1)

    return grouped.sort_values(["model_type", "sample_id"]).reset_index(drop=True)


def compare_before_after(
    before_score: float,
    after_score: float,
) -> dict[str, float]:
    improvement = after_score - before_score
    pct_change = 0.0 if before_score == 0 else (improvement / before_score) * 100.0

    return {
        "before_score": before_score,
        "after_score": after_score,
        "absolute_improvement": improvement,
        "percent_change": pct_change,
    }


def save_summary_tables(
    model_summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "survey_model_summary.csv"
    sample_path = output_dir / "survey_sample_summary.csv"

    model_summary.to_csv(model_path, index=False)
    sample_summary.to_csv(sample_path, index=False)

    return model_path, sample_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    survey_csv = project_root / "outputs" / "survey_results" / "survey_template.csv"
    output_dir = project_root / "outputs" / "survey_results"

    df = load_survey(survey_csv)
    df = clean_survey(df)

    if df.empty:
        print("No completed survey ratings found yet.")
        return

    model_summary = compute_model_summary(df)
    sample_summary = compute_sample_summary(df)

    model_path, sample_path = save_summary_tables(model_summary, sample_summary, output_dir)

    print("Model-level human evaluation summary:")
    print(model_summary)
    print()
    print(f"Saved model summary to: {model_path}")
    print(f"Saved sample summary to: {sample_path}")

    # Example: compare AE vs Transformer if both exist
    model_scores = dict(zip(model_summary["model_type"], model_summary["human_score"]))

    if "AE" in model_scores and "Transformer" in model_scores:
        cmp_result = compare_before_after(
            before_score=float(model_scores["AE"]),
            after_score=float(model_scores["Transformer"]),
        )
        print()
        print("AE vs Transformer comparison:")
        for k, v in cmp_result.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()