from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
INPUT_PATH = PROJECT_DIR / "Dataset" / "borg_traces_data.csv"


def describe_column(df: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")

    print(f"\n{column}")
    print("-" * 60)
    print(values.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))


def main() -> None:
    df = pd.read_csv(
        INPUT_PATH,
        usecols=["time", "start_time", "end_time"],
        nrows=100_000,
        low_memory=False,
    )

    for column in ["time", "start_time", "end_time"]:
        describe_column(df, column)

    start = pd.to_numeric(df["start_time"], errors="coerce")
    end = pd.to_numeric(df["end_time"], errors="coerce")

    duration_raw = end - start
    duration_raw = duration_raw[duration_raw >= 0]

    print("\nRaw duration")
    print("-" * 60)
    print(
        duration_raw.describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        )
    )


if __name__ == "__main__":
    main()