from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "borg_traces_data.csv"

STRUCTURED_COLUMNS = [
    "resource_request",
    "average_usage",
    "maximum_usage",
    "random_sample_usage",
    "cpu_usage_distribution",
    "tail_cpu_usage_distribution",
    "constraint",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH,
        nrows=5_000,
        low_memory=False,
    )

    for column in STRUCTURED_COLUMNS:
        print("\n" + "=" * 100)
        print(f"COLUMN: {column}")
        print("=" * 100)

        if column not in df.columns:
            print("Column not found.")
            continue

        values = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(15)
        )

        for index, value in enumerate(values, start=1):
            print(f"\n[{index}] {value}")


if __name__ == "__main__":
    main()