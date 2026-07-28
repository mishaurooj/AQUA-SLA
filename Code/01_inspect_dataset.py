from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "borg_traces_data.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"Reading dataset from: {DATA_PATH}")
    print(f"File size: {DATA_PATH.stat().st_size / (1024 ** 2):.2f} MB")

    # Read only a small sample first.
    sample = pd.read_csv(DATA_PATH, nrows=10_000, low_memory=False)

    print("\nShape of sample:")
    print(sample.shape)

    print("\nColumn names:")
    for index, column in enumerate(sample.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nFirst five rows:")
    print(sample.head())

    print("\nData types:")
    print(sample.dtypes)

    print("\nMissing-value percentage:")
    missing = sample.isna().mean().mul(100).sort_values(ascending=False)
    print(missing.head(30))

    print("\nUnique values in low-cardinality columns:")
    for column in sample.columns:
        unique_count = sample[column].nunique(dropna=False)
        if unique_count <= 20:
            print(f"\n{column}: {unique_count} values")
            print(sample[column].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()