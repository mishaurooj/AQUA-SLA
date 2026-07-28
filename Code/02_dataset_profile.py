from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "borg_traces_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "dataset_profile.csv"

CHUNK_SIZE = 100_000


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    total_rows = 0
    total_missing = None
    columns = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        if columns is None:
            columns = list(chunk.columns)
            total_missing = pd.Series(0, index=columns, dtype="int64")

        total_rows += len(chunk)
        total_missing = total_missing.add(
            chunk.isna().sum(),
            fill_value=0,
        )

        print(
            f"Processed chunk {chunk_number}: "
            f"{len(chunk):,} rows, total={total_rows:,}"
        )

    profile = pd.DataFrame(
        {
            "column": columns,
            "missing_count": total_missing.values,
        }
    )

    profile["total_rows"] = total_rows
    profile["missing_percent"] = (
        profile["missing_count"] / total_rows * 100
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_PATH, index=False)

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Total columns: {len(columns)}")
    print(f"Profile saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()