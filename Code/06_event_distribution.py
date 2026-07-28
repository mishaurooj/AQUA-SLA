from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
INPUT_PATH = PROJECT_DIR / "Dataset" / "borg_traces_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "event_distribution.csv"

CHUNK_SIZE = 100_000

VIOLATION_EVENTS = {
    "FAIL",
    "LOST",
    "EVICT",
    "KILL",
}


def main() -> None:
    event_counts = pd.Series(dtype="int64")
    total_rows = 0
    failure_rows = 0
    violation_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            usecols=["event", "failed"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        chunk["event"] = (
            chunk["event"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        counts = chunk["event"].value_counts()
        event_counts = event_counts.add(counts, fill_value=0)

        total_rows += len(chunk)
        failure_rows += int((chunk["event"] == "FAIL").sum())
        violation_rows += int(
            chunk["event"].isin(VIOLATION_EVENTS).sum()
        )

        print(
            f"Chunk {chunk_number}: "
            f"processed={total_rows:,}"
        )

    event_counts = event_counts.astype("int64").sort_values(
        ascending=False
    )

    report = event_counts.rename_axis("event").reset_index(
        name="count"
    )
    report["percentage"] = report["count"] / total_rows * 100

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nFull event distribution:")
    print(report.to_string(index=False))

    print(f"\nTotal rows: {total_rows:,}")
    print(
        f"Failure target: {failure_rows:,} "
        f"({failure_rows / total_rows * 100:.2f}%)"
    )
    print(
        f"Composite violation target: {violation_rows:,} "
        f"({violation_rows / total_rows * 100:.2f}%)"
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()