from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

DATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_prepared.csv"
)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH,
        nrows=10_000,
        low_memory=False,
    )

    print("Sample shape:", df.shape)

    print("\nColumns:")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\nFirst rows:")
    print(df.head().to_string())

    print("\nTarget distribution:")
    for target in ["sla_failure", "sla_violation"]:
        print(f"\n{target}")
        print(
            df[target]
            .value_counts(normalize=False)
            .sort_index()
        )
        print(
            df[target]
            .value_counts(normalize=True)
            .sort_index()
            .mul(100)
            .round(2)
        )

    selected = [
        "requested_cpu",
        "average_cpu",
        "maximum_cpu",
        "requested_memory",
        "average_memory",
        "maximum_memory",
        "duration_seconds",
        "average_cpu_request_ratio",
        "maximum_cpu_request_ratio",
        "sla_failure",
        "sla_violation",
    ]

    selected = [
        column for column in selected
        if column in df.columns
    ]

    print("\nSelected feature summary:")
    print(
        df[selected]
        .describe()
        .transpose()
        .to_string()
    )

    print("\nMissing percentages:")
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )
    print(missing.head(30).to_string())


if __name__ == "__main__":
    main()