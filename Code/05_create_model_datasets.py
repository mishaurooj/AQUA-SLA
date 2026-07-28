from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

INPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_prepared.csv"
)

EARLY_OUTPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_early_prediction.csv"
)

RUNTIME_OUTPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_runtime_prediction.csv"
)

METADATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "model_dataset_metadata.json"
)

CHUNK_SIZE = 50_000

IDENTIFIER_COLUMNS = [
    "record_id",
    "collection_id",
    "instance_index",
    "machine_id",
    "cluster",
    "time",
    "time_seconds",
]

TARGET_COLUMNS = [
    "sla_failure",
    "sla_violation",
]

EARLY_FEATURES = [
    "scheduling_class",
    "collection_type",
    "priority",
    "vertical_scaling",
    "scheduler",
    "requested_cpu",
    "requested_memory",
    "requested_cpu_missing",
    "requested_memory_missing",
]

RUNTIME_FEATURES = EARLY_FEATURES + [
    "assigned_memory",
    "page_cache_memory",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "sample_rate",
    "average_cpu",
    "average_memory",
    "maximum_cpu",
    "maximum_memory",
    "random_sample_cpu",
    "duration_seconds",
    "average_cpu_request_ratio",
    "maximum_cpu_request_ratio",
    "average_memory_request_ratio",
    "maximum_memory_request_ratio",
    "cpu_peak_to_average_ratio",
    "memory_peak_to_average_ratio",
    "cpu_request_headroom",
    "memory_request_headroom",
    "cpu_request_exceeded",
    "memory_request_exceeded",
    "average_resource_pressure",
    "maximum_resource_pressure",
    "cpu_distribution_minimum",
    "cpu_distribution_median",
    "cpu_distribution_maximum",
    "cpu_distribution_mean",
    "cpu_distribution_std",
    "cpu_distribution_range",
    "tail_cpu_distribution_minimum",
    "tail_cpu_distribution_median",
    "tail_cpu_distribution_maximum",
    "tail_cpu_distribution_mean",
    "tail_cpu_distribution_std",
    "tail_cpu_distribution_range",
    "cpu_distribution_burst_ratio",
    "tail_cpu_burst_ratio",
    "average_cpu_missing",
    "average_memory_missing",
    "maximum_cpu_missing",
    "maximum_memory_missing",
    "cycles_per_instruction_missing",
    "memory_accesses_per_instruction_missing",
]


def clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=[np.number]).columns

    frame[numeric_columns] = frame[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return frame


def write_chunk(
    frame: pd.DataFrame,
    output_path: Path,
    first_chunk: bool,
) -> None:
    frame.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    EARLY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    total_rows = 0

    early_columns = IDENTIFIER_COLUMNS + EARLY_FEATURES + TARGET_COLUMNS
    runtime_columns = IDENTIFIER_COLUMNS + RUNTIME_FEATURES + TARGET_COLUMNS

    reader = pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        missing_early = [
            column
            for column in early_columns
            if column not in chunk.columns
        ]

        missing_runtime = [
            column
            for column in runtime_columns
            if column not in chunk.columns
        ]

        if missing_early:
            raise ValueError(
                f"Missing early columns: {missing_early}"
            )

        if missing_runtime:
            raise ValueError(
                f"Missing runtime columns: {missing_runtime}"
            )

        early = clean_numeric_frame(
            chunk[early_columns].copy()
        )

        runtime = clean_numeric_frame(
            chunk[runtime_columns].copy()
        )

        write_chunk(
            early,
            EARLY_OUTPUT_PATH,
            first_chunk,
        )

        write_chunk(
            runtime,
            RUNTIME_OUTPUT_PATH,
            first_chunk,
        )

        total_rows += len(chunk)
        first_chunk = False

        print(
            f"Chunk {chunk_number}: "
            f"processed={len(chunk):,}, "
            f"total={total_rows:,}"
        )

    metadata = {
        "source": str(INPUT_PATH),
        "rows": total_rows,
        "early_output": str(EARLY_OUTPUT_PATH),
        "runtime_output": str(RUNTIME_OUTPUT_PATH),
        "early_feature_count": len(EARLY_FEATURES),
        "runtime_feature_count": len(RUNTIME_FEATURES),
        "early_features": EARLY_FEATURES,
        "runtime_features": RUNTIME_FEATURES,
        "targets": TARGET_COLUMNS,
        "excluded_leakage_columns": [
            "event",
            "failed",
            "instance_events_type",
            "collections_events_type",
            "start_time",
            "end_time",
            "start_time_seconds",
            "end_time_seconds",
        ],
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nModel datasets created")
    print(f"Rows: {total_rows:,}")
    print(f"Early features: {len(EARLY_FEATURES)}")
    print(f"Runtime features: {len(RUNTIME_FEATURES)}")
    print(f"Early dataset: {EARLY_OUTPUT_PATH}")
    print(f"Runtime dataset: {RUNTIME_OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()