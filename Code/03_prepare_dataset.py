from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

INPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "borg_traces_data.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_prepared.csv"
)

METADATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_preparation_metadata.json"
)

CHUNK_SIZE = 50_000

COMPOSITE_VIOLATION_EVENTS = {
    "FAIL",
    "LOST",
    "EVICT",
    "KILL",
}

# These are retained for grouping, traceability, and simulation.
IDENTIFIER_COLUMNS = [
    "collection_id",
    "instance_index",
    "machine_id",
    "cluster",
]

# These can be considered for early prediction.
CONTEXT_COLUMNS = [
    "time",
    "scheduling_class",
    "collection_type",
    "priority",
    "alloc_collection_id",
    "vertical_scaling",
    "scheduler",
    "start_time",
    "end_time",
    "assigned_memory",
    "page_cache_memory",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "sample_rate",
]

STRUCTURED_COLUMNS = [
    "resource_request",
    "average_usage",
    "maximum_usage",
    "random_sample_usage",
]


def safe_parse_dictionary(value: Any) -> dict[str, Any]:
    """
    Parse a string such as:
    "{'cpus': 0.02, 'memory': 0.01}"

    Returns an empty dictionary when parsing fails.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if isinstance(value, float) and np.isnan(value):
        return {}

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "{}"}:
        return {}

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def extract_dictionary_value(
    series: pd.Series,
    key: str,
) -> pd.Series:
    """
    Extract a numeric key from a Series of dictionary strings.
    """
    return series.map(
        lambda value: safe_parse_dictionary(value).get(key, np.nan)
    ).pipe(pd.to_numeric, errors="coerce")


def parse_numeric_array(value: Any) -> np.ndarray:
    """
    Parse a NumPy-style string array such as:
    "[0.01 0.02 0.03]"

    Returns an empty array when the value cannot be parsed.
    """
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, float) and np.isnan(value):
        return np.array([], dtype=float)

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "[]"}:
        return np.array([], dtype=float)

    text = text.strip("[]")
    values = np.fromstring(text, sep=" ", dtype=float)

    return values


def add_distribution_features(
    frame: pd.DataFrame,
    source_column: str,
    prefix: str,
) -> None:
    """
    Add stable summary features from CPU usage distributions.
    """
    parsed = frame[source_column].map(parse_numeric_array)

    frame[f"{prefix}_minimum"] = parsed.map(
        lambda values: values[0] if values.size else np.nan
    )

    frame[f"{prefix}_median"] = parsed.map(
        lambda values: np.median(values) if values.size else np.nan
    )

    frame[f"{prefix}_maximum"] = parsed.map(
        lambda values: values[-1] if values.size else np.nan
    )

    frame[f"{prefix}_mean"] = parsed.map(
        lambda values: np.mean(values) if values.size else np.nan
    )

    frame[f"{prefix}_std"] = parsed.map(
        lambda values: np.std(values) if values.size else np.nan
    )

    frame[f"{prefix}_range"] = (
        frame[f"{prefix}_maximum"]
        - frame[f"{prefix}_minimum"]
    )


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Calculate a ratio while avoiding division by zero.
    """
    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = (
        chunk.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    prepared = pd.DataFrame(index=chunk.index)

    # Stable record identifier from the original CSV.
    if "unnamed:_0" in chunk.columns:
        prepared["record_id"] = pd.to_numeric(
            chunk["unnamed:_0"],
            errors="coerce",
        )
    elif "unnamed:_0" not in chunk.columns and "unnamed:_0" not in prepared:
        # The actual normalized form of "Unnamed: 0" remains "unnamed:_0".
        prepared["record_id"] = np.arange(len(chunk))

    for column in IDENTIFIER_COLUMNS + CONTEXT_COLUMNS:
        if column in chunk.columns:
            prepared[column] = pd.to_numeric(
                chunk[column],
                errors="coerce",
            )

    # Event values are used only to construct targets.
    prepared["event"] = (
        chunk["event"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Narrow target: only explicit FAIL records.
    prepared["sla_failure"] = (
        prepared["event"] == "FAIL"
    ).astype("int8")

    # Broader operational disruption target.
    prepared["sla_violation"] = (
        prepared["event"].isin(COMPOSITE_VIOLATION_EVENTS)
    ).astype("int8")

    # Parse requested resources.
    prepared["requested_cpu"] = extract_dictionary_value(
        chunk["resource_request"],
        "cpus",
    )

    prepared["requested_memory"] = extract_dictionary_value(
        chunk["resource_request"],
        "memory",
    )

    # Parse average usage.
    prepared["average_cpu"] = extract_dictionary_value(
        chunk["average_usage"],
        "cpus",
    )

    prepared["average_memory"] = extract_dictionary_value(
        chunk["average_usage"],
        "memory",
    )

    # Parse maximum usage.
    prepared["maximum_cpu"] = extract_dictionary_value(
        chunk["maximum_usage"],
        "cpus",
    )

    prepared["maximum_memory"] = extract_dictionary_value(
        chunk["maximum_usage"],
        "memory",
    )

    # Parse random CPU sample. Memory is None in this dataset sample.
    prepared["random_sample_cpu"] = extract_dictionary_value(
        chunk["random_sample_usage"],
        "cpus",
    )

    prepared["random_sample_memory"] = extract_dictionary_value(
        chunk["random_sample_usage"],
        "memory",
    )

    # Time conversion: trace values are in microseconds.
    prepared["time_seconds"] = prepared["time"] / 1_000_000.0
    prepared["start_time_seconds"] = (
        prepared["start_time"] / 1_000_000.0
    )
    prepared["end_time_seconds"] = (
        prepared["end_time"] / 1_000_000.0
    )

    prepared["duration_seconds"] = (
        prepared["end_time"]
        - prepared["start_time"]
    ) / 1_000_000.0

    prepared.loc[
        prepared["duration_seconds"] < 0,
        "duration_seconds",
    ] = np.nan

    # Resource request-to-usage relationships.
    prepared["average_cpu_request_ratio"] = safe_ratio(
        prepared["average_cpu"],
        prepared["requested_cpu"],
    )

    prepared["maximum_cpu_request_ratio"] = safe_ratio(
        prepared["maximum_cpu"],
        prepared["requested_cpu"],
    )

    prepared["average_memory_request_ratio"] = safe_ratio(
        prepared["average_memory"],
        prepared["requested_memory"],
    )

    prepared["maximum_memory_request_ratio"] = safe_ratio(
        prepared["maximum_memory"],
        prepared["requested_memory"],
    )

    prepared["cpu_peak_to_average_ratio"] = safe_ratio(
        prepared["maximum_cpu"],
        prepared["average_cpu"],
    )

    prepared["memory_peak_to_average_ratio"] = safe_ratio(
        prepared["maximum_memory"],
        prepared["average_memory"],
    )

    prepared["cpu_request_headroom"] = (
        prepared["requested_cpu"]
        - prepared["average_cpu"]
    )

    prepared["memory_request_headroom"] = (
        prepared["requested_memory"]
        - prepared["average_memory"]
    )

    prepared["cpu_request_exceeded"] = (
        prepared["maximum_cpu"]
        > prepared["requested_cpu"]
    ).astype("int8")

    prepared["memory_request_exceeded"] = (
        prepared["maximum_memory"]
        > prepared["requested_memory"]
    ).astype("int8")

    # Combined pressure indicators.
    prepared["average_resource_pressure"] = prepared[
        [
            "average_cpu_request_ratio",
            "average_memory_request_ratio",
        ]
    ].mean(axis=1)

    prepared["maximum_resource_pressure"] = prepared[
        [
            "maximum_cpu_request_ratio",
            "maximum_memory_request_ratio",
        ]
    ].mean(axis=1)

    # Parse CPU distribution vectors.
    add_distribution_features(
        chunk,
        "cpu_usage_distribution",
        "cpu_distribution",
    )

    add_distribution_features(
        chunk,
        "tail_cpu_usage_distribution",
        "tail_cpu_distribution",
    )

    distribution_columns = [
        column
        for column in chunk.columns
        if column.startswith("cpu_distribution_")
        or column.startswith("tail_cpu_distribution_")
    ]

    # add_distribution_features currently writes into chunk.
    for column in distribution_columns:
        prepared[column] = chunk[column]

    # Additional distribution-based burst indicators.
    prepared["cpu_distribution_burst_ratio"] = safe_ratio(
        prepared["cpu_distribution_maximum"],
        prepared["cpu_distribution_median"],
    )

    prepared["tail_cpu_burst_ratio"] = safe_ratio(
        prepared["tail_cpu_distribution_maximum"],
        prepared["tail_cpu_distribution_minimum"],
    )

    # Missingness indicators may carry useful operational information.
    important_columns = [
        "requested_cpu",
        "requested_memory",
        "average_cpu",
        "average_memory",
        "maximum_cpu",
        "maximum_memory",
        "cycles_per_instruction",
        "memory_accesses_per_instruction",
    ]

    for column in important_columns:
        if column in prepared.columns:
            prepared[f"{column}_missing"] = (
                prepared[column].isna().astype("int8")
            )

    # Remove impossible and extreme numeric artifacts.
    numeric_columns = prepared.select_dtypes(
        include=[np.number]
    ).columns

    prepared[numeric_columns] = prepared[
        numeric_columns
    ].replace([np.inf, -np.inf], np.nan)

    return prepared


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_chunk = True
    total_rows = 0
    target_counts = {
        "sla_failure": 0,
        "sla_violation": 0,
    }

    output_columns: list[str] | None = None

    reader = pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        prepared = prepare_chunk(chunk)

        if output_columns is None:
            output_columns = list(prepared.columns)

        prepared.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        total_rows += len(prepared)

        target_counts["sla_failure"] += int(
            prepared["sla_failure"].sum()
        )

        target_counts["sla_violation"] += int(
            prepared["sla_violation"].sum()
        )

        print(
            f"Chunk {chunk_number}: "
            f"saved={len(prepared):,}, "
            f"total={total_rows:,}"
        )

    metadata = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "rows": total_rows,
        "columns": output_columns,
        "column_count": len(output_columns or []),
        "sla_failure_count": target_counts["sla_failure"],
        "sla_failure_percentage": (
            target_counts["sla_failure"] / total_rows * 100
        ),
        "sla_violation_count": target_counts["sla_violation"],
        "sla_violation_percentage": (
            target_counts["sla_violation"] / total_rows * 100
        ),
        "time_unit": "microseconds",
        "duration_output_unit": "seconds",
        "composite_violation_events": sorted(
            COMPOSITE_VIOLATION_EVENTS
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nPreparation complete")
    print(f"Rows: {total_rows:,}")
    print(f"Columns: {len(output_columns or []):,}")

    print(
        "Failure target: "
        f"{target_counts['sla_failure']:,} "
        f"({metadata['sla_failure_percentage']:.2f}%)"
    )

    print(
        "Composite violation target: "
        f"{target_counts['sla_violation']:,} "
        f"({metadata['sla_violation_percentage']:.2f}%)"
    )

    print(f"Prepared dataset: {OUTPUT_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()