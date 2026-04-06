from __future__ import annotations

import pandas as pd


def missing_columns_in_frame(frame: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def validate_json_records(records: object, required_columns: list[str]) -> tuple[bool, str]:
    if not isinstance(records, list):
        return False, "JSON must be an array of customer objects."

    if not all(isinstance(item, dict) for item in records):
        return False, "Every item in JSON array must be an object."

    missing_keys: list[str] = []
    for item in records:
        for column in required_columns:
            if column not in item:
                missing_keys.append(column)

    if missing_keys:
        unique_missing = ", ".join(sorted(set(missing_keys)))
        return False, f"Missing keys in one or more records: {unique_missing}"

    return True, ""
