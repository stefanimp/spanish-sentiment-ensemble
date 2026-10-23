"""CSV contracts and safe loading of numeric prediction artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .ensemble import FILES, LABELS


def read_records(
    path: Path, required: tuple[str, ...], *, unique_ids: bool = True
) -> list[dict[str, str]]:
    """Read UTF-8 CSV records without changing IDs or their row order."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required).issubset(columns):
            raise ValueError(f"{path.name}: expected columns {', '.join(required)}")
        records = list(reader)
    if not records:
        raise ValueError(f"{path.name}: file has no records")
    ids = []
    for row in records:
        if any(row.get(column) is None for column in required) or None in row:
            raise ValueError(f"{path.name}: malformed CSV record")
        identifier = row["ID"]
        if not identifier.strip():
            raise ValueError(f"{path.name}: ID cannot be empty")
        ids.append(identifier)
    if unique_ids and len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate IDs")
    return records


def load_inputs(directory: Path, test_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    records = read_records(test_path, ("ID",))
    logits = {name: np.load(directory / file, allow_pickle=False) for name, file in FILES.items()}
    return [row["ID"] for row in records], logits


def write_submission(
    path: Path, ids: list[str], predictions: np.ndarray, reference: Path | None = None
) -> None:
    """Compare an optional reference before exclusively creating the output file."""
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    rows = [
        {"ID": identifier, "TARGET": LABELS[int(prediction)]}
        for identifier, prediction in zip(ids, predictions, strict=True)
    ]
    if reference is not None:
        expected = read_records(reference, ("ID", "TARGET"))
        if any(set(row) != {"ID", "TARGET"} for row in expected) or expected != rows:
            raise ValueError("Reference mismatch: ID order, labels or schema differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "TARGET"], lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
