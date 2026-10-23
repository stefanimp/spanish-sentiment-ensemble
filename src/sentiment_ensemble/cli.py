"""Command-line entry points for CPU replay and optional transformer training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import RECIPES
from .ensemble import FILES, LABELS, predict
from .io import load_inputs, read_records, write_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spanish sentiment: cached ensemble replay and component training"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Demonstrate the ensemble using synthetic logits")
    for command in ["check", "replay"]:
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--logits-dir", type=Path, required=True)
        command_parser.add_argument("--test-csv", type=Path, required=True)
        if command == "replay":
            command_parser.add_argument("--output", type=Path, required=True)
            command_parser.add_argument(
                "--reference", type=Path, help="Compare IDs and labels before writing"
            )
    train = sub.add_parser("train", help="Train one component in a fresh output directory")
    train.add_argument("--component", choices=list(RECIPES), required=True)
    train.add_argument("--train-csv", type=Path, required=True)
    train.add_argument("--test-csv", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print the recipe without downloading or training",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            logits = {name: np.eye(4, dtype=np.float64) * 4 for name in FILES}
            print(
                json.dumps(
                    {"synthetic": True, "labels": [LABELS[i] for i in predict(logits)]}, indent=2
                )
            )
            return 0
        if args.command == "train":
            if not 0 <= args.seed < 2**32:
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            records = read_records(args.train_csv, ("ID", "TEXT", "TARGET"), unique_ids=False)
            test = read_records(args.test_csv, ("ID", "TEXT"))
            if set(row["TARGET"] for row in records) != set(LABELS):
                raise ValueError("Training CSV must contain all four P/N/NEU/NONE classes")
            if (args.output_dir / args.component).exists() or (
                args.output_dir / FILES[args.component]
            ).exists():
                raise ValueError("Component outputs already exist; choose a new output directory")
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "component": args.component,
                            "recipe": asdict(RECIPES[args.component]),
                            "train_rows": len(records),
                            "test_rows": len(test),
                            "seed": args.seed,
                            "output": str(args.output_dir / FILES[args.component]),
                        },
                        indent=2,
                    )
                )
                return 0
            try:
                from .training import train_component
            except ImportError as error:
                raise ValueError(
                    'Training dependencies are missing; install with: python -m pip install -e ".[train]"'
                ) from error
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            train_component(
                args.component, args.train_csv, args.test_csv, args.output_dir, seed=args.seed
            )
            return 0
        ids, logits = load_inputs(args.logits_dir, args.test_csv)
        predictions = predict(logits, len(ids))
        if args.command == "check":
            print(
                json.dumps(
                    {"status": "valid", "rows": len(ids), "components": list(FILES)}, indent=2
                )
            )
            return 0
        write_submission(args.output, ids, predictions, args.reference)
        summary = {
            "rows": len(ids),
            "counts": dict(Counter(LABELS[i] for i in predictions).most_common()),
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        if args.reference:
            summary["reference_equal"] = True
        print(json.dumps(summary, indent=2))
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
