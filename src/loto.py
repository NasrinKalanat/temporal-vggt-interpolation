"""LOTO fold generation for temporal-VGGT evaluation.

Implements two leave-one-date-out protocols:
  - target_date: hold out date d as t2; d may appear as t1 or t3 in training.
  - strict: hold out date d completely; d cannot be t1, t2, or t3 in training.

Both protocols select a validation date that is non-adjacent to the test date.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def _days_between(a: str, b: str) -> int:
    return abs((_parse_date(a) - _parse_date(b)).days)


def _select_val_date(test_date: str, all_dates: list[str], fixed_val_date: str | None = None) -> str | None:
    """Return val date. If fixed_val_date is given and valid, use it directly."""
    if fixed_val_date is not None and fixed_val_date != test_date:
        return fixed_val_date
    sorted_dates = sorted(all_dates, key=_parse_date)
    idx = sorted_dates.index(test_date)
    adjacent = set()
    if idx > 0:
        adjacent.add(sorted_dates[idx - 1])
    if idx < len(sorted_dates) - 1:
        adjacent.add(sorted_dates[idx + 1])
    adjacent.add(test_date)

    candidates = [d for d in sorted_dates if d not in adjacent]
    if not candidates:
        return None

    return min(candidates, key=lambda d: _days_between(d, test_date))


def build_loto_folds(
    triplets: list[dict[str, Any]],
    crop: str,
    protocol: str,
    fixed_val_date: str | None = None,
) -> list[dict[str, Any]]:
    """Build LOTO folds for one crop and protocol.

    Args:
        triplets: list of triplet dicts from benchmark_triplets.json
        crop: "corn" or "soybean"
        protocol: "target_date" or "strict"

    Returns:
        List of fold dicts, one per test date.
    """
    if protocol not in ("target_date", "strict"):
        raise ValueError(f"Unknown protocol: {protocol}")

    crop_triplets = [t for t in triplets if t["crop"] == crop]

    # Collect every date that appears in any role to find the temporal edges.
    all_crop_dates = sorted({
        d for t in crop_triplets
        for d in (t["left_date"], t["middle_date"], t["right_date"])
    })
    edge_dates = {all_crop_dates[0], all_crop_dates[-1]} if len(all_crop_dates) >= 2 else set(all_crop_dates)

    # Only non-edge middle dates are valid LOTO test dates (they must have a bracket on both sides).
    test_dates = sorted({t["middle_date"] for t in crop_triplets} - edge_dates)

    folds: list[dict[str, Any]] = []
    for test_date in test_dates:
        test_triplets = [t for t in crop_triplets if t["middle_date"] == test_date]

        if protocol == "target_date":
            candidate_train = [t for t in crop_triplets if t["middle_date"] != test_date]
        else:
            candidate_train = [
                t for t in crop_triplets
                if test_date not in (t["left_date"], t["middle_date"], t["right_date"])
            ]

        val_date = _select_val_date(test_date, test_dates, fixed_val_date)

        if val_date is not None:
            val_triplets = [t for t in candidate_train if t["middle_date"] == val_date]
            if protocol == "strict":
                # Also exclude val_date from all roles in training to avoid leakage.
                train_triplets = [
                    t for t in candidate_train
                    if val_date not in (t["left_date"], t["middle_date"], t["right_date"])
                ]
            else:
                train_triplets = [t for t in candidate_train if t["middle_date"] != val_date]
        else:
            val_triplets = []
            train_triplets = list(candidate_train)

        folds.append({
            "fold_id": f"{protocol}_{test_date}_{crop}",
            "crop": crop,
            "protocol": protocol,
            "test_date": test_date,
            "val_date": val_date,
            "n_train": len(train_triplets),
            "n_val": len(val_triplets),
            "n_test": len(test_triplets),
            "train_triplets": train_triplets,
            "val_triplets": val_triplets,
            "test_triplets": test_triplets,
        })

    return folds


def load_triplets(triplets_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(triplets_path.read_text())
    return payload.get("triplets", [])


def build_all_folds(triplets_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Build all folds for both crops and both protocols.

    Returns dict keyed by protocol name, each a list of fold dicts.
    """
    triplets = load_triplets(triplets_path)
    crops = sorted({t["crop"] for t in triplets})
    protocols = ["target_date", "strict"]

    all_folds: dict[str, list[dict[str, Any]]] = {}
    for protocol in protocols:
        folds: list[dict[str, Any]] = []
        for crop in crops:
            folds.extend(build_loto_folds(triplets, crop, protocol))
        all_folds[protocol] = folds

    return all_folds


def compute_tau(left_date: str, middle_date: str, right_date: str) -> float:
    """Relative temporal position of middle date: (t2-t1)/(t3-t1)."""
    t1 = _parse_date(left_date)
    t2 = _parse_date(middle_date)
    t3 = _parse_date(right_date)
    span = (t3 - t1).days
    if span == 0:
        return 0.5
    return (t2 - t1).days / span

