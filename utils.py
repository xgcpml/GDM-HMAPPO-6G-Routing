from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_metrics_json(metrics: Dict[str, float], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
