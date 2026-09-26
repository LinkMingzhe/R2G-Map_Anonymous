from __future__ import annotations

import math

from pathlib import Path

from typing import Dict, List, Sequence, Tuple

import numpy as np

RECALL_K = (1, 5, 10, 20, 30, 50)

NDCG_K = (5, 10, 20, 30, 50)

def load_gt_rows(path: Path) -> List[Tuple[str, List[int]]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = [part.strip() for part in raw.split(",") if part.strip()]
        if len(fields) not in (2, 3):
            raise ValueError(f"{path}:{line_number}: expected one or two GT items")
        rows.append((fields[0], [int(value) for value in fields[1:]]))
    return rows

def multi_gt_metrics(
    rankings: Sequence[Sequence[int]],
    gt_rows: Sequence[Tuple[str, Sequence[int]]],
) -> Dict[str, float]:
    if len(rankings) != len(gt_rows):
        raise ValueError("Ranking/GT row count mismatch")
    recall_values = {k: [] for k in RECALL_K}
    ndcg_values = {k: [] for k in NDCG_K}
    for ranked_ids, (_, gt_items_raw) in zip(rankings, gt_rows):
        gt_items = set(int(value) for value in gt_items_raw)
        if not gt_items:
            raise ValueError("Encountered empty GT set")
        for k in RECALL_K:
            recall_values[k].append(
                len(gt_items.intersection(int(value) for value in ranked_ids[:k]))
                / len(gt_items)
            )
        for k in NDCG_K:
            dcg = 0.0
            for rank, item_id in enumerate(ranked_ids[:k], 1):
                if int(item_id) in gt_items:
                    dcg += 1.0 / math.log2(rank + 1.0)
            ideal_hits = min(len(gt_items), k)
            idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
            ndcg_values[k].append(dcg / idcg)
    metrics = {
        "sample_count": float(len(rankings)),
        "one_gt_samples": float(sum(len(items) == 1 for _, items in gt_rows)),
        "two_gt_samples": float(sum(len(items) == 2 for _, items in gt_rows)),
    }
    metrics.update({f"recall@{k}": float(np.mean(values)) for k, values in recall_values.items()})
    metrics.update({f"ndcg@{k}": float(np.mean(values)) for k, values in ndcg_values.items()})
    return metrics
