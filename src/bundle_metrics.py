from pathlib import Path

import numpy as np

import torch



BUNDLE_METRIC_TOPK = [1, 5, 10]

FEATURE_BANK = None

def load_feature_bank():
    if FEATURE_BANK is None:
        raise ValueError("Set FEATURE_BANK from the released CLHE features.")
    return FEATURE_BANK


def bundle_metrics_from_records(records, topk_values=BUNDLE_METRIC_TOPK):
    grouped = {}
    for record in records:
        grouped.setdefault(str(record["bundle_key"]), []).append(record)
    grouped = {key: value for key, value in grouped.items() if len(value) >= 2}
    if not grouped:
        return {}

    bundle_rows = []
    for recs in grouped.values():
        recs = sorted(recs, key=lambda item: (int(item.get("sample_index", 0)), int(item["target_item_id"])))[:2]
        label_ids = [int(item["target_item_id"]) for item in recs]
        top_ids = []
        for item in recs:
            candidates = [int(x) for x in item.get("top10_candidate_item_ids", [])]
            if not candidates:
                candidates = [-1]
            top_ids.append(candidates)
        bundle_rows.append((label_ids, top_ids))

    feature_bank = load_feature_bank()
    output = {"sample_count": len(grouped)}
    for k in topk_values:
        recalls, precisions, jaccards = [], [], []
        label_matrix = []
        rank_pred_matrices = []
        for label_ids, top_ids in bundle_rows:
            predicted_union = set()
            for rank_idx in range(k):
                rank_pred = []
                for candidates in top_ids:
                    pred_id = candidates[min(rank_idx, len(candidates) - 1)]
                    predicted_union.add(pred_id)
                    rank_pred.append(pred_id)
                rank_pred_matrices.append(rank_pred)
                label_matrix.append(label_ids)
            label_set = set(label_ids)
            recalls.append(len(predicted_union & label_set) / max(len(label_set), 1))
            precisions.append(len(predicted_union & label_set) / max(len(predicted_union), 1))
            jaccards.append(len(predicted_union & label_set) / max(len(predicted_union | label_set), 1))

        label_matrix = np.asarray(label_matrix, dtype=np.int64)
        rank_pred_matrices = np.asarray(rank_pred_matrices, dtype=np.int64)
        pred_feat = feature_bank[rank_pred_matrices]
        label_feat = feature_bank[label_matrix]
        d00 = 1.0 - np.sum(pred_feat[:, 0, :] * label_feat[:, 0, :], axis=1)
        d11 = 1.0 - np.sum(pred_feat[:, 1, :] * label_feat[:, 1, :], axis=1)
        d01 = 1.0 - np.sum(pred_feat[:, 0, :] * label_feat[:, 1, :], axis=1)
        d10 = 1.0 - np.sum(pred_feat[:, 1, :] * label_feat[:, 0, :], axis=1)
        rank_distances = np.minimum(d00 + d11, d01 + d10).reshape(len(bundle_rows), k) / 2.0

        output[f"recall@{k}"] = float(np.mean(recalls))
        output[f"precision@{k}"] = float(np.mean(precisions))
        output[f"jaccard@{k}"] = float(np.mean(jaccards))
        output[f"oas_max@{k}"] = float(np.mean(rank_distances.max(axis=1)))
        output[f"oas_min@{k}"] = float(np.mean(rank_distances.min(axis=1)))
        output[f"oas_mean@{k}"] = float(np.mean(rank_distances.mean(axis=1)))
        output[f"oas_var@{k}"] = float(np.mean(rank_distances.var(axis=1, ddof=1))) if k > 1 else 0.0
    return output
