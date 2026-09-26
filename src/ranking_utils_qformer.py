import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils.operations import gather_object


DEFAULT_RANKING_RAW_META_DIR = ''
DEFAULT_RANKING_ACC_META_DIR = ''
FEATURE_RANKING_METHODS: Tuple[str, ...] = ("feature_t", "feature_c", "feature_cf", "feature_fused")
TOKEN_LOG_LIKELIHOOD_METHOD = "candidate_log_likelihood"
TOKEN_IDF_LIKELIHOOD_METHOD = "candidate_idf_likelihood"
TOKEN_EMBEDDING_METHOD = "token_emb"
DEFAULT_TOKEN_RANKING_METHOD = TOKEN_LOG_LIKELIHOOD_METHOD
TOKEN_RANKING_METHODS: Tuple[str, ...] = (
    TOKEN_LOG_LIKELIHOOD_METHOD,
    TOKEN_IDF_LIKELIHOOD_METHOD,
    TOKEN_EMBEDDING_METHOD,
)
RANKING_METRIC_TOPK: Tuple[int, ...] = (1, 5, 10)
RANKING_NDCG_TOPK: Tuple[int, ...] = (5, 10, 20, 30, 50)
DEFAULT_RANKING_DETAIL_TOPK = 10
DEFAULT_FEATURE_FUSION_WEIGHTS: Tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
DEFAULT_AGGREGATE_RANKING_SUMMARY_FILENAME = "all_checkpoints_ranking_summary.txt"
AGGREGATE_RANKING_PRIMARY_METRIC = "hit@1"
AGGREGATE_RANKING_POOL_ORDER: Tuple[str, ...] = ("raw", "acc", "all")
AGGREGATE_RANKING_METHOD_ORDER: Tuple[str, ...] = FEATURE_RANKING_METHODS + TOKEN_RANKING_METHODS
AGGREGATE_RANKING_PLOT_METRIC_ORDER: Tuple[str, ...] = (
    "hit@1",
    "hit@5",
    "hit@10",
    "ndcg@5",
    "ndcg@10",
)
RANKING_SUMMARY_METRIC_ORDER: Tuple[str, ...] = (
    "hit@1",
    "recall@1",
    "acc@1",
    "hit@5",
    "recall@5",
    "acc@5",
    "hit@10",
    "recall@10",
    "acc@10",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "ndcg@30",
    "ndcg@50",
)


def ranking_is_enabled(ranking_cfg) -> bool:
    return ranking_cfg is not None and bool(ranking_cfg.get("enable", False))


def ranking_compute_raw(ranking_cfg) -> bool:
    return ranking_is_enabled(ranking_cfg) and bool(ranking_cfg.get("compute_raw", True))


def ranking_compute_acc(ranking_cfg) -> bool:
    return ranking_is_enabled(ranking_cfg) and bool(ranking_cfg.get("compute_acc", True))


def ranking_compute_all(ranking_cfg) -> bool:
    return ranking_is_enabled(ranking_cfg) and bool(ranking_cfg.get("compute_all", False))


def resolve_token_ranking_method(ranking_cfg=None) -> str:
    """Resolve the Stage-2 visual-token scorer.

    Equal-mean candidate code log likelihood is the default for new
    DDBC/TIGER runs.  IDF likelihood and the old position-aligned cosine
    scorer remain available only through explicit compatibility settings.
    """
    raw_method = (
        DEFAULT_TOKEN_RANKING_METHOD
        if ranking_cfg is None
        else str(ranking_cfg.get("token_method", DEFAULT_TOKEN_RANKING_METHOD) or "").strip().lower()
    )
    aliases = {
        "candidate_log_likelihood": TOKEN_LOG_LIKELIHOOD_METHOD,
        "candidate_likelihood": TOKEN_LOG_LIKELIHOOD_METHOD,
        "log_likelihood": TOKEN_LOG_LIKELIHOOD_METHOD,
        "logits": TOKEN_LOG_LIKELIHOOD_METHOD,
        "no_idf": TOKEN_LOG_LIKELIHOOD_METHOD,
        "candidate_idf_likelihood": TOKEN_IDF_LIKELIHOOD_METHOD,
        "candidate_specific_idf": TOKEN_IDF_LIKELIHOOD_METHOD,
        "candidate-specific-idf": TOKEN_IDF_LIKELIHOOD_METHOD,
        "idf_likelihood": TOKEN_IDF_LIKELIHOOD_METHOD,
        "position_aligned_cosine": TOKEN_EMBEDDING_METHOD,
        "position-aligned-cosine": TOKEN_EMBEDDING_METHOD,
        "token_emb": TOKEN_EMBEDDING_METHOD,
    }
    if raw_method not in aliases:
        raise ValueError(
            f"Unsupported ranking.token_method={raw_method!r}; valid canonical choices are "
            "['candidate_log_likelihood', 'candidate_idf_likelihood', "
            "'position_aligned_cosine']."
        )
    return aliases[raw_method]


def token_ranking_requires_generated_tokens(ranking_cfg=None) -> bool:
    return resolve_token_ranking_method(ranking_cfg) == TOKEN_EMBEDDING_METHOD


def ranking_generate_visual_tokens(ranking_cfg=None) -> bool:
    """Whether a ranking run should also sample tokens for visual output.

    Full-mask likelihood methods do not use sampled tokens for scoring, but
    DDBC/TIGER Stage-2 outputs still need a generated token sequence for the
    TA-TiTok decoder. Keep that output enabled by default and expose an
    explicit opt-out only for scorer-only diagnostics.
    """
    return ranking_is_enabled(ranking_cfg) and bool(
        ranking_cfg.get("generate_visual_tokens", True)
    )


def attach_visual_generated_tokens(
    records: Sequence[Dict[str, object]],
    sample_indices: Sequence[int],
    generated_tokens: torch.Tensor,
    repeat_index: int,
) -> None:
    """Attach sampled token sequences to ranking details as visual-only data.

    The ranking fields and scores are left untouched.  Multiple candidate-pool
    records for the same sample receive the same visual sequence, and repeated
    samples are retained in generation order.
    """
    if generated_tokens.dim() != 2:
        raise ValueError(
            "generated_tokens must have shape [B,L], got "
            f"{tuple(generated_tokens.shape)}"
        )
    sample_indices = [int(value) for value in sample_indices]
    if len(sample_indices) != int(generated_tokens.shape[0]):
        raise ValueError(
            "sample_indices and generated_tokens batch size differ: "
            f"{len(sample_indices)} vs {int(generated_tokens.shape[0])}"
        )
    if len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must be unique within a generation batch")
    repeat_index = int(repeat_index)
    if repeat_index <= 0:
        raise ValueError(f"repeat_index must be >= 1, got {repeat_index}")

    token_rows = {
        sample_index: [int(token) for token in generated_tokens[row].detach().cpu().tolist()]
        for row, sample_index in enumerate(sample_indices)
    }
    for record in records:
        sample_index = int(record["sample_index"])
        if sample_index not in token_rows:
            continue
        tokens = token_rows[sample_index]
        repeats = record.setdefault("visual_generated_token_repeats", [])
        repeat_indices = record.setdefault("visual_generated_token_repeat_indices", [])
        repeats.append(tokens)
        repeat_indices.append(repeat_index)
        if "visual_generated_tokens" not in record:
            record["visual_generated_tokens"] = tokens
        record["visual_generated_token_length"] = len(tokens)
        record["visual_generated_token_repeat_count"] = len(repeats)
        record["visual_tokens_used_for_ranking"] = False
        record["visual_token_source"] = "iterative_maskgen_sampling"


class CandidateSpecificIDFState(NamedTuple):
    all_item_ids: torch.Tensor
    all_item_tokens: torch.Tensor
    position_idf: torch.Tensor
    all_item_weights: torch.Tensor
    all_item_weight_sums: torch.Tensor
    item_count: int
    sequence_length: int
    vocabulary_size: int


class CandidateLikelihoodState(NamedTuple):
    all_item_ids: torch.Tensor
    all_item_tokens: torch.Tensor
    item_count: int
    sequence_length: int


def prepare_candidate_likelihood_state(
    item_token_lookup: Dict[int, torch.Tensor],
    device: torch.device,
) -> CandidateLikelihoodState:
    """Build the catalog tensor needed by equal-mean likelihood ranking."""
    if not item_token_lookup:
        raise ValueError("Cannot build candidate likelihood state from an empty item-token catalog.")
    item_ids = sorted(int(item_id) for item_id in item_token_lookup)
    token_rows = [item_token_lookup[item_id].reshape(-1).long() for item_id in item_ids]
    sequence_lengths = {int(tokens.numel()) for tokens in token_rows}
    if len(sequence_lengths) != 1:
        raise ValueError(f"Catalog token sequences have inconsistent lengths: {sorted(sequence_lengths)}")
    sequence_length = int(next(iter(sequence_lengths)))
    return CandidateLikelihoodState(
        all_item_ids=torch.tensor(item_ids, dtype=torch.long, device=device),
        all_item_tokens=torch.stack(token_rows, dim=0).contiguous().to(
            device=device, non_blocking=True
        ),
        item_count=len(item_ids),
        sequence_length=sequence_length,
    )


def prepare_candidate_specific_idf_state(
    item_token_lookup: Dict[int, torch.Tensor],
    vocabulary_size: int,
    device: torch.device,
) -> CandidateSpecificIDFState:
    """Build position-specific sqrt-IDF weights from the complete item catalog."""
    if not item_token_lookup:
        raise ValueError("Cannot build candidate-specific IDF state from an empty item-token catalog.")
    item_ids = sorted(int(item_id) for item_id in item_token_lookup)
    token_rows = [item_token_lookup[item_id].reshape(-1).long() for item_id in item_ids]
    sequence_lengths = {int(tokens.numel()) for tokens in token_rows}
    if len(sequence_lengths) != 1:
        raise ValueError(f"Catalog token sequences have inconsistent lengths: {sorted(sequence_lengths)}")
    sequence_length = int(next(iter(sequence_lengths)))
    token_matrix_cpu = torch.stack(token_rows, dim=0).contiguous()
    token_numpy = token_matrix_cpu.numpy()
    if token_numpy.min() < 0 or token_numpy.max() >= int(vocabulary_size):
        raise ValueError(
            f"Catalog token ids must be in [0, {int(vocabulary_size)}), got "
            f"min={int(token_numpy.min())}, max={int(token_numpy.max())}."
        )

    counts = np.zeros((sequence_length, int(vocabulary_size)), dtype=np.int64)
    for position in range(sequence_length):
        counts[position] = np.bincount(
            token_numpy[:, position], minlength=int(vocabulary_size)
        )
    idf = np.log(
        (float(len(item_ids)) + 1.0) / (counts.astype(np.float64) + 1.0)
    )
    position_idf = torch.from_numpy(
        np.sqrt(np.maximum(idf, 0.0)).astype(np.float32)
    ).to(device=device)
    all_item_tokens = token_matrix_cpu.to(device=device, non_blocking=True)
    positions = torch.arange(sequence_length, device=device)
    all_item_weights = position_idf[positions[:, None], all_item_tokens.t()].t().contiguous()
    all_item_weight_sums = all_item_weights.sum(dim=-1).clamp_min(1.0e-12)
    return CandidateSpecificIDFState(
        all_item_ids=torch.tensor(item_ids, dtype=torch.long, device=device),
        all_item_tokens=all_item_tokens,
        position_idf=position_idf,
        all_item_weights=all_item_weights,
        all_item_weight_sums=all_item_weight_sums,
        item_count=len(item_ids),
        sequence_length=sequence_length,
        vocabulary_size=int(vocabulary_size),
    )


@torch.no_grad()
def compute_full_mask_token_ranking_logits(
    model,
    predictions: Dict[str, torch.Tensor],
    use_cfg: bool,
    guidance_scale: float,
) -> torch.Tensor:
    """Return the full-mask logits used by candidate-specific likelihood ranking."""
    latent_tokens = predictions["latent_tokens"]
    pooled_condition = model.build_pooled_condition(
        predictions["pred_e_c"], predictions["pred_e_t"]
    )
    batch_size = int(latent_tokens.shape[0])
    input_ids = torch.full(
        (batch_size, int(model.image_seq_len)),
        int(model.mask_token_id),
        dtype=torch.long,
        device=latent_tokens.device,
    )
    if not use_cfg:
        return model._forward_backbone(
            input_ids=input_ids,
            latent_tokens=latent_tokens,
            pooled_condition=pooled_condition,
        )

    empty_condition = torch.zeros_like(pooled_condition)
    paired_logits = model._forward_backbone(
        input_ids=torch.cat([input_ids, input_ids], dim=0),
        latent_tokens=torch.cat([latent_tokens, latent_tokens], dim=0),
        pooled_condition=torch.cat([pooled_condition, empty_condition], dim=0),
    )
    conditional, unconditional = paired_logits.chunk(2, dim=0)
    return conditional + (conditional - unconditional) * float(guidance_scale)


def resolve_feature_fusion_weights(
    ranking_cfg=None,
    feature_fusion_weights=None,
) -> Tuple[float, float, float]:
    """Return normalized (text, content, CF) inference fusion weights."""
    raw_weights = feature_fusion_weights
    if raw_weights is None and ranking_cfg is not None:
        raw_weights = ranking_cfg.get("feature_fusion_weights", None)
    if raw_weights is None:
        return DEFAULT_FEATURE_FUSION_WEIGHTS

    if hasattr(raw_weights, "get"):
        weights = (
            float(raw_weights.get("text", raw_weights.get("description", raw_weights.get("t", 0.0)))),
            float(raw_weights.get("content", raw_weights.get("c", 0.0))),
            float(raw_weights.get("cf", raw_weights.get("collaborative", 0.0))),
        )
    else:
        values = list(raw_weights)
        if len(values) != 3:
            raise ValueError(
                "ranking.feature_fusion_weights must contain exactly three values in (text, content, cf) order"
            )
        weights = tuple(float(value) for value in values)

    if not all(np.isfinite(value) and value >= 0.0 for value in weights):
        raise ValueError(f"Feature fusion weights must be finite and non-negative, got {weights}")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError(f"Feature fusion weights must have a positive sum, got {weights}")
    return tuple(value / total for value in weights)


def default_ranking_meta_file(split_name: str, acc: bool) -> str:
    split_key = str(split_name).strip().lower()
    if split_key not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported split_name for ranking metadata: {split_name!r}")
    suffix = "_hidden_sampleTimestep_acc_by_bundle.npy" if acc else "_hidden_sampleTimestep_by_bundle.npy"
    return f"{split_key}{suffix}"


def resolve_ranking_meta_path(ranking_cfg, split_name: str, acc: bool) -> Path:
    default_dir = DEFAULT_RANKING_ACC_META_DIR if acc else DEFAULT_RANKING_RAW_META_DIR
    dir_key = "acc_meta_dir" if acc else "raw_meta_dir"
    file_key = "acc_meta_file" if acc else "raw_meta_file"
    meta_dir = str(ranking_cfg.get(dir_key, default_dir)).strip() or default_dir
    meta_file = str(ranking_cfg.get(file_key, default_ranking_meta_file(split_name, acc))).strip()
    if not meta_file:
        meta_file = default_ranking_meta_file(split_name, acc)
    return Path(meta_dir).expanduser() / meta_file


def load_ranking_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Ranking metadata file does not exist: {path}")
    return np.load(path, allow_pickle=True).item()


def extract_ranking_candidate_ids(
    bundle_key: str,
    target_item_id: int,
    target_position: int = 1,
    raw_meta_obj=None,
    acc_meta_obj=None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    raw_candidate_ids = np.asarray([], dtype=np.int64)
    acc_candidate_ids = np.asarray([], dtype=np.int64)
    has_acc_candidate = False

    if raw_meta_obj is not None:
        if bundle_key not in raw_meta_obj:
            raise KeyError(f"Missing bundle_key={bundle_key!r} in raw ranking metadata.")
        raw_entry = raw_meta_obj[bundle_key]
        num_repeats = int(raw_entry.get("num_inference_repeats", -1))
        if num_repeats != 1:
            raise ValueError(
                "Raw ranking metadata only supports num_inference_repeats=1, "
                f"got {num_repeats} for bundle {bundle_key}"
            )
        repeat_entry = raw_entry.get("repeat_01", None)
        if repeat_entry is None:
            raise KeyError(f"Missing repeat_01 in raw ranking metadata for bundle {bundle_key}")
        if "groundtruth_target_item_id" in raw_entry:
            meta_target = int(raw_entry.get("groundtruth_target_item_id", target_item_id))
            if meta_target != int(target_item_id):
                raise ValueError(
                    f"Raw ranking metadata target mismatch for bundle {bundle_key}: "
                    f"expected {target_item_id}, got {meta_target}"
                )
            if "retrieved_target_candidate_item_ids" not in repeat_entry:
                raise KeyError(
                    f"Missing retrieved_target_candidate_item_ids in raw ranking metadata for bundle {bundle_key}"
                )
            raw_candidate_ids = np.asarray(repeat_entry["retrieved_target_candidate_item_ids"], dtype=np.int64)
        else:
            predicted_items = repeat_entry.get("predicted_items", [])
            item_entry = next(
                (
                    item
                    for item in predicted_items
                    if int(item.get("pred_item_position", -1)) == int(target_position)
                ),
                None,
            )
            if item_entry is None:
                raise KeyError(
                    f"Missing raw predicted_items target_position={target_position} for bundle {bundle_key}"
                )
            meta_target = int(item_entry.get("groundtruth_item_id_at_position", target_item_id))
            if meta_target != int(target_item_id):
                raise ValueError(
                    f"Raw ranking metadata target mismatch for bundle {bundle_key}, "
                    f"position={target_position}: expected {target_item_id}, got {meta_target}"
                )
            raw_candidate_ids = np.asarray(item_entry["retrieve_candidate_item_ids"], dtype=np.int64)

    if acc_meta_obj is not None:
        acc_entry = acc_meta_obj.get(bundle_key, None)
        if acc_entry is not None:
            if "groundtruth_target_item_id" in acc_entry:
                meta_target = int(acc_entry.get("groundtruth_target_item_id", target_item_id))
                if meta_target != int(target_item_id):
                    raise ValueError(
                        f"Acc ranking metadata target mismatch for bundle {bundle_key}: "
                        f"expected {target_item_id}, got {meta_target}"
                    )
                if "retrieved_target_candidate_item_ids" not in acc_entry:
                    raise KeyError(
                        f"Missing retrieved_target_candidate_item_ids in acc ranking metadata for bundle {bundle_key}"
                    )
                acc_candidate_ids = np.asarray(acc_entry["retrieved_target_candidate_item_ids"], dtype=np.int64)
            else:
                predicted_items = acc_entry.get("predicted_items", [])
                item_entry = next(
                    (
                        item
                        for item in predicted_items
                        if int(item.get("pred_item_position", -1)) == int(target_position)
                    ),
                    None,
                )
                if item_entry is None:
                    raise KeyError(
                        f"Missing acc predicted_items target_position={target_position} for bundle {bundle_key}"
                    )
                meta_target = int(item_entry.get("groundtruth_item_id_at_position", target_item_id))
                if meta_target != int(target_item_id):
                    raise ValueError(
                        f"Acc ranking metadata target mismatch for bundle {bundle_key}, "
                        f"position={target_position}: expected {target_item_id}, got {meta_target}"
                    )
                acc_candidate_ids = np.asarray(item_entry["retrieve_candidate_item_ids"], dtype=np.int64)
            has_acc_candidate = True

    return raw_candidate_ids, acc_candidate_ids, has_acc_candidate


def _dedupe_preserve_order(values: Sequence[int]) -> np.ndarray:
    deduped: List[int] = []
    seen = set()
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return np.asarray(deduped, dtype=np.int64)


def ensure_gt_in_candidates(
    candidate_item_ids: Sequence[int],
    target_item_id: int,
    require_gt_in_candidate: bool = True,
) -> Tuple[np.ndarray, bool]:
    candidate_array = np.asarray(candidate_item_ids, dtype=np.int64).reshape(-1)
    gt_missing_before_fix = int(target_item_id) not in set(candidate_array.tolist())
    if gt_missing_before_fix and require_gt_in_candidate:
        candidate_array = np.concatenate(
            [np.asarray([int(target_item_id)], dtype=np.int64), candidate_array],
            axis=0,
        )
    candidate_array = _dedupe_preserve_order(candidate_array.tolist())
    return candidate_array, bool(gt_missing_before_fix)


def _safe_float(value) -> float:
    return float(value) if value is not None else 0.0


def _compute_rank_and_topk(
    candidate_item_ids: np.ndarray,
    scores: np.ndarray,
    target_item_id: int,
    topk: int,
) -> Dict[str, object]:
    candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if candidate_item_ids.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Candidate ids and scores length mismatch: {candidate_item_ids.shape[0]} vs {scores.shape[0]}"
        )
    order = np.lexsort((candidate_item_ids.astype(np.int64), -scores))
    sorted_ids = candidate_item_ids[order]
    sorted_scores = scores[order]
    matching = np.where(sorted_ids == int(target_item_id))[0]
    if matching.size == 0:
        raise ValueError(f"Target item id {target_item_id} is missing from the ranking candidate set.")
    rank = int(matching[0]) + 1
    limit = min(int(topk), int(sorted_ids.shape[0]))
    return {
        "gt_rank": rank,
        "gt_score": _safe_float(sorted_scores[rank - 1]),
        "top_candidate_item_ids": [int(x) for x in sorted_ids[:limit].tolist()],
        "top_candidate_scores": [float(x) for x in sorted_scores[:limit].tolist()],
        "candidate_size": int(sorted_ids.shape[0]),
    }


def _zscore(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    mean = float(scores.mean()) if scores.size > 0 else 0.0
    std = float(scores.std()) if scores.size > 0 else 0.0
    return (scores - mean) / (std + 1e-8)


def score_feature_candidates(
    pred_feature: torch.Tensor,
    candidate_item_ids: Sequence[int],
    feature_bank: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    candidate_ids = np.asarray(candidate_item_ids, dtype=np.int64).reshape(-1)
    candidate_features = feature_bank[candidate_ids].to(device=device, dtype=torch.float32, non_blocking=True)
    pred = pred_feature.reshape(1, -1).to(device=device, dtype=torch.float32, non_blocking=True)
    pred = F.normalize(pred, dim=-1)
    candidate_features = F.normalize(candidate_features.reshape(candidate_features.shape[0], -1), dim=-1)
    scores = torch.matmul(candidate_features, pred.t()).squeeze(-1)
    return scores.detach().cpu().numpy()


def build_feature_ranking_records(
    bundle_key: str,
    sample_index: int,
    target_item_id: int,
    candidate_item_ids: Sequence[int],
    pred_e_t: torch.Tensor,
    pred_e_c: torch.Tensor,
    pred_e_cf: torch.Tensor,
    description_feature_bank: torch.Tensor,
    content_feature_bank: torch.Tensor,
    cf_feature_bank: torch.Tensor,
    device: torch.device,
    detail_topk: int,
    require_gt_in_candidate: bool = True,
    pool_name: str = "raw",
    feature_fusion_weights=None,
) -> Dict[str, Dict[str, object]]:
    candidate_ids, gt_missing_before_fix = ensure_gt_in_candidates(
        candidate_item_ids,
        target_item_id=target_item_id,
        require_gt_in_candidate=require_gt_in_candidate,
    )
    scores_t = score_feature_candidates(pred_e_t, candidate_ids, description_feature_bank, device)
    scores_c = score_feature_candidates(pred_e_c, candidate_ids, content_feature_bank, device)
    scores_cf = score_feature_candidates(pred_e_cf, candidate_ids, cf_feature_bank, device)
    weight_t, weight_c, weight_cf = resolve_feature_fusion_weights(
        feature_fusion_weights=feature_fusion_weights
    )
    scores_fused = (
        weight_t * _zscore(scores_t)
        + weight_c * _zscore(scores_c)
        + weight_cf * _zscore(scores_cf)
    )
    score_map = {
        "feature_t": scores_t,
        "feature_c": scores_c,
        "feature_cf": scores_cf,
        "feature_fused": scores_fused,
    }
    records: Dict[str, Dict[str, object]] = {}
    for method_name, method_scores in score_map.items():
        rank_info = _compute_rank_and_topk(candidate_ids, method_scores, target_item_id, detail_topk)
        records[method_name] = {
            "method": method_name,
            "pool_name": str(pool_name),
            "sample_index": int(sample_index),
            "bundle_key": str(bundle_key),
            "target_item_id": int(target_item_id),
            "candidate_item_ids": [int(x) for x in candidate_ids.tolist()],
            "gt_was_missing_before_fix": bool(gt_missing_before_fix),
            "gt_rank": int(rank_info["gt_rank"]),
            "gt_score": float(rank_info["gt_score"]),
            "top10_candidate_item_ids": rank_info["top_candidate_item_ids"],
            "top10_scores": rank_info["top_candidate_scores"],
            "candidate_size": int(rank_info["candidate_size"]),
        }
    return records


def score_candidate_log_likelihood(
    log_probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
) -> torch.Tensor:
    """Score every candidate by the equal mean of its code log probabilities."""
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must have shape [L,V], got {tuple(log_probs.shape)}")
    if candidate_tokens.dim() != 2:
        raise ValueError(
            f"candidate_tokens must have shape [K,L], got {tuple(candidate_tokens.shape)}"
        )
    if candidate_tokens.shape[1] != log_probs.shape[0]:
        raise ValueError(
            "Candidate token sequence length does not match logits: "
            f"{candidate_tokens.shape[1]} vs {log_probs.shape[0]}"
        )
    candidate_tokens = candidate_tokens.to(
        device=log_probs.device, dtype=torch.long, non_blocking=True
    )
    positions = torch.arange(candidate_tokens.shape[1], device=log_probs.device)
    gathered = log_probs[positions[None, :], candidate_tokens]
    return gathered.mean(dim=-1)


def build_candidate_log_likelihood_ranking_record(
    bundle_key: str,
    sample_index: int,
    target_item_id: int,
    candidate_item_ids: Sequence[int],
    log_probs: torch.Tensor,
    item_token_lookup: Dict[int, torch.Tensor],
    detail_topk: int,
    require_gt_in_candidate: bool = True,
    pool_name: str = "raw",
) -> Dict[str, object]:
    candidate_ids, gt_missing_before_fix = ensure_gt_in_candidates(
        candidate_item_ids,
        target_item_id=target_item_id,
        require_gt_in_candidate=require_gt_in_candidate,
    )
    missing_item_ids = [
        int(item_id) for item_id in candidate_ids.tolist() if int(item_id) not in item_token_lookup
    ]
    if missing_item_ids:
        raise KeyError(
            f"Missing {len(missing_item_ids)} candidate item ids in TA-TiTok token lookup; "
            f"first few: {missing_item_ids[:10]}"
        )
    candidate_tokens = torch.stack(
        [item_token_lookup[int(item_id)] for item_id in candidate_ids.tolist()], dim=0
    )
    scores = score_candidate_log_likelihood(
        log_probs=log_probs,
        candidate_tokens=candidate_tokens,
    ).detach().cpu().numpy()
    rank_info = _compute_rank_and_topk(candidate_ids, scores, target_item_id, detail_topk)
    return {
        "method": TOKEN_LOG_LIKELIHOOD_METHOD,
        "pool_name": str(pool_name),
        "sample_index": int(sample_index),
        "bundle_key": str(bundle_key),
        "target_item_id": int(target_item_id),
        "candidate_item_ids": [int(x) for x in candidate_ids.tolist()],
        "gt_was_missing_before_fix": bool(gt_missing_before_fix),
        "gt_rank": int(rank_info["gt_rank"]),
        "gt_score": float(rank_info["gt_score"]),
        "top10_candidate_item_ids": rank_info["top_candidate_item_ids"],
        "top10_scores": rank_info["top_candidate_scores"],
        "candidate_size": int(rank_info["candidate_size"]),
        "token_ranking_context": "one_full_mask_decoder_pass",
        "token_weighting": "equal_mean_over_candidate_code_log_probabilities",
    }


def score_all_items_candidate_log_likelihood(
    log_probs: torch.Tensor,
    likelihood_state: CandidateLikelihoodState,
    item_chunk_size: int = 2048,
) -> torch.Tensor:
    """Score the complete catalog with equal-mean code log likelihood."""
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must have shape [B,L,V], got {tuple(log_probs.shape)}")
    if log_probs.shape[1] != likelihood_state.sequence_length:
        raise ValueError(
            f"Logit sequence length {log_probs.shape[1]} != catalog length "
            f"{likelihood_state.sequence_length}."
        )
    chunk_size = int(item_chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"item_chunk_size must be positive, got {item_chunk_size}")
    batch_size = int(log_probs.shape[0])
    scores = torch.empty(
        (batch_size, likelihood_state.item_count),
        dtype=torch.float32,
        device=log_probs.device,
    )
    for start in range(0, likelihood_state.item_count, chunk_size):
        end = min(start + chunk_size, likelihood_state.item_count)
        token_chunk = likelihood_state.all_item_tokens[start:end]
        gathered = torch.gather(
            log_probs.unsqueeze(1).expand(-1, end - start, -1, -1),
            dim=-1,
            index=token_chunk[None, :, :, None].expand(batch_size, -1, -1, -1),
        ).squeeze(-1)
        scores[:, start:end] = gathered.mean(dim=-1)
    return scores


def score_candidate_specific_idf_likelihood(
    log_probs: torch.Tensor,
    candidate_tokens: torch.Tensor,
    position_idf: torch.Tensor,
) -> torch.Tensor:
    """Score candidate token sequences with candidate-specific sqrt-IDF weights.

    ``log_probs`` has shape ``[L, V]`` and ``candidate_tokens`` has shape
    ``[K, L]``.  The weight at position i comes from the candidate token
    ``t[k, i]`` rather than from a generated hard-token query.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"log_probs must have shape [L,V], got {tuple(log_probs.shape)}")
    if candidate_tokens.dim() != 2:
        raise ValueError(
            f"candidate_tokens must have shape [K,L], got {tuple(candidate_tokens.shape)}"
        )
    if candidate_tokens.shape[1] != log_probs.shape[0]:
        raise ValueError(
            "Candidate token sequence length does not match logits: "
            f"{candidate_tokens.shape[1]} vs {log_probs.shape[0]}"
        )
    positions = torch.arange(candidate_tokens.shape[1], device=log_probs.device)
    candidate_tokens = candidate_tokens.to(
        device=log_probs.device, dtype=torch.long, non_blocking=True
    )
    weights = position_idf[positions[None, :], candidate_tokens]
    gathered = log_probs[positions[None, :], candidate_tokens]
    return (gathered * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0e-12)


def build_candidate_specific_idf_ranking_record(
    bundle_key: str,
    sample_index: int,
    target_item_id: int,
    candidate_item_ids: Sequence[int],
    log_probs: torch.Tensor,
    item_token_lookup: Dict[int, torch.Tensor],
    idf_state: CandidateSpecificIDFState,
    detail_topk: int,
    require_gt_in_candidate: bool = True,
    pool_name: str = "raw",
) -> Dict[str, object]:
    candidate_ids, gt_missing_before_fix = ensure_gt_in_candidates(
        candidate_item_ids,
        target_item_id=target_item_id,
        require_gt_in_candidate=require_gt_in_candidate,
    )
    missing_item_ids = [
        int(item_id) for item_id in candidate_ids.tolist() if int(item_id) not in item_token_lookup
    ]
    if missing_item_ids:
        raise KeyError(
            f"Missing {len(missing_item_ids)} candidate item ids in TA-TiTok token lookup; "
            f"first few: {missing_item_ids[:10]}"
        )
    candidate_tokens = torch.stack(
        [item_token_lookup[int(item_id)] for item_id in candidate_ids.tolist()], dim=0
    )
    scores = score_candidate_specific_idf_likelihood(
        log_probs=log_probs,
        candidate_tokens=candidate_tokens,
        position_idf=idf_state.position_idf,
    ).detach().cpu().numpy()
    rank_info = _compute_rank_and_topk(candidate_ids, scores, target_item_id, detail_topk)
    return {
        "method": TOKEN_IDF_LIKELIHOOD_METHOD,
        "pool_name": str(pool_name),
        "sample_index": int(sample_index),
        "bundle_key": str(bundle_key),
        "target_item_id": int(target_item_id),
        "candidate_item_ids": [int(x) for x in candidate_ids.tolist()],
        "gt_was_missing_before_fix": bool(gt_missing_before_fix),
        "gt_rank": int(rank_info["gt_rank"]),
        "gt_score": float(rank_info["gt_score"]),
        "top10_candidate_item_ids": rank_info["top_candidate_item_ids"],
        "top10_scores": rank_info["top_candidate_scores"],
        "candidate_size": int(rank_info["candidate_size"]),
        "token_ranking_context": "one_full_mask_decoder_pass",
        "token_weighting": "candidate_specific_position_sqrt_idf",
        "idf_catalog_size": int(idf_state.item_count),
    }


def score_all_items_candidate_specific_idf_likelihood(
    log_probs: torch.Tensor,
    idf_state: CandidateSpecificIDFState,
    item_chunk_size: int = 2048,
) -> torch.Tensor:
    """Score the complete item-token catalog without materializing [B,N,L]."""
    if log_probs.dim() != 3:
        raise ValueError(f"log_probs must have shape [B,L,V], got {tuple(log_probs.shape)}")
    if log_probs.shape[1] != idf_state.sequence_length:
        raise ValueError(
            f"Logit sequence length {log_probs.shape[1]} != IDF state length {idf_state.sequence_length}."
        )
    chunk_size = int(item_chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"item_chunk_size must be positive, got {item_chunk_size}")
    batch_size = int(log_probs.shape[0])
    scores = torch.empty(
        (batch_size, idf_state.item_count), dtype=torch.float32, device=log_probs.device
    )
    for start in range(0, idf_state.item_count, chunk_size):
        end = min(start + chunk_size, idf_state.item_count)
        token_chunk = idf_state.all_item_tokens[start:end]
        gathered = torch.gather(
            log_probs.unsqueeze(1).expand(-1, end - start, -1, -1),
            dim=-1,
            index=token_chunk[None, :, :, None].expand(batch_size, -1, -1, -1),
        ).squeeze(-1)
        weights = idf_state.all_item_weights[start:end]
        scores[:, start:end] = (
            (gathered * weights.unsqueeze(0)).sum(dim=-1)
            / idf_state.all_item_weight_sums[start:end].unsqueeze(0)
        )
    return scores


def score_token_embedding_candidates(
    pred_tokens: torch.Tensor,
    candidate_item_ids: Sequence[int],
    item_token_lookup: Dict[int, torch.Tensor],
    quantizer,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    candidate_ids = np.asarray(candidate_item_ids, dtype=np.int64).reshape(-1)
    candidate_tokens: List[torch.Tensor] = []
    missing_item_ids: List[int] = []
    for item_id in candidate_ids.tolist():
        item_token = item_token_lookup.get(int(item_id))
        if item_token is None:
            missing_item_ids.append(int(item_id))
            continue
        candidate_tokens.append(item_token)
    if missing_item_ids:
        raise KeyError(
            f"Missing {len(missing_item_ids)} candidate item ids in TA-TiTok token lookup; "
            f"first few: {missing_item_ids[:10]}"
        )

    pred_tokens = pred_tokens.reshape(-1).to(device=device, dtype=torch.long, non_blocking=True)
    candidate_tokens_tensor = torch.stack(candidate_tokens, dim=0).to(device=device, dtype=torch.long, non_blocking=True)
    pred_embed = quantizer.get_codebook_entry(pred_tokens).reshape(1, pred_tokens.shape[0], -1).float()
    candidate_embed = quantizer.get_codebook_entry(candidate_tokens_tensor.reshape(-1)).reshape(
        candidate_tokens_tensor.shape[0],
        candidate_tokens_tensor.shape[1],
        -1,
    ).float()

    pred_embed = F.normalize(pred_embed, dim=-1)
    candidate_embed = F.normalize(candidate_embed, dim=-1)
    cosine_scores = (candidate_embed * pred_embed).sum(dim=-1).mean(dim=-1)
    l2_scores = -((candidate_embed - pred_embed) ** 2).sum(dim=-1).mean(dim=-1)
    return {
        "token_emb": cosine_scores.detach().cpu().numpy(),
        "token_l2": l2_scores.detach().cpu().numpy(),
    }


def build_token_embedding_ranking_record(
    bundle_key: str,
    sample_index: int,
    target_item_id: int,
    candidate_item_ids: Sequence[int],
    pred_tokens: torch.Tensor,
    item_token_lookup: Dict[int, torch.Tensor],
    quantizer,
    device: torch.device,
    detail_topk: int,
    require_gt_in_candidate: bool = True,
    pool_name: str = "raw",
    repeat_index: int = 1,
) -> Dict[str, object]:
    candidate_ids, gt_missing_before_fix = ensure_gt_in_candidates(
        candidate_item_ids,
        target_item_id=target_item_id,
        require_gt_in_candidate=require_gt_in_candidate,
    )
    score_map = score_token_embedding_candidates(
        pred_tokens=pred_tokens,
        candidate_item_ids=candidate_ids,
        item_token_lookup=item_token_lookup,
        quantizer=quantizer,
        device=device,
    )
    rank_info = _compute_rank_and_topk(candidate_ids, score_map["token_emb"], target_item_id, detail_topk)
    return {
        "method": "token_emb",
        "pool_name": str(pool_name),
        "sample_index": int(sample_index),
        "bundle_key": str(bundle_key),
        "target_item_id": int(target_item_id),
        "candidate_item_ids": [int(x) for x in candidate_ids.tolist()],
        "gt_was_missing_before_fix": bool(gt_missing_before_fix),
        "gt_rank": int(rank_info["gt_rank"]),
        "gt_score": float(rank_info["gt_score"]),
        "top10_candidate_item_ids": rank_info["top_candidate_item_ids"],
        "top10_scores": rank_info["top_candidate_scores"],
        "candidate_size": int(rank_info["candidate_size"]),
        "predicted_tokens_128": [int(x) for x in pred_tokens.reshape(-1).detach().cpu().tolist()],
        "repeat_index": int(repeat_index),
        "token_l2_gt_score": float(score_map["token_l2"][np.asarray(candidate_ids) == int(target_item_id)][0]),
    }


def reduce_best_of_repeat_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int, str], List[Dict[str, object]]] = {}
    for record in records:
        key = (
            str(record["pool_name"]),
            str(record["method"]),
            int(record["sample_index"]),
            str(record["bundle_key"]),
        )
        grouped.setdefault(key, []).append(record)

    reduced_records: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        group_sorted = sorted(
            group,
            key=lambda item: (
                int(item["gt_rank"]),
                -float(item.get("gt_score", 0.0)),
                int(item.get("repeat_index", 0)),
            ),
        )
        best_record = dict(group_sorted[0])
        best_record["token_repeat_ranks"] = [int(item["gt_rank"]) for item in sorted(group, key=lambda x: int(x["repeat_index"]))]
        best_record["token_repeat_scores"] = [
            float(item.get("gt_score", 0.0))
            for item in sorted(group, key=lambda x: int(x["repeat_index"]))
        ]
        best_record["selected_repeat_index"] = int(best_record.get("repeat_index", 1))
        reduced_records.append(best_record)
    return reduced_records


def build_ranking_summary(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not records:
        return {"sample_count": 0}
    ranks = np.asarray([int(record["gt_rank"]) for record in records], dtype=np.float64)
    summary: Dict[str, float] = {
        "sample_count": int(ranks.shape[0]),
        "candidate_size_mean": float(
            np.asarray([int(record["candidate_size"]) for record in records], dtype=np.float64).mean()
        ),
    }
    for k in RANKING_METRIC_TOPK:
        hit = (ranks <= float(k)).astype(np.float64)
        summary[f"hit@{k}"] = float(hit.mean())
        summary[f"recall@{k}"] = float(hit.mean())
        summary[f"acc@{k}"] = float(hit.mean())
        if k in RANKING_NDCG_TOPK:
            ndcg = np.where(hit > 0, 1.0 / np.log2(ranks + 1.0), 0.0)
            summary[f"ndcg@{k}"] = float(ndcg.mean())
    for k in RANKING_NDCG_TOPK:
        if k in RANKING_METRIC_TOPK:
            continue
        hit = (ranks <= float(k)).astype(np.float64)
        ndcg = np.where(hit > 0, 1.0 / np.log2(ranks + 1.0), 0.0)
        summary[f"ndcg@{k}"] = float(ndcg.mean())
    return summary


def gather_detail_records(records: Sequence[Dict[str, object]], accelerator) -> List[Dict[str, object]]:
    local_records = list(records)
    if accelerator.num_processes <= 1:
        return local_records
    gathered_records = gather_object(local_records)
    flattened: List[Dict[str, object]] = []
    for chunk in gathered_records:
        if isinstance(chunk, list):
            flattened.extend(chunk)
        else:
            flattened.append(chunk)
    return flattened


def save_ranking_outputs(
    output_dir: Path,
    summary_by_pool: Dict[str, Dict[str, Dict[str, float]]],
    detail_by_pool: Dict[str, Dict[str, List[Dict[str, object]]]],
    logger=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ranking_summary.json"
    detail_path = output_dir / "ranking_details.npy"
    txt_path = output_dir / "ranking_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_by_pool, f, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as f:
        for pool_name, pool_summary in summary_by_pool.items():
            f.write(f"[{pool_name}]\n")
            for method_name, metrics in pool_summary.items():
                f.write(f"{method_name}: {json.dumps(metrics, ensure_ascii=False, sort_keys=True)}\n")
            f.write("\n")

    np.save(
        detail_path,
        detail_by_pool,
        allow_pickle=True,
    )
    if logger is not None:
        logger.info("Saved ranking summary to %s", summary_path)
        logger.info("Saved ranking details to %s", detail_path)


def save_ranking_topk_item_images(
    output_dir: Path,
    detail_by_pool: Dict[str, Dict[str, List[Dict[str, object]]]],
    item_image_dir: Path,
    method_name: str = DEFAULT_TOKEN_RANKING_METHOD,
    topk: int = 3,
    require_all: bool = True,
    logger=None,
) -> List[Dict[str, object]]:
    """Copy the top-ranked catalog item images next to ranking outputs.

    Image paths are also attached to each ranking detail record.  This helper
    only visualizes an existing ranking list; it never changes scores or order.
    """
    topk = int(topk)
    if topk <= 0:
        return []
    output_dir = Path(output_dir)
    item_image_dir = Path(item_image_dir)
    if not item_image_dir.is_dir():
        raise FileNotFoundError(f"Ranking item image directory does not exist: {item_image_dir}")

    image_root = output_dir / f"top{topk}_images"
    manifest: List[Dict[str, object]] = []
    missing: List[Tuple[str, int]] = []
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp")

    for pool_name, method_payload in detail_by_pool.items():
        records = method_payload.get(str(method_name), [])
        for record in records:
            sample_index = int(record["sample_index"])
            target_item_id = int(record["target_item_id"])
            candidate_ids = [
                int(value) for value in record.get("top10_candidate_item_ids", [])[:topk]
            ]
            sample_dir = (
                image_root
                / str(pool_name)
                / str(method_name)
                / f"sample_{sample_index:06d}_target_{target_item_id}"
            )
            sample_dir.mkdir(parents=True, exist_ok=True)
            saved_paths: List[str] = []
            for rank, item_id in enumerate(candidate_ids, start=1):
                source_path = None
                for extension in supported_extensions:
                    candidate_path = item_image_dir / f"{item_id}{extension}"
                    if candidate_path.is_file():
                        source_path = candidate_path
                        break
                if source_path is None:
                    missing.append((str(pool_name), item_id))
                    manifest.append(
                        {
                            "pool_name": str(pool_name),
                            "method": str(method_name),
                            "sample_index": sample_index,
                            "bundle_key": str(record.get("bundle_key", "")),
                            "target_item_id": target_item_id,
                            "rank": rank,
                            "item_id": item_id,
                            "missing": True,
                        }
                    )
                    continue
                destination = sample_dir / f"rank_{rank:02d}_item_{item_id}{source_path.suffix.lower()}"
                shutil.copy2(source_path, destination)
                relative_path = str(destination.relative_to(output_dir))
                saved_paths.append(relative_path)
                manifest.append(
                    {
                        "pool_name": str(pool_name),
                        "method": str(method_name),
                        "sample_index": sample_index,
                        "bundle_key": str(record.get("bundle_key", "")),
                        "target_item_id": target_item_id,
                        "rank": rank,
                        "item_id": item_id,
                        "image_path": relative_path,
                        "missing": False,
                    }
                )
            record["ranking_topk_image_item_ids"] = candidate_ids
            record["ranking_topk_image_paths"] = saved_paths
            record["ranking_topk_image_count"] = len(saved_paths)

    if missing and require_all:
        raise FileNotFoundError(
            f"Missing {len(missing)} top-ranked catalog images in {item_image_dir}; "
            f"first examples: {missing[:10]}"
        )
    image_root.mkdir(parents=True, exist_ok=True)
    with (image_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    if logger is not None:
        logger.info(
            "Saved ranking Top-%d item images | method=%s | images=%d | missing=%d | output=%s",
            topk,
            method_name,
            sum(not bool(entry["missing"]) for entry in manifest),
            len(missing),
            image_root,
        )
    return manifest


def load_ranking_summary_json(summary_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid ranking summary json: {summary_path}")
    return payload


def _format_metric_value(value) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "N/A"


def _ordered_names(names: Sequence[str], preferred_order: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    name_set = {str(name) for name in names}
    for name in preferred_order:
        if name in name_set and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(name_set):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _group_checkpoint_rows_by_pool_and_method(
    checkpoint_rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in checkpoint_rows:
        pool_name = str(row.get("pool", ""))
        method_name = str(row.get("method", ""))
        grouped.setdefault(pool_name, {}).setdefault(method_name, []).append(row)
    return grouped


def save_aggregated_ranking_summary_txt(
    output_path: Path,
    source_dir: Path,
    metric_order: Sequence[str],
    checkpoint_rows: Sequence[Dict[str, object]],
    best_rows: Sequence[Dict[str, object]],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_rows = _group_checkpoint_rows_by_pool_and_method(checkpoint_rows)
    num_checkpoints = len({str(row.get("checkpoint", "")) for row in checkpoint_rows if row.get("checkpoint", "")})
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"source_dir\t{source_dir}\n")
        f.write(f"num_checkpoints\t{num_checkpoints}\n")
        f.write(f"best_metric\t{AGGREGATE_RANKING_PRIMARY_METRIC}\n")
        f.write("summary_table\n")
        f.write("\t".join(["checkpoint", "summary_timestamp", "pool", "method", *metric_order]) + "\n")
        for row in checkpoint_rows:
            values = [
                str(row["checkpoint"]),
                str(row["summary_timestamp"]),
                str(row["pool"]),
                str(row["method"]),
            ]
            values.extend(_format_metric_value(row.get(metric_name, "")) for metric_name in metric_order)
            f.write("\t".join(values) + "\n")

        for pool_name in _ordered_names(grouped_rows.keys(), AGGREGATE_RANKING_POOL_ORDER):
            f.write("\n")
            f.write(f"{pool_name}_table\n")
            method_rows = grouped_rows.get(pool_name, {})
            for method_name in _ordered_names(method_rows.keys(), AGGREGATE_RANKING_METHOD_ORDER):
                f.write(f"method\t{method_name}\n")
                f.write("\t".join(["checkpoint", "summary_timestamp", *metric_order]) + "\n")
                for row in method_rows.get(method_name, []):
                    values = [str(row["checkpoint"]), str(row["summary_timestamp"])]
                    values.extend(_format_metric_value(row.get(metric_name, "")) for metric_name in metric_order)
                    f.write("\t".join(values) + "\n")
                f.write("\n")

        f.write("\n")
        f.write("best_by_hit@1\n")
        f.write("\t".join(["pool", "method", "checkpoint", "summary_timestamp", *metric_order]) + "\n")
        for row in best_rows:
            values = [
                str(row["pool"]),
                str(row["method"]),
                str(row["checkpoint"]),
                str(row["summary_timestamp"]),
            ]
            values.extend(_format_metric_value(row.get(metric_name, "")) for metric_name in metric_order)
            f.write("\t".join(values) + "\n")


def _to_plot_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _resolve_aggregated_ranking_plot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_plots"


def save_aggregated_ranking_summary_plots(
    output_dir: Path,
    checkpoint_rows: Sequence[Dict[str, object]],
    logger=None,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if logger is not None:
            logger.warning("Skipping aggregated ranking plots because matplotlib is not installed.")
        return []

    grouped_rows = _group_checkpoint_rows_by_pool_and_method(checkpoint_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_styles = {
        "hit@1": {"color": "#2ca02c", "marker": "s"},
        "hit@5": {"color": "#ff7f0e", "marker": "^"},
        "hit@10": {"color": "#d62728", "marker": "D"},
        "ndcg@5": {"color": "#9467bd", "marker": "P"},
        "ndcg@10": {"color": "#8c564b", "marker": "X"},
    }
    saved_paths: List[Path] = []

    for pool_name in _ordered_names(grouped_rows.keys(), AGGREGATE_RANKING_POOL_ORDER):
        method_rows = grouped_rows.get(pool_name, {})
        method_names = _ordered_names(method_rows.keys(), AGGREGATE_RANKING_METHOD_ORDER)
        if not method_names:
            continue

        fig_height = max(3.8 * len(method_names), 5.0)
        fig, axes = plt.subplots(len(method_names), 1, figsize=(14, fig_height), sharex=True)
        if not isinstance(axes, (list, tuple, np.ndarray)):
            axes = [axes]

        for ax, method_name in zip(axes, method_names):
            rows = sorted(method_rows.get(method_name, []), key=lambda row: int(row.get("checkpoint_step", 0)))
            steps = [int(row.get("checkpoint_step", 0)) for row in rows]
            if not steps:
                ax.set_visible(False)
                continue

            for metric_name in AGGREGATE_RANKING_PLOT_METRIC_ORDER:
                style = metric_styles[metric_name]
                metric_values = [_to_plot_float(row.get(metric_name)) for row in rows]
                ax.plot(
                    steps,
                    metric_values,
                    label=metric_name,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.7,
                    markersize=4,
                )

            hit1_values = np.asarray([_to_plot_float(row.get("hit@1")) for row in rows], dtype=np.float64)
            if np.isfinite(hit1_values).any():
                best_idx = int(np.nanargmax(hit1_values))
                ax.scatter([steps[best_idx]], [float(hit1_values[best_idx])], color="#2ca02c", s=30, zorder=4)
                ax.annotate(
                    f"best hit@1 @{steps[best_idx]} = {float(hit1_values[best_idx]):.4f}",
                    xy=(steps[best_idx], float(hit1_values[best_idx])),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="#2ca02c",
                )

            ax.set_title(f"{pool_name} | {method_name}")
            ax.set_ylabel("score")
            ax.grid(True, alpha=0.25)
            handles_left, labels_left = ax.get_legend_handles_labels()
            ax.legend(
                handles_left,
                labels_left,
                loc="upper center",
                ncol=4,
                fontsize=8,
            )

        all_steps = sorted(
            {
                int(row.get("checkpoint_step", 0))
                for method_name in method_names
                for row in method_rows.get(method_name, [])
            }
        )
        if all_steps:
            axes[-1].set_xticks(all_steps)
        axes[-1].set_xlabel("Checkpoint Step")
        fig.suptitle(f"Ranking Metrics Across Checkpoints | pool={pool_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = output_dir / f"{pool_name}_ranking_metrics.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        saved_paths.append(output_path)
        if logger is not None:
            logger.info("Saved aggregated ranking plot to %s", output_path)

    return saved_paths


def aggregate_ranking_summaries_across_checkpoints(
    root_dir: Path,
    logger,
    output_filename: str,
    checkpoint_step_fn: Callable[[Path], int],
    ranking_output_subdir: str = "ranking",
) -> Optional[Path]:
    ranking_output_subdir = str(ranking_output_subdir).strip() or "ranking"
    summary_paths = sorted(
        root_dir.glob(f"checkpoint-*/{ranking_output_subdir}/ranking_summary.json"),
        key=lambda path: checkpoint_step_fn(path.parent.parent),
    )
    if not summary_paths:
        logger.info(
            "Skipping aggregated ranking summary because no checkpoint-*/%s/ranking_summary.json exists under %s",
            ranking_output_subdir,
            root_dir,
        )
        return None

    checkpoint_rows: List[Dict[str, object]] = []
    best_by_pool_method: Dict[Tuple[str, str], Dict[str, object]] = {}

    for summary_path in summary_paths:
        summary_payload = load_ranking_summary_json(summary_path)
        checkpoint_dir = summary_path.parent.parent
        checkpoint_name = checkpoint_dir.name
        checkpoint_step = checkpoint_step_fn(checkpoint_dir)
        summary_timestamp = datetime.fromtimestamp(summary_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        for pool_name, pool_summary in summary_payload.items():
            if not isinstance(pool_summary, dict):
                continue
            for method_name, metrics in pool_summary.items():
                if not isinstance(metrics, dict):
                    continue
                row = {
                    "checkpoint": checkpoint_name,
                    "checkpoint_step": int(checkpoint_step),
                    "summary_timestamp": summary_timestamp,
                    "pool": str(pool_name),
                    "method": str(method_name),
                }
                for metric_name in RANKING_SUMMARY_METRIC_ORDER:
                    row[metric_name] = metrics.get(metric_name, None)
                checkpoint_rows.append(row)

                try:
                    primary_metric = float(metrics[AGGREGATE_RANKING_PRIMARY_METRIC])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (str(pool_name), str(method_name))
                best_row = best_by_pool_method.get(key)
                if best_row is None or primary_metric > float(best_row["primary_metric"]):
                    best_by_pool_method[key] = {
                        "checkpoint": checkpoint_name,
                        "summary_timestamp": summary_timestamp,
                        "pool": str(pool_name),
                        "method": str(method_name),
                        "primary_metric": primary_metric,
                        "metrics": {
                            metric_name: metrics.get(metric_name, None)
                            for metric_name in RANKING_SUMMARY_METRIC_ORDER
                        },
                    }

    best_rows: List[Dict[str, object]] = []
    best_pool_order = _ordered_names(
        [pool_name for pool_name, _ in best_by_pool_method.keys()],
        AGGREGATE_RANKING_POOL_ORDER,
    )
    for pool_name in best_pool_order:
        method_names = [
            method_name
            for candidate_pool, method_name in best_by_pool_method.keys()
            if candidate_pool == pool_name
        ]
        for method_name in _ordered_names(method_names, AGGREGATE_RANKING_METHOD_ORDER):
            best_row = best_by_pool_method[(pool_name, method_name)]
            row = {
                "pool": pool_name,
                "method": method_name,
                "checkpoint": str(best_row["checkpoint"]),
                "summary_timestamp": str(best_row["summary_timestamp"]),
            }
            row.update(best_row["metrics"])
            best_rows.append(row)

    output_path = root_dir / output_filename
    save_aggregated_ranking_summary_txt(
        output_path=output_path,
        source_dir=root_dir,
        metric_order=RANKING_SUMMARY_METRIC_ORDER,
        checkpoint_rows=checkpoint_rows,
        best_rows=best_rows,
    )
    save_aggregated_ranking_summary_plots(
        output_dir=_resolve_aggregated_ranking_plot_dir(output_path),
        checkpoint_rows=checkpoint_rows,
        logger=logger,
    )
    logger.info(
        "Saved aggregated ranking summary across %d checkpoint(s) to %s",
        len(summary_paths),
        output_path,
    )
    return output_path
