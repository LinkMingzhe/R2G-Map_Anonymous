import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
DDFR_ROOT = CURRENT_DIR.parent

from inference_fid_utils import aggregate_fid_summaries_across_checkpoints, compute_and_save_fid_scores
from modeling.tatitok import TATiTok
from ranking_utils_qformer import (
    CandidateLikelihoodState,
    CandidateSpecificIDFState,
    DEFAULT_AGGREGATE_RANKING_SUMMARY_FILENAME,
    DEFAULT_RANKING_DETAIL_TOPK,
    FEATURE_RANKING_METHODS,
    TOKEN_EMBEDDING_METHOD,
    TOKEN_IDF_LIKELIHOOD_METHOD,
    TOKEN_LOG_LIKELIHOOD_METHOD,
    TOKEN_RANKING_METHODS,
    aggregate_ranking_summaries_across_checkpoints,
    attach_visual_generated_tokens,
    build_candidate_log_likelihood_ranking_record,
    build_candidate_specific_idf_ranking_record,
    build_feature_ranking_records,
    build_token_embedding_ranking_record,
    compute_full_mask_token_ranking_logits,
    gather_detail_records,
    prepare_candidate_likelihood_state,
    prepare_candidate_specific_idf_state,
    ranking_compute_acc,
    ranking_compute_all,
    ranking_compute_raw,
    ranking_generate_visual_tokens,
    ranking_is_enabled,
    reduce_best_of_repeat_records,
    resolve_feature_fusion_weights,
    resolve_token_ranking_method,
    save_ranking_outputs,
    save_ranking_topk_item_images,
    score_all_items_candidate_log_likelihood,
    score_all_items_candidate_specific_idf_likelihood,
    token_ranking_requires_generated_tokens,
)
from rg_lsm_model import (
    FixedTextConditionProvider,
    MaskGen_QFormer,
    PromptEmbeddingCache,
    QFormerMaskGenDataset,
    _safe_file_component,
    _tensor_to_uint8_image,
    build_model_condition_batch,
    generate_maskgen_qformer,
    get_decode_modes,
    maybe_prepare_cache,
    maybe_prepare_prompt_cache,
    normalize_condition_source,
    normalize_temb_condition_components,
    group_ranking_records_by_pool_and_method,
    resolve_qformer_base_dataset,
)


def run_generation_quality_eval(config, checkpoint_output_dir, ranking_output_subdir, logger):
    if bool(config.inference.get("generation_quality_eval", {}).get("enable", False)):
        raise ValueError("Use evaluate.py for local ranking and image quality evaluation.")

from runtime_utils import create_loader, get_config, maybe_limit_dataset
from utils.logger import setup_logger


def get_checkpoint_step(checkpoint_path: Path) -> int:
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["global_step"])
    name = checkpoint_path.name
    if name.startswith("checkpoint-"):
        return int(name.split("-")[-1])
    raise ValueError(f"Cannot infer checkpoint step from {checkpoint_path}")


def resolve_checkpoint_paths(config) -> List[Path]:
    checkpoint_mode = str(config.inference.get("checkpoint_mode", "latest")).strip().lower()
    raw_checkpoint_path = config.inference.get("checkpoint_path", None)
    checkpoint_path = "" if raw_checkpoint_path is None else str(raw_checkpoint_path).strip()
    if checkpoint_path.lower() in {"", "none", "null"}:
        checkpoint_path = ""

    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
        return [path]

    output_dir = Path(config.experiment.output_dir)
    candidates = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* found under {output_dir}")
    if checkpoint_mode == "all":
        return candidates
    return [candidates[-1]]


def resolve_model_weight_path(checkpoint_path: Path, use_ema_weights: bool) -> Path:
    subdir = "ema_model" if use_ema_weights else "unwrapped_model"
    weight_path = checkpoint_path / subdir
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing {subdir} under checkpoint: {checkpoint_path}")
    return weight_path


def resolve_checkpoint_output_dir(config, checkpoint_step: int) -> Path:
    raw_image_output_dir = config.inference.get("image_output_dir", None)
    image_output_dir = "" if raw_image_output_dir is None else str(raw_image_output_dir).strip()
    checkpoint_name = f"checkpoint-{checkpoint_step}"
    if image_output_dir.lower() not in {"", "none", "null"}:
        image_output_dir = image_output_dir.format(
            checkpoint_step=checkpoint_step,
            checkpoint_name=checkpoint_name,
        )
        return Path(image_output_dir).expanduser()
    return Path(config.experiment.output_dir) / "inference" / checkpoint_name


def setup_checkpoint_logger(log_output_dir: Path, checkpoint_step: int):
    log_output_dir.mkdir(parents=True, exist_ok=True)
    return setup_logger(
        name=f"VBCMaskGenQFormerV2Infer.{checkpoint_step}",
        log_level="INFO",
        use_accelerate=False,
        output_file=str(log_output_dir / "infer.log"),
    )


def configure_model_shape_from_dataset(config, dataset, text_condition_provider=None):
    sample = dataset[0]
    dataset_hidden_shape = tuple(sample["hidden_condition"].shape)
    condition_source = normalize_condition_source(config.model.get("condition_source", "hidden_latent"))
    if condition_source == "hidden_latent":
        condition_shape = dataset_hidden_shape
    else:
        if text_condition_provider is None:
            raise ValueError("text_condition_provider is required when model.condition_source=text")
        condition_shape = tuple(text_condition_provider.condition_shape)
    config.model.latent_slots = int(condition_shape[0])
    config.model.latent_input_dim = int(condition_shape[1])
    config.model.content_dim = int(sample["e_content"].shape[-1])
    config.model.description_dim = int(sample["e_description"].shape[-1])
    config.model.cf_dim = int(sample["e_cf"].shape[-1])


def load_model(config, checkpoint_path: Path, device: torch.device, logger):
    model = MaskGen_QFormer(config)
    weight_path = resolve_model_weight_path(
        checkpoint_path=checkpoint_path,
        use_ema_weights=bool(config.inference.get("use_ema_weights", False)),
    )
    model.load_pretrained_weight(str(weight_path), strict_loading=True)
    model = model.to(device)
    model.eval()
    logger.info("Loaded model weights from %s", weight_path)
    return model


def maybe_save_gt_image(output_dir: Path, gt_image_dir: Path, target_item_id: int, sample_prefix: str):
    gt_image_path = gt_image_dir / f"{target_item_id}.png"
    if not gt_image_path.exists():
        raise FileNotFoundError(f"Missing gt image for target_item_id={target_item_id}: {gt_image_path}")
    save_path = output_dir / f"{sample_prefix}_gt.png"
    if save_path.exists():
        return save_path
    with Image.open(gt_image_path) as image:
        image.convert("RGB").save(save_path)
    return save_path


def build_sample_prefix(bundle_key: str, target_item_id: int) -> str:
    safe_bundle_key = str(bundle_key).replace("/", "_")
    return f"{safe_bundle_key}_{_safe_file_component(target_item_id)}"


def build_inference_sample_prefix(base_sample_prefix: str, repeat_index: int, repeat_per_bundle: int) -> str:
    if repeat_per_bundle <= 1:
        return base_sample_prefix
    return f"{base_sample_prefix}__repeat{repeat_index + 1:03d}"


def resolve_max_bundles(config) -> int:
    max_bundles = config.inference.get("max_bundles", 0)
    if max_bundles is None:
        return 0
    return max(0, int(max_bundles))


def resolve_repeat_per_bundle(config) -> int:
    repeat_per_bundle = config.inference.get("repeat_per_bundle", 1)
    if repeat_per_bundle is None:
        repeat_per_bundle = 1
    repeat_per_bundle = int(repeat_per_bundle)
    if repeat_per_bundle <= 0:
        raise ValueError(f"inference.repeat_per_bundle must be >= 1, got {repeat_per_bundle}")
    return repeat_per_bundle


def prepare_generation_token_allowed_mask(
    config,
    dataset: QFormerMaskGenDataset,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
    logger,
) -> Optional[torch.Tensor]:
    mode = str(config.inference.get("token_constraint_mode", "none") or "none").strip().lower()
    if mode in {"", "none", "false", "off", "disabled"}:
        return None
    if mode not in {"global_freq", "position_freq"}:
        raise ValueError(
            "inference.token_constraint_mode must be one of none/global_freq/position_freq, "
            f"got {mode!r}"
        )
    if not getattr(dataset, "item_token_lookup", None):
        raise ValueError("Cannot build token constraint mask because dataset.item_token_lookup is empty.")

    token_tensors = []
    for item_tokens in dataset.item_token_lookup.values():
        token_tensor = torch.as_tensor(item_tokens, dtype=torch.long).reshape(-1)
        if token_tensor.numel() != seq_len:
            continue
        token_tensors.append(token_tensor)
    if not token_tensors:
        raise ValueError(
            f"Cannot build token constraint mask: no item token sequences with seq_len={seq_len} were found."
        )

    tokens = torch.stack(token_tensors, dim=0)
    min_count = max(1, int(config.inference.get("token_constraint_min_count", 1) or 1))
    topk = int(config.inference.get("token_constraint_topk", 0) or 0)

    if mode == "global_freq":
        counts = torch.bincount(tokens.reshape(-1), minlength=vocab_size)[:vocab_size]
        allowed = counts >= min_count
        if topk > 0 and topk < vocab_size:
            top_indices = torch.topk(counts, k=topk, largest=True).indices
            top_allowed = torch.zeros_like(allowed)
            top_allowed[top_indices] = True
            allowed = allowed & top_allowed
        if not bool(allowed.any()):
            raise ValueError(
                f"Global token constraint removed every token; min_count={min_count}, topk={topk}."
            )
        logger.info(
            "Generation token constraint | mode=%s | item_sequences=%d | min_count=%d | topk=%d | "
            "allowed_tokens=%d/%d",
            mode,
            len(token_tensors),
            min_count,
            topk,
            int(allowed.sum().item()),
            vocab_size,
        )
        return allowed.to(device=device)

    allowed_rows = []
    allowed_counts = []
    for pos in range(seq_len):
        counts = torch.bincount(tokens[:, pos], minlength=vocab_size)[:vocab_size]
        allowed = counts >= min_count
        if topk > 0 and topk < vocab_size:
            top_indices = torch.topk(counts, k=topk, largest=True).indices
            top_allowed = torch.zeros_like(allowed)
            top_allowed[top_indices] = True
            allowed = allowed & top_allowed
        if not bool(allowed.any()):
            allowed[int(torch.argmax(counts).item())] = True
        allowed_rows.append(allowed)
        allowed_counts.append(int(allowed.sum().item()))
    allowed_matrix = torch.stack(allowed_rows, dim=0)
    logger.info(
        "Generation token constraint | mode=%s | item_sequences=%d | min_count=%d | topk=%d | "
        "allowed_per_pos_min=%d | allowed_per_pos_mean=%.2f | allowed_per_pos_max=%d | vocab=%d",
        mode,
        len(token_tensors),
        min_count,
        topk,
        min(allowed_counts),
        sum(allowed_counts) / float(len(allowed_counts)),
        max(allowed_counts),
        vocab_size,
    )
    return allowed_matrix.to(device=device)


def collect_stage2_feature_ranking_records(
    batch,
    preds,
    dataset: QFormerMaskGenDataset,
    ranking_cfg,
    device: torch.device,
) -> List[Dict[str, object]]:
    if not ranking_is_enabled(ranking_cfg):
        return []

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    require_gt_in_candidate = bool(ranking_cfg.get("require_gt_in_candidate", True))
    sample_indices = batch["sample_index"].detach().cpu().tolist()
    target_item_ids = batch["target_item_id"].detach().cpu().tolist()
    bundle_keys = list(batch["bundle_key"])

    pred_e_t = preds["pred_e_t"].detach()
    pred_e_c = preds["pred_e_c"].detach()
    pred_e_cf = preds["pred_e_cf"].detach()

    records: List[Dict[str, object]] = []
    for batch_idx, sample_index in enumerate(sample_indices):
        bundle_key = str(bundle_keys[batch_idx])
        target_item_id = int(target_item_ids[batch_idx])
        for pool_name, enabled in [("raw", ranking_compute_raw(ranking_cfg)), ("acc", ranking_compute_acc(ranking_cfg))]:
            if not enabled:
                continue
            candidate_item_ids, has_candidates = dataset.get_candidate_item_ids(sample_index, pool_name)
            if pool_name == "acc" and not has_candidates:
                continue
            method_records = build_feature_ranking_records(
                bundle_key=bundle_key,
                sample_index=int(sample_index),
                target_item_id=target_item_id,
                candidate_item_ids=candidate_item_ids,
                pred_e_t=pred_e_t[batch_idx],
                pred_e_c=pred_e_c[batch_idx],
                pred_e_cf=pred_e_cf[batch_idx],
                description_feature_bank=dataset.description_features,
                content_feature_bank=dataset.content_features,
                cf_feature_bank=dataset.cf_features,
                device=device,
                detail_topk=detail_topk,
                require_gt_in_candidate=require_gt_in_candidate,
                pool_name=pool_name,
                feature_fusion_weights=resolve_feature_fusion_weights(ranking_cfg),
            )
            records.extend(method_records.values())
    return records


def prepare_all_item_feature_banks(
    dataset: QFormerMaskGenDataset,
    device: torch.device,
    logger=None,
) -> Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    num_items = min(
        int(dataset.description_features.shape[0]),
        int(dataset.content_features.shape[0]),
        int(dataset.cf_features.shape[0]),
    )
    if num_items <= 0:
        return None
    all_item_ids = torch.arange(num_items, dtype=torch.long, device=device)
    banks = {
        "feature_t": F.normalize(
            dataset.description_features[:num_items].reshape(num_items, -1).to(device=device, dtype=torch.float32),
            dim=-1,
        ),
        "feature_c": F.normalize(
            dataset.content_features[:num_items].reshape(num_items, -1).to(device=device, dtype=torch.float32),
            dim=-1,
        ),
        "feature_cf": F.normalize(
            dataset.cf_features[:num_items].reshape(num_items, -1).to(device=device, dtype=torch.float32),
            dim=-1,
        ),
    }
    if logger is not None:
        logger.info("Prepared all-item feature ranking banks with %d catalog items.", num_items)
    return all_item_ids, banks


def prepare_all_item_token_bank(
    dataset: QFormerMaskGenDataset,
    tokenizer: TATiTok,
    device: torch.device,
    logger=None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not dataset.item_token_lookup:
        return None
    max_item_id = max(int(item_id) for item_id in dataset.item_token_lookup.keys())
    candidate_ids: List[int] = []
    candidate_tokens: List[torch.Tensor] = []
    for item_id in range(max_item_id + 1):
        token_tensor = dataset.item_token_lookup.get(int(item_id))
        if token_tensor is None:
            continue
        candidate_ids.append(int(item_id))
        candidate_tokens.append(token_tensor.long())
    if not candidate_tokens:
        return None
    all_item_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    token_matrix = torch.stack(candidate_tokens, dim=0).to(device=device, dtype=torch.long, non_blocking=True)
    token_embed = tokenizer.quantize.get_codebook_entry(token_matrix.reshape(-1)).reshape(
        token_matrix.shape[0],
        token_matrix.shape[1],
        -1,
    )
    token_embed = F.normalize(token_embed.float(), dim=-1)
    if logger is not None:
        logger.info("Prepared all-item token_emb ranking bank with %d catalog items.", len(candidate_ids))
    return all_item_ids, token_embed


def _zscore_rows(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / (
        scores.std(dim=1, keepdim=True, unbiased=False) + 1e-8
    )


def _build_all_item_records_for_scores(
    batch,
    scores: torch.Tensor,
    method_name: str,
    all_item_ids: torch.Tensor,
    detail_topk: int,
) -> List[Dict[str, object]]:
    sample_indices = batch["sample_index"].detach().cpu().tolist()
    target_item_ids = batch["target_item_id"].detach().cpu().tolist()
    bundle_keys = list(batch["bundle_key"])
    candidate_size = int(all_item_ids.numel())
    if candidate_size <= 0:
        return []

    k = min(int(detail_topk), candidate_size)
    # Float64 offsets implement deterministic ascending-item-ID tie breaking
    # without perturbing distinct float32 scores.
    ordering_scores = scores.double() - all_item_ids.double().unsqueeze(0) * 1.0e-12
    _, top_positions = torch.topk(ordering_scores, k=k, dim=1, largest=True, sorted=True)
    top_scores = torch.gather(scores, dim=1, index=top_positions)
    top_item_ids = all_item_ids[top_positions]

    records: List[Dict[str, object]] = []
    for batch_idx, sample_index in enumerate(sample_indices):
        target_item_id = int(target_item_ids[batch_idx])
        target_position = int(
            torch.searchsorted(all_item_ids, torch.tensor(target_item_id, device=all_item_ids.device)).item()
        )
        if target_position >= candidate_size or int(all_item_ids[target_position].item()) != target_item_id:
            raise ValueError(
                f"Target item id {target_item_id} is missing from the all-item ranking catalog."
            )
        row_scores = scores[batch_idx]
        target_score = row_scores[target_position]
        higher_or_tie_before_gt = (row_scores > target_score) | (
            (row_scores == target_score) & (all_item_ids < target_item_id)
        )
        gt_rank = int(higher_or_tie_before_gt.sum().item()) + 1
        records.append(
            {
                "method": str(method_name),
                "pool_name": "all",
                "sample_index": int(sample_index),
                "bundle_key": str(bundle_keys[batch_idx]),
                "target_item_id": int(target_item_id),
                "candidate_item_ids": [],
                "candidate_item_ids_omitted": True,
                "gt_was_missing_before_fix": False,
                "gt_rank": int(gt_rank),
                "gt_score": float(target_score.detach().cpu().item()),
                "top10_candidate_item_ids": [
                    int(x) for x in top_item_ids[batch_idx].detach().cpu().tolist()
                ],
                "top10_scores": [
                    float(x) for x in top_scores[batch_idx].detach().cpu().tolist()
                ],
                "candidate_size": int(candidate_size),
            }
        )
    return records


def collect_stage2_all_feature_ranking_records(
    batch,
    preds,
    ranking_cfg,
    all_feature_ranking_state: Optional[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> List[Dict[str, object]]:
    if not ranking_compute_all(ranking_cfg) or all_feature_ranking_state is None:
        return []

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    all_item_ids, banks = all_feature_ranking_state
    pred_t = F.normalize(preds["pred_e_t"].detach().reshape(preds["pred_e_t"].shape[0], -1).float(), dim=-1)
    pred_c = F.normalize(preds["pred_e_c"].detach().reshape(preds["pred_e_c"].shape[0], -1).float(), dim=-1)
    pred_cf = F.normalize(preds["pred_e_cf"].detach().reshape(preds["pred_e_cf"].shape[0], -1).float(), dim=-1)

    scores_t = torch.matmul(pred_t, banks["feature_t"].t())
    scores_c = torch.matmul(pred_c, banks["feature_c"].t())
    scores_cf = torch.matmul(pred_cf, banks["feature_cf"].t())
    weight_t, weight_c, weight_cf = resolve_feature_fusion_weights(ranking_cfg)
    scores_fused = (
        weight_t * _zscore_rows(scores_t)
        + weight_c * _zscore_rows(scores_c)
        + weight_cf * _zscore_rows(scores_cf)
    )

    records: List[Dict[str, object]] = []
    for method_name, scores in [
        ("feature_t", scores_t),
        ("feature_c", scores_c),
        ("feature_cf", scores_cf),
        ("feature_fused", scores_fused),
    ]:
        records.extend(
            _build_all_item_records_for_scores(
                batch=batch,
                scores=scores,
                method_name=method_name,
                all_item_ids=all_item_ids,
                detail_topk=detail_topk,
            )
        )
    return records


def collect_stage2_all_candidate_log_likelihood_ranking_records(
    batch,
    log_probs: torch.Tensor,
    ranking_cfg,
    likelihood_state: Optional[CandidateLikelihoodState],
) -> List[Dict[str, object]]:
    if (
        resolve_token_ranking_method(ranking_cfg) != TOKEN_LOG_LIKELIHOOD_METHOD
        or not ranking_compute_all(ranking_cfg)
        or likelihood_state is None
    ):
        return []
    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    item_chunk_size = int(
        ranking_cfg.get(
            "token_likelihood_item_chunk_size",
            ranking_cfg.get("token_idf_item_chunk_size", 2048),
        )
    )
    scores = score_all_items_candidate_log_likelihood(
        log_probs=log_probs,
        likelihood_state=likelihood_state,
        item_chunk_size=item_chunk_size,
    )
    records = _build_all_item_records_for_scores(
        batch=batch,
        scores=scores,
        method_name=TOKEN_LOG_LIKELIHOOD_METHOD,
        all_item_ids=likelihood_state.all_item_ids,
        detail_topk=detail_topk,
    )
    for record in records:
        record["token_ranking_context"] = "one_full_mask_decoder_pass"
        record["token_weighting"] = "equal_mean_over_candidate_code_log_probabilities"
    return records


def collect_stage2_candidate_log_likelihood_ranking_records(
    batch,
    log_probs: torch.Tensor,
    dataset: QFormerMaskGenDataset,
    ranking_cfg,
    likelihood_state: CandidateLikelihoodState,
) -> List[Dict[str, object]]:
    if (
        not ranking_is_enabled(ranking_cfg)
        or resolve_token_ranking_method(ranking_cfg) != TOKEN_LOG_LIKELIHOOD_METHOD
    ):
        return []
    if likelihood_state.sequence_length != int(log_probs.shape[1]):
        raise ValueError(
            "Candidate likelihood catalog/logit sequence length mismatch: "
            f"{likelihood_state.sequence_length} vs {int(log_probs.shape[1])}"
        )

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    require_gt_in_candidate = bool(ranking_cfg.get("require_gt_in_candidate", True))
    sample_indices = batch["sample_index"].detach().cpu().tolist()
    target_item_ids = batch["target_item_id"].detach().cpu().tolist()
    bundle_keys = list(batch["bundle_key"])
    records: List[Dict[str, object]] = []
    for batch_idx, sample_index in enumerate(sample_indices):
        for pool_name, enabled in [
            ("raw", ranking_compute_raw(ranking_cfg)),
            ("acc", ranking_compute_acc(ranking_cfg)),
        ]:
            if not enabled:
                continue
            candidate_item_ids, has_candidates = dataset.get_candidate_item_ids(sample_index, pool_name)
            if pool_name == "acc" and not has_candidates:
                continue
            records.append(
                build_candidate_log_likelihood_ranking_record(
                    bundle_key=str(bundle_keys[batch_idx]),
                    sample_index=int(sample_index),
                    target_item_id=int(target_item_ids[batch_idx]),
                    candidate_item_ids=candidate_item_ids,
                    log_probs=log_probs[batch_idx],
                    item_token_lookup=dataset.item_token_lookup,
                    detail_topk=detail_topk,
                    require_gt_in_candidate=require_gt_in_candidate,
                    pool_name=pool_name,
                )
            )
    return records


def collect_stage2_all_candidate_idf_ranking_records(
    batch,
    log_probs: torch.Tensor,
    ranking_cfg,
    idf_state: Optional[CandidateSpecificIDFState],
) -> List[Dict[str, object]]:
    if (
        resolve_token_ranking_method(ranking_cfg) != TOKEN_IDF_LIKELIHOOD_METHOD
        or not ranking_compute_all(ranking_cfg)
        or idf_state is None
    ):
        return []
    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    item_chunk_size = int(ranking_cfg.get("token_idf_item_chunk_size", 2048))
    scores = score_all_items_candidate_specific_idf_likelihood(
        log_probs=log_probs,
        idf_state=idf_state,
        item_chunk_size=item_chunk_size,
    )
    records = _build_all_item_records_for_scores(
        batch=batch,
        scores=scores,
        method_name=TOKEN_IDF_LIKELIHOOD_METHOD,
        all_item_ids=idf_state.all_item_ids,
        detail_topk=detail_topk,
    )
    for record in records:
        record["token_ranking_context"] = "one_full_mask_decoder_pass"
        record["token_weighting"] = "candidate_specific_position_sqrt_idf"
        record["idf_catalog_size"] = int(idf_state.item_count)
    return records


def collect_stage2_candidate_idf_ranking_records(
    batch,
    log_probs: torch.Tensor,
    dataset: QFormerMaskGenDataset,
    ranking_cfg,
    idf_state: CandidateSpecificIDFState,
) -> List[Dict[str, object]]:
    if (
        not ranking_is_enabled(ranking_cfg)
        or resolve_token_ranking_method(ranking_cfg) != TOKEN_IDF_LIKELIHOOD_METHOD
    ):
        return []

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    require_gt_in_candidate = bool(ranking_cfg.get("require_gt_in_candidate", True))
    sample_indices = batch["sample_index"].detach().cpu().tolist()
    target_item_ids = batch["target_item_id"].detach().cpu().tolist()
    bundle_keys = list(batch["bundle_key"])
    records: List[Dict[str, object]] = []
    for batch_idx, sample_index in enumerate(sample_indices):
        for pool_name, enabled in [
            ("raw", ranking_compute_raw(ranking_cfg)),
            ("acc", ranking_compute_acc(ranking_cfg)),
        ]:
            if not enabled:
                continue
            candidate_item_ids, has_candidates = dataset.get_candidate_item_ids(sample_index, pool_name)
            if pool_name == "acc" and not has_candidates:
                continue
            records.append(
                build_candidate_specific_idf_ranking_record(
                    bundle_key=str(bundle_keys[batch_idx]),
                    sample_index=int(sample_index),
                    target_item_id=int(target_item_ids[batch_idx]),
                    candidate_item_ids=candidate_item_ids,
                    log_probs=log_probs[batch_idx],
                    item_token_lookup=dataset.item_token_lookup,
                    idf_state=idf_state,
                    detail_topk=detail_topk,
                    require_gt_in_candidate=require_gt_in_candidate,
                    pool_name=pool_name,
                )
            )
    return records


def collect_stage2_all_token_ranking_records(
    batch,
    pred_tokens: torch.Tensor,
    ranking_cfg,
    tokenizer: TATiTok,
    all_token_ranking_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    repeat_index: int,
) -> List[Dict[str, object]]:
    if (
        resolve_token_ranking_method(ranking_cfg) != TOKEN_EMBEDDING_METHOD
        or not ranking_compute_all(ranking_cfg)
        or all_token_ranking_state is None
    ):
        return []

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    all_item_ids, token_bank = all_token_ranking_state
    pred_embed = tokenizer.quantize.get_codebook_entry(pred_tokens.reshape(-1)).reshape(
        pred_tokens.shape[0],
        pred_tokens.shape[1],
        -1,
    )
    pred_embed = F.normalize(pred_embed.float(), dim=-1)
    scores = torch.einsum("bld,nld->bn", pred_embed, token_bank) / float(pred_embed.shape[1])
    records = _build_all_item_records_for_scores(
        batch=batch,
        scores=scores,
        method_name="token_emb",
        all_item_ids=all_item_ids,
        detail_topk=detail_topk,
    )
    for record in records:
        record["repeat_index"] = int(repeat_index)
    return records


def collect_stage2_token_ranking_records(
    batch,
    pred_tokens: torch.Tensor,
    dataset: QFormerMaskGenDataset,
    ranking_cfg,
    tokenizer: TATiTok,
    device: torch.device,
    repeat_index: int,
) -> List[Dict[str, object]]:
    if (
        not ranking_is_enabled(ranking_cfg)
        or resolve_token_ranking_method(ranking_cfg) != TOKEN_EMBEDDING_METHOD
    ):
        return []

    detail_topk = int(ranking_cfg.get("topk", DEFAULT_RANKING_DETAIL_TOPK))
    require_gt_in_candidate = bool(ranking_cfg.get("require_gt_in_candidate", True))
    sample_indices = batch["sample_index"].detach().cpu().tolist()
    target_item_ids = batch["target_item_id"].detach().cpu().tolist()
    bundle_keys = list(batch["bundle_key"])

    records: List[Dict[str, object]] = []
    for batch_idx, sample_index in enumerate(sample_indices):
        bundle_key = str(bundle_keys[batch_idx])
        target_item_id = int(target_item_ids[batch_idx])
        for pool_name, enabled in [("raw", ranking_compute_raw(ranking_cfg)), ("acc", ranking_compute_acc(ranking_cfg))]:
            if not enabled:
                continue
            candidate_item_ids, has_candidates = dataset.get_candidate_item_ids(sample_index, pool_name)
            if pool_name == "acc" and not has_candidates:
                continue
            record = build_token_embedding_ranking_record(
                bundle_key=bundle_key,
                sample_index=int(sample_index),
                target_item_id=target_item_id,
                candidate_item_ids=candidate_item_ids,
                pred_tokens=pred_tokens[batch_idx],
                item_token_lookup=dataset.item_token_lookup,
                quantizer=tokenizer.quantize,
                device=device,
                detail_topk=detail_topk,
                require_gt_in_candidate=require_gt_in_candidate,
                pool_name=pool_name,
                repeat_index=int(repeat_index),
            )
            records.append(record)
    return records


def main():
    workspace = os.environ.get("WORKSPACE", "").strip()
    torch_home = os.environ.get("TORCH_HOME", "").strip()
    if torch_home:
        hub_dir = Path(torch_home)
    elif workspace:
        hub_dir = Path(workspace) / "models" / "hub"
    else:
        hub_dir = Path.home() / ".cache" / "torch" / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))

    config = get_config()
    from runtime_guard import validate_config
    validate_config(config, inference=True)
    config.model.condition_source = normalize_condition_source(config.model.get("condition_source", "hidden_latent"))
    config.model.condition = list(
        normalize_temb_condition_components(config.model.get("condition", ["content", "description"]))
    )
    if "gpu_id" in config.inference:
        config.training.gpu_id = config.inference.gpu_id

    requested_device = str(config.inference.get("device", "auto"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no usable CUDA device is available.")
    device = torch.device("cpu" if requested_device == "cpu" or not torch.cuda.is_available() else f"cuda:{int(config.training.gpu_id)}")
    if config.inference.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = config.inference.get("seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    prepare_logger = setup_checkpoint_logger(Path(config.experiment.output_dir) / "tmp_prepare", -1)
    maybe_prepare_cache(
        config.dataset.test,
        config.dataset,
        logger=prepare_logger,
        ranking_cfg=config.get("ranking", None),
        split_name="test",
    )
    maybe_prepare_prompt_cache(config.model, device=device, logger=prepare_logger)

    prompt_cache = PromptEmbeddingCache(str(config.model.prompt_cache_pt))
    text_condition_provider = None
    if config.model.condition_source == "text":
        text_condition_provider = FixedTextConditionProvider(config.model, device=device, logger=prepare_logger)
    test_dataset_full = QFormerMaskGenDataset(
        config.dataset.test,
        config.dataset,
        ranking_cfg=config.get("ranking", None),
        split_name="test",
    )
    configure_model_shape_from_dataset(config, test_dataset_full, text_condition_provider=text_condition_provider)
    max_bundles = resolve_max_bundles(config)
    repeat_per_bundle = resolve_repeat_per_bundle(config)
    test_dataset = maybe_limit_dataset(test_dataset_full, max_bundles)
    test_loader = create_loader(
        test_dataset,
        batch_size=int(config.inference.batch_size),
        shuffle=False,
        num_workers=int(config.inference.num_workers),
        drop_last=False,
    )
    ranking_dataset = resolve_qformer_base_dataset(test_loader.dataset)

    checkpoint_paths = resolve_checkpoint_paths(config)
    decode_modes = get_decode_modes(config.inference)
    ranking_cfg = config.get("ranking", None)
    feature_fusion_weights = resolve_feature_fusion_weights(ranking_cfg)
    token_ranking_method = resolve_token_ranking_method(ranking_cfg)
    prepare_logger.info(
        "Feature fusion weights | text=%.6f | content=%.6f | cf=%.6f",
        feature_fusion_weights[0],
        feature_fusion_weights[1],
        feature_fusion_weights[2],
    )
    # Feature-only checkpoints (for example Stage 1) can evaluate candidate and
    # all-item feature ranking without running the visual-token decoder.  Keep
    # the default enabled so existing inference commands remain unchanged.
    token_ranking_enabled = (
        ranking_is_enabled(ranking_cfg)
        and bool(ranking_cfg.get("compute_token", True))
        and (
            ranking_compute_raw(ranking_cfg)
            or ranking_compute_acc(ranking_cfg)
            or ranking_compute_all(ranking_cfg)
        )
    )
    tokenizer_decode_modes = {"pred_text", "pred_none", "gt_text", "gt_none"}
    legacy_token_ranking = token_ranking_enabled and token_ranking_requires_generated_tokens(ranking_cfg)
    generate_visual_tokens = token_ranking_enabled and ranking_generate_visual_tokens(ranking_cfg)
    needs_tokenizer = bool(tokenizer_decode_modes.intersection(set(decode_modes))) or legacy_token_ranking
    needs_pred_tokens = (
        bool({"pred_text", "pred_none"}.intersection(set(decode_modes)))
        or legacy_token_ranking
        or generate_visual_tokens
    )
    prepare_logger.info(
        "Stage-2 visual-token ranking method = %s | generate_visual_tokens=%s | "
        "visual_tokens_used_for_ranking=%s",
        token_ranking_method,
        generate_visual_tokens,
        legacy_token_ranking,
    )
    tokenizer = None
    if needs_tokenizer:
        tokenizer = TATiTok.from_pretrained(str(config.inference.tatitok_decoder_path))
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        tokenizer = tokenizer.to(device)

    token_allowed_mask = None
    if needs_pred_tokens:
        token_allowed_mask = prepare_generation_token_allowed_mask(
            config=config,
            dataset=ranking_dataset,
            vocab_size=int(config.model.vq_model.codebook_size),
            seq_len=int(config.model.vq_model.num_latent_tokens),
            device=device,
            logger=prepare_logger,
        )

    all_feature_ranking_state = None
    if ranking_compute_all(ranking_cfg):
        all_feature_ranking_state = prepare_all_item_feature_banks(
            ranking_dataset,
            device=device,
            logger=prepare_logger,
        )
    all_token_ranking_state = None
    if ranking_compute_all(ranking_cfg) and tokenizer is not None and legacy_token_ranking:
        all_token_ranking_state = prepare_all_item_token_bank(
            ranking_dataset,
            tokenizer=tokenizer,
            device=device,
            logger=prepare_logger,
        )
    candidate_likelihood_state = None
    candidate_idf_state = None
    if token_ranking_enabled and token_ranking_method == TOKEN_LOG_LIKELIHOOD_METHOD:
        candidate_likelihood_state = prepare_candidate_likelihood_state(
            ranking_dataset.item_token_lookup,
            device=device,
        )
        prepare_logger.info(
            "Prepared equal-mean candidate likelihood state | items=%d | sequence_length=%d",
            candidate_likelihood_state.item_count,
            candidate_likelihood_state.sequence_length,
        )
    if token_ranking_enabled and token_ranking_method == TOKEN_IDF_LIKELIHOOD_METHOD:
        candidate_idf_state = prepare_candidate_specific_idf_state(
            ranking_dataset.item_token_lookup,
            vocabulary_size=int(config.model.vq_model.codebook_size),
            device=device,
        )
        prepare_logger.info(
            "Prepared candidate-specific sqrt-IDF state | items=%d | sequence_length=%d | vocabulary=%d",
            candidate_idf_state.item_count,
            candidate_idf_state.sequence_length,
            candidate_idf_state.vocabulary_size,
        )

    skip_existing_checkpoints = bool(config.inference.get("skip_existing_checkpoints", False))
    skip_existing_marker = str(config.inference.get("skip_existing_marker", "fid_summary.txt")).strip()
    if not skip_existing_marker:
        skip_existing_marker = "fid_summary.txt"
    gt_image_dir = None
    if "gt" in decode_modes:
        gt_image_dir = Path(str(config.inference.gt_image_dir))
        if not gt_image_dir.exists():
            if bool(config.inference.get("use_decoded_gt_for_fid", False)):
                prepare_logger.warning(
                    "gt_image_dir does not exist: %s; using decoded GT tokens as FID reference.",
                    gt_image_dir,
                )
                gt_image_dir = None
            else:
                raise FileNotFoundError(f"gt_image_dir does not exist: {gt_image_dir}")

    aggregate_fid_summary = bool(config.inference.get("aggregate_fid_summary", True))
    aggregate_fid_summary_filename = str(
        config.inference.get("aggregate_fid_summary_filename", "all_checkpoints_fid_summary.txt")
    ).strip()
    if not aggregate_fid_summary_filename:
        aggregate_fid_summary_filename = "all_checkpoints_fid_summary.txt"
    aggregate_ranking_summary = ranking_is_enabled(ranking_cfg) and bool(
        ranking_cfg.get("aggregate_summary", True)
    )
    aggregate_ranking_summary_filename = str(
        ranking_cfg.get("aggregate_summary_filename", DEFAULT_AGGREGATE_RANKING_SUMMARY_FILENAME)
    ).strip() if ranking_cfg is not None else DEFAULT_AGGREGATE_RANKING_SUMMARY_FILENAME
    if not aggregate_ranking_summary_filename:
        aggregate_ranking_summary_filename = DEFAULT_AGGREGATE_RANKING_SUMMARY_FILENAME
    ranking_output_subdir = str(ranking_cfg.get("output_subdir", "ranking")).strip() if ranking_cfg is not None else "ranking"
    if not ranking_output_subdir:
        ranking_output_subdir = "ranking"
    last_checkpoint_output_dir = None
    last_logger = None

    for checkpoint_path in checkpoint_paths:
        checkpoint_step = get_checkpoint_step(checkpoint_path)
        checkpoint_output_dir = resolve_checkpoint_output_dir(config, checkpoint_step)
        ranking_skip_marker_path = checkpoint_output_dir / ranking_output_subdir / "ranking_details.npy"
        if ranking_is_enabled(ranking_cfg) and ranking_skip_marker_path.exists():
            print(
                f"[skip] checkpoint-{checkpoint_step}: found ranking marker "
                f"{ranking_skip_marker_path}"
            )
            continue
        if skip_existing_checkpoints and (checkpoint_output_dir / skip_existing_marker).exists():
            print(
                f"[skip] checkpoint-{checkpoint_step}: found marker "
                f"{checkpoint_output_dir / skip_existing_marker}"
            )
            continue
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, str(checkpoint_output_dir / "resolved_config.yaml"))
        logger = setup_checkpoint_logger(checkpoint_output_dir, checkpoint_step)

        model = load_model(config, checkpoint_path, device, logger)
        model_dtype = next(model.parameters()).dtype
        logger.info("Inference condition source = %s", config.model.condition_source)

        pred_tokens_txt = checkpoint_output_dir / str(
            config.inference.get("pred_tokens_txt", "predicted_tokens.txt")
        )
        pred_tokens_file = pred_tokens_txt.open("w", encoding="utf-8") if needs_pred_tokens else None
        feature_ranking_records_local: List[Dict[str, object]] = []
        token_ranking_records_local: List[Dict[str, object]] = []
        saved_image_records_by_mode: Dict[str, List[tuple[str, Path]]] = {
            mode: [] for mode in decode_modes if mode != "gt"
        }
        if "gt" in decode_modes:
            saved_image_records_by_mode["gt"] = []

        progress = tqdm(
            total=len(test_dataset) * (repeat_per_bundle if needs_pred_tokens else 1),
            desc=f"infer_{checkpoint_step}",
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch in test_loader:
                batch_candidate_idf_records: List[Dict[str, object]] = []
                condition_inputs = build_model_condition_batch(
                    config=config,
                    batch=batch,
                    device=device,
                    dtype=model_dtype,
                    text_condition_provider=text_condition_provider,
                )
                if ranking_is_enabled(config.get("ranking", None)):
                    preds = model.predict_conditions(condition_inputs)
                    feature_ranking_records_local.extend(
                        collect_stage2_feature_ranking_records(
                            batch=batch,
                            preds=preds,
                            dataset=ranking_dataset,
                            ranking_cfg=config.get("ranking", None),
                            device=device,
                        )
                    )
                    feature_ranking_records_local.extend(
                        collect_stage2_all_feature_ranking_records(
                            batch=batch,
                            preds=preds,
                            ranking_cfg=config.get("ranking", None),
                            all_feature_ranking_state=all_feature_ranking_state,
                        )
                    )
                    if token_ranking_enabled and token_ranking_method in {
                        TOKEN_LOG_LIKELIHOOD_METHOD,
                        TOKEN_IDF_LIKELIHOOD_METHOD,
                    }:
                        ranking_logits = compute_full_mask_token_ranking_logits(
                            model=model,
                            predictions=preds,
                            use_cfg=bool(config.inference.get("use_cfg", False)),
                            guidance_scale=float(
                                config.inference.get("guidance_scale", config.model.maskgen.cfg)
                            ),
                        )
                        ranking_log_probs = F.log_softmax(ranking_logits.float(), dim=-1)
                        if token_ranking_method == TOKEN_LOG_LIKELIHOOD_METHOD:
                            if candidate_likelihood_state is None:
                                raise RuntimeError("Candidate likelihood state was not initialized.")
                            batch_candidate_idf_records.extend(
                                collect_stage2_candidate_log_likelihood_ranking_records(
                                    batch=batch,
                                    log_probs=ranking_log_probs,
                                    dataset=ranking_dataset,
                                    ranking_cfg=ranking_cfg,
                                    likelihood_state=candidate_likelihood_state,
                                )
                            )
                            batch_candidate_idf_records.extend(
                                collect_stage2_all_candidate_log_likelihood_ranking_records(
                                    batch=batch,
                                    log_probs=ranking_log_probs,
                                    ranking_cfg=ranking_cfg,
                                    likelihood_state=candidate_likelihood_state,
                                )
                            )
                        else:
                            if candidate_idf_state is None:
                                raise RuntimeError("Candidate-specific IDF state was not initialized.")
                            batch_candidate_idf_records.extend(
                                collect_stage2_candidate_idf_ranking_records(
                                    batch=batch,
                                    log_probs=ranking_log_probs,
                                    dataset=ranking_dataset,
                                    ranking_cfg=ranking_cfg,
                                    idf_state=candidate_idf_state,
                                )
                            )
                            batch_candidate_idf_records.extend(
                                collect_stage2_all_candidate_idf_ranking_records(
                                    batch=batch,
                                    log_probs=ranking_log_probs,
                                    ranking_cfg=ranking_cfg,
                                    idf_state=candidate_idf_state,
                                )
                            )
                        token_ranking_records_local.extend(batch_candidate_idf_records)
                target_item_ids = batch["target_item_id"].tolist()
                bundle_keys = batch["bundle_key"]
                base_sample_prefixes = [
                    build_sample_prefix(bundle_key, target_item_id)
                    for bundle_key, target_item_id in zip(bundle_keys, target_item_ids)
                ]

                text_condition = prompt_cache.get_batch(target_item_ids, device=device, dtype=model_dtype)
                empty_text_condition = prompt_cache.get_empty_batch(
                    len(target_item_ids),
                    device=device,
                    dtype=model_dtype,
                )
                gt_tokens = None
                if "gt_text" in decode_modes or "gt_none" in decode_modes or ("gt" in decode_modes and gt_image_dir is None):
                    gt_tokens = batch["tokens"].to(device=device, non_blocking=True)

                static_decoded_by_mode: Dict[str, torch.Tensor] = {}
                if "gt" in decode_modes and gt_image_dir is None:
                    gt_decode_mode = str(config.inference.get("decoded_gt_mode", "text") or "text").strip().lower()
                    if gt_decode_mode in {"text", "gt_text"}:
                        static_decoded_by_mode["gt"] = tokenizer.decode_tokens(gt_tokens, text_condition)
                    elif gt_decode_mode in {"none", "gt_none"}:
                        static_decoded_by_mode["gt"] = tokenizer.decode_tokens(gt_tokens, empty_text_condition)
                    else:
                        raise ValueError(
                            "inference.decoded_gt_mode must be text/gt_text/none/gt_none, "
                            f"got {gt_decode_mode!r}"
                        )
                if "gt_text" in decode_modes:
                    static_decoded_by_mode["gt_text"] = tokenizer.decode_tokens(gt_tokens, text_condition)
                if "gt_none" in decode_modes:
                    static_decoded_by_mode["gt_none"] = tokenizer.decode_tokens(gt_tokens, empty_text_condition)

                for idx, target_item_id in enumerate(target_item_ids):
                    sample_prefix = base_sample_prefixes[idx]
                    if gt_image_dir is not None:
                        gt_save_path = maybe_save_gt_image(
                            checkpoint_output_dir,
                            gt_image_dir,
                            target_item_id,
                            sample_prefix,
                        )
                        saved_image_records_by_mode["gt"].append((sample_prefix, gt_save_path))
                    for mode, decoded in static_decoded_by_mode.items():
                        image_path = checkpoint_output_dir / f"{sample_prefix}_{mode}.png"
                        image = Image.fromarray(_tensor_to_uint8_image(decoded[idx]))
                        image.save(image_path)
                        saved_image_records_by_mode[mode].append((sample_prefix, image_path))

                if not needs_pred_tokens:
                    progress.update(len(target_item_ids))
                    continue

                for repeat_index in range(repeat_per_bundle):
                    pred_tokens = generate_maskgen_qformer(
                        model=model,
                        hidden_condition=condition_inputs,
                        guidance_scale=float(config.inference.get("guidance_scale", config.model.maskgen.cfg)),
                        guidance_decay=str(config.inference.get("cfg_schedule", config.model.maskgen.cfg_schedule)),
                        guidance_decay_scale_pow=float(
                            config.inference.get("cfg_decay_scale_pow", config.model.maskgen.cfg_decay_scale_pow)
                        ),
                        randomize_temperature=float(
                            config.inference.get("randomize_temperature", config.model.maskgen.randomize_temperature)
                        ),
                        softmax_temperature_annealing=bool(
                            config.inference.get(
                                "softmax_temperature_annealing",
                                config.model.maskgen.softmax_temperature_annealing,
                            )
                        ),
                        num_sample_steps=int(config.inference.get("num_iter", config.model.maskgen.num_iter)),
                        prob_sorting=bool(config.inference.get("prob_sorting", config.model.maskgen.prob_sorting)),
                        use_cfg=bool(config.inference.get("use_cfg", False)),
                        token_allowed_mask=token_allowed_mask,
                        logit_top_k=int(config.inference.get("logit_top_k", 0) or 0),
                        final_step_no_gumbel=bool(config.inference.get("final_step_no_gumbel", False)),
                    )
                    if batch_candidate_idf_records:
                        attach_visual_generated_tokens(
                            records=batch_candidate_idf_records,
                            sample_indices=batch["sample_index"].detach().cpu().tolist(),
                            generated_tokens=pred_tokens,
                            repeat_index=repeat_index + 1,
                        )
                    token_ranking_records_local.extend(
                        collect_stage2_token_ranking_records(
                            batch=batch,
                            pred_tokens=pred_tokens,
                            dataset=ranking_dataset,
                            ranking_cfg=config.get("ranking", None),
                            tokenizer=tokenizer,
                            device=device,
                            repeat_index=repeat_index + 1,
                        )
                    )
                    token_ranking_records_local.extend(
                        collect_stage2_all_token_ranking_records(
                            batch=batch,
                            pred_tokens=pred_tokens,
                            ranking_cfg=config.get("ranking", None),
                            tokenizer=tokenizer,
                            all_token_ranking_state=all_token_ranking_state,
                            repeat_index=repeat_index + 1,
                        )
                    )

                    decoded_by_mode: Dict[str, torch.Tensor] = {}
                    if "pred_text" in decode_modes:
                        decoded_by_mode["pred_text"] = tokenizer.decode_tokens(pred_tokens, text_condition)
                    if "pred_none" in decode_modes:
                        decoded_by_mode["pred_none"] = tokenizer.decode_tokens(pred_tokens, empty_text_condition)

                    for idx, target_item_id in enumerate(target_item_ids):
                        base_sample_prefix = base_sample_prefixes[idx]
                        sample_prefix = build_inference_sample_prefix(
                            base_sample_prefix=base_sample_prefix,
                            repeat_index=repeat_index,
                            repeat_per_bundle=repeat_per_bundle,
                        )
                        pred_tokens_fields = [str(bundle_keys[idx])]
                        if repeat_per_bundle > 1:
                            pred_tokens_fields.append(f"repeat_idx={repeat_index + 1}")
                        pred_tokens_fields.extend(str(x) for x in pred_tokens[idx].tolist())
                        if pred_tokens_file is not None:
                            pred_tokens_file.write(", ".join(pred_tokens_fields) + "\n")
                        for mode, decoded in decoded_by_mode.items():
                            image_path = checkpoint_output_dir / f"{sample_prefix}_{mode}.png"
                            image = Image.fromarray(_tensor_to_uint8_image(decoded[idx]))
                            image.save(image_path)
                            saved_image_records_by_mode[mode].append((base_sample_prefix, image_path))

                    progress.update(len(target_item_ids))

        if pred_tokens_file is not None:
            pred_tokens_file.close()
        progress.close()
        if ranking_is_enabled(config.get("ranking", None)):
            reduced_token_records = (
                list(token_ranking_records_local)
                if token_ranking_method
                in {TOKEN_LOG_LIKELIHOOD_METHOD, TOKEN_IDF_LIKELIHOOD_METHOD}
                else reduce_best_of_repeat_records(token_ranking_records_local)
            )
            all_ranking_records = list(feature_ranking_records_local) + reduced_token_records
            summary_by_pool, detail_by_pool = group_ranking_records_by_pool_and_method(
                all_ranking_records,
                FEATURE_RANKING_METHODS + TOKEN_RANKING_METHODS,
            )
            ranking_output_dir = checkpoint_output_dir / ranking_output_subdir
            ranking_topk_images = int(ranking_cfg.get("save_topk_images", 3) or 0)
            if ranking_topk_images > 0:
                ranking_item_image_dir = str(ranking_cfg.get("item_image_dir", "") or "").strip()
                if not ranking_item_image_dir:
                    ranking_item_image_dir = str(config.inference.gt_image_dir)
                save_ranking_topk_item_images(
                    output_dir=ranking_output_dir,
                    detail_by_pool=detail_by_pool,
                    item_image_dir=Path(ranking_item_image_dir),
                    method_name=token_ranking_method,
                    topk=ranking_topk_images,
                    require_all=bool(ranking_cfg.get("require_topk_images", True)),
                    logger=logger,
                )
            save_ranking_outputs(ranking_output_dir, summary_by_pool, detail_by_pool, logger=logger)
            for pool_name, pool_summary in summary_by_pool.items():
                if "feature_fused" in pool_summary:
                    logger.info("Inference ranking | pool=%s | feature_fused=%s", pool_name, pool_summary["feature_fused"])
                if token_ranking_method in pool_summary:
                    logger.info(
                        "Inference ranking | pool=%s | %s=%s",
                        pool_name,
                        token_ranking_method,
                        pool_summary[token_ranking_method],
                    )
        logger.info(
            "Inference finished | checkpoint=%s | outputs=%s | num_bundles=%d | repeat_per_bundle=%d | "
            "num_inference_runs=%d | decode_modes=%s",
            checkpoint_path,
            checkpoint_output_dir,
            len(test_dataset),
            repeat_per_bundle,
            len(test_dataset) * repeat_per_bundle,
            decode_modes,
        )
        compute_and_save_fid_scores(
            image_output_dir=checkpoint_output_dir,
            decode_modes=decode_modes,
            device=device,
            logger=logger,
            image_suffix_template="_{mode}.png",
            compute_fid=bool(config.inference.get("compute_fid", True)),
            fid_image_size=int(config.inference.get("fid_image_size", 256)),
            fid_batch_size=int(config.inference.get("fid_batch_size", 32)),
            saved_image_records_by_mode=saved_image_records_by_mode,
        )
        run_generation_quality_eval(
            config=config,
            checkpoint_output_dir=checkpoint_output_dir,
            ranking_output_subdir=ranking_output_subdir,
            logger=logger,
        )
        last_checkpoint_output_dir = checkpoint_output_dir
        last_logger = logger

    if aggregate_fid_summary and last_checkpoint_output_dir is not None and last_logger is not None:
        aggregate_root_dir = (
            last_checkpoint_output_dir.parent
            if last_checkpoint_output_dir.name.startswith("checkpoint-")
            else last_checkpoint_output_dir
        )
        aggregate_fid_summaries_across_checkpoints(
            root_dir=aggregate_root_dir,
            logger=last_logger,
            output_filename=aggregate_fid_summary_filename,
            checkpoint_step_fn=get_checkpoint_step,
        )
    if aggregate_ranking_summary and last_checkpoint_output_dir is not None and last_logger is not None:
        aggregate_root_dir = (
            last_checkpoint_output_dir.parent
            if last_checkpoint_output_dir.name.startswith("checkpoint-")
            else last_checkpoint_output_dir
        )
        aggregate_ranking_summaries_across_checkpoints(
            root_dir=aggregate_root_dir,
            logger=last_logger,
            output_filename=aggregate_ranking_summary_filename,
            checkpoint_step_fn=get_checkpoint_step,
            ranking_output_subdir=ranking_output_subdir,
        )


if __name__ == "__main__":
    main()
