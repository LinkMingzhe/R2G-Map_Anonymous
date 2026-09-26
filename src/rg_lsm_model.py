import math

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from diffusers.models.embeddings import PixArtAlphaTextProjection, Timesteps, TimestepEmbedding

from torch.utils.data import Dataset, Subset

from ranking_utils_qformer import AGGREGATE_RANKING_POOL_ORDER, build_ranking_summary, extract_ranking_candidate_ids, load_ranking_metadata, ranking_compute_acc, ranking_compute_raw, ranking_is_enabled, resolve_ranking_meta_path

from prompt_cache import PromptEmbeddingCache, maybe_prepare_prompt_cache

from runtime_utils import _dup_rank, _normalize_step_key, _read_input_gt_pairs, _safe_file_component, _tensor_to_uint8_image, _torch_load_compat, json_load, load_open_clip_text_encoder

from modeling.maskgen import MaskGen_VQ, get_masking_ratio

VALID_STAGES = {"stage1", "stage2"}

STAGE_ALIASES = {
    "1": "stage1",
    "a": "stage1",
    "s1": "stage1",
    "stage1": "stage1",
    "2": "stage2",
    "b": "stage2",
    "c": "stage2",
    "s2": "stage2",
    "stage2": "stage2",
}

VALID_CONDITION_SOURCES = {"hidden_latent", "text"}

VALID_TEMB_CONDITION_COMPONENTS = {"content", "description"}

CACHE_SCHEMA_VERSION = "qformer_v2_hidden_metadata_targets_v1"

def normalize_stage(stage_value) -> str:
    stage_key = str(stage_value).strip().lower()
    if stage_key not in STAGE_ALIASES:
        raise ValueError(f"Unsupported training.stage={stage_value!r}. Valid stages: {sorted(VALID_STAGES)}")
    return STAGE_ALIASES[stage_key]

def normalize_condition_source(source_value) -> str:
    source = str(source_value).strip().lower()
    if source not in VALID_CONDITION_SOURCES:
        raise ValueError(
            f"Unsupported model.condition_source={source_value!r}. "
            f"Valid choices: {sorted(VALID_CONDITION_SOURCES)}"
        )
    return source

def normalize_temb_condition_components(condition_value) -> Tuple[str, ...]:
    if condition_value is None:
        raw_components = ["content", "description"]
    elif isinstance(condition_value, str):
        raw_components = [condition_value]
    else:
        raw_components = list(condition_value)

    normalized_components: List[str] = []
    for component in raw_components:
        component_key = str(component).strip().lower()
        if component_key not in VALID_TEMB_CONDITION_COMPONENTS:
            raise ValueError(
                f"Unsupported model.condition entry={component!r}. "
                f"Valid choices: {sorted(VALID_TEMB_CONDITION_COMPONENTS)}"
            )
        if component_key not in normalized_components:
            normalized_components.append(component_key)

    if not normalized_components:
        raise ValueError("model.condition must contain at least one of: content, description")

    return tuple(normalized_components)

def get_decode_modes(runtime_cfg) -> List[str]:
    modes = runtime_cfg.get("decode_modes", ["pred_text", "pred_none", "gt_text", "gt_none", "gt"])
    if isinstance(modes, str):
        return [modes]
    return list(modes)

def encode_open_clip_prompt(
    prompt: str,
    clip_tokenizer,
    clip_encoder,
    device: torch.device,
) -> torch.Tensor:
    text_tokens = clip_tokenizer([prompt]).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    prompt_embeds = clip_encoder.token_embedding(text_tokens).to(cast_dtype)
    prompt_embeds = prompt_embeds + clip_encoder.positional_embedding.to(cast_dtype)
    prompt_embeds = prompt_embeds.permute(1, 0, 2)
    prompt_embeds = clip_encoder.transformer(prompt_embeds, attn_mask=clip_encoder.attn_mask)
    prompt_embeds = prompt_embeds.permute(1, 0, 2)
    prompt_embeds = clip_encoder.ln_final(prompt_embeds)
    return prompt_embeds.detach().contiguous()

class FixedTextConditionProvider:
    def __init__(self, model_cfg, device: torch.device, logger):
        self.prompt = str(model_cfg.get("generic_prompt", "A fashion item image, on white background."))
        clip_encoder, clip_tokenizer = load_open_clip_text_encoder(
            model_name=str(model_cfg.clip_model),
            pretrained=str(model_cfg.clip_pretrained),
            device=device,
        )
        self.prompt_embeds = encode_open_clip_prompt(
            prompt=self.prompt,
            clip_tokenizer=clip_tokenizer,
            clip_encoder=clip_encoder,
            device=device,
        )
        self.condition_shape = tuple(self.prompt_embeds.shape[1:])
        logger.info(
            "Using CLIP text condition | prompt=%s | condition_shape=%s | clip_model=%s | clip_pretrained=%s",
            self.prompt,
            self.condition_shape,
            model_cfg.clip_model,
            model_cfg.clip_pretrained,
        )
        del clip_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def get_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
        return prompt_embeds.to(device=device, dtype=dtype, non_blocking=True)

def build_model_condition_batch(
    config,
    batch,
    device: torch.device,
    dtype: torch.dtype,
    text_condition_provider: Optional[FixedTextConditionProvider] = None,
) -> torch.Tensor:
    condition_source = normalize_condition_source(config.model.get("condition_source", "hidden_latent"))
    if condition_source == "hidden_latent":
        return batch["hidden_condition"].to(device=device, dtype=dtype, non_blocking=True)
    if text_condition_provider is None:
        raise ValueError("text_condition_provider is required when model.condition_source=text")
    batch_size = int(batch["target_item_id"].shape[0])
    return text_condition_provider.get_batch(batch_size=batch_size, device=device, dtype=dtype)

def use_hidden_sample_timestep_acc(dataset_cfg) -> bool:
    return bool(dataset_cfg.get("hidden_sampleTimestep_acc", False))

def resolve_hidden_condition_npy(split_cfg, dataset_cfg) -> str:
    raw_hidden_npy = str(split_cfg.hidden_npy)
    if not use_hidden_sample_timestep_acc(dataset_cfg):
        return raw_hidden_npy

    explicit_hidden_npy = str(split_cfg.get("hidden_sampleTimestep_acc_npy", "") or "").strip()
    if explicit_hidden_npy:
        return explicit_hidden_npy

    if "hidden_sampleTimestep_acc" in raw_hidden_npy:
        return raw_hidden_npy

    hidden_npy_path = Path(raw_hidden_npy)
    parent_str = str(hidden_npy_path.parent)
    if "hidden_acc" not in parent_str:
        raise ValueError(
            "dataset.hidden_sampleTimestep_acc=true but hidden_sampleTimestep_acc_npy was not provided "
            f"and hidden_npy parent does not contain 'hidden_acc': {raw_hidden_npy}"
        )

    if hidden_npy_path.name.endswith("_predicted_p_x0_hidden_by_bundle.npy"):
        inferred_name = hidden_npy_path.name.replace(
            "_predicted_p_x0_hidden_by_bundle.npy",
            "_hidden_sampleTimestep_acc_by_bundle.npy",
        )
    else:
        split_prefix = hidden_npy_path.stem.split("_", 1)[0]
        inferred_name = f"{split_prefix}_hidden_sampleTimestep_acc_by_bundle.npy"

    inferred_path = Path(parent_str.replace("hidden_acc", "hidden_sampleTimestep_acc")) / inferred_name
    return str(inferred_path)

def resolve_hidden_cache_pt(split_cfg, dataset_cfg) -> str:
    if not use_hidden_sample_timestep_acc(dataset_cfg):
        return str(split_cfg.cache_pt)
    explicit_cache_pt = str(split_cfg.get("hidden_sampleTimestep_acc_cache_pt", "") or "").strip()
    if explicit_cache_pt:
        return explicit_cache_pt
    return str(split_cfg.cache_pt)

def get_hidden_sample_timestep_acc_token_candidates(dataset_cfg) -> List[str]:
    candidates = ["pred_hidden_final", "predicted token", "pred_hidden_4", "predict_hidden_latent_4"]
    explicit_key = str(dataset_cfg.get("hidden_sampleTimestep_acc_token_key", "") or "").strip()
    if explicit_key and explicit_key not in candidates:
        candidates.insert(0, explicit_key)
    return candidates

def _find_position_item_entry(bundle_key: str, bundle_data: Dict, target_position: int) -> Optional[Dict]:
    target_position = int(target_position)
    predicted_items = bundle_data.get("predicted_items", None)
    if predicted_items is None:
        return None
    for item_entry in predicted_items:
        if int(item_entry.get("pred_item_position", -1)) == target_position:
            return item_entry
    raise KeyError(
        f"Missing predicted_items entry for bundle {bundle_key}, target_position={target_position}."
    )

def _extract_condition_latents(
    bundle_key: str,
    bundle_data: Dict,
    dataset_cfg,
    target_position: int = 1,
) -> np.ndarray:
    candidate_keys = get_hidden_sample_timestep_acc_token_candidates(dataset_cfg)
    if use_hidden_sample_timestep_acc(dataset_cfg):
        source_data = bundle_data
        selected_key = next((key for key in candidate_keys if key in source_data), None)
        if selected_key is None:
            item_entry = _find_position_item_entry(
                bundle_key=bundle_key,
                bundle_data=bundle_data,
                target_position=target_position,
            )
            if item_entry is not None:
                source_data = item_entry
                selected_key = next((key for key in candidate_keys if key in source_data), None)
        if selected_key is None:
            raise KeyError(
                "Missing hidden_sampleTimestep_acc latent key for bundle "
                f"{bundle_key}. Tried {candidate_keys!r}; available keys: {sorted(source_data.keys())[:20]!r}"
            )
        selected = np.asarray(source_data[selected_key], dtype=np.float16)
        if selected.ndim != 2:
            raise ValueError(
                f"Expected hidden_sampleTimestep_acc latent with shape [4, dim] for bundle {bundle_key}, got {selected.shape}"
            )
        if selected.shape[0] != 4:
            raise ValueError(
                f"Expected 4 predicted hidden tokens for bundle {bundle_key}, got shape {selected.shape}"
            )
        return selected

    if "repeat_01" in bundle_data:
        repeat_data = bundle_data["repeat_01"]
        source_data = repeat_data
        selected_key = next((key for key in candidate_keys if key in source_data), None)
        if selected_key is None:
            item_entry = _find_position_item_entry(
                bundle_key=bundle_key,
                bundle_data=repeat_data,
                target_position=target_position,
            )
            if item_entry is not None:
                source_data = item_entry
                selected_key = next((key for key in candidate_keys if key in source_data), None)
        if selected_key is None:
            raise KeyError(
                "Missing raw hidden_sampleTimestep latent key for bundle "
                f"{bundle_key}. Tried {candidate_keys!r}; available keys: {sorted(source_data.keys())[:20]!r}"
            )
        selected = np.asarray(source_data[selected_key], dtype=np.float16)
        if selected.ndim != 2 or selected.shape[0] != 4:
            raise ValueError(
                f"Expected raw hidden_sampleTimestep latent with shape [4, dim] for bundle {bundle_key}, got {selected.shape}"
            )
        return selected

    hidden_top_key = str(dataset_cfg.hidden_top_key)
    step_key = _normalize_step_key(dataset_cfg.hidden_step)
    hidden_token_positions = list(dataset_cfg.hidden_token_positions)

    if hidden_top_key not in bundle_data:
        raise KeyError(f"Missing top key {hidden_top_key!r} in hidden bundle {bundle_key}")
    steps = bundle_data[hidden_top_key]
    if step_key not in steps:
        raise KeyError(f"Missing step key {step_key!r} in hidden bundle {bundle_key}")

    hidden_step_data = steps[step_key]
    hidden = hidden_step_data["pre_logits_hidden"]
    selected = np.asarray(hidden[hidden_token_positions], dtype=np.float16)
    if selected.shape != (len(hidden_token_positions), hidden.shape[-1]):
        raise ValueError(f"Unexpected selected latent shape for bundle {bundle_key}: {selected.shape}")
    return selected

def _select_condition_latent_from_source(
    bundle_key: str,
    source_data: Dict,
    dataset_cfg,
    source_name: str,
) -> np.ndarray:
    candidate_keys = get_hidden_sample_timestep_acc_token_candidates(dataset_cfg)
    selected_key = next((key for key in candidate_keys if key in source_data), None)
    if selected_key is None:
        raise KeyError(
            f"Missing hidden latent key in {source_name} for bundle {bundle_key}. "
            f"Tried {candidate_keys!r}; available keys: {sorted(source_data.keys())[:20]!r}"
        )
    selected = np.asarray(source_data[selected_key], dtype=np.float16)
    if selected.ndim != 2 or selected.shape[0] != 4:
        raise ValueError(
            f"Expected hidden latent with shape [4, dim] in {source_name} for bundle {bundle_key}, got {selected.shape}"
        )
    return selected

def _extract_hidden_metadata_condition_entries(
    bundle_key: str,
    bundle_data: Dict,
    dataset_cfg,
) -> List[Dict[str, object]]:
    if use_hidden_sample_timestep_acc(dataset_cfg):
        source_root = bundle_data
    else:
        source_root = bundle_data.get("repeat_01", None)
    if not isinstance(source_root, dict):
        return []

    predicted_items = source_root.get("predicted_items", None)
    if predicted_items:
        target_ids_by_bundle = [
            int(item["groundtruth_item_id_at_position"])
            for item in predicted_items
            if "groundtruth_item_id_at_position" in item
        ]
        entries: List[Dict[str, object]] = []
        for item_entry in predicted_items:
            if "groundtruth_item_id_at_position" not in item_entry:
                continue
            target_position = int(item_entry.get("pred_item_position", len(entries) + 1))
            target_item_id = int(item_entry["groundtruth_item_id_at_position"])
            entries.append(
                {
                    "target_position": target_position,
                    "target_item_id": target_item_id,
                    "target_item_ids_by_bundle": target_ids_by_bundle or [target_item_id],
                    "condition_latent": _select_condition_latent_from_source(
                        bundle_key=bundle_key,
                        source_data=item_entry,
                        dataset_cfg=dataset_cfg,
                        source_name=f"predicted_items[{target_position}]",
                    ),
                    "condition_latents_source": "hidden_metadata_predicted_items",
                }
            )
        return entries

    if "groundtruth_target_item_id" in bundle_data:
        target_item_id = int(bundle_data["groundtruth_target_item_id"])
        return [
            {
                "target_position": 1,
                "target_item_id": target_item_id,
                "target_item_ids_by_bundle": [target_item_id],
                "condition_latent": _select_condition_latent_from_source(
                    bundle_key=bundle_key,
                    source_data=source_root,
                    dataset_cfg=dataset_cfg,
                    source_name="repeat_01",
                ),
                "condition_latents_source": "hidden_metadata_single_target",
            }
        ]

    return []


def maybe_prepare_cache(split_cfg, dataset_cfg, logger, ranking_cfg=None, split_name: str = "train"):
    cache_pt = resolve_hidden_cache_pt(split_cfg, dataset_cfg)
    cache_path = Path(cache_pt)
    hidden_npy = resolve_hidden_condition_npy(split_cfg, dataset_cfg)

    if cache_path.exists():
        cache = _torch_load_compat(str(cache_path), map_location="cpu")
        expected = {
            "source_hidden_npy": str(hidden_npy),
            "source_input_txt": str(split_cfg.input_txt),
            "source_gt_txt": str(split_cfg.gt_txt),
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "hidden_sampleTimestep_acc": bool(use_hidden_sample_timestep_acc(dataset_cfg)),
        }
        if "supervise_all_gt" in dataset_cfg:
            expected["supervise_all_gt"] = bool(dataset_cfg.supervise_all_gt)
        if ranking_is_enabled(ranking_cfg):
            expected["ranking_enabled"] = True
            expected["ranking_compute_raw"] = bool(ranking_compute_raw(ranking_cfg))
            expected["ranking_compute_acc"] = bool(ranking_compute_acc(ranking_cfg))
            if ranking_compute_raw(ranking_cfg):
                expected["ranking_raw_meta_path"] = str(resolve_ranking_meta_path(ranking_cfg, split_name, acc=False))
            if ranking_compute_acc(ranking_cfg):
                expected["ranking_acc_meta_path"] = str(resolve_ranking_meta_path(ranking_cfg, split_name, acc=True))
        if use_hidden_sample_timestep_acc(dataset_cfg):
            expected["condition_latents_source"] = "pred_hidden_final_or_predicted_token"
            expected["hidden_sampleTimestep_acc_token_candidates"] = list(
                get_hidden_sample_timestep_acc_token_candidates(dataset_cfg)
            )
        else:
            expected["condition_latents_source"] = "hidden_metadata_or_pre_logits_positions"
            expected["hidden_step_key"] = _normalize_step_key(dataset_cfg.hidden_step)
            expected["hidden_top_key"] = str(dataset_cfg.hidden_top_key)
            expected["hidden_token_positions"] = list(dataset_cfg.hidden_token_positions)

        cache_matches = all(cache.get(key) == value for key, value in expected.items())
        if cache_matches:
            logger.info("Using existing cache: %s", cache_path)
            return
        logger.info("Cache mismatch detected, rebuilding: %s", cache_path)

    raise ValueError("Anonymous inference requires a matching packaged test cache; rebuilding is disabled.")

class QFormerMaskGenDataset(Dataset):
    def __init__(self, split_cfg, dataset_cfg, ranking_cfg=None, split_name: str = "train"):
        cache = _torch_load_compat(resolve_hidden_cache_pt(split_cfg, dataset_cfg), map_location="cpu")
        with open(dataset_cfg.tatitok_token_json, "r", encoding="utf-8") as f:
            tatitok_tokens = json_load(f)

        self.bundle_keys: List[str] = [str(x) for x in cache["bundle_keys"]]
        self.base_bundle_ids: List[str] = [str(x) for x in cache["base_bundle_ids"]]
        self.input_item_ids = torch.as_tensor(cache["input_item_ids"], dtype=torch.long)
        self.target_item_ids = torch.as_tensor(cache["target_item_ids"], dtype=torch.long)
        self.target_positions = torch.as_tensor(
            cache.get("target_positions", np.ones(len(self.bundle_keys), dtype=np.int64)),
            dtype=torch.long,
        )
        self.hidden_condition = torch.as_tensor(cache["condition_latents"], dtype=torch.float32).detach().contiguous()
        self.ranking_enabled = bool(ranking_is_enabled(ranking_cfg))
        self.ranking_compute_raw = bool(ranking_compute_raw(ranking_cfg))
        self.ranking_compute_acc = bool(ranking_compute_acc(ranking_cfg))
        self.split_name = str(split_name)

        self.content_features = torch.as_tensor(
            _torch_load_compat(str(dataset_cfg.content_feature_path), map_location="cpu"),
            dtype=torch.float32,
        ).detach().contiguous()
        self.description_features = torch.as_tensor(
            _torch_load_compat(str(dataset_cfg.description_feature_path), map_location="cpu"),
            dtype=torch.float32,
        ).detach().contiguous()
        self.cf_features = torch.as_tensor(
            _torch_load_compat(str(dataset_cfg.cf_feature_path), map_location="cpu"),
            dtype=torch.float32,
        ).detach().contiguous()

        self.target_tokens: List[torch.Tensor] = []
        self.item_token_lookup: Dict[int, torch.Tensor] = {}
        missing_token_ids: List[str] = []
        for token_key, tokens in tatitok_tokens.items():
            if len(tokens) != 128:
                raise ValueError(
                    f"Expected 128 TA-TiTok tokens for item {token_key}, got {len(tokens)}"
                )
            self.item_token_lookup[int(token_key)] = torch.tensor(tokens, dtype=torch.long)
        for target_item_id in self.target_item_ids.tolist():
            if int(target_item_id) not in self.item_token_lookup:
                missing_token_ids.append(str(target_item_id))
                continue
            self.target_tokens.append(self.item_token_lookup[int(target_item_id)])

        if missing_token_ids:
            raise KeyError(
                f"Missing {len(missing_token_ids)} item ids in tatitok json, first few: {missing_token_ids[:10]}"
            )

        cache_raw_candidates = cache.get("raw_candidate_item_ids", None)
        cache_acc_candidates = cache.get("acc_candidate_item_ids", None)
        cache_has_acc = cache.get("has_acc_candidate", None)
        if self.ranking_enabled:
            if self.ranking_compute_raw and cache_raw_candidates is None:
                raise KeyError(
                    f"Ranking is enabled for split={self.split_name}, but raw_candidate_item_ids are missing from cache."
                )
            if self.ranking_compute_acc and cache_acc_candidates is None:
                raise KeyError(
                    f"Ranking is enabled for split={self.split_name}, but acc_candidate_item_ids are missing from cache."
                )
        self.raw_candidate_item_ids: List[np.ndarray] = [
            np.asarray(x, dtype=np.int64)
            for x in (cache_raw_candidates if cache_raw_candidates is not None else [np.asarray([], dtype=np.int64)] * len(self.bundle_keys))
        ]
        self.acc_candidate_item_ids: List[np.ndarray] = [
            np.asarray(x, dtype=np.int64)
            for x in (cache_acc_candidates if cache_acc_candidates is not None else [np.asarray([], dtype=np.int64)] * len(self.bundle_keys))
        ]
        if cache_has_acc is None:
            self.has_acc_candidate = np.zeros(len(self.bundle_keys), dtype=np.bool_)
        else:
            self.has_acc_candidate = np.asarray(cache_has_acc, dtype=np.bool_)

        max_bundles = int(split_cfg.get("max_bundles", 0) or 0)
        if max_bundles > 0:
            limit = min(len(self.bundle_keys), max_bundles)
            self.bundle_keys = self.bundle_keys[:limit]
            self.base_bundle_ids = self.base_bundle_ids[:limit]
            self.input_item_ids = self.input_item_ids[:limit]
            self.target_item_ids = self.target_item_ids[:limit]
            self.target_positions = self.target_positions[:limit]
            self.hidden_condition = self.hidden_condition[:limit]
            self.target_tokens = self.target_tokens[:limit]
            self.raw_candidate_item_ids = self.raw_candidate_item_ids[:limit]
            self.acc_candidate_item_ids = self.acc_candidate_item_ids[:limit]
            self.has_acc_candidate = self.has_acc_candidate[:limit]

        if self.hidden_condition.shape[0] != len(self.target_tokens):
            raise ValueError(
                f"Cache size mismatch: hidden_condition={self.hidden_condition.shape[0]}, "
                f"target_tokens={len(self.target_tokens)}"
            )

    def __len__(self):
        return len(self.bundle_keys)

    def __getitem__(self, idx):
        target_item_id = int(self.target_item_ids[idx].item())
        if target_item_id >= self.content_features.shape[0]:
            raise IndexError(
                f"target_item_id {target_item_id} out of range for content feature shape "
                f"{tuple(self.content_features.shape)}"
            )
        return {
            "sample_index": torch.tensor(int(idx), dtype=torch.long),
            "bundle_key": self.bundle_keys[idx],
            "base_bundle_id": self.base_bundle_ids[idx],
            "input_item_ids": self.input_item_ids[idx],
            "target_item_id": self.target_item_ids[idx],
            "target_position": self.target_positions[idx],
            "hidden_condition": self.hidden_condition[idx].detach(),
            "e_content": self.content_features[target_item_id].unsqueeze(0).detach(),
            "e_description": self.description_features[target_item_id].unsqueeze(0).detach(),
            "e_cf": self.cf_features[target_item_id].unsqueeze(0).detach(),
            "tokens": self.target_tokens[idx],
        }

    def get_candidate_item_ids(self, sample_index: int, pool_name: str) -> Tuple[np.ndarray, bool]:
        sample_index = int(sample_index)
        if pool_name == "raw":
            return np.asarray(self.raw_candidate_item_ids[sample_index], dtype=np.int64), True
        if pool_name == "acc":
            return np.asarray(self.acc_candidate_item_ids[sample_index], dtype=np.int64), bool(
                self.has_acc_candidate[sample_index]
            )
        raise ValueError(f"Unsupported ranking pool_name={pool_name!r}")

    def get_item_tokens(self, item_ids: Sequence[int]) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        missing_item_ids: List[int] = []
        for item_id in item_ids:
            token_tensor = self.item_token_lookup.get(int(item_id))
            if token_tensor is None:
                missing_item_ids.append(int(item_id))
                continue
            tokens.append(token_tensor)
        if missing_item_ids:
            raise KeyError(
                f"Missing {len(missing_item_ids)} item ids in TA-TiTok token lookup; "
                f"first few: {missing_item_ids[:10]}"
            )
        return torch.stack(tokens, dim=0)

def resolve_qformer_base_dataset(dataset) -> QFormerMaskGenDataset:
    base_dataset = dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    if not isinstance(base_dataset, QFormerMaskGenDataset):
        raise TypeError(
            f"Expected QFormerMaskGenDataset or Subset[QFormerMaskGenDataset], got {type(base_dataset).__name__}"
        )
    return base_dataset

def group_ranking_records_by_pool_and_method(
    records: Sequence[Dict[str, object]],
    methods: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, List[Dict[str, object]]]]]:
    summary_by_pool: Dict[str, Dict[str, Dict[str, float]]] = {}
    detail_by_pool: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for pool_name in AGGREGATE_RANKING_POOL_ORDER:
        pool_summary: Dict[str, Dict[str, float]] = {}
        pool_details: Dict[str, List[Dict[str, object]]] = {}
        for method_name in methods:
            method_records = [
                record
                for record in records
                if str(record.get("pool_name")) == pool_name and str(record.get("method")) == method_name
            ]
            if not method_records:
                continue
            method_records.sort(
                key=lambda item: (
                    int(item.get("sample_index", 0)),
                    str(item.get("bundle_key", "")),
                )
            )
            pool_summary[method_name] = build_ranking_summary(method_records)
            pool_details[method_name] = method_records
        if pool_summary:
            summary_by_pool[pool_name] = pool_summary
            detail_by_pool[pool_name] = pool_details
    return summary_by_pool, detail_by_pool

class MemoryAdapter(nn.Module):
    def __init__(self, latent_input_dim: int, latent_slots: int, qformer_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(latent_input_dim)
        self.proj = nn.Linear(latent_input_dim, qformer_dim)
        self.level_embedding = nn.Parameter(torch.zeros(latent_slots, qformer_dim))

    def forward(self, hidden_condition: torch.Tensor) -> torch.Tensor:
        hidden_condition = hidden_condition.to(dtype=self.proj.weight.dtype)
        memory = self.proj(self.norm(hidden_condition))
        return memory + self.level_embedding.unsqueeze(0)

class QFormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q_self = self.norm_self(queries)
        queries = queries + self.self_attn(q_self, q_self, q_self, need_weights=False)[0]

        q_cross = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(memory)
        queries = queries + self.cross_attn(q_cross, kv, kv, need_weights=False)[0]

        queries = queries + self.ffn(self.norm_ff(queries))
        return queries

class LightweightQFormer(nn.Module):
    def __init__(self, num_queries: int, dim: int, num_layers: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(num_queries, dim))
        self.layers = nn.ModuleList(
            [QFormerBlock(dim=dim, num_heads=num_heads, ffn_hidden_dim=ffn_hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory)
        return queries

class IntermediateHeads(nn.Module):
    def __init__(self, input_dim: int, content_dim: int, description_dim: int, cf_dim: int):
        super().__init__()
        self.head_c = nn.Linear(input_dim, content_dim)
        self.head_t = nn.Linear(input_dim, description_dim)
        self.head_cf = nn.Linear(input_dim, cf_dim)

    def forward(self, z_c: torch.Tensor, z_t: torch.Tensor, z_cf: torch.Tensor):
        return (
            self.head_c(z_c).unsqueeze(1),
            self.head_t(z_t).unsqueeze(1),
            self.head_cf(z_cf).unsqueeze(1),
        )

class ContrastiveHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(projection_dim)
        self.shared_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred.reshape(pred.shape[0], -1)
        gt = gt.reshape(gt.shape[0], -1)
        if pred.shape[-1] != self.input_dim or gt.shape[-1] != self.input_dim:
            raise ValueError(
                f"ContrastiveHead expected last dim {self.input_dim}, got pred={tuple(pred.shape)} gt={tuple(gt.shape)}"
            )
        proj_dtype = self.shared_proj.weight.dtype
        pred_proj = self.shared_proj(pred.to(dtype=proj_dtype))
        with torch.no_grad():
            gt_proj = self.shared_proj(gt.to(dtype=proj_dtype))
        return F.normalize(pred_proj.float(), dim=-1), F.normalize(gt_proj.float(), dim=-1)

class TypeSpecificContrastiveHeads(nn.Module):
    def __init__(self, content_dim: int, description_dim: int, cf_dim: int, projection_dim: int):
        super().__init__()
        self.head_c = ContrastiveHead(content_dim, projection_dim)
        self.head_t = ContrastiveHead(description_dim, projection_dim)
        self.head_cf = ContrastiveHead(cf_dim, projection_dim)

    def forward(
        self,
        pred_e_c: torch.Tensor,
        gt_content: torch.Tensor,
        pred_e_t: torch.Tensor,
        gt_description: torch.Tensor,
        pred_e_cf: torch.Tensor,
        gt_cf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        contrastive_pred_c, contrastive_gt_c = self.head_c(pred_e_c, gt_content)
        contrastive_pred_t, contrastive_gt_t = self.head_t(pred_e_t, gt_description)
        contrastive_pred_cf, contrastive_gt_cf = self.head_cf(pred_e_cf, gt_cf)
        return {
            "contrastive_pred_c": contrastive_pred_c,
            "contrastive_gt_c": contrastive_gt_c,
            "contrastive_pred_t": contrastive_pred_t,
            "contrastive_gt_t": contrastive_gt_t,
            "contrastive_pred_cf": contrastive_pred_cf,
            "contrastive_gt_cf": contrastive_gt_cf,
        }

class TimeConditionEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        dtype = self.timestep_embedder.linear_1.weight.dtype
        timestep_proj = self.time_proj(timestep.reshape(-1))
        return self.timestep_embedder(timestep_proj.to(dtype=dtype))

class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.proj = PixArtAlphaTextProjection(
            in_features=self.input_dim,
            hidden_size=output_dim,
            out_features=output_dim,
            act_fn="silu",
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.reshape(feature.shape[0], -1)
        if feature.shape[-1] != self.input_dim:
            raise ValueError(
                f"Unexpected feature shape {tuple(feature.shape)}. Expected last dim {self.input_dim}."
            )
        dtype = self.proj.linear_1.weight.dtype
        return self.proj(self.norm(feature).to(dtype=dtype))

class MaskGen_QFormer(MaskGen_VQ):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        latent_input_dim = int(config.model.latent_input_dim)
        latent_slots = int(config.model.latent_slots)
        qformer_dim = int(config.model.qformer_dim)
        qformer_layers = int(config.model.qformer_layers)
        qformer_heads = int(config.model.qformer_heads)
        qformer_ffn_mult = int(config.model.get("qformer_ffn_mult", 4))
        decoder_embed_dim = int(config.model.maskgen.decoder_embed_dim)
        content_dim = int(config.model.content_dim)
        description_dim = int(config.model.description_dim)
        cf_dim = int(config.model.cf_dim)
        num_queries = int(config.model.num_qformer_queries)
        contrastive_dim = int(config.losses.get("contrastive_dim", qformer_dim))
        self.condition_drop_prob = float(config.model.maskgen.get("text_drop_prob", 0.0))
        if not 0.0 <= self.condition_drop_prob <= 1.0:
            raise ValueError(
                "model.maskgen.text_drop_prob must be in [0, 1], "
                f"got {self.condition_drop_prob}"
            )
        self.temb_condition_components = normalize_temb_condition_components(
            config.model.get("condition", ["content", "description"])
        )

        if num_queries != 6:
            raise ValueError(f"Expected model.num_qformer_queries=6, got {num_queries}")

        self.latent_token_proj = nn.Linear(latent_input_dim, decoder_embed_dim)
        self.memory_adapter = MemoryAdapter(latent_input_dim, latent_slots, qformer_dim)
        self.qformer = LightweightQFormer(
            num_queries=num_queries,
            dim=qformer_dim,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
            ffn_hidden_dim=qformer_dim * qformer_ffn_mult,
        )
        self.intermediate_heads = IntermediateHeads(
            input_dim=qformer_dim,
            content_dim=content_dim,
            description_dim=description_dim,
            cf_dim=cf_dim,
        )
        self.contrastive_heads = TypeSpecificContrastiveHeads(
            content_dim=content_dim,
            description_dim=description_dim,
            cf_dim=cf_dim,
            projection_dim=contrastive_dim,
        )
        self.phi_c = FeatureProjector(content_dim, decoder_embed_dim)
        self.phi_t = FeatureProjector(description_dim, decoder_embed_dim)

        self.latent_token_proj.apply(self._init_weights)
        self.memory_adapter.apply(self._init_weights)
        self.qformer.apply(self._init_weights)
        self.intermediate_heads.apply(self._init_weights)
        self.contrastive_heads.apply(self._init_weights)
        self.phi_c.apply(self._init_weights)
        self.phi_t.apply(self._init_weights)
        nn.init.trunc_normal_(self.memory_adapter.level_embedding, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.qformer.query_tokens, mean=0.0, std=0.02)
        self.use_timestep_condition = bool(config.model.get("use_timestep_condition", False))
        if self.use_timestep_condition:
            # Preserve the original RNG stream and original weights at initialization.
            with torch.random.fork_rng(devices=[]):
                self.time_condition = TimeConditionEmbedding(decoder_embed_dim)
                self.time_condition.apply(self._init_weights)
                nn.init.zeros_(self.time_condition.timestep_embedder.linear_2.weight)
                nn.init.zeros_(self.time_condition.timestep_embedder.linear_2.bias)

    def input_timesteps(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Invert the unchanged arccos mask schedule for the current input state."""
        if self.mask_schedule_strategy != "arccos":
            raise ValueError("stage2_tsp requires the original arccos mask schedule")
        fraction = (input_ids == self.mask_token_id).float().mean(dim=1)
        return torch.cos(fraction * (math.pi / 2)).clamp(0, 1)

    def masking_input_tokens(self, input_tokens, return_timesteps: bool = False, timesteps: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        if timesteps is None:
            timesteps = torch.zeros((batch_size,), device=device, dtype=torch.float32).uniform_(0, 1.0)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.float32).reshape(batch_size)
        mask_ratio = get_masking_ratio(timesteps, self.mask_schedule_strategy)
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.0)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < num_token_masked.unsqueeze(1)
        masked_tokens = torch.where(masks, self.mask_token_id, input_tokens)
        if return_timesteps:
            return masked_tokens, masks, timesteps
        return masked_tokens, masks

    def predict_conditions(self, hidden_condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden_condition = hidden_condition.to(dtype=self.latent_token_proj.weight.dtype)
        latent_tokens = self.latent_token_proj(hidden_condition)
        memory = self.memory_adapter(hidden_condition)
        queries = self.qformer(memory)
        z_c = queries[:, 0:2].mean(dim=1)
        z_t = queries[:, 2:4].mean(dim=1)
        z_cf = queries[:, 4:6].mean(dim=1)
        pred_e_c, pred_e_t, pred_e_cf = self.intermediate_heads(z_c, z_t, z_cf)
        return {
            "latent_tokens": latent_tokens,
            "memory": memory,
            "queries": queries,
            "z_c": z_c,
            "z_t": z_t,
            "z_cf": z_cf,
            "pred_e_c": pred_e_c,
            "pred_e_t": pred_e_t,
            "pred_e_cf": pred_e_cf,
        }

    def build_pooled_condition(
        self,
        e_content: torch.Tensor,
        e_description: torch.Tensor,
    ) -> torch.Tensor:
        pooled_conditions: List[torch.Tensor] = []
        if "content" in self.temb_condition_components:
            pooled_conditions.append(self.phi_c(e_content))
        if "description" in self.temb_condition_components:
            pooled_conditions.append(self.phi_t(e_description))
        if len(pooled_conditions) == 1:
            return pooled_conditions[0]
        return pooled_conditions[0] + pooled_conditions[1]

    def project_contrastive_features(
        self,
        pred_e_c: torch.Tensor,
        gt_content: torch.Tensor,
        pred_e_t: torch.Tensor,
        gt_description: torch.Tensor,
        pred_e_cf: torch.Tensor,
        gt_cf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.contrastive_heads(
            pred_e_c=pred_e_c,
            gt_content=gt_content,
            pred_e_t=pred_e_t,
            gt_description=gt_description,
            pred_e_cf=pred_e_cf,
            gt_cf=gt_cf,
        )

    def _forward_backbone(
        self,
        input_ids: torch.Tensor,
        latent_tokens: torch.Tensor,
        pooled_condition: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        x = embeddings + self.pos_embed[:, : embeddings.shape[1]]
        condition = latent_tokens.to(dtype=embeddings.dtype)
        temb = pooled_condition.to(dtype=embeddings.dtype)
        if self.use_timestep_condition:
            if timesteps is None:
                timesteps = self.input_timesteps(input_ids)
            # Add time after feature dropout: CFG's null feature branch retains time.
            temb = temb + self.time_condition(timesteps).to(dtype=embeddings.dtype)
        for blk in self.blocks:
            condition, x = blk(x, condition, temb)
        x = self.norm(x, temb)
        return self.lm_head(x)

    def apply_condition_dropout(
        self,
        pooled_condition: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop only the pooled CFG branch while preserving latent tokens.

        This mirrors inference-time CFG in ``generate_maskgen_qformer``: the
        conditional and unconditional passes share predicted latent tokens,
        and the unconditional pass receives an all-zero pooled condition.
        """
        batch_size = int(pooled_condition.shape[0])
        if drop_mask is None:
            if self.training and self.condition_drop_prob > 0.0:
                drop_mask = torch.rand(batch_size, device=pooled_condition.device) < self.condition_drop_prob
            else:
                drop_mask = torch.zeros(batch_size, device=pooled_condition.device, dtype=torch.bool)
        else:
            drop_mask = drop_mask.to(device=pooled_condition.device, dtype=torch.bool).reshape(batch_size)

        broadcast_shape = (batch_size,) + (1,) * (pooled_condition.ndim - 1)
        dropped_condition = pooled_condition.masked_fill(drop_mask.reshape(broadcast_shape), 0)
        return dropped_condition, drop_mask

    def forward(
        self,
        input_tokens: Optional[torch.Tensor],
        hidden_condition: torch.Tensor,
        e_content_gt: Optional[torch.Tensor] = None,
        e_description_gt: Optional[torch.Tensor] = None,
        e_cf_gt: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        input_is_masked: bool = False,
        enable_teacher: bool = False,
        predictor_only: bool = False,
        build_contrastive_features: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        preds = self.predict_conditions(hidden_condition)
        if build_contrastive_features and any(value is not None for value in [e_content_gt, e_description_gt, e_cf_gt]):
            if e_content_gt is None or e_description_gt is None or e_cf_gt is None:
                raise ValueError("e_content_gt, e_description_gt, and e_cf_gt must be provided together.")
            preds.update(
                self.project_contrastive_features(
                    pred_e_c=preds["pred_e_c"],
                    gt_content=e_content_gt,
                    pred_e_t=preds["pred_e_t"],
                    gt_description=e_description_gt,
                    pred_e_cf=preds["pred_e_cf"],
                    gt_cf=e_cf_gt,
                )
            )
        if predictor_only:
            preds.update(
                {
                    "student_logits": None,
                    "teacher_logits": None,
                    "masks": None,
                    "timesteps": timesteps,
                    "input_ids": input_tokens,
                }
            )
            return preds

        if input_tokens is None:
            raise ValueError("input_tokens must be provided when predictor_only=False")

        if input_is_masked:
            if timesteps is None:
                raise ValueError("timesteps must be provided when input_is_masked=True")
            input_ids = input_tokens
            masks = None
        elif self.training:
            input_ids, masks, timesteps = self.masking_input_tokens(input_tokens, return_timesteps=True)
        else:
            if timesteps is None:
                raise ValueError("timesteps must be provided during eval/inference")
            input_ids = input_tokens
            masks = None

        student_pooled = self.build_pooled_condition(preds["pred_e_c"], preds["pred_e_t"])
        student_pooled, condition_drop_mask = self.apply_condition_dropout(student_pooled)
        student_logits = self._forward_backbone(input_ids, preds["latent_tokens"], student_pooled, timesteps)

        teacher_logits = None
        if enable_teacher:
            if e_content_gt is None or e_description_gt is None:
                raise ValueError("e_content_gt and e_description_gt are required when enable_teacher=True")
            with torch.no_grad():
                teacher_pooled = self.build_pooled_condition(e_content_gt, e_description_gt)
                teacher_pooled, _ = self.apply_condition_dropout(teacher_pooled, condition_drop_mask)
                teacher_logits = self._forward_backbone(input_ids, preds["latent_tokens"], teacher_pooled, timesteps)

        preds.update(
            {
                "student_logits": student_logits,
                "teacher_logits": teacher_logits,
                "masks": masks,
                "timesteps": timesteps,
                "input_ids": input_ids,
                "condition_drop_mask": condition_drop_mask,
            }
        )
        return preds

    def set_stage_trainable(self, stage: str):
        stage = normalize_stage(stage)
        for param in self.parameters():
            param.requires_grad_(False)

        predictor_modules = [
            self.memory_adapter,
            self.qformer,
            self.intermediate_heads,
            self.contrastive_heads,
        ]
        joint_modules = [
            self.latent_token_proj,
            self.phi_c,
            self.phi_t,
            self.embeddings,
            self.blocks,
            self.norm,
            self.lm_head,
        ]
        if self.use_timestep_condition:
            joint_modules.append(self.time_condition)

        if stage == "stage1":
            for module in predictor_modules:
                module.requires_grad_(True)
        else:
            for module in predictor_modules + joint_modules:
                module.requires_grad_(True)
            self.pos_embed.requires_grad_(True)

def generate_maskgen_qformer(
    model,
    hidden_condition,
    guidance_scale=6.0,
    randomize_temperature=1.5,
    softmax_temperature_annealing=True,
    num_sample_steps=16,
    guidance_decay="cosine",
    guidance_decay_scale_pow=1.0,
    prob_sorting=True,
    use_cfg=False,
    token_allowed_mask: Optional[torch.Tensor] = None,
    logit_top_k: int = 0,
    final_step_no_gumbel: bool = False,
):
    assert guidance_decay in ["linear", "cosine", "none", "flippedcosine"]
    num_samples = hidden_condition.shape[0]
    device = hidden_condition.device

    # For inference-time CFG, keep the predicted latent tokens fixed and only
    # drop the pooled temb branch in the unconditional pass.
    preds = model.predict_conditions(hidden_condition)
    latent_tokens = preds["latent_tokens"]
    pooled_condition = model.build_pooled_condition(preds["pred_e_c"], preds["pred_e_t"])
    empty_pooled_condition = torch.zeros_like(pooled_condition)

    ids = torch.full(
        (num_samples, model.image_seq_len),
        model.mask_token_id,
        device=device,
        dtype=torch.long,
    )
    cfg_scale = guidance_scale if guidance_decay == "none" else 0.0

    def log(t, eps=1e-20):
        return torch.log(t.clamp(min=eps))

    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -log(-log(noise))

    def add_gumbel_noise(t, temperature):
        if temperature <= 0:
            return t
        return t + temperature * gumbel_noise(t)

    def apply_token_filters(logits: torch.Tensor) -> torch.Tensor:
        if token_allowed_mask is not None:
            allowed = token_allowed_mask.to(device=logits.device, dtype=torch.bool)
            if allowed.dim() == 1:
                allowed = allowed.view(1, 1, -1)
            elif allowed.dim() == 2:
                allowed = allowed.unsqueeze(0)
            else:
                raise ValueError(f"token_allowed_mask must have shape [V] or [L,V], got {tuple(allowed.shape)}")
            logits = logits.masked_fill(~allowed, -float("inf"))

        top_k = int(logit_top_k or 0)
        if top_k > 0 and top_k < logits.shape[-1]:
            top_values = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < top_values, -float("inf"))
        return logits

    for step in range(num_sample_steps):
        ratio = float(step + 1) / float(num_sample_steps)
        annealed_temp = randomize_temperature * (1.0 - ratio)
        is_mask = ids == model.mask_token_id

        if guidance_decay == "cosine":
            scale_pow = torch.ones((1,), device=device) * guidance_decay_scale_pow
            scale_step = (1 - torch.cos((ratio**scale_pow) * torch.pi)) * 0.5
            cfg_scale = (guidance_scale - 1) * scale_step + 1
        elif guidance_decay == "flippedcosine":
            scale_pow = torch.ones((1,), device=device) * guidance_decay_scale_pow
            scale_step = torch.cos((ratio**scale_pow) * torch.pi) * 0.5
            cfg_scale = (guidance_scale - 1) * scale_step + 1
        elif guidance_decay == "linear":
            cfg_scale = ratio * (guidance_scale - 1) + 1

        if use_cfg and cfg_scale != 0:
            logits = model._forward_backbone(
                input_ids=torch.cat([ids, ids], dim=0),
                latent_tokens=torch.cat([latent_tokens, latent_tokens], dim=0),
                pooled_condition=torch.cat([pooled_condition, empty_pooled_condition], dim=0),
            )
            cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
            logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = model._forward_backbone(
                input_ids=ids,
                latent_tokens=latent_tokens,
                pooled_condition=pooled_condition,
            )

        if softmax_temperature_annealing:
            softmax_temperature = 0.5 + 0.8 * (1 - ratio)
        else:
            softmax_temperature = annealed_temp
        logits = logits / softmax_temperature
        logits = apply_token_filters(logits)

        prob_ids = logits
        sample_temperature = 0.0 if final_step_no_gumbel and step == num_sample_steps - 1 else annealed_temp
        sampled_ids = add_gumbel_noise(prob_ids, sample_temperature).argmax(dim=-1)
        sampled_logits = torch.squeeze(
            torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)),
            -1,
        )
        sampled_ids = torch.where(is_mask, sampled_ids, ids)
        sampled_logits = torch.where(
            is_mask,
            sampled_logits,
            torch.full_like(sampled_logits, float("inf")),
        ).float()

        mask_ratio = get_masking_ratio(ratio, model.mask_schedule_strategy)
        mask_len = torch.floor(model.image_seq_len * mask_ratio).to(device)
        mask_len = torch.maximum(
            torch.tensor([1], device=device),
            torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1, mask_len),
        )[0].squeeze()

        confidence = add_gumbel_noise(sampled_logits, annealed_temp) if prob_sorting else sampled_logits
        sorted_confidence, _ = torch.sort(confidence, axis=-1)
        cut_off = sorted_confidence[:, mask_len.long() - 1 : mask_len.long()]
        masking = confidence <= cut_off
        if step == num_sample_steps - 1:
            ids = sampled_ids
        else:
            ids = torch.where(masking, model.mask_token_id, sampled_ids)

    return ids
