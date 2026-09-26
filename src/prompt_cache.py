import json

from pathlib import Path

from typing import List, Sequence, Tuple

import torch

from tqdm.auto import tqdm

from runtime_utils import _torch_load_compat, load_open_clip_text_encoder

def build_prompt_mappings(
    item_info_path: str,
    cate_id_to_text_path: str,
    include_category_in_prompt: bool,
    prompt_template: str,
    generic_prompt: str,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    with open(item_info_path, "r", encoding="utf-8") as f:
        item_info = json.load(f)
    with open(cate_id_to_text_path, "r", encoding="utf-8") as f:
        cate_id_to_text = json.load(f)

    max_item_id = max(int(item_id) for item_id in item_info.keys())
    item_to_prompt_index = torch.full((max_item_id + 1,), -1, dtype=torch.long)

    if include_category_in_prompt:
        prompt_keys = sorted({str(info.get("cate_id", "") or "") for info in item_info.values()})
        if any(not key for key in prompt_keys):
            raise ValueError("Found empty cate_id while building prompt mappings")
        missing = [key for key in prompt_keys if key not in cate_id_to_text]
        if missing:
            raise KeyError(f"Missing cate_id_to_text entries for cate_ids: {missing[:10]}")
        prompt_texts = [str(prompt_template).format(category=str(cate_id_to_text[key])) for key in prompt_keys]
        key_to_index = {key: idx for idx, key in enumerate(prompt_keys)}
        for raw_item_id, info in item_info.items():
            item_id = int(raw_item_id)
            cate_id = str(info.get("cate_id", "") or "")
            item_to_prompt_index[item_id] = key_to_index[cate_id]
    else:
        prompt_keys = ["__generic__"]
        prompt_texts = [str(generic_prompt)]
        for raw_item_id in item_info.keys():
            item_to_prompt_index[int(raw_item_id)] = 0

    return item_to_prompt_index, prompt_keys, prompt_texts

def encode_open_clip_prompts(
    prompts: Sequence[str],
    clip_tokenizer,
    clip_encoder,
    device: torch.device,
) -> torch.Tensor:
    text_tokens = clip_tokenizer(list(prompts)).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    prompt_embeds = clip_encoder.token_embedding(text_tokens).to(cast_dtype)
    prompt_embeds = prompt_embeds + clip_encoder.positional_embedding.to(cast_dtype)
    prompt_embeds = prompt_embeds.permute(1, 0, 2)
    prompt_embeds = clip_encoder.transformer(prompt_embeds, attn_mask=clip_encoder.attn_mask)
    prompt_embeds = prompt_embeds.permute(1, 0, 2)
    prompt_embeds = clip_encoder.ln_final(prompt_embeds)
    return prompt_embeds

def maybe_prepare_prompt_cache(model_cfg, device: torch.device, logger):
    cache_path = Path(str(model_cfg.prompt_cache_pt))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    item_to_prompt_index, prompt_keys, prompt_texts = build_prompt_mappings(
        item_info_path=str(model_cfg.item_info_path),
        cate_id_to_text_path=str(model_cfg.cate_id_to_text_path),
        include_category_in_prompt=bool(model_cfg.get("include_category_in_prompt", True)),
        prompt_template=str(model_cfg.prompt_template),
        generic_prompt=str(model_cfg.get("generic_prompt", "A fashion item image, on white background.")),
    )

    expected_metadata = {
        "cache_format_version": 1,
        "item_info_path": str(model_cfg.item_info_path),
        "cate_id_to_text_path": str(model_cfg.cate_id_to_text_path),
        "include_category_in_prompt": bool(model_cfg.get("include_category_in_prompt", True)),
        "prompt_template": str(model_cfg.prompt_template),
        "generic_prompt": str(model_cfg.get("generic_prompt", "A fashion item image, on white background.")),
        "clip_model": str(model_cfg.clip_model),
        "clip_pretrained": str(model_cfg.clip_pretrained),
        "prompt_keys": list(prompt_keys),
        "prompt_texts": list(prompt_texts),
        "num_items": int(item_to_prompt_index.shape[0]),
    }

    if cache_path.exists():
        payload = _torch_load_compat(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        cached_mapping = payload.get("item_to_prompt_index")
        if (
            all(metadata.get(key) == value for key, value in expected_metadata.items())
            and torch.is_tensor(cached_mapping)
            and torch.equal(cached_mapping.cpu().to(dtype=torch.long), item_to_prompt_index)
            and "empty_prompt_embeds" in payload
        ):
            logger.info("Using prompt cache %s", cache_path)
            return
        logger.info("Prompt cache mismatch, rebuilding %s", cache_path)

    clip_encoder, clip_tokenizer = load_open_clip_text_encoder(
        model_name=str(model_cfg.clip_model),
        pretrained=str(model_cfg.clip_pretrained),
        device=device,
    )

    prompt_embeds_chunks: List[torch.Tensor] = []
    batch_size = int(model_cfg.get("prompt_batch_size", 32))
    logger.info(
        "Building OpenCLIP prompt cache %s | num_prompts=%d | include_category_in_prompt=%s | clip_model=%s",
        cache_path,
        len(prompt_texts),
        bool(model_cfg.get("include_category_in_prompt", True)),
        model_cfg.clip_model,
    )

    with torch.no_grad():
        for start in tqdm(range(0, len(prompt_texts), batch_size), desc=f"build-prompt-cache:{cache_path.name}"):
            batch_prompts = prompt_texts[start : start + batch_size]
            batch_prompt_embeds = encode_open_clip_prompts(
                prompts=batch_prompts,
                clip_tokenizer=clip_tokenizer,
                clip_encoder=clip_encoder,
                device=device,
            )
            prompt_embeds_chunks.append(batch_prompt_embeds.detach().cpu().contiguous())

        empty_prompt_embeds = encode_open_clip_prompts(
            prompts=[""],
            clip_tokenizer=clip_tokenizer,
            clip_encoder=clip_encoder,
            device=device,
        )

    payload = {
        "metadata": expected_metadata,
        "prompt_keys": list(prompt_keys),
        "prompt_texts": list(prompt_texts),
        "item_to_prompt_index": item_to_prompt_index.cpu(),
        "prompt_embeds": torch.cat(prompt_embeds_chunks, dim=0).contiguous(),
        "empty_prompt_embeds": empty_prompt_embeds.detach().cpu().contiguous(),
    }
    torch.save(payload, cache_path)
    logger.info("Saved prompt cache to %s", cache_path)

    del clip_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

class PromptEmbeddingCache:
    def __init__(self, cache_path: str):
        payload = _torch_load_compat(cache_path, map_location="cpu")
        self.metadata = dict(payload["metadata"])
        self.prompt_keys = [str(key) for key in payload["prompt_keys"]]
        self.prompt_texts = [str(text) for text in payload["prompt_texts"]]
        self.item_to_prompt_index = payload["item_to_prompt_index"].to(dtype=torch.long, device="cpu").contiguous()
        self.prompt_embeds = payload["prompt_embeds"].to(device="cpu").contiguous()
        self.empty_prompt_embeds = payload["empty_prompt_embeds"].to(device="cpu").contiguous()
        self.prompt_seq_len = int(self.prompt_embeds.shape[1])
        self.prompt_embed_dim = int(self.prompt_embeds.shape[2])

    def _resolve_prompt_indices(self, item_ids) -> torch.Tensor:
        item_ids_tensor = torch.as_tensor(item_ids, dtype=torch.long, device="cpu").reshape(-1)
        if torch.any(item_ids_tensor < 0) or torch.any(item_ids_tensor >= self.item_to_prompt_index.shape[0]):
            raise IndexError("target item_id is out of range for prompt cache")
        prompt_indices = self.item_to_prompt_index.index_select(0, item_ids_tensor)
        if torch.any(prompt_indices < 0):
            missing_item_ids = item_ids_tensor[prompt_indices < 0][:10].tolist()
            raise KeyError(f"missing prompt mapping for item_ids={missing_item_ids}")
        return prompt_indices

    def get_batch(self, item_ids, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prompt_indices = self._resolve_prompt_indices(item_ids)
        prompt_embeds = self.prompt_embeds.index_select(0, prompt_indices)
        return prompt_embeds.to(device=device, dtype=dtype, non_blocking=True)

    def get_empty_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        empty_prompt_embeds = self.empty_prompt_embeds.expand(batch_size, -1, -1)
        return empty_prompt_embeds.to(device=device, dtype=dtype, non_blocking=True)
