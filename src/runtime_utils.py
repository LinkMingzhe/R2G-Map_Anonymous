import re

import json

from typing import Dict, List, Tuple

import numpy as np

import torch

from omegaconf import OmegaConf

from torch.utils.data import DataLoader, Subset

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)

def _dup_rank(bundle_key: str) -> int:
    match = re.search(r"__dup(\d+)$", bundle_key)
    if match is None:
        return 0
    return int(match.group(1))

def _normalize_step_key(step_value) -> str:
    if isinstance(step_value, str):
        if step_value.startswith("step_"):
            return step_value
        if step_value.isdigit():
            return f"step_{int(step_value):04d}"
    return f"step_{int(step_value):04d}"

def _safe_file_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))

def _parse_bundle_line(line: str) -> Tuple[str, List[int]]:
    parts = [part.strip() for part in line.strip().split(",") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid bundle line: {line!r}")
    bundle_id = parts[0]
    item_ids = [int(part) for part in parts[1:]]
    return bundle_id, item_ids

def _read_input_gt_pairs(input_txt: str, gt_txt: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(input_txt, "r", encoding="utf-8") as fin, open(gt_txt, "r", encoding="utf-8") as fgt:
        for line_idx, (input_line, gt_line) in enumerate(zip(fin, fgt), start=1):
            input_id, input_items = _parse_bundle_line(input_line)
            gt_id, gt_items = _parse_bundle_line(gt_line)
            if input_id != gt_id:
                raise ValueError(
                    f"Bundle id mismatch at line {line_idx}: input_id={input_id}, gt_id={gt_id}"
                )
            if not gt_items:
                raise ValueError(f"Expected at least one gt item at line {line_idx}, got {gt_items}")
            for target_position, target_item_id in enumerate(gt_items, start=1):
                rows.append(
                    {
                        "bundle_id": input_id,
                        "input_item_ids": input_items,
                        "target_item_id": target_item_id,
                        "target_position": target_position,
                        "target_item_ids": list(gt_items),
                    }
                )

        extra_input = fin.readline()
        extra_gt = fgt.readline()
        if extra_input or extra_gt:
            raise ValueError(f"File length mismatch between {input_txt} and {gt_txt}")

    return rows

def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def json_load(fp):
    return json.load(fp)

def load_open_clip_text_encoder(model_name: str, pretrained: str, device: torch.device):
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'open_clip'. Please install 'open_clip_torch' to enable CLIP-guided validation decode."
        ) from exc

    clip_encoder, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer(model_name)
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval()
    clip_encoder.requires_grad_(False)
    clip_encoder = clip_encoder.to(device)
    return clip_encoder, clip_tokenizer

def _tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_uint8 = (image_tensor * 255.0).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
    return image_uint8

def create_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

def maybe_limit_dataset(dataset, max_bundles: int):
    if max_bundles is None or int(max_bundles) <= 0:
        return dataset
    limit = min(len(dataset), int(max_bundles))
    return Subset(dataset, range(limit))
