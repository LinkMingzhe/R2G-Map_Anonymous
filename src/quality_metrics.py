from __future__ import annotations

from pathlib import Path

from typing import Optional

from PIL import Image

import numpy as np

def require_numpy() -> None:
    if np is None:
        raise RuntimeError("This evaluator requires numpy. Install it with: pip install numpy")

def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")

def reference_inception_score(probs: np.ndarray, splits: int) -> dict[str, float]:
    """Reference/PFITB IS: exp(E[KL(p(y|x) || uniform_y)]) on the 50-way classifier."""
    if probs.shape[0] == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "splits": 0.0,
            "entropy_mean": float("nan"),
            "entropy_std": float("nan"),
        }
    splits = max(1, min(int(splits), probs.shape[0]))
    scores = []
    entropies = []
    eps = 1e-16
    uniform = np.full((1, probs.shape[1]), 1.0 / probs.shape[1], dtype=np.float64)
    for split_probs in np.array_split(probs, splits):
        if split_probs.size == 0:
            continue
        entropy = -split_probs * np.log(split_probs + eps)
        entropies.append(float(np.mean(np.sum(entropy, axis=1))))
        kl = split_probs * (np.log(split_probs + eps) - np.log(uniform))
        scores.append(float(np.exp(np.mean(np.sum(kl, axis=1)))))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "splits": float(len(scores)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
    }

def torch_load_compat(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_category_map(path: Optional[Path]) -> dict[int, str]:
    if path is None:
        return {}
    mapping: dict[int, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"category map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        header: Optional[list[str]] = None
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = parts
                continue
            row = {header[i]: parts[i] for i in range(min(len(header), len(parts)))}
            item_value = row.get("item_index") or row.get("item_id") or row.get("target_item_id")
            category_value = row.get("semantic_category") or row.get("category")
            if item_value is None or category_value is None:
                continue
            mapping[int(item_value)] = category_value
    return mapping

def load_category_labels(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                labels.append(value)
    return labels

def reorder_category_labels_for_classifier(
    labels: list[str],
    index_map_path: Optional[Path],
) -> list[str]:
    """Put semantic labels in classifier-output order.

    DiFashion stores ``cid_to_label.npy`` as a mapping from the semantic
    category id to the corresponding fine-tuned Inception output index.  The
    reference evaluator applies that mapping to every ground-truth category
    before comparing it with the model top-k indices.  Reordering the display
    labels here is exactly equivalent and lets the rest of this evaluator keep
    comparing semantic category strings.
    """
    if index_map_path is None:
        return labels
    if not index_map_path.exists():
        raise FileNotFoundError(f"category index map not found: {index_map_path}")
    require_numpy()
    raw_mapping = np.load(index_map_path, allow_pickle=True).item()
    if not isinstance(raw_mapping, dict):
        raise TypeError(f"category index map must contain a dict: {index_map_path}")
    if len(raw_mapping) != len(labels):
        raise ValueError(
            f"category index map has {len(raw_mapping)} entries but {len(labels)} labels were provided"
        )
    output_labels: list[Optional[str]] = [None] * len(labels)
    for category_id, output_index in raw_mapping.items():
        category_id = int(category_id)
        output_index = int(output_index)
        if not 0 <= category_id < len(labels):
            raise ValueError(f"category id {category_id} is outside [0, {len(labels)})")
        if not 0 <= output_index < len(labels):
            raise ValueError(f"classifier output index {output_index} is outside [0, {len(labels)})")
        if output_labels[output_index] is not None:
            raise ValueError(f"duplicate classifier output index {output_index} in {index_map_path}")
        output_labels[output_index] = labels[category_id]
    if any(label is None for label in output_labels):
        raise ValueError(f"category index map does not cover every classifier output: {index_map_path}")
    return [str(label) for label in output_labels]

def load_category_inception_model(checkpoint_path: Path, labels: list[str], device):
    import torch
    from torch import nn
    from torchvision import models

    model = models.inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=False)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    if getattr(model, "AuxLogits", None) is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, len(labels))
    state_dict = torch_load_compat(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model

def collect_reference_inception_probs(
    image_paths: list[Path],
    model,
    device,
    batch_size: int,
    image_size: int,
    desc: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm.auto import tqdm

    preprocess = transforms.ToTensor()
    probs: list[np.ndarray] = []
    use_cuda = device.type == "cuda"
    for start in tqdm(range(0, len(image_paths), batch_size), desc=desc, unit="batch"):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            image = load_image_rgb(path)
            tensors.append(preprocess(image))
            image.close()
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=use_cuda)
        batch = F.interpolate(batch, size=(image_size, image_size), mode="bilinear", align_corners=False)
        batch = 2 * batch - 1
        with torch.no_grad():
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            batch_probs = torch.softmax(logits, dim=1)
        probs.append(batch_probs.detach().cpu().numpy().astype(np.float64))
    if not probs:
        return np.zeros((0, len(getattr(model, "fc").bias)), dtype=np.float64)
    return np.concatenate(probs, axis=0)
