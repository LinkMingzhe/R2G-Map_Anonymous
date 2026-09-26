from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from tqdm.auto import tqdm


_INFERENCE_REPEAT_SUFFIX = re.compile(r"__repeat\d+$")


def _strip_inference_repeat_suffix(sample_id: str) -> str:
    return _INFERENCE_REPEAT_SUFFIX.sub("", sample_id)


def _list_mode_image_files(
    image_output_dir: Path,
    mode: str,
    image_suffix_template: str,
) -> dict[str, list[Path]]:
    suffix = image_suffix_template.format(mode=mode)
    matched: dict[str, list[Path]] = {}
    for image_path in sorted(image_output_dir.glob(f"*{suffix}")):
        sample_id = _strip_inference_repeat_suffix(image_path.name[: -len(suffix)])
        matched.setdefault(sample_id, []).append(image_path)
    return matched


def _group_saved_image_records(
    image_records: Sequence[tuple[str, Path]],
) -> dict[str, list[Path]]:
    matched: dict[str, list[Path]] = {}
    for sample_id, image_path in image_records:
        normalized_sample_id = _strip_inference_repeat_suffix(str(sample_id))
        matched.setdefault(normalized_sample_id, []).append(Path(image_path))
    return matched


def _compute_feature_stats(
    image_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    desc: str,
):
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    total = None
    sigma = None
    count = 0
    use_cuda = device.type == "cuda"

    for i in tqdm(
        range(0, len(image_paths), batch_size),
        desc=desc,
        total=(len(image_paths) + batch_size - 1) // batch_size,
        unit="batch",
    ):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for image_path in batch_paths:
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
                batch_tensors.append(preprocess(img))

        batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=use_cuda)
        with torch.no_grad():
            feats = model(batch)

        feats = feats.detach().cpu().numpy().astype(np.float64)
        if total is None:
            dim = feats.shape[1]
            total = np.zeros(dim, dtype=np.float64)
            sigma = np.zeros((dim, dim), dtype=np.float64)

        total += feats.sum(axis=0)
        sigma += feats.T @ feats
        count += feats.shape[0]

    if count < 2:
        raise RuntimeError("Need at least 2 images to compute covariance for FID.")

    mu = total / count
    cov = (sigma - np.outer(total, total) / count) / (count - 1)
    return mu, cov


def _compute_fid_score(
    real_paths: Sequence[Path],
    fake_paths: Sequence[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    desc_prefix: str,
):
    mu_real, sigma_real = _compute_feature_stats(
        real_paths,
        model,
        device,
        image_size,
        batch_size,
        desc=f"{desc_prefix} real feats",
    )
    mu_fake, sigma_fake = _compute_feature_stats(
        fake_paths,
        model,
        device,
        image_size,
        batch_size,
        desc=f"{desc_prefix} fake feats",
    )

    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        np.sum((mu_real - mu_fake) ** 2)
        + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    )


def save_mode_fid_txt(
    fid_path: Path,
    mode: str,
    image_output_dir: Path,
    gt_count: int,
    mode_count: int,
    matched_count: int,
    matched_bundle_count: Optional[int] = None,
    gt_used_count: Optional[int] = None,
    mode_used_count: Optional[int] = None,
    fid_score: Optional[float] = None,
    error: Optional[str] = None,
):
    fid_path.parent.mkdir(parents=True, exist_ok=True)
    with fid_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"image_output_dir\t{image_output_dir}\n")
        f.write("reference_mode\tgt\n")
        f.write(f"target_mode\t{mode}\n")
        f.write(f"gt_count\t{gt_count}\n")
        f.write(f"mode_count\t{mode_count}\n")
        if matched_bundle_count is not None:
            f.write(f"matched_bundle_count\t{matched_bundle_count}\n")
        if gt_used_count is not None:
            f.write(f"gt_used_for_fid\t{gt_used_count}\n")
        if mode_used_count is not None:
            f.write(f"mode_used_for_fid\t{mode_used_count}\n")
        f.write(f"used_for_fid\t{matched_count}\n")
        if error is None:
            f.write("status\tok\n")
            f.write(f"fid\t{fid_score:.6f}\n")
        else:
            f.write("status\terror\n")
            f.write(f"error\t{error}\n")


def save_fid_summary_txt(summary_path: Path, summary_rows: Sequence[dict[str, str]]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("mode\tstatus\tgt_count\tmode_count\tused_for_fid\tfid_or_error\n")
        for row in summary_rows:
            f.write(
                f"{row['mode']}\t{row['status']}\t{row['gt_count']}\t{row['mode_count']}\t"
                f"{row['used_for_fid']}\t{row['fid_or_error']}\n"
            )


def load_fid_summary_txt(summary_path: Path) -> tuple[str, list[dict[str, str]]]:
    expected_header = ["mode", "status", "gt_count", "mode_count", "used_for_fid", "fid_or_error"]
    with summary_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Invalid fid summary file (too few lines): {summary_path}")

    key, sep, timestamp = lines[0].partition("\t")
    if key != "timestamp" or not sep or not timestamp:
        raise ValueError(f"Invalid fid summary timestamp line in {summary_path}: {lines[0]!r}")

    header = lines[1].split("\t")
    if header != expected_header:
        raise ValueError(
            f"Invalid fid summary header in {summary_path}: expected {expected_header}, got {header}"
        )

    rows = []
    for line_no, line in enumerate(lines[2:], start=3):
        parts = line.split("\t", len(expected_header) - 1)
        if len(parts) != len(expected_header):
            raise ValueError(f"Invalid fid summary row at {summary_path}:{line_no}: {line!r}")
        rows.append(dict(zip(expected_header, parts)))
    return timestamp, rows


def save_aggregated_fid_summary_txt(
    output_path: Path,
    source_dir: Path,
    mode_order: Sequence[str],
    checkpoint_rows: Sequence[dict[str, str]],
    best_rows: Sequence[dict[str, str]],
    raw_rows: Sequence[dict[str, str]],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"source_dir\t{source_dir}\n")
        f.write(f"num_checkpoints\t{len(checkpoint_rows)}\n")
        f.write("summary_table\n")
        f.write("\t".join(["checkpoint", "summary_timestamp", *mode_order]) + "\n")
        for row in checkpoint_rows:
            values = [row["checkpoint"], row["summary_timestamp"]] + [row.get(mode, "") for mode in mode_order]
            f.write("\t".join(values) + "\n")

        f.write("\n")
        f.write("best_by_mode\n")
        f.write("mode\tcheckpoint\tsummary_timestamp\tfid\n")
        for row in best_rows:
            f.write(f"{row['mode']}\t{row['checkpoint']}\t{row['summary_timestamp']}\t{row['fid']}\n")

        f.write("\n")
        f.write("raw_rows\n")
        f.write("checkpoint\tsummary_timestamp\tmode\tstatus\tgt_count\tmode_count\tused_for_fid\tfid_or_error\n")
        for row in raw_rows:
            f.write(
                f"{row['checkpoint']}\t{row['summary_timestamp']}\t{row['mode']}\t{row['status']}\t"
                f"{row['gt_count']}\t{row['mode_count']}\t{row['used_for_fid']}\t{row['fid_or_error']}\n"
            )


def aggregate_fid_summaries_across_checkpoints(
    root_dir: Path,
    logger,
    output_filename: str,
    checkpoint_step_fn: Callable[[Path], int],
) -> Optional[Path]:
    summary_paths = sorted(
        root_dir.glob("checkpoint-*/fid_summary.txt"),
        key=lambda path: checkpoint_step_fn(path.parent),
    )
    if not summary_paths:
        logger.info(
            "Skipping aggregated FID summary because no checkpoint-*/fid_summary.txt exists under %s",
            root_dir,
        )
        return None

    mode_order: list[str] = []
    checkpoint_rows: list[dict[str, str]] = []
    best_by_mode: dict[str, dict[str, object]] = {}
    raw_rows: list[dict[str, str]] = []

    for summary_path in summary_paths:
        summary_timestamp, rows = load_fid_summary_txt(summary_path)
        checkpoint_name = summary_path.parent.name
        checkpoint_row = {
            "checkpoint": checkpoint_name,
            "summary_timestamp": summary_timestamp,
        }
        for row in rows:
            mode = row["mode"]
            if mode not in mode_order:
                mode_order.append(mode)
            checkpoint_row[mode] = row["fid_or_error"]
            raw_rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "summary_timestamp": summary_timestamp,
                    **row,
                }
            )
            if row["status"] != "ok":
                continue
            try:
                fid_value = float(row["fid_or_error"])
            except ValueError:
                continue

            best_row = best_by_mode.get(mode)
            if best_row is None or fid_value < float(best_row["fid_value"]):
                best_by_mode[mode] = {
                    "checkpoint": checkpoint_name,
                    "summary_timestamp": summary_timestamp,
                    "fid": f"{fid_value:.6f}",
                    "fid_value": fid_value,
                }
        checkpoint_rows.append(checkpoint_row)

    best_rows = []
    for mode in mode_order:
        best_row = best_by_mode.get(mode)
        if best_row is None:
            best_rows.append(
                {
                    "mode": mode,
                    "checkpoint": "",
                    "summary_timestamp": "",
                    "fid": "N/A",
                }
            )
            continue
        best_rows.append(
            {
                "mode": mode,
                "checkpoint": str(best_row["checkpoint"]),
                "summary_timestamp": str(best_row["summary_timestamp"]),
                "fid": str(best_row["fid"]),
            }
        )

    output_path = root_dir / output_filename
    save_aggregated_fid_summary_txt(
        output_path=output_path,
        source_dir=root_dir,
        mode_order=mode_order,
        checkpoint_rows=checkpoint_rows,
        best_rows=best_rows,
        raw_rows=raw_rows,
    )
    logger.info(
        "Saved aggregated FID summary across %d checkpoint(s) to %s",
        len(checkpoint_rows),
        output_path,
    )
    return output_path


def compute_and_save_fid_scores(
    image_output_dir: Path,
    decode_modes: Sequence[str],
    device: torch.device,
    logger,
    image_suffix_template: str = "_{mode}.png",
    compute_fid: bool = True,
    fid_image_size: int = 256,
    fid_batch_size: int = 32,
    saved_image_records_by_mode: Optional[Mapping[str, Sequence[tuple[str, Path]]]] = None,
):
    if not compute_fid:
        logger.info("Skipping FID because compute_fid is disabled.")
        return

    if "gt" not in decode_modes:
        logger.warning("Skipping FID because decode_modes does not include 'gt'.")
        return

    if saved_image_records_by_mode is not None and "gt" in saved_image_records_by_mode:
        gt_group_map = _group_saved_image_records(saved_image_records_by_mode["gt"])
    else:
        gt_group_map = _list_mode_image_files(image_output_dir, "gt", image_suffix_template=image_suffix_template)

    gt_duplicates = [sample_id for sample_id, paths in gt_group_map.items() if len(paths) > 1]
    if gt_duplicates:
        logger.warning(
            "Found duplicated gt images for %d sample(s); only the first file per sample will be used for FID.",
            len(gt_duplicates),
        )
    gt_map = {sample_id: paths[0] for sample_id, paths in gt_group_map.items() if paths}
    if len(gt_map) < 2:
        logger.warning("Skipping FID because fewer than 2 gt images were saved in %s", image_output_dir)
        return

    try:
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'torchvision'. Please install it to enable FID computation."
        ) from exc

    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    summary_rows = []
    for mode in decode_modes:
        if mode == "gt":
            continue

        if saved_image_records_by_mode is not None and mode in saved_image_records_by_mode:
            mode_map = _group_saved_image_records(saved_image_records_by_mode[mode])
        else:
            mode_map = _list_mode_image_files(image_output_dir, mode, image_suffix_template=image_suffix_template)
        matched_ids = sorted(set(gt_map.keys()) & set(mode_map.keys()))
        mode_count = sum(len(paths) for paths in mode_map.values())
        gt_paths = [gt_map[sample_id] for sample_id in matched_ids]
        mode_paths = [path for sample_id in matched_ids for path in mode_map[sample_id]]
        fid_txt_path = image_output_dir / f"fid_{mode}.txt"

        if len(gt_paths) < 2 or len(mode_paths) < 2:
            error = (
                "Need at least 2 images on both sides to compute FID. "
                f"gt_count={len(gt_map)}, mode_count={mode_count}, matched_bundles={len(matched_ids)}, "
                f"gt_used={len(gt_paths)}, mode_used={len(mode_paths)}"
            )
            save_mode_fid_txt(
                fid_path=fid_txt_path,
                mode=mode,
                image_output_dir=image_output_dir,
                gt_count=len(gt_map),
                mode_count=mode_count,
                matched_count=len(mode_paths),
                matched_bundle_count=len(matched_ids),
                gt_used_count=len(gt_paths),
                mode_used_count=len(mode_paths),
                error=error,
            )
            summary_rows.append(
                {
                    "mode": mode,
                    "status": "error",
                    "gt_count": str(len(gt_map)),
                    "mode_count": str(mode_count),
                    "used_for_fid": str(len(mode_paths)),
                    "fid_or_error": error,
                }
            )
            logger.warning("Skipping FID for mode=%s: %s", mode, error)
            continue

        fid_score = _compute_fid_score(
            real_paths=gt_paths,
            fake_paths=mode_paths,
            model=inception,
            device=device,
            image_size=fid_image_size,
            batch_size=fid_batch_size,
            desc_prefix=f"FID[{mode}]",
        )
        save_mode_fid_txt(
            fid_path=fid_txt_path,
            mode=mode,
            image_output_dir=image_output_dir,
            gt_count=len(gt_map),
            mode_count=mode_count,
            matched_count=len(mode_paths),
            matched_bundle_count=len(matched_ids),
            gt_used_count=len(gt_paths),
            mode_used_count=len(mode_paths),
            fid_score=fid_score,
        )
        summary_rows.append(
            {
                "mode": mode,
                "status": "ok",
                "gt_count": str(len(gt_map)),
                "mode_count": str(mode_count),
                "used_for_fid": str(len(mode_paths)),
                "fid_or_error": f"{fid_score:.6f}",
            }
        )
        logger.info(
            "Saved FID for mode=%s to %s | fid=%.6f | gt=%d | mode=%d | matched_bundles=%d | "
            "gt_used=%d | mode_used=%d",
            mode,
            fid_txt_path,
            fid_score,
            len(gt_map),
            mode_count,
            len(matched_ids),
            len(gt_paths),
            len(mode_paths),
        )

    save_fid_summary_txt(image_output_dir / "fid_summary.txt", summary_rows)
