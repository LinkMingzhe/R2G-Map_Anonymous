# RG-LSM — anonymous inference release

This package provides **test-only** RG-LSM inference and local evaluation for bundle recommendation and **sequential recommendation**. It consumes supplied DDBC/TIGER hidden latents. It contains no training entry point, train/validation split, train/validation latent cache, Stage1 initialization, or upstream backbone checkpoint/exporter.

The four inference configurations are `ddbc_bundle`, `tiger_bundle`, `ddbc_sequential`, and `tiger_sequential`. The full release trains each task for Stage1 10,000 steps and TSP Stage2 **18,000 steps** by default. This anonymous package remains inference-only. Its packaged sequential checkpoints come from the completed 9,000-step run; updating the training defaults does not change those weights.

## Layout and installation

Use the dedicated anonymous asset archives:

```text
release/
  RG-LSM_Anonymous/
  dataset_anonymous/
  ckpt_anonymous/
```

The defaults resolve to `../dataset_anonymous` and `../ckpt_anonymous`. The full-release data/weight archives have a broader scope and are not the anonymous assets.

Use Python 3.9 and the pinned requirements. Install a PyTorch build suitable for your CUDA installation; the following is the release environment:

```bash
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU execution use the PyTorch `https://download.pytorch.org/whl/cpu` wheel index. Main-model inference does not require FlashAttention. Model loading defaults to offline mode; retain the supplied prompt caches and evaluation weights.

## Packaged checkpoints

| Task | Main-model inference weights, relative to the checkpoint root |
|---|---|
| `ddbc_bundle` | `rg_lsm/ddbc_bundle/stage2_tsp/checkpoint-9000/ema_model/` |
| `tiger_bundle` | `rg_lsm/tiger_bundle/stage2_tsp/checkpoint-6000/ema_model/` |
| `ddbc_sequential` | `rg_lsm/ddbc_sequential/stage2_full/checkpoint-9000/ema_model/` |
| `tiger_sequential` | `rg_lsm/tiger_sequential/stage2_full/checkpoint-9000/ema_model/` |

The checkpoint root is `../ckpt_anonymous`. It contains only main Stage2 EMA inference weights, the TA-TiTok decoder, and local quality-evaluation weights. Sequential models are the fixed final step 9,000 after training on the current augmented dataset (50,464 contexts / 75,384 supervised targets). This states model provenance; training records are not included in the anonymous package.

## Main-model inference

Run inside this code directory. All four task names run RG-LSM; the prefix selects the DDBC or TIGER conditioning latent.

```bash
python inference.py --task ddbc_sequential --config-only
python inference.py --task ddbc_sequential --output runs/ddbc_sequential
python inference.py --task tiger_sequential --output runs/tiger_sequential
python inference.py --task ddbc_bundle --output runs/ddbc_bundle
python inference.py --task tiger_bundle --output runs/tiger_bundle
```

Defaults are the packaged **Stage2 EMA**, **CFG scale 6**, 16 sampling steps, cosine CFG schedule, seed 42, inference batch size 16 for sequential / 32 for bundle, and test split. `--no-cfg` disables classifier-free guidance. `--ranking-only` skips image decoding. `--dataset-root` and `--checkpoint-root` override the sibling asset directories. Resolved configs and rebased caches are written under the output directory.

For a quick CPU smoke test:

```bash
python inference.py --task tiger_sequential --device cpu --max-samples 2 --batch-size 2 --num-workers 0 --sampling-steps 2 --output runs/smoke
```

A limited-sample smoke test is not a full benchmark. `--max-samples 0` means the full split; the limit counts generation target rows (two per bundle, one per sequential context).

Outputs are under `<output>/inference/`, including generated visual tokens, `*_pred_text.png`, `*_pred_none.png`, `*_gt.png`, and `ranking/ranking_details.npy`.

## Evaluation

```bash
python evaluate.py --task ddbc_sequential --output runs/ddbc_sequential/inference --mode both
python evaluate.py --task tiger_sequential --output runs/tiger_sequential/inference --mode both
python evaluate.py --task ddbc_bundle --output runs/ddbc_bundle/inference --mode both
```

Use `--mode ranking|quality|both`; `--device cpu` and `--batch-size` also apply to image evaluation. Full sequential ranking requires the complete split and reads `*_gt_all.txt`.

Only `--split test` is supported in this package. Sequential data/latent/cache provenance is checked against packaged SHA-256 hashes before inference; replacing a latent file without its matching cache is rejected. The anonymous package has no cache-rebuild or training workflow.

Sequential checkpoints are the **fixed final step 9,000** of the current augmented-data experiment, after Stage1 10,000 steps. The preserved bundle checkpoints used historical post-hoc selection involving test image quality; they must not be presented as validation-only selection.

Candidate ranking inserts missing GT items into the upstream candidate pool and therefore measures reranking conditional on coverage. All ranking uses the entire 31,217-item catalog. Sequential ranking evaluates all one or two GTs. Quality generates one image per sequential context and pairs it with the **first GT**, with `pred_text` and `pred_none` decoders. Text decoding uses the benchmark target-category prompt. IS is the POG **reference-uniform-50-way** score, not conventional marginal-distribution IS; IS-acc@1/3 are category accuracies. LPIPS uses AlexNet and lower is better.

## Test-data scope

| Task | Test contexts | Input items / context | GT items / context | GT target rows | Generated images / decoder |
|---|---:|---:|---:|---:|---:|
| bundle recommendation | 2,521 | 2 | 2 | 5,042 | 5,042 |
| sequential recommendation | 2,483 | 3 | 1–2 | 3,171 | 2,483 |

Sequential test has 1,795 one-GT contexts and 688 two-GT contexts. Both backbone test latent exports are included for each task (four files total). All catalog features, visual token IDs and numeric item metadata needed for full-catalog ranking are retained. **Only the 3,266 unique GT images referenced by the bundle/sequential test sets** are included; the image folder is not the full catalog image collection. No train or validation records are included. See [test-data README](../dataset_anonymous/README.md).

The sequential input records are POG-derived outfit records arranged as a prediction sequence; this release does not claim they are original chronological user-interaction logs. The internal `clhe/` asset directory preserves benchmark compatibility without changing the public task name.

## Validation, privacy and licensing

The delivery-level `VALIDATION.json` records historical execution checks; `RELEASE_VALIDATION.json`, `PRIVACY_AUDIT.json` and `ARCHIVE_VALIDATION.json` describe the current packaging checks. These optional records are not loaded by the training or inference entrypoints. The current 18,000-step training defaults have only been checked statically; no training or inference was run for that configuration update. Packaging strips private author/server identifiers, training service metadata and source image text/EXIF metadata. Model weights are exported without optimizer/RNG state or private trainer objects; archive ownership is normalized. Public upstream license notices and necessary third-party attribution are retained.

Original RG-LSM code is Apache-2.0; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the separate scope of third-party code, benchmark data and upstream weights. Verify `SHA256SUMS` after extraction.
