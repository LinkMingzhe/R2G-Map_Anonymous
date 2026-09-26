# R2G-Map — anonymous inference release

This project provides R2G-Map inference and local evaluation for bundle recommendation and **sequential recommendation**. It consumes supplied DDBC/TIGER hidden latents. It contains no training entry point, train/validation split, train/validation latent cache, Stage1 initialization, or upstream backbone checkpoint/exporter.
The four inference configurations are `ddbc_bundle`, `tiger_bundle`, `ddbc_sequential`, and `tiger_sequential`. 
## Layout and installation

Use the dedicated anonymous asset archives:

```text
release/
  R2G-Map_Anonymous/
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

The [dataset](https://drive.google.com/file/d/1hxYw1cRo3BiHs3d_2zTNPWWR36fQAuJK/view?usp=sharing) and [pretrained checkpoints](https://drive.google.com/file/d/1v9v-CdY6Zci_3-e081Pib6DcibPesTEr/view?usp=drive_link) are available for download.

## Main-model inference

Run inside this code directory. All four task names run R2G-Map; the prefix selects the DDBC or TIGER conditioning latent.

```bash
python inference.py --task ddbc_sequential --config-only
python inference.py --task ddbc_sequential --output runs/ddbc_sequential
python inference.py --task tiger_sequential --output runs/tiger_sequential
python inference.py --task ddbc_bundle --output runs/ddbc_bundle
python inference.py --task tiger_bundle --output runs/tiger_bundle
```

Defaults are the packaged **Stage2 EMA**, **CFG scale 6**, 16 sampling steps, cosine CFG schedule, seed 42, inference batch size 16 for sequential / 32 for bundle, and test split. `--no-cfg` disables classifier-free guidance. `--ranking-only` skips image decoding. `--dataset-root` and `--checkpoint-root` override the sibling asset directories. 



## Evaluation

```bash
python evaluate.py --task ddbc_sequential --output runs/ddbc_sequential/inference --mode both
python evaluate.py --task tiger_sequential --output runs/tiger_sequential/inference --mode both
python evaluate.py --task ddbc_bundle --output runs/ddbc_bundle/inference --mode both
```

Use `--mode ranking|quality|both`; `--device cpu` and `--batch-size` also apply to image evaluation. Full sequential ranking requires the complete split and reads `*_gt_all.txt`.

Sequential checkpoints are the **fixed final step 9,000** of the current augmented-data experiment, after Stage1 10,000 steps.

## Test-data scope

| Task | Test contexts | Input items / context | GT items / context | GT target rows | Generated images / decoder |
|---|---:|---:|---:|---:|---:|
| bundle recommendation | 2,521 | 2 | 2 | 5,042 | 5,042 |
| sequential recommendation | 2,483 | 3 | 1–2 | 3,171 | 2,483 |

Sequential test has 1,795 one-GT contexts and 688 two-GT contexts. Both backbone test latent exports are included for each task (four files total). All catalog features, visual token IDs and numeric item metadata needed for full-catalog ranking are retained. 


## Validation, privacy and licensing

The delivery-level `VALIDATION.json` records historical execution checks; `RELEASE_VALIDATION.json`, `PRIVACY_AUDIT.json` and `ARCHIVE_VALIDATION.json` describe the current packaging checks. These optional records are not loaded by the training or inference entrypoints. The current 18,000-step training defaults have only been checked statically; no training or inference was run for that configuration update. Packaging strips private author/server identifiers, training service metadata and source image text/EXIF metadata. Model weights are exported without optimizer/RNG state or private trainer objects; archive ownership is normalized. Public upstream license notices and necessary third-party attribution are retained.

Original R2G-Map code is Apache-2.0; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the separate scope of third-party code, benchmark data and upstream weights. Verify `SHA256SUMS` after extraction.
