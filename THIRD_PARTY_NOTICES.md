# Third-party notices

The Apache-2.0 license at the package root applies to original RG-LSM code and release tooling. It does not replace licenses of upstream code, images, datasets, features or model weights. Existing source headers and public attribution links are retained; public upstream author names are not private release metadata.

* `src/modeling/`, portions of `src/runtime_utils.py`, `src/prompt_cache.py`, and training utilities derive from [Bytedance 1d-tokenizer / TA-TiTok / MaskGen](https://github.com/bytedance/1d-tokenizer). Copyright (2024) Bytedance Ltd. and/or its affiliates; Apache-2.0. See `licenses/Apache-2.0.txt`. The release extracts the necessary dependency closure and adapts imports and runtime paths.
* `src/modeling/modules/maskgit_vqgan.py` preserves the Google LLC / Hugging Face team Apache-2.0 header and [Open-MUSE](https://github.com/huggingface/open-muse) reference. Other upstream references in the model files remain intact.
* TA-TiTok decoder weights originate from [the upstream MaskGen model release](https://github.com/bytedance/1d-tokenizer/blob/main/README_MaskGen.md). Their upstream model terms continue to apply.
* The data originate from the [BundleConstruction dataset](https://huggingface.co/datasets/xhLiu/BundleConstruction), with task splits and features used by [DDBC](https://github.com/LiAi16/DDBC) and [CLHE](https://github.com/Xiaohao-Liu/CLHE). Product images and titles remain third-party content. No blanket Apache-2.0 grant is asserted for them. No explicit dataset-wide license was found in the supplied local data; the linked upstream dataset page did not expose one when checked for this release.
* The fine-tuned 50-class Inception evaluator and category-index mapping are included as evaluation assets with their original third-party provenance (DiFashion/reference evaluator). The supplied files do not contain a standalone license grant; they are not relicensed here.

Release changes include portable paths, local evaluation, offline cache use, metadata cleanup and weight-only exports. Numerical model tensors are preserved. A source snapshot's exact upstream commit was not recorded in the supplied working tree for every vendored component; no commit identity is fabricated.

## LPIPS evaluation assets

LPIPS uses the installed `lpips==0.1.4` package (BSD-style license, retained in `licenses/LPIPS-LICENSE.txt`) and its learned version-0.1 calibration weights. The included AlexNet checkpoint `evaluation/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` is the torchvision ImageNet pretrained AlexNet asset used by LPIPS. Torchvision's BSD license is retained in `licenses/TORCHVISION-LICENSE.txt`; these upstream assets are not relicensed as original RG-LSM weights.
