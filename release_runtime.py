"""Portable release paths and checked, relocatable data/cache preparation."""
from pathlib import Path
import os
from omegaconf import OmegaConf

TASKS = ['ddbc_bundle', 'tiger_bundle', 'ddbc_sequential', 'tiger_sequential']
PACKAGE = Path(__file__).resolve().parent
ANONYMOUS = True


def add_paths(parser):
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--dataset-root', type=Path, default=PACKAGE.parent/'dataset_anonymous')
    parser.add_argument('--checkpoint-root', type=Path, default=PACKAGE.parent/'ckpt_anonymous')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config-only', action='store_true', help='Validate assets, split counts and cache/GT alignment without loading models.')


def configure(args, mode):
    if ANONYMOUS and (mode != 'inference' or args.split != 'test'):
        raise ValueError('The anonymous release supports test inference only.')
    cfg = OmegaConf.load(PACKAGE/'configs'/mode/(args.task+'.yaml'))
    cfg.data_root = str(args.dataset_root.resolve())
    cfg.ckpt_root = str(args.checkpoint_root.resolve())
    cfg.run_root = str((args.output or PACKAGE/'runs'/args.task).resolve())
    if mode == 'inference':
        task = 'ddbc' if args.task.endswith('bundle') else 'clhe'
        bb = args.task.split('_')[0]
        sp = args.split
        cfg.dataset.test.hidden_npy = f'{cfg.data_root}/{task}/hidden/{bb}/{sp}_hidden_sampleTimestep_by_bundle.npy'
        cfg.dataset.test.input_txt = f'{cfg.data_root}/{task}/Bundle/{sp}_input_4.txt' if task == 'ddbc' else f'{cfg.data_root}/{task}/splits/{sp}_input.txt'
        cfg.dataset.test.gt_txt = f'{cfg.data_root}/{task}/Bundle/{sp}_gt_4.txt' if task == 'ddbc' else f'{cfg.data_root}/{task}/splits/{sp}_gt_first.txt'
        cache_name = sp+'.pt' if task == 'ddbc' else sp+'_eval.pt'
        cfg.dataset.test.cache_pt = f'{cfg.run_root}/cache/{cache_name}'
        if task == 'clhe':
            cfg.dataset.test.gt_all_txt = f'{cfg.data_root}/clhe/splits/{sp}_gt_all.txt'
        cfg.ranking.raw_meta_file = f'{sp}_hidden_sampleTimestep_by_bundle.npy'
        cfg.inference.use_cfg = not args.no_cfg
        cfg.inference.guidance_scale = args.guidance_scale
        cfg.inference.batch_size = args.batch_size
        cfg.inference.num_workers = args.num_workers
        cfg.inference.max_bundles = args.max_samples
        cfg.inference.num_iter = args.sampling_steps
        cfg.inference.device = args.device
        cfg.inference.decode_modes = [] if args.ranking_only else args.decode_modes
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def cache_assets(args, cfg, mode):
    family = 'ddbc' if args.task.endswith('bundle') else 'clhe'
    asset = Path(cfg.data_root)/family/'cache'/args.task.split('_')[0]
    splits = [args.split] if mode == 'inference' else ['train', 'valid']
    for split in splits:
        slot = cfg.dataset.test if mode == 'inference' else cfg.dataset[split]
        name = split if family == 'ddbc' else split+('_eval' if mode == 'inference' else '_all_gt')
        yield split, slot, asset/(name+'.pt')


def _read_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        parts = line.strip().split(',')
        if len(parts) < 2:
            raise ValueError(f'{path}:{line_number}: missing item IDs')
        rows.append((parts[0], [int(item) for item in parts[1:]]))
    return rows


def validate_sequential_cache(args, cfg, split, slot, payload):
    """Reject a stale split or a first-GT-only training cache before starting a GPU job."""
    import numpy as np
    inputs = _read_rows(slot.input_txt)
    truth = _read_rows(Path(cfg.data_root)/'clhe/splits'/(split+'_gt_all.txt'))
    first = _read_rows(Path(cfg.data_root)/'clhe/splits'/(split+'_gt_first.txt'))
    expected_counts = cfg.dataset.release_counts[split]
    if len(inputs) != expected_counts.contexts or sum(len(items) for _, items in truth) != expected_counts.targets:
        raise ValueError(f'{split}: wrong release dataset counts; expected {dict(expected_counts)}')
    if len(truth) != len(inputs) or len(first) != len(inputs):
        raise ValueError(f'{split}: input/GT row count mismatch')
    if len({key for key, _ in inputs}) != len(inputs):
        raise ValueError(f'{split}: every augmented context must have a distinct row ID')
    supervise_all = bool(cfg.dataset.supervise_all_gt)
    if payload is not None and bool(payload.get('supervise_all_gt', False)) != supervise_all:
        raise ValueError(f'{split}: incompatible supervise_all_gt cache flag')
    expected_keys, expected_inputs, expected_targets, expected_positions = [], [], [], []
    full_truth = {}
    for (key, context), (gt_key, targets), (first_key, first_targets) in zip(inputs, truth, first):
        if key != gt_key or key != first_key or len(context) != 3 or len(targets) not in (1, 2) or first_targets != targets[:1]:
            raise ValueError(f'{split}: malformed context/GT alignment at {key}')
        if set(context) & set(targets):
            raise ValueError(f'{split}: input/GT overlap at {key}')
        if not all(0 <= item < 31217 for item in context+targets):
            raise ValueError(f'{split}: item outside the 31,217-item catalog')
        full_truth[key] = targets
        selected = targets if supervise_all else targets[:1]
        for position, target in enumerate(selected, 1):
            expected_keys.append(key); expected_inputs.append(context)
            expected_targets.append(target); expected_positions.append(position)
    if payload is None:
        print(f'Validated {args.task}/{split}: {len(inputs):,} contexts, {len(expected_targets):,} expected cache rows; cache will be rebuilt')
        return
    if len(payload['raw_candidate_item_ids']) != len(expected_targets):
        raise ValueError(f'{split}: candidate pool row count mismatch')
    if [str(key) for key in payload['base_bundle_ids']] != expected_keys:
        raise ValueError(f'{split}: cache context IDs/order do not match the released split')
    for name, expected in [('input_item_ids', expected_inputs), ('target_item_ids', expected_targets), ('target_positions', expected_positions)]:
        if not np.array_equal(np.asarray(payload[name]), np.asarray(expected)):
            raise ValueError(f'{split}: cache {name} differs from the released split')
    latent = np.asarray(payload['condition_latents'])
    dim = 64 if args.task.startswith('ddbc') else 128
    if latent.shape != (len(expected_targets), 4, dim) or not np.isfinite(latent).all():
        raise ValueError(f'{split}: invalid hidden latent shape or nonfinite values: {latent.shape}')
    for key, pool in zip(expected_keys, payload['raw_candidate_item_ids']):
        if not set(full_truth[key]).issubset(set(int(item) for item in pool)):
            raise ValueError(f'{split}: candidate pool is missing a full-GT label for {key}')
    print(f'Validated {args.task}/{split}: {len(inputs):,} contexts, {len(expected_targets):,} cache rows; all-GT supervision={supervise_all}')


def check_inputs(args, cfg, mode):
    required = [cfg.model.item_info_path, cfg.model.cate_id_to_text_path, cfg.dataset.tatitok_token_json,
                cfg.model.content_feature_path, cfg.model.description_feature_path, cfg.model.cf_feature_path]
    assets = list(cache_assets(args, cfg, mode))
    for split, slot, asset in assets:
        required += [slot.hidden_npy, slot.input_txt, slot.gt_txt, asset.parent/'prompt.pt']
        if not getattr(args, 'rebuild_cache', False):
            required.append(asset)
        if args.task.endswith('sequential'):
            required += [Path(cfg.data_root)/'clhe/splits'/(split+'_gt_all.txt'), Path(cfg.data_root)/'clhe/splits'/(split+'_gt_first.txt')]
    if mode == 'inference':
        required += [Path(cfg.inference.checkpoint_path)/'ema_model/pytorch_model.bin',
                     Path(cfg.inference.checkpoint_path)/'ema_model/config.json']
        if cfg.inference.decode_modes:
            required += [Path(cfg.inference.tatitok_decoder_path)/'model.safetensors', cfg.inference.gt_image_dir]
    elif mode == 'stage2_tsp':
        required += [Path(cfg.training.stage_init_checkpoint)/'pytorch_model.bin']
    missing = [str(path) for path in required if not Path(str(path)).exists()]
    if missing:
        raise FileNotFoundError('Missing release assets:\n'+'\n'.join(missing))
    if args.task.endswith('sequential'):
        if getattr(args, 'rebuild_cache', False):
            for split, slot, _ in assets:
                validate_sequential_cache(args, cfg, split, slot, None)
        else:
            verify_provenance(cfg, assets)
            import torch
            for split, slot, asset in assets:
                payload = torch.load(asset, map_location='cpu', weights_only=False)
                validate_sequential_cache(args, cfg, split, slot, payload)


def verify_provenance(cfg, assets):
    import hashlib
    import json
    root = Path(cfg.data_root)
    manifest = root/'clhe/cache_provenance.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing release cache provenance manifest: {manifest}')
    hashes = json.loads(manifest.read_text())['files']
    paths = set()
    for _, slot, asset in assets:
        paths.update([Path(slot.hidden_npy), Path(slot.input_txt), Path(slot.gt_txt), Path(slot.gt_all_txt), asset, asset.parent/'prompt.pt'])
        paths.add(Path(slot.gt_all_txt).with_name(Path(slot.gt_all_txt).name.replace('_gt_all.txt', '_gt_first.txt')))
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if relative not in hashes:
            raise ValueError(f'Unregistered release asset: {relative}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != hashes[relative]:
            raise ValueError(f'Release asset changed: {relative}. Use --rebuild-cache in the full release after intentional backbone/data changes.')


def prepare_caches(args, cfg, mode):
    import torch
    assets = list(cache_assets(args, cfg, mode))
    for split, slot, asset in assets:
        payload = torch.load(asset, map_location='cpu', weights_only=False)
        payload.update(source_hidden_npy=str(slot.hidden_npy), source_input_txt=str(slot.input_txt), source_gt_txt=str(slot.gt_txt),
                       ranking_raw_meta_path=str(slot.hidden_npy))
        dest = Path(slot.cache_pt)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, dest)
    payload = torch.load(assets[0][2].parent/'prompt.pt', map_location='cpu', weights_only=False)
    payload['metadata']['item_info_path'] = str(cfg.model.item_info_path)
    payload['metadata']['cate_id_to_text_path'] = str(cfg.model.cate_id_to_text_path)
    dest = Path(cfg.model.prompt_cache_pt)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)


def run_engine(args, cfg, engine, mode):
    import subprocess, sys
    if ANONYMOUS and (mode != 'inference' or engine != 'inference_engine.py' or args.split != 'test'):
        raise ValueError('The anonymous release supports test inference only.')
    check_inputs(args, cfg, mode)
    out = Path(cfg.run_root); out.mkdir(parents=True, exist_ok=True)
    config_path = out/(mode+'_resolved.yaml')
    OmegaConf.save(cfg, config_path)
    if args.config_only:
        print('Validated configuration:', config_path)
        return
    prepare_caches(args, cfg, mode)
    env = os.environ.copy()
    env.setdefault('WANDB_MODE', 'disabled')
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('OPENBLAS_NUM_THREADS', '4')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    if getattr(args, 'device', None) == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run([sys.executable, str(PACKAGE/'src'/engine), 'config='+str(config_path)], check=True, env=env, cwd=PACKAGE)
