#!/usr/bin/env python3
"""Visual-token ranking and POG 50-class reference-uniform image quality."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def ranking(args):
    import bundle_metrics
    import sequential_metrics
    details = np.load(args.output/'ranking/ranking_details.npy', allow_pickle=True).item()
    result = {}
    for pool in ['raw', 'all']:
        records = details[pool]['candidate_log_likelihood']
        if args.task.endswith('bundle'):
            bank = torch.load(args.dataset_root/'ddbc/clhe.pt', map_location='cpu', weights_only=False).float().numpy()
            bundle_metrics.FEATURE_BANK = bank/np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
            metrics = bundle_metrics.bundle_metrics_from_records(records)
            if not metrics:
                raise ValueError('Bundle evaluation requires both target rows per bundle.')
            for k in [1, 5, 10]:
                precision, recall = metrics[f'precision@{k}'], metrics[f'recall@{k}']
                metrics[f'f1@{k}'] = 2*precision*recall/max(precision+recall, 1e-12)
                metrics[f'oas_similarity@{k}'] = 1-metrics[f'oas_mean@{k}']
        else:
            rows = sequential_metrics.load_gt_rows(args.dataset_root/'clhe/splits'/(args.split+'_gt_all.txt'))
            bykey = {str(record['bundle_key']): record for record in records}
            if len(bykey) != len(records) or set(bykey) != {key for key, _ in rows}:
                raise ValueError('Sequential ranking requires exactly one prediction for every split context.')
            rankings = [bykey[key]['top10_candidate_item_ids'] for key, _ in rows]
            # The historical field name contains "top10" but the release stores top 50.
            if any(len(items) < 50 for items in rankings):
                raise ValueError('Recall/NDCG through K=50 require saved top-50 candidate IDs.')
            metrics = sequential_metrics.multi_gt_metrics(rankings, rows)
        result[pool] = metrics
    return result


def lpips_scores(args, paths, targets, device):
    import lpips
    from torchvision import transforms
    hub = args.checkpoint_root/'evaluation/torch/hub'
    weights = hub/'checkpoints/alexnet-owt-7be5be79.pth'
    if not weights.is_file():
        raise FileNotFoundError(f'Missing packaged AlexNet weights for offline LPIPS: {weights}')
    torch.hub.set_dir(str(hub.resolve()))
    model = lpips.LPIPS(net='alex').eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    import quality_metrics as q
    values = []
    for start in range(0, len(paths), args.batch_size):
        generated, truth = [], []
        for path, target in zip(paths[start:start+args.batch_size], targets[start:start+args.batch_size]):
            image = q.load_image_rgb(path)
            gt = q.load_image_rgb(args.dataset_root/'images'/f'{target}.png')
            generated.append(transform(image)); truth.append(transform(gt))
            image.close(); gt.close()
        with torch.no_grad():
            batch = model(torch.stack(generated).to(device), torch.stack(truth).to(device))
        values.extend(batch.detach().cpu().flatten().tolist())
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'count': len(values), 'net': 'alex', 'image_size': 256}


def quality(args):
    import quality_metrics as q
    import sequential_metrics
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = q.load_category_labels(args.dataset_root/'categories/labels.txt')
    labels = q.reorder_category_labels_for_classifier(labels, args.checkpoint_root/'evaluation/cid_to_label.npy')
    cats = q.load_category_map(args.dataset_root/'categories/item_categories.tsv')
    model = q.load_category_inception_model(args.checkpoint_root/'evaluation/Inception-finetune-epoch300', labels, device)
    expected = None
    if args.task.endswith('sequential'):
        rows = sequential_metrics.load_gt_rows(args.dataset_root/'clhe/splits'/(args.split+'_gt_all.txt'))
        expected = {key: targets[0] for key, targets in rows}
    result = {}
    for mode in ['pred_text', 'pred_none']:
        paths = sorted(args.output.glob('*_'+mode+'.png'))
        if not paths:
            continue
        pairs = [path.name[:-len('_'+mode+'.png')].rsplit('_', 1) for path in paths]
        target = [int(item) for _, item in pairs]
        if expected is not None:
            observed = {key: int(item) for key, item in pairs}
            if any(expected.get(key) != item for key, item in observed.items()) or len(observed) != len(paths):
                raise ValueError('Sequential visual quality must pair one image/context with its first GT.')
            if not args.allow_partial_quality and observed != expected:
                raise ValueError('Full visual quality requires every split context; use --allow-partial-quality only for smoke tests.')
        probs = q.collect_reference_inception_probs(paths, model, device, args.batch_size, 299, mode)
        truth = np.array([labels.index(cats[item]) for item in target])
        order = np.argsort(-probs, axis=1)
        result[mode] = {
            'samples': len(paths), 'is': q.reference_inception_score(probs, 1)['mean'],
            'is_variant': 'reference_uniform_50way',
            'is_acc_top1': float(np.mean(order[:, 0] == truth)),
            'is_acc_top3': float(np.mean(np.any(order[:, :3] == truth[:, None], axis=1))),
            'lpips': lpips_scores(args, paths, target, device),
        }
    if not result:
        raise FileNotFoundError('No pred_text/pred_none images found.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['ddbc_bundle', 'tiger_bundle', 'ddbc_sequential', 'tiger_sequential'], required=True)
    parser.add_argument('--output', type=Path, required=True, help='Directory containing generated PNGs and ranking/.')
    parser.add_argument('--dataset-root', type=Path, default=ROOT.parent/'dataset_anonymous')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT.parent/'ckpt_anonymous')
    parser.add_argument('--split', choices=['test'], default='test')
    parser.add_argument('--mode', choices=['ranking', 'quality', 'both'], default='both')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--allow-partial-quality', action='store_true', help='Permit incomplete visual outputs for smoke tests only.')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    result = {}
    if args.mode in ['ranking', 'both']:
        result['ranking'] = ranking(args)
    if args.mode in ['quality', 'both']:
        result['quality'] = quality(args)
    dest = args.output/'release_metrics.json'
    dest.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2)); print('Saved:', dest)


if __name__ == '__main__':
    main()
