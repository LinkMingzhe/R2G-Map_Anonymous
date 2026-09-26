#!/usr/bin/env python3
"""RG-LSM TSP inference from bundled recommender hidden latents."""
import argparse
from release_runtime import add_paths, configure, run_engine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_paths(p)
    p.add_argument('--split',choices=['test'],default='test')
    p.add_argument('--batch-size',type=int,help='Default: 16 sequential, 32 bundle; keep the default for released visual metrics.')
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--max-samples',type=int,default=0,help='0 evaluates the full split; positive values are smoke tests.')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--no-cfg',action='store_true',help='Explicit ablation only; released metrics use CFG=6.')
    p.add_argument('--guidance-scale',type=float,default=6.0)
    p.add_argument('--sampling-steps',type=int,default=16)
    p.add_argument('--ranking-only',action='store_true')
    p.add_argument('--decode-modes',nargs='+',choices=['pred_text','pred_none','gt'],default=['pred_text','pred_none','gt'])
    args=p.parse_args()
    if args.batch_size is None:args.batch_size=16 if args.task.endswith('sequential') else 32
    if args.max_samples<0 or args.batch_size<1 or args.sampling_steps<1:p.error('Invalid sample/batch/step count.')
    cfg=configure(args,'inference')
    run_engine(args,cfg,'inference_engine.py','inference')

if __name__=='__main__':main()
