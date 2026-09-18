"""Persist atomic assembly and all physical intermediates for four fixed examples."""
import argparse,json
from pathlib import Path
import torch
from dataset import collate
from models import build_pair
from scale_dataset import load_scale
from protocol import verify
ROOT=Path(__file__).resolve().parents[1]

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    return value

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale',choices=['1k','10k','full'],required=True);args=parser.parse_args()
    frozen=verify();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    graphs,_=load_scale(ROOT,'full',splits=('common_fresh_test',),limit=4)
    cfg=json.loads((ROOT/'configs/grid.json').read_text())['configs'][0]
    seeds=json.loads((ROOT/'configs/execution.json').read_text())['seeds_by_scale'][args.scale]
    manifest=json.loads((ROOT/'data'/f'manifest_{args.scale}.json').read_text())
    norm=manifest['train_spectrum_rms'];x,y=collate(graphs,'cuda',norm)
    for seed in seeds:
        pair,_=build_pair(cfg,manifest['train_median_atoms'],seed=seed,spectrum_rms=norm)
        model=pair['mto'];ck=torch.load(ROOT/'runs'/f'{args.scale}_seed{seed}'/'mto_best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['state_dict']);model.cuda().eval()
        pred,extra=model(**x,export=True)
        artifact={'ids':torch.tensor([g['id'] for g in graphs]),'input':cpu_tree(x),
                  'normalized_prediction':pred.cpu(),'source_target':(y*norm).cpu(),
                  'intermediates':cpu_tree(extra),'protocol_id':frozen['protocol_id'],'seed':seed,
                  'scope':'Four preselected common-test examples, no quality-based selection. MTO factors and spectrum-only state decomposition are latent.'}
        torch.save(artifact,ROOT/'results'/args.scale/f'mto_assembly_examples_seed{seed}.pt')
        model.cpu();del pair,model,extra;torch.cuda.empty_cache()
    print('Atomic assembly exports completed',args.scale,flush=True)

if __name__=='__main__':main()
