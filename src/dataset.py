import gzip,json
import torch


def load_database(path):
    with gzip.open(path,'rt') as f: records=json.load(f)
    graphs=[]
    for r in records:
        pos=torch.tensor(r['pos'],dtype=torch.float32)
        z=torch.tensor(r['z'],dtype=torch.long)
        d=torch.cdist(pos,pos)
        mask=(d<5.)&~torch.eye(len(z),dtype=torch.bool)
        if bool(((d<1e-7)&mask).any()): raise ValueError('Coincident atoms')
        graphs.append({'z':z,'pos':pos,'edge_index':mask.nonzero().T.contiguous(),
                       'spectrum':torch.tensor(r['spectrum'],dtype=torch.float32),
                       'id':r['number'],'split':r['split'],'identity':r['canonical_smiles']})
    return graphs


def collate(graphs,device='cpu',scale=1.):
    offset=0; edges=[]
    for g in graphs:
        edges.append(g['edge_index']+offset); offset+=len(g['z'])
    x={'z':torch.cat([g['z'] for g in graphs]).to(device),
       'pos':torch.cat([g['pos'] for g in graphs]).to(device),
       'edge_index':torch.cat(edges,dim=1).to(device),
       'batch':torch.cat([torch.full((len(g['z']),),i,dtype=torch.long) for i,g in enumerate(graphs)]).to(device),
       'n':len(graphs)}
    y=torch.stack([g['spectrum'] for g in graphs]).to(device)/scale
    return x,y


def metrics(pred,target):
    error=pred-target
    mse=error.square().mean(-1)
    mae=error.abs().mean(-1)
    cosine=torch.nn.functional.cosine_similarity(pred,target,dim=-1,eps=1e-12)
    pearson=torch.nn.functional.cosine_similarity(pred-pred.mean(-1,keepdim=True),target-target.mean(-1,keepdim=True),dim=-1,eps=1e-12)
    return {'mse':mse,'rmse':mse.sqrt(),'mae':mae,'cosine':cosine,'pearson':pearson}
