"""Zero-update diagnostic of the preserved GPU identity-preflight failure."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
import json,math
import numpy as np
import torch
from common import ROOT,read,sha,immutable_json
from partition_data import Partition,index_sha
from fresh_model import build_fresh,build_original,base_state,tensor_hash,TYPES,math_module
from runtime import setup

def differences(a,b):
    a=a.detach().double();b=b.detach().double();d=(a-b).abs()
    return {'bitwise_equal':torch.equal(a,b),'max_abs':float(d.max()),
        'max_relative_with_1e_minus12_floor':float((d/torch.maximum(a.abs(),b.abs()).clamp_min(1e-12)).max())}

def prediction_difference(a,b):
    def f(p):return p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*27.211386245988)
    return {'E':differences(a[0],b[0]),'A':differences(a[1],b[1]),'native_f':differences(f(a),f(b))}

def decode(model,scalar,tensors):
    d=model.decoder;v=d.trunk(scalar)
    energy=torch.nn.functional.softplus(d.energy_head(v).squeeze(-1)+d.energy_offset)
    beta=d.beta_head(v).squeeze(-1)
    q=(d.tensor_gate(v).tanh()[...,None]*tensors).sum(-2)/math.sqrt(d.tensor_channels)
    Q=torch.einsum('...m,mij->...ij',q,d.cartesian_basis)
    C=beta[...,None,None]/math.sqrt(3)*d.identity+Q
    return energy,C@C.transpose(-1,-2)

def main():
    review=read(ROOT/'IDENTITY_DIAGNOSTIC_REVIEW.json');assert review['passed']
    assert review['scope']=='round05_two_train_geometry_zero_update_identity_diagnosis'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    assert not list((ROOT/'private_preflight').glob('*after_update1.pt'))
    cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json');setup(cfg)
    part=Partition('train');rows=part.indices[:2]
    z=torch.from_numpy(part.selected('dataset.npz','z',rows)).cuda()
    pos=torch.from_numpy(part.selected('dataset.npz','pos',rows)).cuda()
    edges=torch.from_numpy(part.selected('dataset.npz','edge',rows)).cuda().long()
    mask=z!=0;counts=mask.sum(1);offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]));em=edges[:,0]>=0
    x=dict(z=z[mask],pos=pos[mask],batch=torch.arange(2,device='cuda')[:,None].expand_as(z)[mask],n=2,
        edge_index=(edges+offsets[:,None,None]).transpose(1,2)[em].T.contiguous())
    models={'original':build_original(mc,stats).cuda().eval(),
        'control':build_fresh(mc,stats,False).cuda().eval(),'adapter':build_fresh(mc,stats,True).cuda().eval()}
    assert len(set(tensor_hash(m.state_dict() if k=='original' else base_state(m)) for k,m in models.items()))==1
    report={'model_updates':0,'unique_train_geometry_rows':2,'train_indices_sha256':index_sha(rows),
        'target_arrays_decoded':False,'actual_decode_ledger':part.ledger,'repeat_full_forward':{},'cross_model':{}}
    allpred={};allm={}
    with torch.no_grad():
        for name,model in models.items():
            values=[model(**x) for _ in range(3)];allpred[name]=values[0]
            report['repeat_full_forward'][name]=[prediction_difference(values[0],v) for v in values[1:]]
        for name in ('control','adapter'):report['cross_model'][name]=prediction_difference(allpred['original'],allpred[name])
        wrapper=models['adapter'];p,raw=wrapper(**x,return_aux=True)
        p2,raw2=wrapper(**x,return_aux=True)
        report['repeat_raw_M']={t:differences(raw[t],raw2[t]) for t in TYPES}
        native_s,native_t,_=models['original'].cg(raw)
        report['shared_raw_M_coupling']={}
        for name in ('control','adapter'):
            s,t=models[name].couple(raw)
            report['shared_raw_M_coupling'][name]={'scalar':differences(native_s,s),'tensor':differences(native_t,t),
                'decoder':prediction_difference(decode(models['original'],native_s,native_t),decode(models[name],s,t))}
        right={t:raw[t][:,1:] for t in TYPES};adapted=wrapper.right_adapter(right)
        report['identity_F_on_shared_raw_M']={t:differences(right[t],adapted[t]) for t in TYPES}
    report['shared_latent_operations_bitwise']=all(
        entry['scalar']['bitwise_equal'] and entry['tensor']['bitwise_equal'] and
        all(v['bitwise_equal'] for v in entry['decoder'].values()) for entry in report['shared_raw_M_coupling'].values())
    report['same_model_full_forward_variability_observed']=any(
        not v['bitwise_equal'] for rows in report['repeat_full_forward'].values() for row in rows for v in row.values())
    report['source_hashes']=review['source_hashes'];report['diagnostic_only_no_acceptance_tolerance_change']=True
    report['passed']=True
    immutable_json(report,ROOT/'GPU_IDENTITY_DIAGNOSTIC.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
