import hashlib,json,pathlib,shutil,subprocess,sys
import numpy as np
import torch
from dataset import ROOT,Data
from model_factory import build
from trainer import setup,state_hash,atomic_json,atomic_save

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    for n in ('data','configs','initial','reports','runs','logs'): (ROOT/n).mkdir(exist_ok=True)
    assert not any((ROOT/'runs').glob('*/last.pt')),'Never reinitialize an active campaign'
    old=ROOT.parent/'qm9s_pouter_trace_wE_20260928'
    tree=json.loads((ROOT/'reports/reference_tree.json').read_text(encoding='utf-8-sig'))
    verified={}
    for p in list((ROOT/'frozen_reference').rglob('*'))+list((ROOT/'pinned_reference').rglob('*')):
        if not p.is_file() or '__pycache__' in str(p):continue
        prefix='experiments/qm9s_full_EA_20260925/' if p.is_relative_to(ROOT/'frozen_reference') else 'experiments/qm9s_eta_Ef_20260926/'
        base=ROOT/('frozen_reference' if prefix.endswith('full_EA_20260925/') else 'pinned_reference')
        name=prefix+p.relative_to(base).as_posix()
        rows=[r for r in tree['tree'] if r['path']==name]
        if not rows:continue
        raw=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==rows[0]['sha'],name
        verified[name]=dict(git_blob=blob,sha256=sha(p))
    atomic_json(dict(commit='71e575f573cfc94c7f0a301b663b0169290ce9bd',files=verified),ROOT/'reports/source_verification.json')
    hashes=json.loads((ROOT/'frozen_reference/data/hashes.json').read_text())
    for name in ('dataset.npz','raw_labels.npz','normalization.json','splits.json','identity_audit_v2.json'):
        if name in hashes:assert sha(old/'data'/name)==hashes[name],name
        shutil.copy2(old/'data'/name,ROOT/'data'/name)
    audit=json.loads((ROOT/'pinned_reference/reports/data_audit.json').read_text())
    assert sha(ROOT/'data/raw_labels.npz')==audit['raw_labels_sha256']
    shutil.copy2(ROOT/'pinned_reference/reports/data_audit.json',ROOT/'reports/data_audit.json')
    # Frozen E^2 weight constant for the trace loss: one global mean over train-only valid
    # (molecule,state) entries, computed once here, shared verbatim by training and validation.
    # Derived from truth E only, so it is a constant, not a function of any prediction.
    # It is written to its OWN file rather than into data/normalization.json: that file is read-only and
    # is pinned by the frozen data/hashes.json, and the sE2/sA2/E_state_mean inside it are the original
    # cross-experiment constants this run must not alter. A separate frozen file keeps the data hashes
    # byte-identical to their origin while still fixing the new constant once, here.
    norm_path=ROOT/'data/normalization.json';stats=json.loads(norm_path.read_text())
    with np.load(ROOT/'data/dataset.npz') as d, np.load(ROOT/'data/raw_labels.npz') as r:
        assert np.array_equal(d['ids'],r['ids']),'dataset.npz and raw_labels.npz disagree on ids'
        # dataset.npz ships E as float32 while raw_labels.npz is float64, so the two copies agree only
        # to float32 precision; compare within that, and compute the constant from the float32 copy
        # that training actually sees so the recorded value is the one the loss will divide by.
        assert np.allclose(d['E'].astype(np.float64),r['E'],rtol=1e-6,atol=1e-6,equal_nan=True),'dataset.npz and raw_labels.npz disagree on E'
        onehot=np.zeros(d['E'].shape[0],dtype=bool);onehot[d['train']]=True
        ma,me=r['mask_A'],r['mask_E']
        assert np.array_equal(ma,me),'mask_A and mask_E differ; the trace weight uses their intersection'
        valid=onehot[:,None]&ma&me
        assert int(valid.sum())==int(onehot.sum())*int(ma.shape[1]),'a train entry is invalid; the train split is expected to be mask-complete'
        assert not valid[onehot==False].any(),'a non-train entry leaked into the weight constant'
        Ef=d['E'].astype(np.float64)
        mean_E2=float(np.mean(np.square(Ef[valid])))
        # Independent path: the same quantity rebuilt from the float64 label file.
        independent=float(np.mean(np.square(r['E'][valid]).astype(np.float64)))
        all_split_mean=float(np.mean(np.square(r['E'][ma&me]).astype(np.float64)))
        assert abs(mean_E2-independent)<1e-5*mean_E2,(mean_E2,independent)
        assert np.isfinite(mean_E2) and mean_E2>0
        shipped_rms=float(np.sqrt(mean_E2))
    definition='mean over train-only (molecule,state) entries with mask_A & mask_E of E_true^2; computed once, frozen, shared by training and validation; never rescaled per state or per batch'
    # Keep the recorded declaration in the same key order the loader would have produced from
    # normalization.json, so the only change to this dict is the one new key.
    stats=dict(stats,mean_E2_train=mean_E2,definition_mean_E2_train=definition,
        mean_E2_train_source='train split only; no validation or test statistics')
    weight_path=ROOT/'data/trace_weight.json'
    atomic_json(stats,weight_path)
    atomic_json(dict(mean_E2_train=mean_E2,definition=definition,
        train_valid_entries=int(valid.sum()),independent_recomputation=independent,
        independent_relative_difference=abs(mean_E2-independent)/mean_E2,
        shipped_train_E_rms_eV=shipped_rms,
        all_split_mean_for_contrast=all_split_mean,
        relative_difference_from_all_split=abs(mean_E2-all_split_mean)/mean_E2,
        train_only_equals_all_split_required=False,
        frozen_stats_file='data/trace_weight.json',
        frozen_stats_sha256=sha(weight_path),
        normalization_sha256_unchanged=sha(norm_path),
        source='train only, mask_A & mask_E, computed once in prepare.py'),ROOT/'reports/trace_weight.json')
    # data/hashes.json is regenerated AFTER every data file is final. The five original data files keep
    # their original sha256, so the frozen table stays a valid cross-check on them; the new frozen
    # constant file is listed alongside them.
    atomic_json({p.name:sha(p) for p in sorted((ROOT/'data').iterdir())},ROOT/'data/hashes.json')
    frozen_hashes=json.loads((ROOT/'frozen_reference/data/hashes.json').read_text())
    produced=json.loads((ROOT/'data/hashes.json').read_text())
    for name,frozen in frozen_hashes.items():
        assert produced[name]==frozen,('a supplied data file no longer matches its frozen hash',name)
    cfg=json.loads((ROOT/'pinned_reference/configs/mto_eta1.json').read_text())
    # The ONLY change in this experiment: MTO channel count 16 -> 64.
    # The pinned config is git-blob-verified and must stay byte-identical, so the
    # value is overridden here rather than edited at its source.
    CHANNELS=64
    assert cfg['mto_channels']==16,'pinned config no longer ships the c=16 baseline'
    for n,arch,energy,gpu in [('G1','original',True,1),('G2','original',False,2),('G3','p_outer',True,4),('G4','p_outer',False,6)]:
        # trace_weight_E2 is on for every group: E labels set a fixed weight in the loss.
        # eta still controls whether an E *prediction* loss exists, so G2/G4 remain trace-only.
        c=dict(cfg,name=n,architecture=arch,supervise_E=energy,eta=int(energy),gpu=gpu,trace_weight_E2=True,mto_channels=CHANNELS)
        atomic_json(c,ROOT/'configs'/f'{n}.json');(ROOT/'runs'/n).mkdir(exist_ok=True)
    setup(11);data=Data()
    assert len(data.ids)==133727 and {k:len(v) for k,v in data.parts.items()}==dict(train=120355,val=6686,test=6686)
    assert len(np.unique(np.concatenate(list(data.parts.values()))))==133727
    # CHANNELS must reach the initial-weight construction too, not only the run configs:
    # the frozen g1/g3 checkpoints and the alpha_init calibrated from them are c=CHANNELS.
    base=dict(cfg,architecture='original',supervise_E=True,mto_channels=CHANNELS)
    g1=build(base,data.stats,initial=False).cuda().eval()
    g3=build(dict(base,architecture='p_outer'),data.stats,initial=False).cuda().eval()
    g3.core.load_state_dict(g1.core.state_dict());g3.mto.load_state_dict(g1.mto.state_dict())
    shared={k:v for k,v in g1.state_dict().items() if k.startswith(('core.','mto.'))}
    assert all(torch.equal(v,g3.state_dict()[k]) for k,v in shared.items())
    indices=data.parts['train'][:512];s1=0.;s3=0.;count=0
    with torch.no_grad():
        for a in range(0,512,64):
            x,_=data.batch(indices[a:a+64]);_,A=g1(**x);_,_,mu=g3(**x,export=True)
            s1+=float(A.diagonal(dim1=-2,dim2=-1).sum(-1).double().sum());s3+=float(mu.double().square().sum(-1).sum());count+=mu.shape[0]*mu.shape[1]
        assert s3>0 and np.isfinite(s3)
        alpha=float(np.sqrt(s1/s3));g3.decoder.alpha_init.fill_(alpha)
        calibrated=0.
        for a in range(0,512,64):
            x,_=data.batch(indices[a:a+64]);_,_,mu=g3(**x,export=True);calibrated+=float(mu.double().square().sum(-1).sum())
    assert abs(calibrated/s1-1)<2e-6
    atomic_save({k:v.cpu() for k,v in g1.state_dict().items()},ROOT/'initial/original.pt')
    atomic_save({k:v.cpu() for k,v in g3.state_dict().items()},ROOT/'initial/p_outer.pt')
    orders=[];rng=np.random.default_rng(11)
    for epoch in range(1,1001):orders.append(dict(epoch=epoch,sha256=hashlib.sha256(rng.permutation(data.parts['train']).tobytes()).hexdigest()))
    atomic_json(orders,ROOT/'reports/epoch_order_hashes.json')
    atomic_json(dict(groups=dict(G1=state_hash(g1.state_dict()),G2=state_hash(g1.state_dict()),G3=state_hash(g3.state_dict()),G4=state_hash(g3.state_dict())),
        shared_state_sha256=state_hash(shared),alpha_init=alpha,calibration_indices=indices.tolist(),calibration_ids=data.ids[indices].tolist(),
        G1_mean_trace=s1/count,raw_mean_mu2=s3/count,calibrated_mean_mu2=calibrated/count,calibration_partition='first 512 indices of frozen train array',
        initial_files={p.name:sha(p) for p in (ROOT/'initial').glob('*.pt')},new_CG_weights='fresh seed 11 construction; never copied from original CG'),ROOT/'reports/initialization.json')
    atomic_json(dict(python=sys.executable,python_version=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)),ROOT/'reports/environment.json')
    print('PREPARED',alpha,flush=True)
if __name__=='__main__':main()
