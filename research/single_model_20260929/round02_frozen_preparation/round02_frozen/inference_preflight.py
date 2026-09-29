"""CPU composite-checkpoint geometry replay with forbidden source-data accesses."""
import builtins
import io
import json
from pathlib import Path
import torch
from model_frozen import SOURCE,PARENT,build,frozen_digests
from predictor import load_predictor
from freeze import sha
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(20260930)
    pc=json.loads((PARENT/'round_config.json').read_text())
    sc=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    stats=json.loads((SOURCE/'data/normalization.json').read_text())
    base=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)
    model=build(sc,stats,base['model']);del base
    with torch.no_grad():
        for layer in model.right_adapter.mix.values():layer.weight.normal_(std=.01)
    model.requires_grad_(False)
    z=torch.tensor([6,1,1,1,1,8,1,1],dtype=torch.long)
    pos=torch.tensor([[0.,0.,0.],[.6,.6,.6],[.6,-.6,-.6],[-.6,.6,-.6],[-.6,-.6,.6],
                      [0.,0.,0.],[.8,.6,0.],[-.8,.6,0.]],dtype=torch.float32)
    batch=torch.tensor([0,0,0,0,0,1,1,1],dtype=torch.long)
    edges=torch.tensor([(i,j) for i in range(len(z)) for j in range(len(z)) if i!=j and batch[i]==batch[j]],dtype=torch.long).T
    geometry=dict(z=z,pos=pos,batch=batch,n=2,edge_index=edges)
    with torch.no_grad():expected=model(**geometry)
    blob=io.BytesIO();torch.save({'model':model.state_dict(),'source_config':sc,'stats':stats,
        'adapter_enabled':True,'geometry_only_inference':True,
        'frozen_parameter_buffer_hashes':frozen_digests(model)},blob)
    blob.seek(0)
    forbidden=[str(SOURCE/'data'),str(ROOT/'cache'),str(pc['source_checkpoint'])]
    opened=[];old_open=builtins.open;old_io_open=io.open
    def checked(function):
        def wrapper(file,*args,**kwargs):
            if isinstance(file,(str,bytes,Path)):
                path=str(Path(file).resolve());opened.append(path)
                assert not any(path==prefix or path.startswith(prefix+'/') for prefix in forbidden),('Forbidden inference input',path)
            return function(file,*args,**kwargs)
        return wrapper
    try:
        builtins.open=checked(old_open);io.open=checked(old_io_open)
        loaded=load_predictor(blob,'cpu')
        with torch.no_grad():actual=loaded(**geometry)
    finally:
        builtins.open=old_open;io.open=old_io_open
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert frozen_digests(loaded)==frozen_digests(model)
    result={'passed':True,'device':'cpu','fixture':'two synthetic geometries, nonzeroF, one composite checkpoint in memory',
        'E_A_bitwise_equal':True,'frozen_parameters_and_nonpersistent_buffers_verified':True,
        'original_base_checkpoint_opened_during_inference':False,'dataset_or_cache_opened_during_inference':False,
        'filesystem_paths_opened_during_guard':opened,'qc_labels_used':False,'fit_performed':False,
        'source_hashes':{str(ROOT/name):sha(ROOT/name) for name in ('inference_preflight.py','predictor.py','model_frozen.py')}}
    (ROOT/'INFERENCE_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
