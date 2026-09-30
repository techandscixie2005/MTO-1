"""CPU geometry-only one-checkpoint fixture with frozen eta0 and hand-set coefficients."""
from common import require_cpu
require_cpu()
import builtins
import copy
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from common import ROOT, ROUND03, SOURCE, settings, sha, read_json, atomic_json, source_hashes, require_preparation_authority
from backbone import MTOEA, tensor_hash
from predictor import ScalarSplineMTO, export_record, load_predictor
from scalar_map import apply_map, affine_parameters


def main():
    require_preparation_authority(); torch.set_num_threads(2); cfg = settings()
    checkpoint = SOURCE/'runs/mto_eta0/best.pt'
    assert sha(checkpoint) == cfg['base_checkpoint_sha256']
    config_path = SOURCE/'configs/mto_eta0.json'; stats_path = SOURCE/'data/normalization.json'
    frozen = read_json(ROUND03/'FROZEN_MANIFEST.json')['source_hashes']
    for path in (config_path, stats_path):
        assert sha(path) == frozen[str(path)]
    config = read_json(config_path); stats = read_json(stats_path)
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    base = MTOEA(config, stats)
    base.load_state_dict(original['model'], strict=True); base.requires_grad_(False).eval()
    before_state = tensor_hash(base.state_dict()); before_buffers = tensor_hash(dict(base.named_buffers()))
    assert before_state == cfg['base_tensor_sha256']
    z = torch.tensor([6,1,1,1,1,8,1,1])
    pos = torch.tensor([[0.,0.,0.],[.6,.6,.6],[.6,-.6,-.6],[-.6,.6,-.6],[-.6,-.6,.6],
                        [0.,0.,0.],[.8,.6,0.],[-.8,.6,0.]])
    batch = torch.tensor([0,0,0,0,0,1,1,1])
    edge = torch.tensor([(i,j) for i in range(8) for j in range(8) if i!=j and batch[i]==batch[j]]).T
    geometry = dict(z=z,pos=pos,batch=batch,n=2,edge_index=edge)
    with torch.no_grad():
        reference_E, reference_A = base(**geometry)
    fixtures = [('synthetic_hand_set_nonlinear', np.array([-.002,.047,-.013,.009])),
                ('existing_heldout_affine_anchor', affine_parameters(.9025853252242239,.0018285582112617521))]
    outputs = []; opened = []; old_open = builtins.open; old_io = io.open
    def guard(function):
        def wrapped(file, *args, **kwargs):
            if isinstance(file, (str,bytes,Path)):
                path = str(Path(file).resolve()); opened.append(path)
                if '/data/' in path or '/runs/' in path or '/cache/' in path or path.endswith(('.pt','.pth','.npz','.npy','.json')):
                    raise PermissionError('One-file inference attempted external model/data/config access: '+path)
            return function(file,*args,**kwargs)
        return wrapped
    for name, theta in fixtures:
        record = export_record(base,config,stats,theta,{'fixture':name,'fitted_head':False})
        assert record['source_config'] == config and record['stats'] == stats
        assert record['base_tensor_sha256'] == before_state
        assert record['base_buffer_tensor_sha256'] == before_buffers
        blob = io.BytesIO(); torch.save(record,blob); blob.seek(0)
        with patch('builtins.open',guard(old_open)),patch('io.open',guard(old_io)):
            loaded = load_predictor(blob,'cpu')
            with torch.no_grad(): actual = loaded(**geometry)
        assert torch.equal(reference_E,actual['E']) and torch.equal(reference_A,actual['A'])
        reference_f = apply_map(actual['native_f'].numpy(),theta)
        delta = float(np.max(np.abs(actual['f'].numpy()-reference_f)))
        assert np.allclose(actual['f'].numpy(),reference_f,atol=1e-12,rtol=1e-12)
        assert not any(parameter.requires_grad for parameter in loaded.parameters())
        assert tensor_hash(loaded.base.state_dict()) == before_state
        assert tensor_hash(dict(loaded.base.named_buffers())) == before_buffers
        outputs.append({'fixture':name,'E_A_bitwise_equal':True,'scalar_numpy_torch_max_abs':delta,
                        'all_parameters_frozen':True,'base_state_and_buffers_unchanged':True,
                        'one_checkpoint_bytes':blob.getbuffer().nbytes})
    tampered = copy.deepcopy(record); tampered['model']['scale'] = torch.tensor(.1,dtype=torch.float64)
    blob = io.BytesIO(); torch.save(tampered,blob); blob.seek(0)
    try:
        load_predictor(blob,'cpu')
    except ValueError:
        scale_rejected = True
    else:
        raise AssertionError('Tampered scale accepted')
    assert tensor_hash(base.state_dict()) == before_state
    assert tensor_hash(dict(base.named_buffers())) == before_buffers
    result = {'passed':True,'CPU_only':True,'synthetic_geometries_only':True,'real_head_fit':False,
        'real_geometry_or_target_arrays_opened':False,'outer_validation_or_test_access':False,
        'base_checkpoint_sha256':sha(checkpoint),'base_tensor_sha256':before_state,
        'base_buffer_tensor_sha256':before_buffers,'fixtures':outputs,
        'external_paths_opened_during_one_file_load_or_forward':opened,
        'original_base_data_cache_or_source_access_during_inference':False,
        'tampered_scale_rejected':scale_rejected,
        'input_hashes':{str(config_path):sha(config_path),str(stats_path):sha(stats_path),str(checkpoint):sha(checkpoint)},
        'source_hashes':source_hashes(['inference_preflight.py','predictor.py','backbone.py','scalar_map.py','common.py','settings.json'])}
    atomic_json(result,ROOT/'INFERENCE_PREFLIGHT.json'); print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
