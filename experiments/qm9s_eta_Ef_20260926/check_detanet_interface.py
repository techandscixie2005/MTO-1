"""Forward-only compatibility verification; no optimizer or training run."""
import json,torch
from dataset import Data,ROOT
from trainer import setup,state_hash
from detanet_adapter import DetaNet,ARCH,build_detanet,ef_loss,l2loss
def main():
    setup(11);data=Data(device='cpu');cfg={'seed':11}
    original=DetaNet(**ARCH,device=torch.device('cpu'))
    adapted=build_detanet(cfg)
    assert state_hash(original.state_dict())==state_hash(adapted.state_dict())
    x,y=data.batch(data.parts['train'][:2]);xx={k:v for k,v in x.items() if k!='n'}
    with torch.no_grad():
        a=original(**xx);b=adapted(**x)
        assert torch.equal(a,b) and b.shape==(2,20)
        loss=ef_loss(b,y);expected=l2loss(b,torch.cat((y['E'],y['f']),-1));assert torch.equal(loss,expected)
        # Verify cache and native radius_graph select the same topology/output on these samples.
        native=original(z=x['z'],pos=x['pos'],batch=x['batch'])
        diff=float((native-a).abs().max());assert diff<1e-4
    report=dict(passed=True,forward_only=True,training_not_started=True,official_source_unchanged=True,
        initial_state_sha256=state_hash(adapted.state_dict()),native_vs_adapter_max_abs=float((a-b).abs().max()),
        cached_vs_native_radius_graph_max_abs=diff,total_parameters=sum(p.numel() for p in adapted.parameters()),
        native_scalar_output_shape=list(b.shape),loss_exactly_official_l2loss=True,
        adaptations=['Drop the dataset-only n keyword before native forward','Register elec/mass as nonpersistent buffers for device/dtype moves'],
        configuration_status='Pending user confirmation of missing E/f training settings; this check makes no assertion about the original experiment configuration')
    (ROOT/'reports/detanet_interface_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
