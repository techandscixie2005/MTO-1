"""Read only small checkpoint metadata; never export tensors or optimizer arrays."""
import hashlib
import json
import tarfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
HISTORY=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    run=SOURCE/'runs/mto_eta0';records={}
    for name in ('best.pt','last.pt'):
        p=run/name;ck=torch.load(p,map_location='cpu',weights_only=False)
        state=ck.get('state',{});opt=ck.get('optimizer',{})
        records[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
            'top_level_keys':list(ck),'epoch':ck.get('epoch'),
            'state':{k:state.get(k) for k in ('epoch','cursor','best_epoch','steps','bad_epochs') if k in state},
            'optimizer_state_count':len(opt.get('state',{})),
            'learning_rates':[v['lr'] for v in opt.get('param_groups',[])],
            'scheduler_last_epoch':ck.get('scheduler',{}).get('last_epoch')}
        del ck
    assert records['best.pt']['epoch']==33 and records['best.pt']['optimizer_state_count']==0
    assert records['last.pt']['state']['epoch']==237 and records['last.pt']['state']['cursor']==0
    assert records['last.pt']['scheduler_last_epoch']==236
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    selected={row['epoch']:{'epoch':row['epoch'],'lr':row['lr'],'original_LE_plus_Ls':row['val'][0],
        'best_epoch':row['best_epoch']} for row in history if row['epoch'] in (33,34,236)}
    refs=[SOURCE/'trainer.py',run/'history.jsonl',HISTORY/'SCIENTIFIC_AUDIT_AND_PROTOCOL.md',
          HISTORY/'architecture/ARCHITECTURE_PROTOCOL.md',Path(__file__)]
    report={'passed':True,'selected_epoch33_matching_Adam_available':False,
        'exact_preserved_moment_control_feasible_from_retained_selected_checkpoint':False,
        'checkpoints':records,'run_checkpoint_inventory':[str(p) for p in sorted(run.rglob('*.pt'))],
        'original_trajectory_reference':selected,
        'original_trajectory_is_not_matched_gentle_continuation':True,
        'historical_reset_evidence':['SCIENTIFIC_AUDIT_AND_PROTOCOL.md:21','architecture/ARCHITECTURE_PROTOCOL.md:43'],
        'scope':'Source-run metadata and available eta0-named path inventory; not a claim about unknown external backups.',
        'no_optimizer_reconstruction_no_model_inference_no_fit':True,'no_tensor_or_array_export':True,
        'source_hashes':{str(p):sha(p) for p in refs}}
    backup=SOURCE/'backups/mto_completed_20260927.tar.gz'
    archived=[]
    with tarfile.open(backup,'r:gz') as stream:
        for name in ('best.pt','last.pt'):
            member=stream.getmember('runs/mto_eta0/'+name)
            h=hashlib.sha256()
            with stream.extractfile(member) as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            assert h.hexdigest()==records[name]['sha256']
            archived.append({'member':member.name,'sha256':h.hexdigest(),'identical_to_current':True})
    report['completed_backup_checked']={'path':str(backup),'members':archived,'archive_not_downloaded':True}
    (ROOT/'OPTIMIZER_METADATA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'epoch33_optimizer_saved':False,'retained_full_state_completed_epoch':236}))

if __name__=='__main__':main()
