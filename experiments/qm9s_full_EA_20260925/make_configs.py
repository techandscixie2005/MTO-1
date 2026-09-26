import json,pathlib
ROOT=pathlib.Path(__file__).parent
base=dict(seed=11,model='mto',states=10,mto_channels=16,query_dim=32,router_hidden=128,head_hidden=128,
    initial_energy_min_eV=2.,initial_energy_max_eV=13.,sigma_eV=.2,lr=.001,weight_decay=0.,batch_size=64,
    max_epochs=1000,min_epochs=200,early_patience=150,plateau_patience=50,min_lr=1e-6,grad_clip=5.,
    loss_weights=[1.,1.],dtype='float32',amp=False,tf32=False,energy_initialization='train_state_means',
    optimizer='Adam_AMSGrad',num_threads=2,checkpoint_seconds=600)
variants=[('detanet_reference',0,dict(model='detanet')),
    ('mto_reference',1,{}),('mto_lr3e4',2,dict(lr=.0003)),('mto_lr3e3',4,dict(lr=.003)),
    ('mto_channels32',5,dict(mto_channels=32)),('mto_batch128',6,dict(batch_size=128)),
    ('mto_seed23',7,dict(seed=23))]
(ROOT/'configs').mkdir(exist_ok=True)
for name,gpu,overrides in variants:
    c={**base,**overrides,'name':name,'gpu':gpu}
    (ROOT/'configs'/f'{name}.json').write_text(json.dumps(c,indent=2))
(ROOT/'campaign.json').write_text(json.dumps(dict(runs=[name for name,_,_ in variants],gpu_excluded=[3],split=[.9,.05,.05],
    selection='minimum best validation LE+LA; test evaluated only after ALL fits complete',
    matched_comparison=['detanet_reference','mto_reference'],repeat_seed=['mto_reference','mto_seed23']),indent=2))
