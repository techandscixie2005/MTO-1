import pathlib,sys,torch
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen_reference'))
from models_ea import MTOEA
def build(cfg,stats):
    torch.manual_seed(cfg['seed'])
    if cfg['model']=='mto':return MTOEA(cfg,stats)
    raise RuntimeError('DetaNet E/f training configuration is awaiting user confirmation')
