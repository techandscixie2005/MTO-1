"""Common evaluation, with validation selection frozen before test access."""
import hashlib,json,math,pathlib,time
import numpy as np
import torch
from dataset import Data,ROOT
from model_factory import build
from trainer import setup,atomic_json
EV_PER_HARTREE=27.211386245988
GRID=np.arange(1051,dtype=np.float64)*.02
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def regression(pred,true,mask):
    p=pred[mask].astype(np.float64);t=true[mask].astype(np.float64);d=p-t
    assert np.isfinite(p).all() and np.isfinite(t).all()
    ss=np.sum((t-t.mean())**2)
    return dict(count=int(t.size),MAE=float(np.abs(d).mean()),RMSE=float(np.sqrt(np.mean(d*d))),R2=float(1-np.sum(d*d)/ss) if ss else None)
def tensor_error(pred,true,mask):
    d=(pred-true)[mask].astype(np.float64);fro=np.linalg.norm(d,axis=(-2,-1))
    return dict(count=int(len(fro)),element_MAE=float(np.abs(d).mean()),element_RMSE=float(np.sqrt(np.mean(d*d))),
        Frobenius_MAE=float(fro.mean()),Frobenius_RMSE=float(np.sqrt(np.mean(fro*fro))))
def broaden(e,f,mask):
    e=np.where(mask,e,0);f=np.where(mask,f,0)
    return (f[...,None]*np.exp(-.5*((GRID-e[...,None])/.2)**2)/(.2*np.sqrt(2*np.pi))).sum(1)
def metrics(pred,truth):
    e=pred['E'].astype('float64');f=pred['f'].astype('float64');et=truth['E'];ft=truth['f']
    me=truth['mask_E'];mf=truth['mask_f'];ma=truth['mask_A']
    spec_mse=[];spec_mae=[]
    for start in range(0,len(e),128):
        sl=slice(start,start+128);mask=me[sl]&mf[sl]
        d=broaden(e[sl],f[sl],mask)-broaden(et[sl],ft[sl],mask)
        spec_mse.extend(np.mean(d*d,axis=1));spec_mae.extend(np.mean(np.abs(d),axis=1))
    out=dict(E_eV=regression(e,et,me),f=regression(f,ft,mf),spectrum=dict(MSE=float(np.mean(spec_mse)),MAE=float(np.mean(spec_mae))),
        predicted_f_negative_count=int((f[mf]<0).sum()),predicted_E_negative_count=int((e[me]<0).sum()))
    per=[]
    if 'A' in pred:
        a=pred['A'].astype('float64');at=truth['A'];s=np.trace(a,axis1=-2,axis2=-1);st=np.trace(at,axis1=-2,axis2=-1)
        q=a-s[...,None,None]*np.eye(3)/3;qt=at-st[...,None,None]*np.eye(3)/3
        out.update(trace=regression(s,st,ma),Q=tensor_error(q,qt,ma),A=tensor_error(a,at,ma),
            minimum_A_eigenvalue=float(np.linalg.eigvalsh(a[ma]).min()),A_symmetry_max_abs=float(np.abs(a-a.swapaxes(-1,-2)).max()))
    for k in range(10):
        row=dict(state=k+1,E_eV=regression(e[:,k],et[:,k],me[:,k]),f=regression(f[:,k],ft[:,k],mf[:,k]))
        if 'A' in pred:row.update(trace=regression(s[:,k],st[:,k],ma[:,k]),Q=tensor_error(q[:,k],qt[:,k],ma[:,k]),A=tensor_error(a[:,k],at[:,k],ma[:,k]))
        per.append(row)
    out['per_state']=per
    return out,np.asarray(spec_mse),np.asarray(spec_mae)
@torch.no_grad()
def predict(model,data,idx,cfg):
    accum={k:[] for k in (('E','f','A') if cfg['model']=='mto' else ('E','f'))}
    model.eval()
    for start in range(0,len(idx),64):
        x,_=data.batch(idx[start:start+64]);result=model(**x)
        if cfg['model']=='mto':
            e,a=result;accum['A'].append(a.cpu().numpy())
            # Derive f in float64 from the saved predictions, so exports reproduce metrics exactly.
            en=e.cpu().numpy();an=a.cpu().numpy().astype('float64');f=(2/3)*(en.astype('float64')/EV_PER_HARTREE)*np.trace(an,axis1=-2,axis2=-1)
        else:
            en=result[:,:10].cpu().numpy();f=result[:,10:].cpu().numpy()
        accum['E'].append(en);accum['f'].append(f)
    return {k:np.concatenate(v) for k,v in accum.items()}
def make_model(cfg,stats):
    if cfg['model']=='mto':return build(cfg,stats)
    from detanet_adapter import build_detanet
    return build_detanet(cfg)
def main():
    setup(11);campaign=json.loads((ROOT/'campaign.json').read_text())
    assert campaign['detanet_configuration_status']=='confirmed' and len(campaign['runs'])==4,'Four-run configuration is not frozen'
    specs=campaign['runs'];names=[s['name'] for s in specs]
    assert all((ROOT/'runs'/n/'FIT_COMPLETE.json').exists() for n in names),'Training incomplete'
    data=Data();raw=np.load(ROOT/'data/raw_labels.npz');results={};checkpoints={};cfgs={}
    for name in names:
        out=ROOT/'runs'/name;cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text());cfgs[name]=cfg
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg
        model=make_model(cfg,data.stats).cuda();model.load_state_dict(ck['model'])
        if cfg['model']=='mto':
            from trainer import fingerprint as source_fingerprint
        else:
            from trainer_detanet import fingerprint as source_fingerprint
        assert ck['fingerprint']==source_fingerprint(), 'Training code fingerprint changed before evaluation'
        checkpoints[name]=dict(best_pt_sha256=sha(out/'best.pt'),best_epoch=ck['epoch'],best_step=ck.get('steps'),config_sha256=sha(ROOT/'configs'/f'{name}.json'))
        idx=data.parts['val'];pred=predict(model,data,idx,cfg);truth={k:raw[k][idx] for k in ('E','A','f','mask_E','mask_A','mask_f')}
        result,pmse,pmae=metrics(pred,truth);results[name]={'val':result,'checkpoint':checkpoints[name]}
        np.savez_compressed(out/'val_predictions.npz',ids=data.ids[idx],indices=idx,**pred,**{k+'_true':v for k,v in truth.items()},spectrum_MSE_per_molecule=pmse,spectrum_MAE_per_molecule=pmae)
        atomic_json(result,out/'val_metrics.json');del model;torch.cuda.empty_cache()
    mto=[n for n in names if cfgs[n]['model']=='mto']
    ranking=sorted(mto,key=lambda n:results[n]['val']['spectrum']['MSE'])
    selection=dict(time=time.time(),classification=campaign['classification'],primary_metric='common validation spectrum MSE of own-L_eta-best checkpoint',
        selected_mto=ranking[0],mto_ranking=ranking,validation_spectrum_MSE={n:results[n]['val']['spectrum']['MSE'] for n in names},checkpoints=checkpoints,
        protocol_sha256=sha(ROOT/'campaign.json'),test_used_for_selection=False)
    selpath=ROOT/'reports/selection_before_test.json'
    if selpath.exists():
        previous=json.loads(selpath.read_text());assert previous['checkpoints']==checkpoints and previous['selected_mto']==ranking[0] and previous['protocol_sha256']==selection['protocol_sha256']
    else:atomic_json(selection,selpath)
    # No model predictions or metrics on test are produced above this point.
    atomic_json(dict(time=time.time(),selection_sha256=sha(selpath)),ROOT/'reports/test_evaluation_started.json')
    for name in names:
        cfg=cfgs[name];out=ROOT/'runs'/name;ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        assert sha(out/'best.pt')==checkpoints[name]['best_pt_sha256']
        model=make_model(cfg,data.stats).cuda();model.load_state_dict(ck['model'])
        for part in ('test','train'):
            idx=data.parts[part];pred=predict(model,data,idx,cfg);truth={k:raw[k][idx] for k in ('E','A','f','mask_E','mask_A','mask_f')}
            result,pmse,pmae=metrics(pred,truth);results[name][part]=result
            np.savez_compressed(out/f'{part}_predictions.npz',ids=data.ids[idx],indices=idx,**pred,**{k+'_true':v for k,v in truth.items()},spectrum_MSE_per_molecule=pmse,spectrum_MAE_per_molecule=pmae)
            atomic_json(result,out/f'{part}_metrics.json')
        del model;torch.cuda.empty_cache()
    atomic_json(results,ROOT/'reports/results.json')
    report(results,selection,campaign)
    atomic_json(dict(time=time.time(),runs=names,selection_sha256=sha(selpath),results_sha256=sha(ROOT/'reports/results.json')),ROOT/'reports/ANALYSIS_COMPLETE.json')
def report(results,selection,campaign):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    histories={}
    for n in results:
        hist=[json.loads(s) for s in (ROOT/'runs'/n/'history.jsonl').read_text().splitlines()];histories[n]=hist
        ep=[r['epoch'] for r in hist]
        axes[0].plot(ep,[r['train'][0] for r in hist],label=n)
        axes[1].plot(ep,[r['val'][0] for r in hist],label=n)
        axes[2].plot(ep,[r['val_spectrum']['MSE'] for r in hist],label=n)
    for ax,title in zip(axes,['Training objective (different definitions)','Validation objective (different definitions)','Common validation spectrum MSE']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'reports/learning_curves.png',dpi=180);plt.close(fig)
    lines=['# QM9S η 与原版 DetaNet E/f 对照','',
        '本轮为追加探索性实验：旧测试集此前已被分析。仅 seed=11，不推断跨种子稳定优势。',
        '三组 MTO 各自按验证 Lη 选 best；跨 η 按该 best 的统一验证光谱 MSE 选型。测试指标不用于继续调参。',
        f"验证集选定 MTO：**{selection['selected_mto']}**。",'',
        '| 组别 | best轮次 | 验证光谱MSE | 测试光谱MSE | 测试光谱MAE | 测试E MAE(eV) | 测试E RMSE(eV) | 测试E R² | 测试f MAE | 测试f RMSE | 测试f R² |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in results.items():
        t=r['test'];e=t['E_eV'];f=t['f']
        vals=[n,str(r['checkpoint']['best_epoch'])]+[f'{v:.7g}' for v in (r['val']['spectrum']['MSE'],t['spectrum']['MSE'],t['spectrum']['MAE'],e['MAE'],e['RMSE'],e['R2'],f['MAE'],f['RMSE'],f['R2'])]
        lines.append('| '+' | '.join(vals)+' |')
    lines+=['','## MTO 张量误差','', '| 组别 | 迹 MAE | 迹 RMSE | Q Frobenius RMSE | A Frobenius RMSE |','|---|---:|---:|---:|---:|']
    for n,r in results.items():
        t=r['test']
        if 'A' in t:lines.append(f"| {n} | {t['trace']['MAE']:.7g} | {t['trace']['RMSE']:.7g} | {t['Q']['Frobenius_RMSE']:.7g} | {t['A']['Frobenius_RMSE']:.7g} |")
    ref=results['mto_eta1']
    lines+=['','## 对比解读','']
    for n in ('mto_eta0','mto_eta01'):
        v=results[n]['val']['spectrum']['MSE']/ref['val']['spectrum']['MSE']-1
        t=results[n]['test']['spectrum']['MSE']/ref['test']['spectrum']['MSE']-1
        lines.append(f'- {n} 相对 η=1：验证光谱 MSE 变化 {v:+.2%}，测试光谱 MSE 变化 {t:+.2%}。本结论仅描述本轮种子和划分。')
    lines+=['','## 评价定义与限制','',
        '- 光谱网格 0–21 eV（1051 点），步长0.02 eV，单位积分 Gaussian σ=0.2 eV；有限区间不再归一化。MSE/MAE 对分子及网格点等权平均。',
        '- 真值使用原始 oscillator_strength；MTO f=(2/3)(E/27.211386245988)tr(A)，DetaNet直接使用标量读出f。所有预测均不裁剪。',
        '- E/f 总体R²按全部有效分子/态的全局均值定义；逐态R²另列在各run的metrics.json。',
        '- 迹为tr(A)，Q为最终A的无迹部分；张量Frobenius RMSE定义为sqrt(mean(sum_ij(error²)))，element误差另存JSON。',
        '- 完整训练/验证曲线见history.jsonl与learning_curves.png。train/val/test_predictions.npz包含ID、预测、原始真值、mask及逐分子光谱误差。',
        '- 冻结总体133727个分子，共120355/6686/6686；保留冻结分组划分。与论文的QM9S数据版本和划分不同，不视为严格复现其R²。',
        '- DetaNet采用正文明确的Adam+AMSGrad、lr=1e-3、batch=64、MSE、验证间隔50、降率减半及1e-5/1,000,000停止规则。结合官方Trainer把Epoch解释为更新次数，故每50次更新全验证、最多1m更新；不套用MTO早停。此计数解释和缺失判据补全详见detanet_protocol.md及用户确认，不视为E/f专用配置的严格复现。',
        '- 数据中原始f的打印精度与A派生值不同，零值与所有冻结样本完整保留，未删除离群点。详见data_audit.json。',
        '- GPU健康、启动失败及恢复记录均保留在reports和runs中。']
    (ROOT/'reports/comparison_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
