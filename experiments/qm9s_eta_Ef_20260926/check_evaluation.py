"""Small independent checks of evaluator definitions; never reads test labels."""
import json,math
import numpy as np
from evaluate import regression,tensor_error,broaden,metrics,GRID
from dataset import ROOT
def main():
    mask=np.ones((3,10),bool);e=np.tile(np.arange(10)*.5+2,(3,1));f=np.full_like(e,.02)
    a=np.zeros((3,10,3,3));a[...,0,0]=.2
    truth=dict(E=e,A=a,f=f,mask_E=mask,mask_A=mask,mask_f=mask)
    pred=dict(E=e.copy(),A=a.copy(),f=f.copy())
    m,ms,ma=metrics(pred,truth)
    assert m['E_eV']['RMSE']==0 and m['spectrum']['MSE']==0 and m['A']['Frobenius_RMSE']==0
    assert m['E_eV']['R2']==1 and m['f']['R2'] in (None,1.)
    y=broaden(np.array([[5.]]),np.array([[1.]]),np.array([[True]]))[0]
    assert abs(np.trapz(y,GRID)-1)<1e-12
    neg=broaden(np.array([[5.]]),np.array([[-1.]]),np.array([[True]]))[0]
    assert np.array_equal(neg,-y)
    e2=e.copy();e2[0,0]=np.nan;f2=f.copy();f2[0,0]=np.nan;mask[0,0]=False
    assert np.isfinite(broaden(e2,f2,mask)).all()
    r=regression(np.array([2.,3.,4.]),np.array([1.,2.,3.]),np.ones(3,bool))
    assert r['MAE']==1 and r['RMSE']==1 and r['R2']==-.5
    one=np.ones((2,3,3));zero=np.zeros_like(one);t=tensor_error(one,zero,np.ones(2,bool))
    assert t['Frobenius_RMSE']==3 and t['element_RMSE']==1
    (ROOT/'reports/evaluation_checks.json').write_text(json.dumps(dict(passed=True,identity_zero_error=True,unit_integral_gaussian=True,negative_f_not_clipped=True,masked_nans=True,R2_known_case=True,Frobenius_known_case=True),indent=2))
    print('Evaluation definition checks passed')
if __name__=='__main__':main()
