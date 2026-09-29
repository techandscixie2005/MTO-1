"""Small CPU mathematical counterexamples; no repository model or data loads."""
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def projector(n):return np.eye(n)-np.ones((n,n))/n
def covariance(r,k):
    c=projector(len(r));p=c@k@c
    return p,r.T@p@r
def kernel(r,w=None,length=1.):
    d2=np.square(r[:,None,:]-r[None,:,:]).sum(-1)
    k=np.exp(-d2/(2*length**2))
    return k if w is None else w[:,None]*k*w[None,:]
def err(a,b):return float(np.max(np.abs(a-b)))

def main():
    rng=np.random.default_rng(20260930)
    r=rng.normal(size=(6,3));w=np.exp(rng.normal(size=6)*.2)
    k=kernel(r,w);p,a=covariance(r,k)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    reflection=np.diag([-1.,1.,1.]);perm=rng.permutation(6)
    checks={}
    for name,o in [('rotation',q),('reflection',reflection),('inversion',-np.eye(3))]:
        _,at=covariance(r@o.T,kernel(r@o.T,w))
        checks[name]=err(at,o@a@o.T)
    _,at=covariance(r+np.array([8.,-3.,4.]),kernel(r+np.array([8.,-3.,4.]),w))
    checks['translation']=err(at,a)
    pp,ap=covariance(r[perm],kernel(r[perm],w[perm]))
    checks['atom_permutation_A']=err(ap,a)
    checks['atom_permutation_P']=err(pp,p[perm][:,perm])
    checks['neutrality']=float(np.max(np.abs(p@np.ones(6))))
    checks['minimum_P_eigenvalue']=float(np.linalg.eigvalsh(p).min())
    checks['minimum_A_eigenvalue']=float(np.linalg.eigvalsh(a).min())
    assert max(v for key,v in checks.items() if not key.startswith('minimum'))<1e-12
    assert checks['minimum_P_eigenvalue']>-1e-12 and checks['minimum_A_eigenvalue']>=0
    dimer=np.array([[-1.,0.,0.],[1.,0.,0.]])
    _,identity=covariance(dimer,np.eye(2));_,gaussian=covariance(dimer,kernel(dimer))
    invariant_features=np.array([[1.,2.,4.],[1.,2.,4.]])
    _,lowrank=covariance(dimer,invariant_features@invariant_features.T)
    b=kernel(dimer);_,pair_factor=covariance(dimer,b@b.T)
    assert identity[0,0]==2 and gaussian[0,0]>0 and pair_factor[0,0]>0 and err(lowrank,np.zeros((3,3)))==0
    # Point charges cannot span a dipole perpendicular to all nuclear positions.
    plane=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]])
    _,plane_a=covariance(plane,kernel(plane))
    target=np.diag([0.,0.,1.])
    assert np.all(plane_a[2]==0) and np.all(plane_a[:,2]==0)
    plane_reflection=np.diag([1.,1.,-1.])
    assert err(plane@plane_reflection.T,plane)==0
    assert err(plane_reflection@target@plane_reflection.T,target)==0
    _,atom_a=covariance(np.array([[2.,3.,4.]]),np.ones((1,1)))
    assert not np.any(atom_a)
    # A neutral, nonzero covariance can be invisible in its dipole projection.
    line=np.array([[-1.,0.,0.],[0.,0.,0.],[1.,0.,0.]])
    dark=np.array([1.,-2.,1.]);pd=np.outer(dark,dark)
    _,dark_a=covariance(line,pd)
    _,line_a=covariance(line,np.eye(3));_,line_plus=covariance(line,np.eye(3)+pd)
    diagonal=float(sum(pd[i,i]*np.dot(line[i],line[i]) for i in range(3)))
    cross=float(sum(pd[i,j]*np.dot(line[i],line[j]) for i in range(3) for j in range(3) if i!=j))
    assert err(dark_a,np.zeros((3,3)))<1e-12 and err(line_a,line_plus)<1e-12
    # Positive entries and symmetry alone do not make a PSD kernel.
    bad=np.array([[1.,.9,.9],[.9,1.,.1],[.9,.1,1.]])
    bad_p,_=covariance(line,bad)
    assert np.linalg.eigvalsh(bad_p).min()<0
    # Smooth no-eigenconstruction has finite derivatives at repeated K eigenvalues.
    torch.set_num_threads(1)
    tt=torch.eye(3,dtype=torch.float64,requires_grad=True)
    rr=torch.tensor(line,dtype=torch.float64);cc=torch.tensor(projector(3),dtype=torch.float64)
    aa=rr.T@cc@tt@tt.T@cc@rr
    derivative=torch.autograd.grad(aa.trace(),tt)[0]
    assert torch.isfinite(derivative).all()
    zero=torch.zeros((3,3),dtype=torch.float64,requires_grad=True)
    az=rr.T@cc@zero@zero.T@cc@rr
    grad_zero=torch.autograd.grad((az-torch.eye(3,dtype=torch.float64)).square().sum(),zero)[0]
    assert torch.count_nonzero(grad_zero)==0
    delta=np.eye(3)*.1;base=np.eye(3)
    report={'passed':True,'CPU_only':True,'dataset_or_model_loaded':False,'fit':False,
        'symmetry_and_PSD':checks,
        'centrosymmetric_dimer':{'P_C_trace_A':float(identity.trace()),'Gaussian_trace_A':float(gaussian.trace()),
            'invariant_fixed_rank_scalar_factor_trace_A':float(lowrank.trace()),'atom_indexed_pair_factor_trace_A':float(pair_factor.trace())},
        'span_restriction':{'planar_out_of_plane_Azz':float(plane_a[2,2]),
            'reflection_invariant_target_Azz':1.,'irreducible_squared_error_lower_bound_for_that_target':1.,
            'linear_perpendicular_Ayy_Azz':[float(line_a[1,1]),float(line_a[2,2])],
            'one_atom_trace_A':float(atom_a.trace()),'general_rank_bound':'rank(A)<=min(3,rank(CR),rank(P))'},
        'coherent_dark_example':{'neutral_charge_sum':float(dark.sum()),'nonzero_P_Frobenius_norm':float(np.linalg.norm(pd)),
            'trace_diagonal_terms':diagonal,'trace_pair_cross_terms':cross,'trace_A':float(dark_a.trace())},
        'nonidentifiability':{'different_neutral_P_same_A_max_abs':err(line_a,line_plus),
            'P_difference_norm':float(np.linalg.norm(pd)),'factor_right_orthogonal_gauge':'B and B O give same P'},
        'kernel_constraints':{'symmetric_positive_entry_kernel_min_eig':float(np.linalg.eigvalsh(bad).min()),
            'centered_bad_kernel_min_eig':float(np.linalg.eigvalsh(bad_p).min()),
            'BBt_no_eigendecomposition_derivative_finite_at_identity':True},
        'residual_rejection':{'additive_PSD_trace_before':float(base.trace()),'after':float((base+delta).trace()),
            'zero_initialized_factor_loss_gradient_max_abs':float(grad_zero.abs().max())},
        'verdict':'Reject general replacement. Atom-indexed covariance avoids scalar-factor inversion collapse but nuclear-span and physical-identifiability constraints remain. No model design or fit proposed.'}
    (ROOT/'SYNTHETIC_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
