#!/usr/bin/env python3
"""Stable block-merged FP64 moments and fixed-λ ridge solvers."""
import json,math
import numpy as np
import torch
from features import cfg
class Moments:
    def __init__(self,d):
        self.d=d;self.n=0
        self.mean=np.zeros(d,dtype=np.float64)
        self.mean_r=0.
        self.M=np.zeros((d,d),dtype=np.float64)
        self.c=np.zeros(d,dtype=np.float64)
        self.rr=0.
        self.minimum=np.full(d,np.inf,dtype=np.float64)
        self.maximum=np.full(d,-np.inf,dtype=np.float64)
    def add(self,X,r,device="cpu"):
        X=np.asarray(X,dtype=np.float64);r=np.asarray(r,dtype=np.float64)
        assert X.ndim==2 and X.shape[1]==self.d and r.shape==(len(X),)
        assert np.isfinite(X).all() and np.isfinite(r).all()
        n=len(X);assert n>0
        self.minimum=np.minimum(self.minimum,X.min(0))
        self.maximum=np.maximum(self.maximum,X.max(0))
        if device=="cuda":
            x=torch.as_tensor(X,device="cuda");y=torch.as_tensor(r,device="cuda")
            mx=x.mean(0);my=y.mean()
            xc=x-mx;yc=y-my
            M=(xc.T@xc).cpu().numpy()
            c=(xc.T@yc).cpu().numpy()
            rr=float((yc@yc).cpu())
            mb=mx.cpu().numpy();rb=float(my.cpu())
        else:
            mb=X.mean(0);rb=float(r.mean())
            xc=X-mb;yc=r-rb
            M=xc.T@xc;c=xc.T@yc;rr=float(yc@yc)
        if self.n==0:
            self.n=n;self.mean=mb;self.mean_r=rb
            self.M=M;self.c=c;self.rr=rr;return
        total=self.n+n
        dx=mb-self.mean;dy=rb-self.mean_r
        factor=self.n*n/total
        self.M+=M+factor*np.outer(dx,dx)
        self.c+=c+factor*dx*dy
        self.rr+=rr+factor*dy*dy
        self.mean+=dx*n/total
        self.mean_r+=dy*n/total
        self.n=total
    def finalize(self):
        assert self.n>0 and np.isfinite(self.M).all()
        symmetry=float(np.max(np.abs(self.M-self.M.T)))
        scale=max(float(np.max(np.abs(self.M))),1.)
        assert symmetry/scale<1e-11,("moment symmetry",symmetry/scale)
        M=.5*(self.M+self.M.T)
        constant=self.minimum==self.maximum
        variance=np.diag(M)/self.n
        assert np.all(variance[~constant]>0)
        assert np.all(variance[constant]>=-1e-20)
        std=np.ones(self.d,dtype=np.float64)
        std[~constant]=np.sqrt(variance[~constant])
        active=~constant
        C=np.zeros((self.d,self.d),dtype=np.float64)
        c=np.zeros(self.d,dtype=np.float64)
        C[np.ix_(active,active)]=M[np.ix_(active,active)]/self.n/np.outer(std[active],std[active])
        c[active]=self.c[active]/self.n/std[active]
        C=.5*(C+C.T)
        return {"n":self.n,"mean":self.mean.copy(),"std":std,"active":active,
            "constant_values":self.minimum[constant].copy(),"constant_indices":np.flatnonzero(constant),
            "mean_r":self.mean_r,"C":C,"c":c,"r_variance":self.rr/self.n,
            "raw_centered_symmetry_relative":symmetry/scale,
            "minimum":self.minimum.copy(),"maximum":self.maximum.copy()}
def solve_one(stats,ncols,lam):
    assert ncols in (129,657) and lam==.001
    active=stats["active"][:ncols]
    ix=np.flatnonzero(active)
    C=stats["C"][np.ix_(ix,ix)]
    c=stats["c"][ix]
    assert C.shape==(len(ix),len(ix)) and np.isfinite(C).all() and np.isfinite(c).all()
    symmetry=float(np.max(np.abs(C-C.T))) if len(ix) else 0.
    assert symmetry<1e-10
    A=C+lam*np.eye(len(ix),dtype=np.float64)
    eig=np.linalg.eigvalsh(C) if len(ix) else np.empty(0,dtype=np.float64)
    if len(ix):
        assert float(eig.min())>=-1e-9,("non-PSD covariance",float(eig.min()))
        L=np.linalg.cholesky(A)
        w=np.linalg.solve(L.T,np.linalg.solve(L,c))
        independent=np.linalg.solve(A,c)
        assert np.allclose(w,independent,rtol=1e-9,atol=1e-10)
        normal=float(np.linalg.norm(A@w-c)/max(float(np.linalg.norm(c)),1e-12))
        assert normal<=1e-10,("normal equation residual",normal)
    else:
        w=np.empty(0,dtype=np.float64);normal=0.
    full=np.zeros(ncols,dtype=np.float64);full[ix]=w
    J=float(stats["r_variance"]-2*w@c+w@C@w+lam*(w@w))
    baseline=float(stats["r_variance"]+stats["mean_r"]**2)
    assert J<=baseline+1e-10
    edf=float(np.sum(eig/(eig+lam))) if len(eig) else 0.
    condition=float((eig.max()+lam)/(eig.min()+lam)) if len(eig) else 1.
    return {"ncols":ncols,"active_count":int(active.sum()),"active_indices":ix,
        "w":full,"intercept_centered":float(stats["mean_r"]),"objective":J,
        "zero_correction_baseline_objective":baseline,"normal_residual":normal,
        "covariance_symmetry_error":symmetry,
        "min_covariance_eigenvalue":float(eig.min()) if len(eig) else 0.,
        "max_covariance_eigenvalue":float(eig.max()) if len(eig) else 0.,
        "condition_estimate":condition,"effective_degrees_of_freedom":edf,
        "coefficient_norm":float(np.linalg.norm(w))}
def solve_both(stats,lam=.001):
    a=solve_one(stats,129,lam);b=solve_one(stats,657,lam)
    assert b["objective"]<=a["objective"]+1e-10,("nested regularized objective",a["objective"],b["objective"])
    assert stats["C"][:129,:129].shape==(129,129)
    return {"A":a,"B":b}
def row_matrix(cache,gram,a,b):
    h=cache["h"][a:b].reshape(-1,128).astype(np.float64)
    base=cache["base64"][a:b].reshape(-1,1)
    g=np.asarray(gram[a:b]).reshape(-1,528)
    X=np.concatenate((h,base,g),axis=1)
    assert X.shape==((b-a)*10,657) and np.isfinite(X).all()
    return X
def residual(cache,a,b):
    y=cache["f_true"][a:b].reshape(-1)
    base=cache["base64"][a:b].reshape(-1)
    r=(y-base)/cfg()["f_std_train"]
    assert np.isfinite(r).all()
    return r

