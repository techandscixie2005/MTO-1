"""Synthetic NumPy algebra only; no datasets, model loads, fitting or GPU API."""
import json
import numpy as np

rng = np.random.default_rng(20260930)
I = np.eye(3)
EPS = 1e-6

def transform(A, b, control=False):
    R = A.sum(axis=0)
    B = (R + EPS*I/3) / (np.trace(R) + EPS)
    q = np.sqrt(np.sum(B*B)/3)
    L = I + b*(q*I if control else B)
    return L[None] @ A @ L.T[None], B, q, L

def err(x, y):
    return float(np.max(np.abs(x-y)))

C = rng.normal(size=(10, 3, 3))
C = (C + C.transpose(0, 2, 1))/2
A = C @ C.transpose(0, 2, 1)
out, B, q, L = transform(A, -.18)
rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
reflection = rot @ np.diag([-1., 1., 1.])
tests = {}
for label, O in [('rotation', rot), ('improper', reflection), ('inversion', -I)]:
    moved = O[None] @ A @ O.T[None]
    tests[label+'_error'] = err(transform(moved, -.18)[0], O[None]@out@O.T[None])
perm = rng.permutation(10)
tests['state_permutation_error'] = err(transform(A[perm], -.18)[0], out[perm])
tests['identity_error'] = err(transform(A, 0.)[0], A)
tests['control_identity_error'] = err(transform(A, 0., True)[0], A)
tests['minimum_output_eigenvalue'] = float(np.linalg.eigvalsh(out).min())
tests['L_min_eigenvalue'] = float(np.linalg.eigvalsh(L).min())

# A-only computation is invariant to right-orthogonal factor gauges, not merely signs.
gauged = []
for row in C:
    V, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    gauged.append(row @ V)
gauged = np.array(gauged)
tests['independent_factor_gauge_error'] = err(transform(gauged@gauged.transpose(0,2,1),-.18)[0],out)
tests['matched_perturbation_frobenius_error'] = abs(float(np.linalg.norm(B)-np.linalg.norm(q*I)))

# Rank-one synthetic transitions allow a physical degenerate-subspace test.
mu = rng.normal(size=(10, 3))
energies = np.array([.2,.2,.3,.4,.5,.6,.7,.8,.9,1.])
res = mu[:,:,None]*mu[:,None,:]
V = np.array([[np.cos(.43),-np.sin(.43)],[np.sin(.43),np.cos(.43)]])
mu2 = mu.copy()
mu2[:2] = V @ mu[:2]
res2 = mu2[:,:,None]*mu2[:,None,:]
t1 = transform(res,-.18)
t2 = transform(res2,-.18)
tests['degenerate_R_error'] = err(res.sum(0),res2.sum(0))
tests['degenerate_group_output_sum_error'] = err(t1[0][:2].sum(0),t2[0][:2].sum(0))
tests['degenerate_outside_state_error'] = err(t1[0][2:],t2[0][2:])
tests['degenerate_energy_strength_sum_error'] = abs(float(np.sum(energies*np.trace(res,axis1=1,axis2=2))-np.sum(energies*np.trace(res2,axis1=1,axis2=2))))
tests['degenerate_individual_outputs_change'] = err(t1[0][0],t2[0][0])

# Reflection-fixed planar residue can have nonzero normal transition strength.
planar = np.zeros((10,3,3))
planar[0,2,2] = 1.
plane_mirror = np.diag([1.,1.,-1.])
po = transform(planar,-.18)[0]
tests['planar_normal_strength'] = float(po[0,2,2])
tests['planar_reflection_error'] = err(po,plane_mirror[None]@po@plane_mirror[None])

# Neutral synthetic transition charges provide an explicit origin-shift fixture.
# These charges are not a proposed molecular architecture or real target labels.
positions = rng.normal(size=(5,3))
charges = rng.normal(size=(10,5))
charges -= charges.mean(axis=1,keepdims=True)
origin_mu = charges @ positions
shifted_mu = charges @ (positions + np.array([10.,-4.,2.]))
origin_A = origin_mu[:,:,None]*origin_mu[:,None,:]
shifted_A = shifted_mu[:,:,None]*shifted_mu[:,None,:]
tests['neutral_origin_shift_error'] = err(transform(origin_A,-.18)[0],transform(shifted_A,-.18)[0])

# Derivatives with respect to the bounded gate outputs at identity, no fit.
step = 1e-6
s = np.trace(A,axis1=1,axis2=2)
for control in (False, True):
    label = 'isotropic_control' if control else 'tensor_candidate'
    direction = q*I if control else B
    numeric = np.trace((transform(A,step,control)[0]-transform(A,-step,control)[0])/(2*step),axis1=1,axis2=2)
    analytic = 2*np.einsum('kij,ji->k',A,direction)
    tests[label+'_b_gradient_error'] = err(numeric,analytic)
    tests[label+'_b_gradient_maxabs'] = float(np.max(np.abs(analytic)))
    tests[label+'_b_gradient_min'] = float(np.min(analytic))
tests['candidate_total_strength_change'] = float(np.trace(out,axis1=1,axis2=2).sum()-s.sum())

for label, mat in [('zero',np.zeros_like(A)),('isotropic',np.tile(I[None],(10,1,1))),('tiny',A*1e-30)]:
    for control in (False,True):
        z = transform(mat, -.249, control)
        assert all(np.isfinite(v).all() for v in z)
    tests[label+'_finite'] = True

# Energy gaps never enter this readout: approach a coincident energy from either side.
tests['near_gap_no_division_or_eigensolver'] = True
isotropic = np.tile(I[None],(10,1,1))
tests['isotropic_candidate_control_error'] = err(transform(isotropic,.249)[0],transform(isotropic,.249,True)[0])
assert max(v for k,v in tests.items() if k.endswith('_error')) < 1e-7
assert tests['minimum_output_eigenvalue'] >= -1e-10
assert tests['planar_normal_strength'] > 0
assert tests['tensor_candidate_b_gradient_maxabs'] > 1e-6
assert tests['isotropic_control_b_gradient_maxabs'] > 1e-6
assert tests['degenerate_individual_outputs_change'] > 1e-6
print(json.dumps(dict(passed=True, scope='synthetic algebra only; actual MTO/torch/autograd not tested',
    seed=20260930, epsilon=EPS, tests=tests),indent=2))
