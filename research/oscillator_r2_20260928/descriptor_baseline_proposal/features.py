"""Frozen 803-dimensional Z+coordinate descriptor schema. No labels or split access."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import deque

import numpy as np

ELEMENTS=(1,6,7,8,9)  # H,C,N,O,F
RADII={1:0.31,6:0.76,7:0.71,8:0.66,9:0.57}  # Angstrom
PAIR_KEYS=tuple(itertools.combinations_with_replacement(ELEMENTS,2))
CENTERS=np.arange(0.5,8.0+0.25,0.5,dtype=np.float64)
WIDTH=0.5
FEATURES=803
GRAPH_FACTOR=1.3

def _json(value):
    return json.dumps(value,separators=(",",":"),ensure_ascii=True).encode("utf-8")

def _hash(value):
    return hashlib.sha256(_json(value)).digest()

def names():
    names=[f"count_Z{z}" for z in ELEMENTS]+["atoms_total","heavy_atoms"]
    names += [f"atom_hash_{i:03d}" for i in range(512)]
    names += [f"edge_Z{a}_Z{b}" for a,b in PAIR_KEYS]
    names += [f"degree_{x}" for x in ("0","1","2","3","4","5plus")]
    names += ["components","cycle_rank"]
    for a,b in PAIR_KEYS:
        names.append(f"pair_Z{a}_Z{b}_count")
        names.extend(f"pair_Z{a}_Z{b}_radial_{c:.1f}" for c in CENTERS)
    names += ["cov_eig1","cov_eig2","cov_eig3","radius_gyration","q2","q3"]
    assert len(names)==FEATURES and len(set(names))==FEATURES
    return tuple(names)

def _components(neighbors):
    n=len(neighbors);seen=set();components=0
    for start in range(n):
        if start in seen:continue
        components+=1;todo=deque([start]);seen.add(start)
        while todo:
            for v in neighbors[todo.popleft()]:
                if v not in seen:seen.add(v);todo.append(v)
    return components

def extract(z_padded,pos_padded):
    """Return FP32 feature vector and TRAIN-only threshold diagnostic."""
    z=np.asarray(z_padded);pos=np.asarray(pos_padded)
    if z.ndim!=1 or pos.shape!=(len(z),3):
        raise ValueError("Z/coordinate shape")
    if z.dtype.kind not in "iu":raise ValueError("atomic numbers must be integral")
    active=z!=0
    n=int(np.count_nonzero(active))
    if n<1 or n>29:raise ValueError("nonempty atom count")
    atoms=z[active].astype(np.int64,copy=False)
    if not set(map(int,atoms)).issubset(RADII):raise ValueError("unsupported element")
    xyz=np.asarray(pos[active],dtype=np.float64)
    if not np.isfinite(xyz).all():raise ValueError("nonfinite coordinate")
    pair_dist=np.linalg.norm(xyz[:,None,:]-xyz[None,:,:],axis=-1)
    vec=np.zeros(FEATURES,dtype=np.float64)
    for i,a in enumerate(ELEMENTS):vec[i]=int(np.count_nonzero(atoms==a))
    vec[5]=n;vec[6]=n-int(np.count_nonzero(atoms==1))
    neighbors=[[] for _ in range(n)]
    edge_counts={k:0 for k in PAIR_KEYS}
    pair_dists={k:[] for k in PAIR_KEYS}
    min_boundary=float("inf");near_boundary=0;edges=0
    for i in range(n):
        for j in range(i+1,n):
            a,b=sorted((int(atoms[i]),int(atoms[j])),key=ELEMENTS.index)
            key=(a,b);d=float(pair_dist[i,j])
            pair_dists[key].append(d)
            boundary=GRAPH_FACTOR*(RADII[int(atoms[i])]+RADII[int(atoms[j])])
            gap=abs(d-boundary);min_boundary=min(min_boundary,gap)
            near_boundary+=int(gap<=1e-10)
            if d<=boundary:
                neighbors[i].append(j);neighbors[j].append(i)
                edge_counts[key]+=1;edges+=1
    labels=[str(int(a)) for a in atoms]
    for radius in range(3):
        for label in labels:
            digest=_hash([radius,label])
            bucket=int.from_bytes(digest,"big")%512
            vec[7+bucket]+=1
        if radius<2:
            labels=[_hash([radius+1,labels[i],sorted(labels[j] for j in neighbors[i])]).hex()
                    for i in range(n)]
    offset=7+512
    vec[offset:offset+15]=[edge_counts[key] for key in PAIR_KEYS]
    offset+=15
    for i,degree in enumerate(map(len,neighbors)):
        vec[offset+min(degree,5)]+=1
    components=_components(neighbors)
    vec[offset+6]=components
    vec[offset+7]=edges-n+components
    offset+=8
    for key in PAIR_KEYS:
        ds=np.asarray(pair_dists[key],dtype=np.float64)
        vec[offset]=len(ds)
        if len(ds):
            vec[offset+1:offset+17]=np.exp(
                -0.5*np.square((ds[:,None]-CENTERS[None,:])/WIDTH)).mean(axis=0)
        offset+=17
    centered=xyz-xyz.mean(axis=0)
    covariance=centered.T@centered/n
    eig=np.linalg.eigvalsh(covariance)[::-1]
    trace=float(np.trace(covariance))
    if eig[-1]<-1e-12*max(trace,1.0):raise ValueError("covariance eigenvalue")
    eig=np.maximum(eig,0.0)
    eig0=float(eig[0])
    vec[offset:offset+3]=eig
    vec[offset+3]=math.sqrt(float(eig.sum()))
    vec[offset+4]=float(eig[1]/eig0) if eig0>1e-20 else 0.0
    vec[offset+5]=float(eig[2]/eig0) if eig0>1e-20 else 0.0
    if offset+6!=FEATURES or not np.isfinite(vec).all():raise ValueError("descriptor length/finite")
    out=vec.astype(np.float32)
    if not np.isfinite(out).all():raise ValueError("FP32 descriptor finite")
    return out,{"atoms":n,"edges":edges,"components":components,
                "near_graph_threshold_pairs":near_boundary,
                "minimum_graph_threshold_gap_angstrom":min_boundary if n>1 else None}

SCHEMA_NAMES=names()

