"""Synthetic-only compatibility gate tests; no current seed/scratch predictions."""
import json,tempfile,time
from pathlib import Path
import numpy as np
import seed_mask_compat as c

def fails(fn,label):
 try:fn()
 except RuntimeError:return label
 raise AssertionError("accepted "+label)
def run():
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/"seed.npz"
  rows={"indices":np.array([3,9],dtype=np.int64),"ids":np.array([17,18],dtype=np.int64),"truth":np.ones((2,10),dtype=np.float64),"E_true":np.full((2,10),4.0,dtype=np.float64),"mask":np.ones((2,10),dtype=np.bool_)}
  base={"indices":rows["indices"],"ids":rows["ids"],"f_true":rows["truth"],"E_true":rows["E_true"],"f":np.full((2,10),.5,dtype=np.float64)}
  def put(**changes):np.savez(p,**(base|changes));return c.H(p)
  digest=put();assert np.array_equal(c.load_seed(p,rows,digest),base["f"])
  checks=[fails(lambda:c.load_seed(p,rows,"0"*64),"hash"),fails(lambda:c.load_seed(p,{**rows,"mask":np.zeros((2,10),dtype=bool)},digest),"not-all-valid reference mask")]
  for label,changes in (("ids",{"ids":np.array([17,99])}),("truth",{"f_true":np.zeros((2,10))}),("E truth",{"E_true":np.zeros((2,10))}),("nonfinite",{"f":np.full((2,10),np.nan)}),("wrong optional mask",{"mask_f_true":np.zeros((2,10),dtype=bool)})):
   d=put(**changes);checks.append(fails(lambda:c.load_seed(p,rows,d),label))
  called=[]
  orig=c.ORIGINAL
  try:
   c.ORIGINAL=lambda *a:called.append(a) or "original"
   assert c.dispatch(p,rows,digest)=="original" and len(called)==1
  finally:c.ORIGINAL=orig
  checks.append("nonseed original loader retained")
 return {"passed":True,"time":time.time(),"classification":"synthetic only; no active/completed family outcome arrays opened","adapter_sha256":c.H(c.__file__),"protocol_sha256":c.H(c.HERE/"COMPATIBILITY_PROTOCOL.md"),"checks":checks,"test_access":False}
if __name__=="__main__":
 out=run();p=c.HERE/"COMPATIBILITY_PREFLIGHT.json";p.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8");print(json.dumps({"passed":True,"checks":len(out["checks"]),"adapter_sha256":out["adapter_sha256"]}))
