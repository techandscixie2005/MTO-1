"""Physical GPU health and occupancy for short MTO architecture runs."""
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ADMISSION=Path("/home/inspur/MTO-1/research/oscillator_r2_20260928/resource_audit/ADMISSION_REPORT.json")
def snapshot(index):
    root=ET.fromstring(subprocess.check_output(["nvidia-smi","-q","-x"],text=True))
    g=root.findall("gpu")[int(index)]
    ecc=g.find("ecc_errors")
    values={}
    for scope in ("volatile","aggregate"):
        values[scope]={c.tag:int(c.text) if c.text and c.text.isdigit() else c.text
                       for c in ecc.find(scope)}
    rr=g.find("remapped_rows")
    values["remap"]={c.tag:c.text for c in rr if c.tag!="row_remapper_histogram"}
    values["repair"]={k:ecc.findtext(k) for k in ("channel_repair_pending","tpc_repair_pending")}
    uuid=g.findtext("uuid")
    apps=[]
    text=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader"],text=True)
    for line in text.splitlines():
        bits=[s.strip() for s in line.split(",")]
        if len(bits)==2 and bits[0]==uuid and bits[1].isdigit():apps.append(int(bits[1]))
    return {"index":int(index),"uuid":uuid,"memory_MiB":int(g.findtext("fb_memory_usage/used").split()[0]),
            "temperature_C":g.findtext("temperature/gpu_temp"),"ecc":values,"apps":apps}
def eligible(row,guarded=False):
    e=row["ecc"];r=e["remap"]
    if row["apps"] or row["memory_MiB"]>200:return False
    for scope in ("aggregate","volatile"):
        for name,value in e[scope].items():
            if "uncorrectable" in name and value != 0:return False
        if e[scope].get("sram_threshold_exceeded","No")!="No":return False
    if any(r[k]!="No" for k in ("remapped_row_pending","remapped_row_failure")):return False
    if r["remapped_row_unc"]!="0" or r["remapped_row_corr"]!="0":return False
    if any(v!="No" for v in e["repair"].values()):return False
    if guarded:
        if row["index"]!=5:return False
        admission=json.loads(ADMISSION.read_text())
        assert admission["admission"]["gpu5_guarded_candidate"] is True
        pinned=admission["after"]["5"]["ecc"]
        for scope in ("aggregate","volatile","remap"):
            if e[scope]!=pinned[scope]:return False
    else:
        for scope in ("aggregate","volatile"):
            for name,value in e[scope].items():
                if "correctable" in name and value != 0:return False
    return True
def unchanged(before,after):
    return before["uuid"]==after["uuid"] and before["ecc"]==after["ecc"]

