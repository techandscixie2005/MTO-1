"""Read-only physical GPU admission shared by bounded checks and future fits."""
import subprocess,xml.etree.ElementTree as ET
EXPECTED={1:'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800',2:'GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa',
          4:'GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184',6:'GPU-2431a641-8045-a10b-4aa0-2769010e0708'}
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'

def admit(index):
    assert index in EXPECTED
    xml=subprocess.check_output(['nvidia-smi','-q','-x','-i',str(index)],text=True)
    gpu=ET.fromstring(xml).find('gpu');assert gpu.findtext('uuid')==EXPECTED[index]
    assert int(gpu.findtext('fb_memory_usage/used').split()[0])<1000
    assert gpu.findtext('ecc_mode/current_ecc')=='Enabled' and gpu.findtext('gpu_recovery_action')=='None'
    for node in gpu.findall('processes/process_info'):assert 'C' not in node.findtext('type','')
    for period in ('volatile','aggregate'):
        for field in ('sram_uncorrectable_parity','sram_uncorrectable_secded','dram_uncorrectable'):
            assert gpu.findtext('ecc_errors/'+period+'/'+field)=='0',('Missing/nonzero ECC',period,field)
        for node in gpu.findall('ecc_errors/'+period+'/*'):
            if 'uncorrectable' in node.tag:assert node.text=='0'
    for tag in ('remapped_row_pending','remapped_row_failure'):assert gpu.findtext('remapped_rows/'+tag)=='No'
    for tag in ('channel_repair_pending','tpc_repair_pending'):assert gpu.findtext('ecc_errors/'+tag) in ('No','N/A')
    return xml
