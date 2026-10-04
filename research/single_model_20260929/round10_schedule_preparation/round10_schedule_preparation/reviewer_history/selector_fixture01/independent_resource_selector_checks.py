"""Five temporary, stdlib-only selector scenarios. No live probes or GPU work."""
import builtins
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

# The selector does not call process_identity. Avoid importing monitoring state.
with patch.dict(sys.modules, {'monitor': types.SimpleNamespace(process_identity=lambda pid: None)}):
    import run_gpu_preflight as wrapper

ROOT = Path(__file__).resolve().parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def healthy_xml(gpu, busy=False, wrong_uuid=False):
    fields = ''.join('<%s>0</%s>' % (x, x) for x in
                     ('sram_uncorrectable_parity', 'sram_uncorrectable_secded', 'dram_uncorrectable'))
    uuid = 'GPU-wrong' if wrong_uuid else wrapper.EXPECTED[gpu]
    memory = 2000 if busy else 10
    return ('<nvidia_smi_log><gpu><uuid>%s</uuid><fb_memory_usage><used>%s MiB</used></fb_memory_usage>'
            '<ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode><gpu_recovery_action>None</gpu_recovery_action>'
            '<processes></processes><ecc_errors><volatile>%s</volatile><aggregate>%s</aggregate>'
            '<channel_repair_pending>No</channel_repair_pending><tpc_repair_pending>No</tpc_repair_pending></ecc_errors>'
            '<remapped_rows><remapped_row_pending>No</remapped_row_pending><remapped_row_failure>No</remapped_row_failure>'
            '</remapped_rows></gpu></nvidia_smi_log>') % (uuid, memory, fields, fields)

def scenario(name, *, locked=(), busy=(), bad_uuid=(), unexpected=(), selected=None, probes=()):
    locks = {}; observed = []; original_open = builtins.open

    class FakeLock:
        def __init__(self, gpu): self.gpu=gpu; self.closed=False; self.held=False
        def fileno(self): return 100+self.gpu
        def close(self): self.closed=True; self.held=False

    def fake_open(path, *args, **kwargs):
        value = str(path)
        if value.startswith('/tmp/mto_pouter_gpu_'):
            gpu = int(value.removeprefix('/tmp/mto_pouter_gpu_').removesuffix('.lock'))
            assert gpu in (1,2,4,6) and gpu not in locks
            lock = FakeLock(gpu); locks[gpu]=lock; return lock
        return original_open(path, *args, **kwargs)

    def flock(fd, flags):
        gpu=fd-100; assert flags == wrapper.fcntl.LOCK_EX | wrapper.fcntl.LOCK_NB
        if gpu in locked: raise BlockingIOError('synthetic occupied lock')
        locks[gpu].held=True

    def probe(command, **kwargs):
        gpu=int(command[-1])
        assert command==['nvidia-smi','-q','-x','-i',str(gpu)] and kwargs=={'text':True}
        assert locks[gpu].held and not locks[gpu].closed
        observed.append(gpu)
        if gpu in unexpected: raise OSError('synthetic probe failure')
        return healthy_xml(gpu, gpu in busy, gpu in bad_uuid)

    with tempfile.TemporaryDirectory(prefix='round10_reviewer_selector_') as tmp:
        out=Path(tmp)
        with patch('builtins.open', fake_open), patch.object(wrapper.fcntl,'flock',flock), patch.object(wrapper.subprocess,'check_output',probe):
            if unexpected:
                try: wrapper.select_locked_device(out)
                except OSError: pass
                else: raise AssertionError('Unexpected probe failure must stop')
            else:
                lock,gpu,xml=wrapper.select_locked_device(out)
                assert gpu==selected
                if selected is None: assert lock is None and xml is None
                else:
                    assert lock is locks[selected] and lock.held and not lock.closed
                    assert xml==healthy_xml(selected)
                receipt=json.loads((out/'RESOURCE_SELECTION.json').read_text())
                assert receipt['ordered_allowed']==[1,2,4,6]
                for entry in receipt['observations']:
                    if entry['probe_path'] is None:
                        assert entry['gpu'] in locked and entry['probe_sha256'] is None
                    else:
                        p=out/entry['probe_path']; assert sha(p)==entry['probe_sha256']
                        assert p.read_text()==healthy_xml(entry['gpu'],entry['gpu'] in busy,entry['gpu'] in bad_uuid)
                if selected is not None: lock.close()
        assert observed==list(probes), (name,observed)
        assert all(lock.closed for lock in locks.values())
        assert list(locks)==list((1,2,4,6)[:len(locks)])
    return name

assert not (ROOT/'INDEPENDENT_RESOURCE_SELECTOR_CHECKS.json').exists()
checks=[scenario('first_available_selected_and_lock_retained',selected=1,probes=[1]),
        scenario('locked1_busy2_then4_preserves_rejected_XML',locked=[1],busy=[2],selected=4,probes=[2,4]),
        scenario('one_pass_none_eligible_releases_all',busy=[1,2,4,6],selected=None,probes=[1,2,4,6]),
        scenario('unexpected_probe_error_stops_without_later_device',unexpected=[1],probes=[1]),
        scenario('UUID_mismatch_rejected_then2_selected',bad_uuid=[1],selected=2,probes=[1,2])]
result={'passed':True,'checks':checks,'checks_count':5,'live_resource_probes':0,'model_or_data_access':False,
        'optimizer_updates':0,'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in
        ('run_gpu_preflight.py','resources.py','common.py','independent_resource_selector_checks.py')}}
(ROOT/'INDEPENDENT_RESOURCE_SELECTOR_CHECKS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
