"""Extract observed Gaussian TD labels without imposing a root count or protocol."""
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import math
import pathlib
import re
import shutil
import time
import numpy as np

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[DEde][-+]?\d+)?'
STATE = re.compile(r'Excited State\s+(\d+):\s+(\S+)\s+('+NUM+r')\s+eV\s+('+NUM+r')\s+nm\s+f=\s*('+NUM+r')(?:\s+<S\*\*2>=\s*('+NUM+r'))?')
ORIENT = re.compile(r'(Input|Standard|Z-Matrix) orientation:\s*\n\s*-+\n.*?\n.*?\n\s*-+\n(.*?)\n\s*-+', re.S)
ROUTE = re.compile(r'^\s*-{5,}\s*\n\s*(#[^\n]*(?:\n(?!\s*-{5,})[^\n]*)*)\n\s*-{5,}', re.M)
HEAD = 'Ground to excited state transition electric dipole moments (Au):'
ELECTRONIC = 'Electronic transition elements'
HARTREE_EV = 27.211386245988

def number(x):
    return float(x.replace('D','E').replace('d','e'))

def table(text, heading, columns):
    starts = [m.end() for m in re.finditer(re.escape(heading), text)]
    if len(starts) != 1:
        return {}, len(starts)
    rows = {}
    started = False
    for line in text[starts[0]:].splitlines():
        p = line.split()
        if len(p) == columns + 1 and p[0].isdigit():
            try:
                rows[int(p[0])] = [number(v) for v in p[1:]]
                started = True
            except ValueError:
                if started: break
        elif started:
            break
    return rows, len(starts)

def parse(path):
    raw = path.read_bytes()
    t = raw.decode('utf-8', errors='replace')
    r = dict(molecule_id=int(path.name.split('-')[0]), source_file=path.name,
             source_bytes=len(raw), source_sha256=hashlib.sha256(raw).hexdigest(),
             warnings=[], states=[])
    r['route_sections'] = [' '.join(m.group(1).split()) for m in ROUTE.finditer(t)]
    r['normal_termination_count'] = t.count('Normal termination of Gaussian')
    r['error_termination_count'] = t.count('Error termination')
    r['response_convergence_reported'] = bool(re.search(r'Convergence (?:achieved )?on expansion vectors\.',t))
    r['rpa_flags'] = re.findall(r'\bDoRPA=([TF])', t)
    r['normal_termination'] = r['normal_termination_count'] > 0 and t.rfind('Normal termination of Gaussian') > t.rfind('Error termination')
    cm = re.findall(r'Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)', t)
    r['charge'] = int(cm[-1][0]) if cm else None
    r['multiplicity'] = int(cm[-1][1]) if cm else None
    r['formula_printed'] = (re.findall(r'Stoichiometry\s+(\S+)',t) or [None])[-1]
    r['basis_printed'] = (re.findall(r'Standard basis:\s*([^\n]+)', t) or [None])[-1]
    r['program_version'] = (re.findall(r'Gaussian \d+:\s*([^\n]+)',t) or [None])[-1]
    r['expected_natoms_printed'] = int((re.findall(r'NAtoms=\s*(\d+)',t) or [0])[0])
    dip, dip_blocks = table(t, HEAD, 5)
    vel, vel_blocks = table(t, 'Ground to excited state transition velocity dipole moments (Au):', 5)
    mag, _ = table(t, 'Ground to excited state transition magnetic dipole moments (Au):', 3)
    r['electric_dipole_table_count'] = dip_blocks
    # Do not silently join repeated calculations or choose a root from another job.
    matches = list(STATE.finditer(t))
    first_label = min([m.start() for m in matches]+[t.find(HEAD) if HEAD in t else len(t)])
    geometries = [m for m in ORIENT.finditer(t) if m.start() < first_label]
    r['orientation_table_count_before_labels'] = len(geometries)
    r['atomic_numbers'] = []
    r['positions_angstrom'] = []
    r['coordinate_frame'] = None
    if geometries:
        standards = [m for m in geometries if m.group(1) == 'Standard']
        g = standards[-1] if standards else geometries[-1]
        r['coordinate_frame'] = g.group(1)
        for line in g.group(2).splitlines():
            p = line.split()
            if len(p) == 6:
                r['atomic_numbers'].append(int(p[1]))
                r['positions_angstrom'].append([number(x) for x in p[3:6]])
    # Preserve the higher-precision printed transition block as an auxiliary field.
    # Only rows 2--4 that agree with the labeled electric table are promoted as mu.
    electronic = {}
    if t.count(ELECTRONIC) == 1:
        block = t.split(ELECTRONIC,1)[1].split('1\\1\\',1)[0]
        ids = []
        for line in block.splitlines():
            p = line.split()
            if p and all(x.isdigit() for x in p):
                ids = [int(x) for x in p]
            elif ids and len(p) == len(ids)+1 and p[0].isdigit():
                try:
                    vals = [number(x) for x in p[1:]]
                    for sid,v in zip(ids,vals): electronic.setdefault(sid,{})[int(p[0])] = v
                except ValueError:
                    pass
    counts = collections.Counter(int(m[1]) for m in matches)
    for m in matches:
        sid = int(m[1])
        s = dict(state_index=sid, symmetry_printed=m[2], energy_eV=number(m[3]),
                 wavelength_nm=number(m[4]), oscillator_strength=number(m[5]),
                 spin_squared=number(m[6]) if m[6] else None)
        if sid in dip:
            mu = dip[sid][:3]
            s.update(transition_dipole_table_au=mu, dipole_strength_table_au2=dip[sid][3],
                     oscillator_strength_table=dip[sid][4], transition_dipole_au=mu,
                     transition_dipole_source='labeled_electric_table_4_decimals')
            e = electronic.get(sid,{})
            if all(k in e for k in (2,3,4)):
                precise = [e[k] for k in (2,3,4)]
                if max(abs(a-b) for a,b in zip(mu,precise)) <= 5.1e-5:
                    s['transition_dipole_au'] = precise
                    s['transition_dipole_source'] = 'electronic_transition_elements_rows_2_3_4_validated_against_electric_table'
                else:
                    r['warnings'].append('higher_precision_dipole_mismatch_state_'+str(sid))
            mu = s['transition_dipole_au']
            s['A_au2'] = [[x*y for y in mu] for x in mu]
            s['oscillator_strength_from_E_mu'] = 2/3*s['energy_eV']/HARTREE_EV*sum(x*x for x in mu)
            s['f_absolute_residual'] = abs(s['oscillator_strength_from_E_mu'] - s['oscillator_strength'])
            if abs(s['oscillator_strength_table']-s['oscillator_strength']) > 1e-8:
                r['warnings'].append('oscillator_strength_table_mismatch_state_'+str(sid))
        if sid in vel:
            s.update(velocity_dipole_au=vel[sid][:3], velocity_dipole_strength_au2=vel[sid][3], oscillator_strength_velocity=vel[sid][4])
        if sid in mag: s['magnetic_dipole_au'] = mag[sid]
        s['unique_root_index'] = counts[sid] == 1
        s['scalar_fields_present'] = all(math.isfinite(s[k]) for k in ('energy_eV','wavelength_nm','oscillator_strength'))
        s['vector_fields_present'] = 'transition_dipole_au' in s and all(math.isfinite(x) for x in s['transition_dipole_au'])
        r['states'].append(s)
    if len(r['route_sections']) != 1: r['warnings'].append('not_exactly_one_route_section')
    if dip_blocks != 1: r['warnings'].append('not_exactly_one_electric_dipole_table')
    if len(counts) != len(matches): r['warnings'].append('duplicate_root_indices')
    if not r['normal_termination']: r['warnings'].append('no_final_normal_termination')
    if not r['response_convergence_reported']: r['warnings'].append('response_convergence_marker_absent')
    if not matches: r['warnings'].append('no_excited_state_labels')
    if not r['atomic_numbers'] or len(r['atomic_numbers']) != r['expected_natoms_printed']:
        r['warnings'].append('missing_or_incomplete_geometry')
    if any(s['energy_eV'] <= 0 or s['oscillator_strength'] < 0 for s in r['states']):
        r['warnings'].append('nonpositive_energy_or_negative_strength')
    if any(a['energy_eV'] > b['energy_eV'] for a,b in zip(r['states'],r['states'][1:])):
        r['warnings'].append('energies_not_monotonic')
    if dip and set(dip) != set(counts): r['warnings'].append('dipole_root_set_mismatch')
    # Masks distinguish successful, unambiguous records; they do not delete records.
    r['unambiguous_single_calculation'] = len(r['route_sections']) == 1 and dip_blocks == 1 and len(counts) == len(matches) and len(geometries)==1
    r['geometry_present'] = bool(r['atomic_numbers']) and len(r['atomic_numbers']) == r['expected_natoms_printed']
    return r

def safe_parse(path):
    try: return parse(path)
    except Exception as ex:
        raw = path.read_bytes()
        return dict(molecule_id=int(path.name.split('-')[0]),source_file=path.name,
                    source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),
                    warnings=['parse_exception:'+type(ex).__name__+':'+str(ex)],states=[],
                    atomic_numbers=[],positions_angstrom=[],route_sections=[])

def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def write_shard(records, out, index):
    name = 'part-%05d'%index
    with gzip.open(out/'records'/(name+'.jsonl.gz'),'wt',encoding='utf-8',compresslevel=3) as f:
        for r in records: f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    n = len(records)
    a = max([len(r['atomic_numbers']) for r in records]+[1])
    k = max([len(r['states']) for r in records]+[1])
    arr = dict(molecule_id=np.array([r['molecule_id'] for r in records],dtype=np.int32),
        atomic_numbers=np.zeros((n,a),np.int16),positions_angstrom=np.full((n,a,3),np.nan),
        atom_mask=np.zeros((n,a),bool),state_index=np.zeros((n,k),np.int16),
        state_mask=np.zeros((n,k),bool),scalar_label_mask=np.zeros((n,k),bool),
        vector_label_mask=np.zeros((n,k),bool),normal_termination=np.zeros(n,bool),
        response_convergence_reported=np.zeros(n,bool),unambiguous_single_calculation=np.zeros(n,bool),
        geometry_present=np.zeros(n,bool),charge=np.full(n,np.nan),multiplicity=np.full(n,np.nan))
    for key in ('energy_eV','wavelength_nm','oscillator_strength','oscillator_strength_from_E_mu','dipole_strength_table_au2','oscillator_strength_velocity'):
        arr[key]=np.full((n,k),np.nan)
    for key in ('transition_dipole_au','transition_dipole_table_au','velocity_dipole_au','magnetic_dipole_au'):
        arr[key]=np.full((n,k,3),np.nan)
    arr['A_au2']=np.full((n,k,3,3),np.nan)
    for i,r in enumerate(records):
        na = len(r['atomic_numbers'])
        arr['atomic_numbers'][i,:na] = r['atomic_numbers']
        if na: arr['positions_angstrom'][i,:na] = r['positions_angstrom']
        arr['atom_mask'][i,:na] = True
        for key in ('normal_termination','response_convergence_reported','unambiguous_single_calculation','geometry_present'):
            arr[key][i]=r.get(key,False)
        for key in ('charge','multiplicity'):
            if r.get(key) is not None: arr[key][i]=r[key]
        for j,s in enumerate(r['states']):
            arr['state_mask'][i,j]=True
            arr['scalar_label_mask'][i,j]=s['scalar_fields_present'] and s['unique_root_index']
            arr['vector_label_mask'][i,j]=s['vector_fields_present'] and s['unique_root_index']
            for key in ('state_index','energy_eV','wavelength_nm','oscillator_strength','oscillator_strength_from_E_mu','dipole_strength_table_au2','oscillator_strength_velocity','transition_dipole_au','transition_dipole_table_au','velocity_dipole_au','magnetic_dipole_au','A_au2'):
                if key in s: arr[key][i,j]=s[key]
    np.savez_compressed(out/'arrays'/(name+'.npz'),**arr)
    return dict(name=name,records=n,first_id=records[0]['molecule_id'],last_id=records[-1]['molecule_id'],max_atoms=a,max_observed_states=k)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--shard-size',type=int,default=5000)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--executor',choices=['thread','process'],default='process')
    args=ap.parse_args(); src=pathlib.Path(args.source);out=pathlib.Path(args.output)
    out.mkdir(parents=True,exist_ok=args.resume)
    (out/'records').mkdir(exist_ok=args.resume);(out/'arrays').mkdir(exist_ok=args.resume)
    if pathlib.Path(__file__).resolve() != (out/'extract_qm9s.py').resolve():
        shutil.copy2(__file__,out/'extract_qm9s.py')
    files=sorted(src.glob('*-td.log'),key=lambda p:int(p.name.split('-')[0]))
    if args.limit: files=files[:args.limit]
    routes=collections.Counter(); warnings=collections.Counter(); roots=collections.Counter()
    elements=collections.Counter();frames=collections.Counter();sources=collections.Counter();rpa=collections.Counter();versions=collections.Counter()
    normal=converged=unambiguous=geometry=scalar=vector=0
    residuals=[];strengths=[];energies=[];weak=[];strong=[];relative=[]
    shards=[];pending=[];total_bytes=0;start=time.time()
    with (out/'source_manifest.jsonl').open('w',encoding='utf-8') as mf, (out/'anomalies.jsonl').open('w',encoding='utf-8') as af:
        executor=concurrent.futures.ProcessPoolExecutor if args.executor=='process' else concurrent.futures.ThreadPoolExecutor
        with executor(max_workers=args.workers) as pool:
            # Bounded work avoids submitting 133k futures at once.
            for offset in range(0,len(files),args.shard_size):
                current_files=files[offset:offset+args.shard_size]
                jp=out/'records'/('part-%05d.jsonl.gz'%len(shards))
                cached=None
                if args.resume and jp.exists():
                    try:
                        with gzip.open(jp,'rt',encoding='utf-8') as f: cached=[json.loads(line) for line in f]
                        if [r['source_file'] for r in cached] != [p.name for p in current_files]: cached=None
                    except (OSError,EOFError,json.JSONDecodeError): cached=None
                if cached is not None:
                    # Re-evaluate records affected by the two equivalent convergence
                    # phrasings or a route-like string inside Gaussian's archive.
                    fix=[i for i,r in enumerate(cached) if any(w in r['warnings'] for w in ('response_convergence_marker_absent','not_exactly_one_route_section'))]
                    for i,r in zip(fix,pool.map(safe_parse,[current_files[i] for i in fix])): cached[i]=r
                    iterator=cached
                else:
                    iterator=pool.map(safe_parse,current_files)
                for r in iterator:
                    pending.append(r); total_bytes+=r['source_bytes']
                    mf.write(json.dumps({key:r.get(key) for key in ('molecule_id','source_file','source_bytes','source_sha256')})+'\n')
                    routes.update(r['route_sections']);warnings.update(r['warnings']);roots[len(r['states'])]+=1
                    elements.update(str(x) for x in set(r['atomic_numbers']));frames[str(r.get('coordinate_frame'))]+=1
                    rpa.update(r.get('rpa_flags',[]));versions[str(r.get('program_version'))]+=1
                    normal+=r.get('normal_termination',False);converged+=r.get('response_convergence_reported',False)
                    unambiguous+=r.get('unambiguous_single_calculation',False);geometry+=r.get('geometry_present',False)
                    if r['warnings']: af.write(json.dumps({'molecule_id':r['molecule_id'],'source_file':r['source_file'],'warnings':r['warnings']})+'\n')
                    for s in r['states']:
                        scalar+=s['scalar_fields_present'];vector+=s['vector_fields_present']
                        energies.append(s['energy_eV']);strengths.append(s['oscillator_strength'])
                        if 'f_absolute_residual' in s:
                            v=s['f_absolute_residual'];residuals.append(v)
                            (weak if s['oscillator_strength']<1e-3 else strong).append(v)
                            if s['oscillator_strength']>=1e-3: relative.append(v/s['oscillator_strength'])
                        sources[s.get('transition_dipole_source','missing')]+=1
                shards.append(write_shard(pending,out,len(shards)));pending=[]
                print(json.dumps({'processed':min(offset+args.shard_size,len(files)),'total':len(files),'seconds':round(time.time()-start,1),'warning_counts':dict(warnings)}),flush=True)
    def stats(x):
        return dict(count=len(x),min=float(np.min(x)),max=float(np.max(x)),mean=float(np.mean(x)),p50=float(np.quantile(x,.5)),p99=float(np.quantile(x,.99))) if x else {'count':0}
    ids=[int(p.name.split('-')[0]) for p in files]
    manifest=dict(schema_version='1.0',source_directory=str(src),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_log_count=len(files),source_bytes=total_bytes,observed_id_range=[min(ids),max(ids)],missing_ids_within_range=sorted(set(range(min(ids),max(ids)+1))-set(ids)),
        duplicate_ids=[k for k,v in collections.Counter(ids).items() if v>1],observed_state_count_distribution=dict(roots),
        route_section_distribution=dict(routes),program_version_distribution=dict(versions),rpa_flag_distribution=dict(rpa),
        coordinate_frame_distribution=dict(frames),molecules_by_atomic_number=dict(elements),normal_termination_count=normal,
        response_convergence_reported_count=converged,unambiguous_single_calculation_count=unambiguous,complete_geometry_count=geometry,
        scalar_label_count=scalar,vector_label_count=vector,transition_dipole_source_distribution=dict(sources),warnings=dict(warnings),
        energy_eV=stats(energies),oscillator_strength=stats(strengths),f_absolute_residual=stats(residuals),
        f_absolute_residual_weak_f_below_1e_minus_3=stats(weak),f_absolute_residual_strong=stats(strong),f_relative_residual_strong=stats(relative),
        units={'positions':'angstrom','energy':'eV','wavelength':'nm','transition_dipole':'atomic units','A':'atomic units squared','oscillator_strength':'dimensionless'},
        label_policy='All observed labeled roots retained in printed order. No predefined number of states, broadening, train/test split, or sample exclusion.',
        coordinate_policy='Last Standard orientation before labels, otherwise last available orientation. Multiple geometry tables are flagged by unambiguous_single_calculation=False.',
        connectivity='Not supplied by extraction; no bonds or SMILES inferred. Geometry optimization protocol not inferred from TD geom=check.',
        derived_fields={'A_au2':'outer product of transition_dipole_au','oscillator_strength_from_E_mu':'(2/3)*(energy_eV/27.211386245988)*sum(mu**2)'},
        precision_policy='Retain electric table verbatim numerically. Use electronic transition elements rows 2,3,4 only when every component agrees with the labeled electric table within 0.000051 au. Energy and f remain from Excited State lines.',
        references=['https://pyscf.org/user/tddft.html','https://pyscf.org/_modules/pyscf/gw/bse.html'],shards=shards)
    dump(out/'dataset_manifest.json',manifest)
    (out/'dataset_audit.md').write_text('# 实际日志提取审计\n\n'
        f'- 来源：`{src}`\n- 日志：{len(files):,}；未删除任何记录。\n'
        f'- 正常结束：{normal:,}；明确报告响应收敛：{converged:,}；完整几何：{geometry:,}。\n'
        f'- 实际态数分布：`{dict(roots)}`。\n- 标量标签：{scalar:,}；向量标签：{vector:,}。\n'
        f'- 实际 route：`{dict(routes)}`。\n- 坐标框架：`{dict(frames)}`。\n'
        f'- 异常统计：`{dict(warnings)}`；详见 anomalies.jsonl。\n'
        f'- 振子强度与能量/偶极的一致性绝对残差：`{stats(residuals)}`。\n\n'
        '保留全部实际打印的跃迁，不把迭代求解过程中的临时 Root 当作最终光谱。原始波长、能量、f、电偶极表和可用的速度/磁偶极均保留。'
        '没有施加总纲里的态数、展宽、划分或筛选设定。此处光谱是日志计算出的离散跃迁，不声称覆盖无限高能激发态或实验谱。\n\n'
        '坐标来自同一 TD 日志的 orientation 表，单位 Å，原子顺序不变。优化协议、连接关系和分子身份去重未从其他来源推断。'
        '长度规范电偶极采用带标题的原子单位表；经逐分量核对后使用末尾 transition elements 中的高精度分量，并保留原表。'
        'A=μμᵀ 是明确标记的派生量，f_from_E_mu 仅用于一致性审计，不替换原始 f。暗态的打印零保持为零，缺失值使用 NaN/mask。\n\n'
        '一致性关系参考 [PySCF 官方实现](https://pyscf.org/_modules/pyscf/gw/bse.html)，规范说明参考 [PySCF TDDFT 文档](https://pyscf.org/user/tddft.html)。\n',encoding='utf-8')
    print(json.dumps({'complete':True,'output':str(out),'logs':len(files),'labels':scalar,'seconds':round(time.time()-start,1)}),flush=True)

if __name__=='__main__': main()
