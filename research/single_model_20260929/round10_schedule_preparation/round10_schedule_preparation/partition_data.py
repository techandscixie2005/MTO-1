"""Selected-row NPZ decoding; no full numeric target-array load or test API."""
import hashlib
from pathlib import Path
import zipfile
import numpy as np
from common import ROOT,DATA,SPLIT,PINS,sha,read

FIELDS={'dataset.npz':frozenset(('ids','z','pos','edge','E','A')),
        'raw_labels.npz':frozenset(('ids','E','A','f','mask_E','mask_A','mask_f'))}
TARGETS=frozenset(('E','A','f','mask_E','mask_A','mask_f'))
def index_sha(rows):return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()
def convert_block(raw,dtype,shape,global_rows):
    """The sole numeric target decoding site, instrumented by synthetic preflight."""
    assert len(global_rows)==shape[0]
    return np.frombuffer(raw,dtype=dtype).copy().reshape(shape)

class Partition:
    def __init__(self,name,validation_authorized=False):
        if name not in ('train','val'):raise PermissionError('No test partition target reader exists')
        if name=='val' and not validation_authorized:raise PermissionError('Validation reader requires verified production evaluation context')
        for filename in ('SPLIT_MANIFEST.json','INDEPENDENT_SPLIT_VERIFICATION.json'):
            assert sha(SPLIT/filename)==PINS[filename]
        manifest=read(SPLIT/'SPLIT_MANIFEST.json');verification=read(SPLIT/'INDEPENDENT_SPLIT_VERIFICATION.json')
        assert manifest['passed'] and verification['passed']
        assert verification['split_manifest_sha256']==PINS['SPLIT_MANIFEST.json']
        entry=manifest['arrays'][name+'_indices.npy'];assert sha(entry['path'])==entry['sha256']
        self.indices=np.load(entry['path'],allow_pickle=False)
        assert self.indices.dtype==np.int64 and self.indices.ndim==1
        assert np.all(np.diff(self.indices)>0) and index_sha(self.indices)==entry['content_sha256']
        id_entry=manifest['arrays']['ids.npy'];assert sha(id_entry['path'])==id_entry['sha256']
        all_ids=np.load(id_entry['path'],allow_pickle=False)
        assert index_sha(all_ids)==id_entry['content_sha256']
        self.ids=all_ids[self.indices];self.name=name;self.count=manifest['counts'][name]
        self.corpus_size=len(all_ids);assert len(self.indices)==self.count
        self.allowed=np.zeros(self.corpus_size,dtype=bool);self.allowed[self.indices]=True
        self.ledger=[];self.verified_sources={}

    def selected(self,archive_name,field,rows=None):
        if archive_name not in FIELDS or field not in FIELDS[archive_name]:raise PermissionError('Unapproved source/member')
        rows=self.indices if rows is None else np.asarray(rows)
        # All validation is before opening a ZIP member or interpreting values.
        if rows.ndim!=1 or rows.dtype!=np.int64 or len(rows)==0:raise ValueError('Nonempty int64 rows required')
        if np.any(rows<0) or np.any(rows>=self.corpus_size) or np.any(np.diff(rows)<=0):raise ValueError('Rows must be unique, ascending and in range')
        if not self.allowed[rows].all():raise PermissionError('Target/geometry rows outside the approved partition')
        path=DATA/archive_name
        if archive_name not in self.verified_sources:
            assert sha(path)==PINS[archive_name];self.verified_sources[archive_name]=PINS[archive_name]
        chunks=[];actual_rows=[];numeric_calls=0
        with zipfile.ZipFile(path) as archive:
            member=field+'.npy';assert archive.namelist().count(member)==1
            with archive.open(member) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('Unreviewed NPY version')
                assert not fortran and not dtype.hasobject and shape[0]==self.corpus_size
                row_bytes=int(np.prod(shape[1:]))*dtype.itemsize;origin=stream.tell()
                # Consecutive selected rows share one decode; no unselected row is included.
                boundaries=np.r_[0,np.flatnonzero(np.diff(rows)!=1)+1,len(rows)]
                for start,end in zip(boundaries[:-1],boundaries[1:]):
                    block=rows[start:end];assert np.array_equal(block,np.arange(block[0],block[0]+len(block)))
                    stream.seek(origin+int(block[0])*row_bytes);raw=stream.read(len(block)*row_bytes)
                    assert len(raw)==len(block)*row_bytes
                    chunks.append(convert_block(raw,dtype,(len(block),)+shape[1:],block))
                    actual_rows.append(block.copy());numeric_calls+=1
        decoded=np.concatenate(actual_rows);assert np.array_equal(decoded,rows)
        output=np.concatenate(chunks,axis=0);assert len(output)==len(rows)
        self.ledger.append({'archive':str(path),'archive_sha256':PINS[archive_name],'field':field,
            'partition':self.name,'numeric_rows':len(decoded),'numeric_global_indices_sha256':index_sha(decoded),
            'numeric_decode_blocks':numeric_calls,'is_target_or_mask':field in TARGETS,
            'shape':list(output.shape),'dtype':str(output.dtype)})
        return output

    def load_training_arrays(self,rows=None):
        rows=self.indices if rows is None else rows
        actual_ids=self.selected('dataset.npz','ids',rows)
        expected=self.ids[np.searchsorted(self.indices,rows)];assert np.array_equal(actual_ids,expected)
        result={key:self.selected('dataset.npz',key,rows) for key in ('z','pos','edge','E','A')}
        assert np.array_equal(self.selected('raw_labels.npz','ids',rows),expected)
        for key in ('f','mask_E','mask_A','mask_f'):result[key]=self.selected('raw_labels.npz',key,rows)
        result['raw_E']=self.selected('raw_labels.npz','E',rows)
        assert all(result[k].all() for k in ('mask_E','mask_A','mask_f')),'Unexpected invalid labels; no silent exclusions'
        assert all(np.isfinite(result[k]).all() for k in ('E','A','f','raw_E','pos'))
        return result
