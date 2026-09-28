"""Map ZIP_STORED NumPy members and index only requested TRAIN/validation rows.

This implements the same offset/header procedure as the frozen Round09
geometry_error_audit.npz_member_memmap, without importing its analysis module.
"""
from pathlib import Path
import struct,zipfile
import numpy as np

def member(npz_path,key):
    path=Path(npz_path)
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(key+".npy")
        if info.compress_type!=zipfile.ZIP_STORED:
            raise RuntimeError(f"NPZ member not ZIP_STORED: {path}:{key}")
        with path.open("rb") as stream:
            stream.seek(info.header_offset)
            header=stream.read(30)
            sig,ver,flag,method,mtime,mdate,crc,csize,usize,nlen,elen=struct.unpack(
                "<IHHHHHIIIHH",header)
            if sig!=0x04034b50 or method!=0:
                raise RuntimeError("unexpected NPZ local member header")
            stream.seek(info.header_offset+30+nlen+elen)
            version=np.lib.format.read_magic(stream)
            if version==(1,0):
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version in ((2,0),(3,0)):
                shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise RuntimeError("unsupported NPY member version")
            offset=stream.tell()
    return np.memmap(path,mode="r",dtype=dtype,offset=offset,shape=shape,
                     order="F" if fortran else "C")

