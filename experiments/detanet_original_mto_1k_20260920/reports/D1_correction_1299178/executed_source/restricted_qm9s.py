"""Read a bounded engineering sample without executing pickle globals or loading all tensors.

Only the observed PyG container and tensor reconstruction schema is supported.
Unknown globals fail closed. Input dataset and original split are never modified.
"""
import argparse
import collections
import hashlib
import json
import pickle
import struct
import zipfile
from pathlib import Path


class Container:
    pass


class Storage:
    def __init__(self, key, dtype, size):
        self.key, self.dtype, self.size = key, dtype, size


class Tensor:
    def __init__(self, storage, offset, shape, stride, *unused):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = shape, stride


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('torch_geometric.data.data', 'Data'): Container,
            ('torch_geometric.data.storage', 'GlobalStorage'): Container,
            ('torch._utils', '_rebuild_tensor_v2'): Tensor,
            ('torch', 'FloatStorage'): 'f',
            ('torch', 'DoubleStorage'): 'd',
            ('torch', 'LongStorage'): 'q',
            ('torch', 'IntStorage'): 'i',
            ('torch', 'BoolStorage'): '?',
            ('collections', 'OrderedDict'): collections.OrderedDict,
        }
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported pickle global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        kind, dtype, key, location, size = pid
        if kind != 'storage':
            raise ValueError(f'Unsupported persistent identifier {kind}')
        return Storage(key, dtype, size)


def decode(t, archive, prefix):
    if not isinstance(t, Tensor):
        return t
    s = t.storage
    raw = archive.read(f'{prefix}data/{s.key}')
    width = struct.calcsize('<' + s.dtype)
    def at(dim, offset):
        if dim == len(t.shape):
            return struct.unpack_from('<' + s.dtype, raw, offset * width)[0]
        return [at(dim + 1, offset + i * t.stride[dim]) for i in range(t.shape[dim])]
    return at(0, t.offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--count', type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.source) as archive:
        meta = next(n for n in archive.namelist() if n.endswith('data.pkl'))
        prefix = meta[:-8]
        byteorder = prefix + 'byteorder'
        if byteorder in archive.namelist() and archive.read(byteorder) != b'little':
            raise ValueError('Only little-endian archives supported')
        with archive.open(meta) as stream:
            records = MetadataReader(stream).load()
        print('metadata records', len(records), flush=True)
        samples = []
        allowed = ['z', 'pos', 'number', 'smile', 'tran_energy', 'tran_dipole']
        for record in records[:args.count]:
            mapping = record.__dict__.get('_store', record).__dict__.get('_mapping', record.__dict__)
            samples.append({key: decode(mapping[key], archive, prefix) for key in allowed if key in mapping})
        manifest = {
            'source': str(Path(args.source).resolve()),
            'source_bytes': Path(args.source).stat().st_size,
            'source_mtime_ns': Path(args.source).stat().st_mtime_ns,
            'source_records': len(records), 'extracted_records': len(samples),
            'purpose': 'engineering_only_not_benchmark; exclude these identities from future official test',
            'extraction': 'restricted metadata unpickler; only selected tensor payloads read',
            'source_fields': sorted(mapping),
            'units': {'pos': None, 'tran_energy': None, 'tran_dipole': None},
            'protocol': None, 'gauge': None, 'orientation_alignment': 'unverified',
            'independent_oscillator_strength': False,
            'formal_training_allowed': False,
        }
    payload = json.dumps(samples, indent=2)
    manifest['sample_sha256'] = hashlib.sha256(payload.encode()).hexdigest()
    (out / 'engineering_sample.json').write_text(payload)
    (out / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print('first sample', json.dumps(samples[0])[:2500], flush=True)


if __name__ == '__main__':
    main()
