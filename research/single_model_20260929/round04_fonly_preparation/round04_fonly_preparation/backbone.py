"""Original architecture import and tensor hashing; no checkpoint/data read on import."""
import hashlib
import sys
from common import SOURCE

sys.path.insert(0, str(SOURCE / 'frozen_reference'))
from models_ea import MTOEA


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
