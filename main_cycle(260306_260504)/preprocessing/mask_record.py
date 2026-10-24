"""Processing utilities for mask record."""
import sys as _sys
from pathlib import Path as _Path
_CYCLE_ROOT = _Path(__file__).resolve().parents[1]
if str(_CYCLE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_CYCLE_ROOT))
import base64
import csv
import hashlib
import zlib
import numpy as np
csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
FIELDS = ['mask_encoding', 'mask_height', 'mask_width', 'mask_data']

def pack(mask):
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not binary.size:
        raise ValueError('Invalid input or state in pack (rejected condition: binary.ndim != 2 or not binary.size)')
    bits = np.packbits(binary.reshape(-1)).tobytes()
    return dict(mask_encoding='packbits_zlib_base64_v1', mask_height=binary.shape[0], mask_width=binary.shape[1], mask_data=base64.b64encode(zlib.compress(bits)).decode('ascii'), mask_sha256=hashlib.sha256(binary.tobytes()).hexdigest(), mask_path='')

def unpack(row):
    if row.get('mask_encoding') != 'packbits_zlib_base64_v1':
        raise ValueError("Invalid input or state in unpack (rejected condition: row.get('mask_encoding') != 'packbits_zlib_base64_v1')")
    (height, width) = (int(row['mask_height']), int(row['mask_width']))
    if height <= 0 or width <= 0:
        raise ValueError('Invalid input or state in unpack (rejected condition: height <= 0 or width <= 0)')
    bits = zlib.decompress(base64.b64decode(row['mask_data'], validate=True))
    if len(bits) != (height * width + 7) // 8:
        raise ValueError('Invalid input or state in unpack (rejected condition: len(bits) != (height * width + 7) // 8)')
    binary = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))[:height * width].reshape(height, width)
    if hashlib.sha256(binary.tobytes()).hexdigest() != row['mask_sha256']:
        raise ValueError("Invalid input or state in unpack (rejected condition: hashlib.sha256(binary.tobytes()).hexdigest() != row['mask_sha256'])")
    return binary * 255
