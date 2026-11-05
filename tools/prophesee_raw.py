"""Minimal numpy decoder for Prophesee EVT 2.0 .raw files (as used by the FES dataset, Gen3 VGA sensor)."""
import numpy as np


def read_header(path):
    header, offset = [], 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line.startswith(b'%'):
                break
            header.append(line.decode('latin-1').strip())
            offset = f.tell()
    return header, offset


def read_evt2(path):
    """Returns ts (int64, microseconds), x, y (uint16), p (int8, 0/1)."""
    _, offset = read_header(path)
    words = np.fromfile(path, dtype='<u4', offset=offset)
    typ = words >> 28
    is_th = typ == 0x8
    # running TIME_HIGH value (upper 28 bits of the 34-bit timestamp)
    th = np.where(is_th, words & 0x0FFFFFFF, 0).astype(np.int64)
    idx = np.where(is_th, np.arange(len(words)), 0)
    np.maximum.accumulate(idx, out=idx)
    th = th[idx]
    # CD events before the first TIME_HIGH have an unknown time base -> drop them
    first_th = np.argmax(is_th) if is_th.any() else len(words)
    is_cd = ((typ == 0x0) | (typ == 0x1)) & (np.arange(len(words)) > first_th)
    w = words[is_cd]
    ts = (th[is_cd] << 6) | ((w >> 22) & 0x3F).astype(np.int64)
    x = ((w >> 11) & 0x7FF).astype(np.uint16)
    y = (w & 0x7FF).astype(np.uint16)
    p = (typ[is_cd] == 0x1).astype(np.int8)
    return ts, x, y, p
