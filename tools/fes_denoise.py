"""Denoise EVREAL-format event sequences: hot-pixel removal + background-activity (BA) filter.

- Hot pixels: pixels whose event count exceeds `hot_factor` x the 99th-percentile per-pixel count are dropped.
- BA filter: an event is kept only if one of its 8 neighbours fired within `ba_dt_ms` before it
  (isolated events are sensor noise; real edges move across neighbouring pixels).

usage: python tools/fes_denoise.py data/FES data/FES_denoised [--seqs 20211008_1_03 ...]
"""
import argparse
import json
import os
import shutil

import numba
import numpy as np


@numba.njit(cache=True)
def ba_filter(xs, ys, ts, height, width, dt):
    last = np.full((height + 2, width + 2), -1e9)  # padded so neighbours never go out of bounds
    keep = np.zeros(len(ts), dtype=np.bool_)
    for i in range(len(ts)):
        x, y, t = xs[i] + 1, ys[i] + 1, ts[i]
        support = False
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (dx != 0 or dy != 0) and t - last[y + dy, x + dx] <= dt:
                    support = True
        keep[i] = support
        last[y, x] = t
    return keep


def denoise(in_dir, out_dir, hot_factor=5.0, ba_dt_ms=5.0):
    meta = json.load(open(os.path.join(in_dir, 'metadata.json')))
    h, w = meta['sensor_resolution']
    ts = np.load(os.path.join(in_dir, 'events_ts.npy'))
    xy = np.load(os.path.join(in_dir, 'events_xy.npy'))
    p = np.load(os.path.join(in_dir, 'events_p.npy'))
    xs, ys = xy[:, 0].astype(np.int64), xy[:, 1].astype(np.int64)

    counts = np.bincount(ys * w + xs, minlength=h * w)
    hot_thr = hot_factor * max(np.percentile(counts[counts > 0], 99), 1)
    not_hot = counts[ys * w + xs] <= hot_thr
    keep = not_hot & ba_filter(xs, ys, ts, h, w, ba_dt_ms / 1000.0)

    tmp = os.path.join(os.path.dirname(out_dir), '.tmp_' + os.path.basename(out_dir))
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(tmp, 'events_ts.npy'), ts[keep])
    np.save(os.path.join(tmp, 'events_xy.npy'), xy[keep])
    np.save(os.path.join(tmp, 'events_p.npy'), p[keep])
    meta.update({'num_events': int(keep.sum()), 'denoise': {
        'hot_factor': hot_factor, 'ba_dt_ms': ba_dt_ms, 'hot_pixels': int((counts > hot_thr).sum()),
        'kept_fraction': float(keep.mean())}})
    json.dump(meta, open(os.path.join(tmp, 'metadata.json'), 'w'))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    return meta['denoise']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_root')
    ap.add_argument('out_root')
    ap.add_argument('--seqs', nargs='*')
    ap.add_argument('--hot_factor', type=float, default=5.0)
    ap.add_argument('--ba_dt_ms', type=float, default=5.0)
    args = ap.parse_args()
    seqs = args.seqs or sorted(d for d in os.listdir(args.in_root) if not d.startswith('.'))
    for s in seqs:
        out = os.path.join(args.out_root, s)
        if not args.seqs and os.path.exists(os.path.join(out, 'metadata.json')):
            continue
        d = denoise(os.path.join(args.in_root, s), out, args.hot_factor, args.ba_dt_ms)
        print(f"{s}: kept {d['kept_fraction']:.1%}, hot pixels {d['hot_pixels']}", flush=True)


if __name__ == '__main__':
    main()
