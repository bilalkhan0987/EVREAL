"""Convert FES (Faces in Event Streams) raw Prophesee recordings to EVREAL's npy sequence format.

Input layout:  <raw_root>/<subject>/<exp>.raw   (from raw_dataset/event_streams/lab/<subject>.zip)
Output layout: <out_root>/<subject>_<exp>/{events_ts,events_xy,events_p}.npy + metadata.json

FES has no ground-truth intensity frames, so only no-reference metrics (brisque, niqe, maniqa) apply.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prophesee_raw import read_evt2  # noqa: E402

SENSOR_RESOLUTION = [360, 480]  # [height, width] as observed in the FES lab recordings


def convert(raw_path, out_dir):
    ts, x, y, p = read_evt2(raw_path)
    if len(ts) == 0:
        raise ValueError("no events decoded")
    assert np.all(np.diff(ts) >= 0), "timestamps are not monotonic"
    assert x.max() < SENSOR_RESOLUTION[1] and y.max() < SENSOR_RESOLUTION[0], "events outside sensor"
    # write to a hidden temp dir and rename at the end, so eval.py never globs a half-written sequence
    tmp_dir = os.path.join(os.path.dirname(out_dir), '.tmp_' + os.path.basename(out_dir))
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, 'events_ts.npy'), (ts - ts[0]).astype(np.float64) / 1e6)
    np.save(os.path.join(tmp_dir, 'events_xy.npy'), np.stack([x, y], axis=1).astype(np.int16))
    np.save(os.path.join(tmp_dir, 'events_p.npy'), p.astype(np.uint8))
    meta = {"sensor_resolution": SENSOR_RESOLUTION, "source": os.path.abspath(raw_path),
            "num_events": int(len(ts)), "t0_us": int(ts[0]), "duration_s": float((ts[-1] - ts[0]) / 1e6)}
    with open(os.path.join(tmp_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    os.rename(tmp_dir, out_dir)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_root')
    ap.add_argument('out_root')
    ap.add_argument('--subjects', nargs='*', help='subject folders to convert (default: all)')
    ap.add_argument('--max_exp', type=int, default=15, help='convert experiments 0..max_exp (0-15 = large-face)')
    args = ap.parse_args()

    subjects = args.subjects or sorted(os.listdir(args.raw_root))
    for subj in subjects:
        for raw_path in sorted(glob.glob(os.path.join(args.raw_root, subj, '*.raw'))):
            exp = int(os.path.splitext(os.path.basename(raw_path))[0])
            if exp > args.max_exp:
                continue
            out_dir = os.path.join(args.out_root, f"{subj}_{exp:02d}")
            if os.path.exists(os.path.join(out_dir, 'metadata.json')):
                continue
            try:
                meta = convert(raw_path, out_dir)
                print(f"{out_dir}: {meta['num_events']} events, {meta['duration_s']:.2f}s", flush=True)
            except Exception as e:  # keep going, like bag_to_npy.py
                print(f"FAILED {raw_path}: {e}", flush=True)


if __name__ == '__main__':
    main()
