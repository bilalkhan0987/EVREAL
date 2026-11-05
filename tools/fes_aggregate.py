"""Aggregate EVREAL no-reference scores over all FES sequences.

Reads outputs/<config>/FES/<subject>/<subject>_<exp>/<method>/<metric>.txt (one "frame_idx score" line per frame)
and reports frame-weighted averages (same weighting as eval.py's table), overall and per experiment ID.
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np

METRICS = ['brisque', 'niqe', 'maniqa']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/t33ms/FES')
    ap.add_argument('--methods', nargs='+', default=['E2VID+', 'FireNet+'])
    ap.add_argument('--csv', default=None, help='optional path for per-sequence csv')
    args = ap.parse_args()

    rows = []  # (subject, exp, method, vals)
    incomplete = []
    for seq_dir in sorted(glob.glob(os.path.join(args.root, '*', '*'))):
        seq = os.path.basename(seq_dir)
        subject, exp = seq.rsplit('_', 1)
        for m in args.methods:
            ts_file = os.path.join(seq_dir, m, 'timestamps.txt')
            if not os.path.exists(ts_file):
                continue
            n_frames = sum(1 for _ in open(ts_file))
            vals = {}
            for metric in METRICS:
                f = os.path.join(seq_dir, m, metric + '.txt')
                if os.path.exists(f) and os.path.getsize(f) > 0:
                    vals[metric] = np.loadtxt(f, ndmin=2)[:, 1]
            if len(vals) == len(METRICS) and all(len(v) == n_frames for v in vals.values()):
                rows.append((subject, int(exp), m, vals))
            else:
                incomplete.append(f"{seq}/{m}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} incomplete runs (e.g. metric OOM) skipped: {' '.join(incomplete)}")

    if not rows:
        print('no results found under', args.root)
        return

    def table(group_key, title):
        acc = defaultdict(lambda: defaultdict(list))
        for subject, exp, m, vals in rows:
            for metric in METRICS:
                acc[(group_key(subject, exp), m)][metric].append(vals[metric])
        print(f"\n{title}")
        print(f"{'group':>10} {'method':>9} {'seqs':>5} {'frames':>7} " + ' '.join(f'{k.upper():>8}' for k in METRICS))
        for (g, m) in sorted(acc):
            d = acc[(g, m)]
            allv = {k: np.concatenate(v) for k, v in d.items()}
            print(f"{str(g):>10} {m:>9} {len(d['brisque']):>5} {len(allv['brisque']):>7} "
                  + ' '.join(f'{allv[k].mean():8.3f}' for k in METRICS))

    subjects = sorted({r[0] for r in rows})
    print(f"{len(subjects)} subjects, {len({(r[0], r[1]) for r in rows})} sequences with results")
    table(lambda s, e: 'all', 'Overall (frame-weighted; BRISQUE/NIQE lower=better, MANIQA higher=better)')
    table(lambda s, e: e, 'Per experiment ID')

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('subject,exp,method,frames,' + ','.join(METRICS) + '\n')
            for subject, exp, m, vals in rows:
                f.write(f"{subject},{exp},{m},{len(vals['brisque'])}," +
                        ','.join(f'{vals[k].mean():.5f}' for k in METRICS) + '\n')
        print('\nwrote', args.csv)


if __name__ == '__main__':
    main()
