"""List FES subjects whose evaluation is incomplete (missing sequences, or metric files shorter than the number
of frames, e.g. after a CUDA OOM), and print the command to re-run just those subjects.

usage: python tools/fes_check_complete.py --cfg t33ms_rwarm --dataset FES_denoised
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--dataset', required=True, help='FES or FES_denoised')
    ap.add_argument('--methods', nargs='+', default=['E2VID+', 'FireNet+'])
    ap.add_argument('--metrics', nargs='+', default=['brisque', 'niqe', 'maniqa'])
    ap.add_argument('--max_exp', type=int, default=15)
    ap.add_argument('--subjects_file', default=os.path.join(HERE, '..', 'scripts', 'fes', 'subjects.txt'))
    args = ap.parse_args()

    subjects = [l.strip() for l in open(args.subjects_file) if l.strip() and not l.startswith('#')]
    root = os.path.join('outputs', args.cfg, args.dataset)
    bad = {}
    for s in subjects:
        problems = []
        for e in range(args.max_exp + 1):
            for m in args.methods:
                d = os.path.join(root, s, f'{s}_{e:02d}', m)
                ts = os.path.join(d, 'timestamps.txt')
                if not os.path.exists(ts):
                    problems.append(f'{s}_{e:02d}/{m}: missing')
                    continue
                n = sum(1 for _ in open(ts))
                for k in args.metrics:
                    f = os.path.join(d, k + '.txt')
                    got = sum(1 for _ in open(f)) if os.path.exists(f) else 0
                    if n == 0 or got < n:
                        problems.append(f'{s}_{e:02d}/{m}/{k}: {got}/{n} frames')
        if problems:
            bad[s] = problems

    total = len(subjects) * (args.max_exp + 1) * len(args.methods)
    print(f'{root}: {len(subjects) - len(bad)}/{len(subjects)} subjects complete ({total} sequence runs expected)')
    for s, p in bad.items():
        print(f'  {s}: {len(p)} problem(s), e.g. {p[0]}')
    if bad:
        print('\nre-run with:\n  SUBJECTS="' + ' '.join(bad) + f'" scripts/fes/run_eval.sh {args.cfg} {args.dataset}')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
