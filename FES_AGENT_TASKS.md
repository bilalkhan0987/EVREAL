# FES evaluation — task brief for an AI agent

You are setting up and running the EVREAL evaluation of event-to-video reconstruction on the **FES (Faces in
Event Streams)** dataset, on a new GPU server. The work was first done on another server; this file captures what
was learned there so you don't repeat its mistakes. Work from the repo root (`EVREAL/`, branch `fes`).

**Goal:** reconstruct intensity frames from the FES lab recordings with **E2VID+** and **FireNet+**, using the
**reversed-time warm-up** and **event denoising**, score them with the no-reference metrics **BRISQUE, NIQE,
MANIQA**, and report the results (overall, per experiment, worst sequences, raw vs denoised).

## Ground rules

- This is a **shared** server. Check `nvidia-smi` and `df -h` before starting heavy jobs; use only GPUs that are
  idle and leave headroom for other users.
- **Never** delete or modify files you did not create, other users' folders, or anything outside `$FES_ROOT`
  and this repo. Ask the user before deleting anything large, before pushing to git, and before starting a run
  that will occupy GPUs for many hours.
- Run long jobs with `nohup` (or in `tmux`) and write logs to disk; don't depend on your session staying alive.
- When you report results, show the numbers and the sample images you looked at. If something failed or was
  skipped, say so.

## Background: what you need to know

1. **Use raw events only.** The local PNG "event frames" some users have (e.g. `FES_frames_fixed_duration_*`)
   and the Hugging Face `preprocessed_dataset/*.h5` files are both *binned histograms* with no per-event
   timing; EVREAL cannot use them. The real events are the Prophesee `.raw` files in
   `raw_dataset/event_streams/lab/<subject>.zip` of the public, ungated dataset
   [`issai/Faces_in_Event_Streams`](https://huggingface.co/datasets/issai/Faces_in_Event_Streams)
   (~9 GB per subject).
2. **Format:** EVT 2.0, sensor 480×360 (W×H), ~10 s per recording, experiments 0–59 per subject. We use
   experiments **0–15** (large face) of the **39 subjects** in `scripts/fes/subjects.txt` → **624 sequences**.
3. **No ground truth.** FES has no intensity frames, so only no-reference metrics apply (no MSE/SSIM/LPIPS).
   MANIQA barely penalises washed-out early frames, so **also inspect images**, don't trust scores alone.
4. **Early frames are blank** because the recurrent networks start from an empty state and subjects sit still
   at first. Fix (already in `eval.py`): `warmup_s` in the eval config primes the network with the first 1 s
   of events **played backwards in time with flipped polarity**, then reconstructs from t=0. Use config
   `t33ms_rwarm` (33 ms windows, `warmup_s: 1.0`). *Do not* replace this with a forward warm-up that rewinds:
   it produces ghost overlays of later head poses.
5. **Noise.** All recordings show 100 Hz light flicker; some (e.g. `20211008_1_03`) have heavy background
   noise and hundreds of hot pixels and reconstruct poorly throughout. `tools/fes_denoise.py` removes hot
   pixels and applies a background-activity filter (keep an event only if a neighbour fired within
   `--ba_dt_ms`). With the default 5 ms it keeps only ~46–77 % of events: less speckle, but it visibly
   loses face detail, and in a 2-sequence test FireNet+ BRISQUE got much **worse** (35.4 vs ~20 raw).
   **Denoising parameters must be validated (Task 4) before the full run.**

## Tasks

### 1. Inspect the server and pick a data location

```bash
nvidia-smi; df -h; nproc; free -g
```
Choose `FES_ROOT` on a disk with **≥ 800 GB free** (zips 334 GB + extracted raw 88 GB + converted 194 GB +
denoised up to ~190 GB + outputs ~45 GB per run). Default is `/data/FES`. If space is tight, ask the user
whether the zips may be deleted after extraction. Don't put data on a nearly full root/home disk.

### 2. Python environment

```bash
conda env create -f evreal_env.yml          # or, if that fails:
# conda create -n evreal python=3.10 -y && conda activate evreal
# pip install -r requirements_fes_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu121
conda activate evreal
python -c "import torch, h5py, numba, pyiqa; print(torch.__version__, torch.cuda.is_available())"
```
Must print `True`. If the GPU driver needs a different CUDA build, install the matching `torch`/`torchvision`
from pytorch.org first. Use the env's `python`, not the system one.

### 3. Configure, link, and smoke-test one subject

```bash
export FES_ROOT=/data/FES PYTHON=$(which python) GPUS="<idle gpu ids>" WORKERS_PER_GPU=2
scripts/fes/setup_links.sh t33ms_rwarm                    # data/FES, data/FES_denoised, outputs/t33ms_rwarm
SUBJECTS=20211005_0 scripts/fes/fetch_fes.sh               # ~10 min: download, extract, convert, denoise
```
Check: 16 folders `data/FES/20211005_0_00..15`, each `metadata.json` with `duration_s` ≈ 10 s.
Reference: `20211005_0_00` = **6,818,420 events, 10.60 s**. A duration of hundreds of seconds means the
EVT2 decoder bug is back (events before the first TIME_HIGH word must be dropped; see `tools/prophesee_raw.py`).

### 4. Validate the denoising settings (required before the full run)

Compare raw vs denoised reconstructions on three sequences — `20211005_0_00` (clean, still subject),
`20211015_0_04` (moving), `20211008_1_03` (noisy) — for a few `--ba_dt_ms` values, e.g. 5, 10, 20, and
1000 (≈ hot-pixel removal only). The two non-`20211005_0` subjects must be fetched first
(`SUBJECTS="20211015_0 20211008_1" scripts/fes/fetch_fes.sh`). Write each variant to its own root, e.g.:
```bash
python tools/fes_denoise.py $FES_ROOT/evreal $FES_ROOT/denoise_dt10 --ba_dt_ms 10 \
       --seqs 20211005_0_00 20211015_0_04 20211008_1_03
```
then make a small dataset config per variant (`{"root_path": "data/<link>", "sequences": {...}}` with a
`data/<link>` symlink to that root) and run
`python eval.py -m E2VID+ FireNet+ -c t33ms_rwarm -d <variant> -qm brisque niqe maniqa`.
Look at frames ~0, 20, 150, 250 of each sequence side by side and compare scores. Pick the setting that removes
noise **without losing facial detail for either method**. If no denoised variant beats raw, say so and
recommend running raw only. Report the comparison to the user (images + table) and get their go-ahead,
then regenerate `$FES_ROOT/evreal_denoised` with the chosen setting
(`DENOISE_ARGS="--ba_dt_ms <X>"` for `fetch_fes.sh`, or delete the folder and re-run `fes_denoise.py`).
Clean up the variant folders and temporary configs afterwards.

### 5. Fetch all 39 subjects

```bash
nohup env DENOISE_ARGS="--ba_dt_ms <X>" scripts/fes/fetch_fes.sh > $FES_ROOT/fetch.log 2>&1 &
```
Several hours (39 × ~9 GB). Safe to re-run; finished steps are skipped. Done when the log ends with
`ALL DONE: 624 converted, 624 denoised`. Check for `DOWNLOAD FAILED` / `UNZIP WARN` / `FAILED` lines.

### 6. Run the evaluation

Main run (denoised + reversed warm-up), and the raw run for comparison:
```bash
nohup scripts/fes/run_eval.sh t33ms_rwarm FES_denoised > $FES_ROOT/run_denoised.log 2>&1 &
# after it finishes (or on other GPUs):
nohup scripts/fes/run_eval.sh t33ms_rwarm FES > $FES_ROOT/run_raw.log 2>&1 &
```
Each run is ~1 h per subject per worker (≈ 7 h with 6 workers). Output layout:
`outputs/t33ms_rwarm/<FES|FES_denoised>/<subject>/<subject>_<exp>/<method>/` with `frame_*.png`,
`brisque.txt`, `niqe.txt`, `maniqa.txt` (per-frame), `timestamps.txt`, `event_rate.txt`.
Logs: `$FES_ROOT/logs/t33ms_rwarm/<dataset>/<subject>.log`.

**GPU memory:** MANIQA needs ~3 GB per batch of 4 frames. On the old server, other users' jobs filled the
GPUs and frames silently lost their MANIQA scores; one subject lost 15 sequences. Now handled: metrics retry
on CUDA OOM (`utils/eval_metrics.py`, split the batch + back off), a failing sequence no longer aborts the rest
(`SEQUENCE FAILED` in the log), and the scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Still
keep `WORKERS_PER_GPU` at 2 or less on shared GPUs.

### 7. Verify completeness and re-run gaps

```bash
python tools/fes_check_complete.py --cfg t33ms_rwarm --dataset FES_denoised
python tools/fes_check_complete.py --cfg t33ms_rwarm --dataset FES
grep -l "SEQUENCE FAILED\|EVAL FAILED" $FES_ROOT/logs/t33ms_rwarm/*/*.log
```
The checker prints a `SUBJECTS="..." scripts/fes/run_eval.sh ...` command for any incomplete subjects; run it
and check again until all 39 subjects are complete for both methods.

### 8. Aggregate and report

```bash
python tools/fes_aggregate.py --root outputs/t33ms_rwarm/FES_denoised --csv $FES_ROOT/scores_denoised.csv
python tools/fes_aggregate.py --root outputs/t33ms_rwarm/FES          --csv $FES_ROOT/scores_raw.csv
```
Report to the user:
- overall table (both methods × raw/denoised): BRISQUE ↓, NIQE ↓, MANIQA ↑, and sequence/frame counts;
- per-experiment-ID breakdown;
- the 10 worst sequences per method (from the csv), with a sample frame from each;
- visual check that early frames (0–20) are good and have no ghosting;
- anything incomplete or anomalous.

Reference results from the old server (raw data, **no** warm-up, config `t33ms`; E2VID+ was missing 22 runs
due to OOM):

| Method | runs | BRISQUE | NIQE | MANIQA |
|---|---|---|---|---|
| E2VID+ | 601 | 7.63 | 4.67 | 0.320 |
| FireNet+ | 623 | 3.98 | 4.41 | 0.398 |

## File map

| Path | Purpose |
|---|---|
| `scripts/fes/env.sh` | settings: `FES_ROOT`, `PYTHON`, `GPUS`, `WORKERS_PER_GPU`, `MAX_EXP`, `SUBJECTS`, `METHODS`, `METRICS` |
| `scripts/fes/subjects.txt` | the 39 subjects |
| `scripts/fes/setup_links.sh` | `data/` and `outputs/` symlinks into `$FES_ROOT` |
| `scripts/fes/fetch_fes.sh` | download → extract exps 0–15 (`unzip -j`: zip paths start with `../`) → convert → denoise |
| `scripts/fes/run_eval.sh` | parallel per-subject `eval.py` runs |
| `tools/prophesee_raw.py` | EVT 2.0 decoder |
| `tools/fes_raw_to_npy.py` | `.raw` → EVREAL npy format (`events_ts/xy/p.npy`, `metadata.json`) |
| `tools/fes_denoise.py` | hot-pixel + background-activity filter |
| `tools/fes_check_complete.py` | find incomplete subjects, print re-run command |
| `tools/fes_aggregate.py` | frame-weighted averages, per-sequence csv |
| `config/eval/t33ms_rwarm.json` | 33 ms windows + 1 s reversed warm-up (use this) |
| `config/eval/t33ms.json` | 33 ms windows, no warm-up (baseline only) |

Git: if SSH port 22 is blocked, use `ssh://git@ssh.github.com:443/bilalkhan0987/EVREAL.git`.
