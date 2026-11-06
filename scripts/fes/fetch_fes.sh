#!/bin/bash
# Download FES raw lab recordings from Hugging Face, extract experiments 0..MAX_EXP, convert to EVREAL npy
# (data/FES) and denoise (data/FES_denoised). Idempotent: finished steps are skipped on re-run.
# usage: fetch_fes.sh            (all subjects in subjects.txt)   |   SUBJECTS="20211005_0" fetch_fes.sh
set -u
source "$(dirname "$0")/env.sh"
URL=https://huggingface.co/datasets/issai/Faces_in_Event_Streams/resolve/main/raw_dataset/event_streams/lab
mkdir -p "$FES_ROOT"/{raw_zips,raw,evreal,evreal_denoised}
cd "$EVREAL_DIR"
for s in $SUBJECTS; do
  zip="$FES_ROOT/raw_zips/$s.zip"
  if [ ! -f "$zip" ]; then
    curl -sfL --retry 5 -C - -o "$zip.part" "$URL/$s.zip" && mv "$zip.part" "$zip" || { echo "DOWNLOAD FAILED $s"; continue; }
  fi
  # zip members are stored as ../<s>/<s>/<exp>.raw  -> extract flat (-j) so nothing escapes the target dir
  unzip -o -q -j "$zip" $(for i in $(seq 0 "$MAX_EXP"); do echo "*/$i.raw"; done) -d "$FES_ROOT/raw/$s" \
    || echo "UNZIP WARN $s"
  $PYTHON tools/fes_raw_to_npy.py "$FES_ROOT/raw" "$FES_ROOT/evreal" --subjects "$s" --max_exp "$MAX_EXP"
  $PYTHON tools/fes_denoise.py "$FES_ROOT/evreal" "$FES_ROOT/evreal_denoised" ${DENOISE_ARGS:-}
  echo "SUBJECT DONE $s"
done
echo "ALL DONE: $(ls "$FES_ROOT/evreal" | grep -vc '^\.') converted, $(ls "$FES_ROOT/evreal_denoised" | grep -vc '^\.') denoised"
