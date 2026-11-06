#!/bin/bash
# Create the data/ and outputs/ symlinks that point EVREAL at $FES_ROOT.  usage: setup_links.sh [eval_config ...]
source "$(dirname "$0")/env.sh"
cd "$EVREAL_DIR" && mkdir -p data outputs "$FES_ROOT/evreal" "$FES_ROOT/evreal_denoised"
ln -sfn "$FES_ROOT/evreal" data/FES
ln -sfn "$FES_ROOT/evreal_denoised" data/FES_denoised
for cfg in "${@:-t33ms t33ms_rwarm}"; do
  for c in $cfg; do mkdir -p "$FES_ROOT/outputs_$c"; ln -sfn "$FES_ROOT/outputs_$c" "outputs/$c"; done
done
ls -l data outputs | grep -- '->'
