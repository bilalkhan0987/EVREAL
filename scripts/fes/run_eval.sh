#!/bin/bash
# Evaluate METHODS on FES, one eval.py process per subject, WORKERS_PER_GPU processes on each GPU in GPUS.
# usage: run_eval.sh <eval_config> <dataset: FES | FES_denoised>      e.g.  GPUS="0 1" run_eval.sh t33ms_rwarm FES_denoised
# Outputs: outputs/<eval_config>/<dataset>/<subject>/<subject>_<exp>/<method>/   Logs: $FES_ROOT/logs/<cfg>/<dataset>/
set -u
source "$(dirname "$0")/env.sh"
if [ "${1:-}" = "--one" ]; then   # internal: evaluate one subject on one GPU
  cfg=$2; ds=$3; s=$4; gpu=$5
  cd "$EVREAL_DIR"
  $PYTHON - "$s" "$ds" "$MAX_EXP" <<'PY'
import json, sys
s, ds, max_exp = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = {"root_path": f"data/{ds}", "sequences": {f"{s}_{e:02d}": {} for e in range(max_exp + 1)}}
json.dump(cfg, open(f"config/dataset/{ds}/{s}.json", "w"), indent=2)
PY
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON eval.py -m $METHODS -c "$cfg" -d "$ds/$s" -qm $METRICS \
    > "$FES_ROOT/logs/$cfg/$ds/$s.log" 2>&1 && echo "EVAL DONE $s" || echo "EVAL FAILED $s"
  exit 0
fi
cfg=$1; ds=$2
[ -f "$EVREAL_DIR/config/eval/$cfg.json" ] || { echo "no config/eval/$cfg.json"; exit 1; }
mkdir -p "$EVREAL_DIR/config/dataset/$ds" "$FES_ROOT/logs/$cfg/$ds" "$FES_ROOT/outputs_$cfg"
[ -e "$EVREAL_DIR/outputs/$cfg" ] || ln -s "$FES_ROOT/outputs_$cfg" "$EVREAL_DIR/outputs/$cfg"
gpus=($GPUS); n=$(( ${#gpus[@]} * WORKERS_PER_GPU )); i=0
for s in $SUBJECTS; do echo "$s ${gpus[$(( i % ${#gpus[@]} ))]}"; i=$((i+1)); done \
  | xargs -P "$n" -L 1 bash "$0" --one "$cfg" "$ds"
echo "ALL EVAL DONE ($cfg / $ds)"
