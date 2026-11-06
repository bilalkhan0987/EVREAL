# Shared settings for the FES scripts. Override any of these by exporting them before running a script.
FES_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVREAL_DIR=${EVREAL_DIR:-$(cd "$FES_HERE/../.." && pwd)}
FES_ROOT=${FES_ROOT:-/data/FES}                 # big disk: zips, raw, converted events, outputs, logs
PYTHON=${PYTHON:-python}                         # python of the activated 'evreal' conda env
MAX_EXP=${MAX_EXP:-15}                           # experiments 0..15 = large-face recordings
GPUS=${GPUS:-0}                                  # space-separated GPU ids to use, e.g. "0 1 2"
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}            # MANIQA needs ~3 GB/batch; keep this small on shared GPUs
METHODS=${METHODS:-"E2VID+ FireNet+"}
METRICS=${METRICS:-"brisque niqe maniqa"}
SUBJECTS=${SUBJECTS:-$(grep -v '^#' "$FES_HERE/subjects.txt")}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
