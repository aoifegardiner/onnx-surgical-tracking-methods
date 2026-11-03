#!/usr/bin/env bash
# Run one or more STIR ONNX evaluations.
#
#   export STIR_DATA=/path/to/STIRChallenge_2024
#   bash scripts/run_stir.sh --gpu 0 --methods RAFT --num-seq 1
#
# Scaled methods use scales 1, 2 and 4. CoTracker, LiteTracker and TAPNext++
# run at scale 1 only, because their graphs are fixed-resolution.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
NUM_SEQ="${NUM_SEQ:--1}"
JSONSUFFIX="${JSONSUFFIX:-}"
WALL_SCOPE="${WALL_SCOPE:-track}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/stir}"
METHODS="${METHODS:-RAFT WAFT MFT ALLTRACKER PIPS LOCOTRACK COTRACKER LITETRACKER TAPNEXTPP}"
SCALED="${SCALED:-1 2 4}"
DATADIR="${STIR_DATA:-}"

usage() {
  cat <<'EOF'
Usage: scripts/run_stir.sh [options]

  --gpu N                 CUDA_VISIBLE_DEVICES index. Default: 0
  --methods "RAFT WAFT"   Methods to run
  --num-seq N             How many sequences, -1 for all. Default: -1
  --datadir DIR           STIR root. Default: $STIR_DATA
  --output-root DIR       Where JSON results are written
  --python PATH           Python interpreter
  --skip-existing         Skip a method/scale that already has a positions file
  --stop-on-error         Stop the sweep at the first failure
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --num-seq) NUM_SEQ="$2"; shift 2 ;;
    --datadir) DATADIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --stop-on-error) CONTINUE_ON_ERROR=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DATADIR" ]]; then
  echo "Set STIR_DATA or pass --datadir. See README.md for the expected layout." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/litetracker${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ "$NUM_SEQ" -lt 0 ]]; then
  NUM_DATA_NAME="all"
else
  NUM_DATA_NAME="$NUM_SEQ"
fi

scales_for() {
  case "$1" in
    COTRACKER|LITETRACKER|COTRACKER_ONLINE|TAPNEXTPP) echo "1" ;;
    *) echo "$SCALED" ;;
  esac
}

echo "STIR ONNX sweep"
echo "  data:    $DATADIR"
echo "  gpu:     $GPU"
echo "  methods: $METHODS"
echo "  output:  $OUTPUT_ROOT"

failed=()
skipped=()
succeeded=()

for method in $METHODS; do
  for scale in $(scales_for "$method"); do
    outdir="${OUTPUT_ROOT}/${method}/scale${scale}"
    mkdir -p "$outdir"
    logfile="${OUTPUT_ROOT}/logs/${method}_scale${scale}${JSONSUFFIX}.log"
    expected="${outdir}/positions_${NUM_DATA_NAME}_${method}_scale${scale}${JSONSUFFIX}.json"
    if [[ "$SKIP_EXISTING" -eq 1 && -f "$expected" ]]; then
      echo "[SKIP] $method scale=${scale}"
      skipped+=("${method}_scale${scale}")
      continue
    fi
    echo "[START] ${method} scale=${scale}"
    set +e
    "$PYTHON" -u "${ROOT}/export_onnx/run_all_onnx_stir.py" \
      --datadir "$DATADIR" \
      --method "$method" \
      --scale "$scale" \
      --num_seq "$NUM_SEQ" \
      --jsonsuffix "$JSONSUFFIX" \
      --wall-scope "$WALL_SCOPE" \
      --outdir "$outdir" \
      > "$logfile" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      echo "[OK]   ${method} scale=${scale}"
      succeeded+=("${method}_scale${scale}")
    elif [[ $status -eq 2 ]]; then
      echo "[SKIP] ${method} scale=${scale} (model file not present)"
      skipped+=("${method}_scale${scale}")
    else
      echo "[FAIL] ${method} scale=${scale}  see ${logfile}"
      failed+=("${method}_scale${scale}")
      if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
        break 2
      fi
    fi
  done
done

echo "succeeded: ${succeeded[*]:-(none)}"
echo "skipped:   ${skipped[*]:-(none)}"
echo "failed:    ${failed[*]:-(none)}"
if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi
