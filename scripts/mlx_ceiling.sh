#!/usr/bin/env bash
# Context-ceiling sweep for the MLX backend (plan §1.2), write-ahead logged so the
# result survives a sudden OOM/kernel panic. After a hard reboot, the highest
# `ATTEMPT ... ctx=N` with no matching `RESULT ... ctx=N` in the journal is the ceiling.
# Resume-safe: re-running skips any size already marked RESULT.
set -uo pipefail        # NOT -e: a failed N must still journal and continue

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="${PY:-$ROOT/.venv-mlx/bin/python}"
M="${MLX_MODEL:-$ROOT/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit}"
SIZES="${SIZES:-30000 65000 86000 100000 120000}"
TAG="${TAG:-1.2}"                 # result tag; override for KV-quant runs (e.g. 1.2-kv8)
KVBITS="${KVBITS:-}"              # if set, pass --kv-bits N (quantized KV cache)
EXTRA="${EXTRA:-}"               # any extra mlx_lm.generate flags
KVARGS=""; [ -n "$KVBITS" ] && KVARGS="--kv-bits $KVBITS --quantized-kv-start 0"
BENCH="$ROOT/data/mlx-bench"
J="$BENCH/RESULTS.md"
mkdir -p "$BENCH/logs"

[ -x "$PY" ] || { echo "ERROR: mlx venv python not found: $PY" >&2; exit 1; }
[ -d "$M" ]  || { echo "ERROR: MLX model not found: $M" >&2; exit 1; }

log(){ printf '%s | %s\n' "$(date -u +%FT%TZ)" "$*" >> "$J"; sync; }

log "=== $TAG ceiling sweep start; sizes=[$SIZES]; kv-bits=${KVBITS:-fp16}; mlx_lm=$("$PY" -c 'import mlx_lm;print(mlx_lm.__version__)' 2>/dev/null) ==="

for N in $SIZES; do
  if grep -q "RESULT $TAG ctx=$N" "$J" 2>/dev/null; then
    echo "skip ctx=$N (already resolved)"; continue
  fi
  P="$BENCH/ctx_$N.txt"
  "$PY" - "$N" > "$P" <<'PY'
import sys
n = int(sys.argv[1])
para = "The hybrid GDN plus MoE architecture caches KV on only a subset of layers. "
# ~12 tokens per repetition; overshoot slightly so actual prompt >= target
print(para * (n // 12))
PY

  log "ATTEMPT $TAG ctx=$N — loading model now (machine may die here)"   # write-ahead + sync
  LOG="$BENCH/logs/$TAG-ctx-$N.log"
  # shellcheck disable=SC2086
  /usr/bin/time -l "$PY" -m mlx_lm.generate \
     --model "$M" --prompt "$(cat "$P")" --temp 0 --max-tokens 100 $KVARGS $EXTRA \
     > "$LOG" 2>&1
  rc=$?
  foot=$(grep -i "peak memory footprint" "$LOG" | awk '{printf "%.1fGB",$1/1e9}')
  [ -z "$foot" ] && foot="n/a"
  if [ "$rc" -eq 0 ]; then
    log "RESULT $TAG ctx=$N: OK rc=0 peakFootprint=$foot"
  else
    log "RESULT $TAG ctx=$N: FAIL rc=$rc peakFootprint=$foot (OOM/hang?) log=$LOG"
  fi
done

log "RESULT $TAG sweep complete"
echo "done — see $J"
