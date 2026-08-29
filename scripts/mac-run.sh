#!/bin/bash
# Run the Dataroom stack natively on an Apple Silicon Mac (Metal) - no Docker, no NVIDIA.
#
#   inference server (BACKEND)      ->  :8080  (OpenAI-compatible)
#   FastAPI app  (.venv, server.app) ->  :8000  (web UI + dashboard + API)
#
# The Pi agent and the v5-nano embedder run inside the app process tree (embedder on CPU).
# See docs/MAC.md for the full setup. This script only LAUNCHES; install steps live there.
#
# BACKEND selects the :8080 server:
#   dspark (default)  mlx-dspark serve - Apple-native MLX with DSpark/DFlash speculative
#                     decoding, 8-bit KV, and prefix caching (the compaction loop re-prefills
#                     its growing session every turn; the prefix cache is the big win there).
#                     Serves mlx-community/Qwen3.8-27B-4bit; the matched drafter auto-resolves.
#   llamacpp          llama-server (Homebrew, Metal) using the GGUF's built-in MTP head
#                     (--spec-type draft-mtp, community-measured +33-145% decode) with a
#                     q4_0 KV cache, which is required for ctx > ~90K next to 17 GB weights.
#
# The default model is now **Qwen3.8-27B** (dense hybrid, 262K native context, Apache 2.0).
# It is slower in raw tok/s than the old Qwen3.6-35B-A3B MoE (~130-184 prefill, ~25-38 decode
# on an M4 Pro via mlx-dspark vs ~960/~91-145 for the MoE) but its linear-attention cache
# (~0.086 GB/1k tokens, halved by --kv-bits 8) pushes the practical context ceiling from
# ~85K to 128K+ inside 36 GB - the ceiling the dataroom's compaction loop was bound by.
#
# Env knobs: BACKEND, DSPARK_MODEL, DSPARK_KV_BITS, DSPARK_CTX_CAP, MODEL_FILE, CTX_SIZE,
# SPEC_ARGS, CHAT_TEMPLATE_FILE, NGL, MODEL_ID. See docs/MAC.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

set -a; [ -f .env ] && . ./.env; set +a

JINA_API_KEY="${JINA_API_KEY:-}"
if [ -z "$JINA_API_KEY" ] || [ "$JINA_API_KEY" = "jina_xxxx" ]; then
  echo "ERROR: set a real JINA_API_KEY in .env (free key: https://jina.ai/api-dashboard/)" >&2
  exit 1
fi

# Inference backend: dspark (mlx-dspark, default) | llamacpp (GGUF via llama.cpp).
BACKEND="${BACKEND:-dspark}"
case "$BACKEND" in
  mlx)
    echo "NOTE: BACKEND=mlx (mlx_lm.server) is retired; routing to its successor BACKEND=dspark (mlx-dspark)." >&2
    BACKEND=dspark ;;
  dspark|llamacpp) ;;
  *) echo "ERROR: BACKEND must be 'dspark' or 'llamacpp' (got '$BACKEND')" >&2; exit 1 ;;
esac

[ -x "$ROOT/.venv/bin/python" ] || { echo "ERROR: .venv missing. See docs/MAC.md (uv venv + uv pip install)." >&2; exit 1; }
command -v pi >/dev/null || { echo "ERROR: pi not found. Install: npm install -g @earendil-works/pi-coding-agent@0.78.0" >&2; exit 1; }
if [ "$BACKEND" = "dspark" ]; then
  [ -x "$ROOT/.venv-dspark/bin/mlx-dspark" ] || { echo "ERROR: .venv-dspark is missing mlx-dspark. Create it: uv venv .venv-dspark && VIRTUAL_ENV=\$PWD/.venv-dspark uv pip install mlx-dspark  (see docs/MAC.md)" >&2; exit 1; }
else
  command -v llama-server >/dev/null || { echo "ERROR: llama-server not found. Install: brew install llama.cpp (need a build with the qwen3_5 hybrid path; older builds silently produce garbage on this arch)" >&2; exit 1; }
fi

CTX_SIZE="${CTX_SIZE:-65536}"

if [ "$BACKEND" = "dspark" ]; then
  # --- mlx-dspark backend knobs ---
  DSPARK_MODEL="${DSPARK_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
  # 8-bit KV roughly halves the cache share (measured 0.086 GB per 1k tokens at bf16 KV), which
  # is what makes a 131072 context fit 36 GB. Set DSPARK_KV_BITS= (empty) for fp16 KV; the cap
  # then drops to 98304.
  DSPARK_KV_BITS="${DSPARK_KV_BITS-8}"
  if [ -n "$DSPARK_KV_BITS" ]; then
    DSPARK_CTX_CAP="${DSPARK_CTX_CAP:-131072}"
    DSPARK_KV_ARGS=(--kv-bits "$DSPARK_KV_BITS")
  else
    DSPARK_CTX_CAP="${DSPARK_CTX_CAP:-98304}"
    DSPARK_KV_ARGS=()
  fi
  # A repo id (org/name) downloads into the HF cache on first run; anything that looks like a
  # path (leading /, ./, ../, or an existing directory) must already exist.
  if [ -d "$DSPARK_MODEL" ]; then
    : # local model dir, fine
  elif [[ "$DSPARK_MODEL" == /* || "$DSPARK_MODEL" == ./* || "$DSPARK_MODEL" == ../* ]]; then
    echo "ERROR: DSPARK_MODEL path not found: $DSPARK_MODEL  (see docs/MAC.md)" >&2
    exit 1
  elif [[ "$DSPARK_MODEL" != */* ]]; then
    echo "ERROR: DSPARK_MODEL must be a HF repo id (org/name) or a path: $DSPARK_MODEL" >&2
    exit 1
  else
    echo "NOTE: BACKEND=dspark will download $DSPARK_MODEL (~18 GB) into the HF cache on first run."
  fi
  # Extra flags passed verbatim to `mlx-dspark serve` (e.g. --drafter <repo> for a non-registry
  # target, --api-key, --max-batch).
  DSPARK_EXTRA_ARGS="${DSPARK_EXTRA_ARGS:-}"
  # mlx-dspark resolves the request's `model` field against the loaded model (basename match,
  # per its own pi config example), so pin Pi's MODEL_ID to the basename.
  export MODEL_ID="${MODEL_ID:-${DSPARK_MODEL##*/}}"
else
  # --- llama.cpp backend knobs ---
  MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
  NGL="${NGL:-999}"
  # The MTP head ships inside the unsloth GGUF (blk.*.nextn tensors); draft-mtp is the community
  # recipe for this arch. Set SPEC_ARGS= to disable.
  SPEC_ARGS="${SPEC_ARGS:---spec-type draft-mtp --spec-draft-n-max 2}"
  # Empty = use the GGUF's embedded (Unsloth-style) chat template via --jinja. The bundled
  # templates/chat_template.jinja is Qwen3.6-specific; set CHAT_TEMPLATE_FILE explicitly for
  # non-Qwen3.8 GGUFs (a wrong template silently corrupts tool-calling).
  CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
  MODEL_PATH="$ROOT/models/$MODEL_FILE"
  [ -f "$MODEL_PATH" ] || { echo "ERROR: model not found: $MODEL_PATH  (see docs/MAC.md to download the GGUF)" >&2; exit 1; }
  export MODEL_ID="${MODEL_ID:-qwen3.8}"   # free label; llama.cpp ignores it
fi

mkdir -p logs "${JOBS_DIR:-./data/jobs}"

# Put the venv (jina CLI) and pi on PATH for the agent's bash tool.
export PATH="$ROOT/.venv/bin:$(dirname "$(command -v pi)"):$PATH"
export PI_BIN="$(command -v pi)"
export PI_SKIP_VERSION_CHECK=1

# Wait for the :8080 server to answer /health, or tail its log and bail.
# WAIT_SECS covers a first-run model/drafter download plus the Metal load; raise it on a
# slow connection.
WAIT_SECS="${WAIT_SECS:-900}"
wait_for_server() {
  local label="$1" logf="$2"
  echo -n "waiting for $label"
  for i in $(seq 1 $((WAIT_SECS / 2))); do
    if curl -fsS "http://127.0.0.1:8080/health" >/dev/null 2>&1; then echo " ready"; return 0; fi
    echo -n "."; sleep 2
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " DIED"; tail -30 "$logf"; exit 1
    fi
  done
  echo " TIMEOUT after ${WAIT_SECS}s (server still running; re-run this script once it answers /health)"; tail -30 "$logf"; exit 1
}

# --- 1. inference server (:8080) ----------------------------------------------
if curl -fsS "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
  echo "inference server already up on :8080"
elif [ "$BACKEND" = "dspark" ]; then
  echo "=== starting mlx-dspark serve (Metal) - loads ~18 GB; first run downloads the model ==="
  # --mode auto (the default) resolves the measured-best drafter for the target (DFlash 2 for
  # Qwen3.8-27B); serve's default bind is 127.0.0.1:8080, matching the llama.cpp contract.
  nohup "$ROOT/.venv-dspark/bin/mlx-dspark" serve \
    --model "$DSPARK_MODEL" \
    --context-window "$CTX_SIZE" \
    ${DSPARK_KV_ARGS[@]+"${DSPARK_KV_ARGS[@]}"} \
    $DSPARK_EXTRA_ARGS \
    > "$ROOT/logs/dspark.log" 2>&1 &
  echo "mlx-dspark PID: $!  (logs: logs/dspark.log; stop with pkill -f mlx-dspark)"
  SERVER_PID=$!
  wait_for_server "mlx-dspark" "$ROOT/logs/dspark.log"
else
  echo "=== starting llama-server (Metal) - loads ~18 GB, first run ~30-60s ==="
  template_args=()
  if [ -n "$CHAT_TEMPLATE_FILE" ]; then template_args=(--chat-template-file "$CHAT_TEMPLATE_FILE"); fi
  # shellcheck disable=SC2086
  nohup llama-server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 --port 8080 \
    --metrics \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --flash-attn on \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -ngl "$NGL" \
    -ub 256 -b 2048 \
    --n-predict 8192 \
    --jinja \
    ${template_args[@]+"${template_args[@]}"} \
    $SPEC_ARGS \
    > "$ROOT/logs/llama.log" 2>&1 &
  echo "llama-server PID: $!  (logs: logs/llama.log)"
  SERVER_PID=$!
  wait_for_server "llama-server" "$ROOT/logs/llama.log"
fi

# --- 2. FastAPI app -----------------------------------------------------------
export LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8080}"
export JOBS_DIR="${JOBS_DIR:-$ROOT/data/jobs}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-$CTX_SIZE}"
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PORT="${PORT:-8000}"

echo "=== starting Dataroom app on :$PORT ==="
echo "    web UI:     http://localhost:$PORT/"
echo "    backend:    $BACKEND"
if [ "$BACKEND" = "dspark" ]; then
  echo "    LLAMA_URL:  $LLAMA_URL    ctx=$CTX_SIZE (cap $DSPARK_CTX_CAP)    model=$DSPARK_MODEL    embedder=$EMBED_DEVICE"
  if [ -n "$DSPARK_KV_BITS" ]; then
    echo "    note:       ${DSPARK_KV_BITS}-bit KV; dashboard tok/s + KV gauges are llama.cpp-only"
  else
    echo "    note:       fp16 KV (DSPARK_KV_BITS empty); dashboard tok/s + KV gauges are llama.cpp-only"
  fi
else
  echo "    LLAMA_URL:  $LLAMA_URL    ctx=$CTX_SIZE    ngl=$NGL    embedder=$EMBED_DEVICE"
  [ -z "$SPEC_ARGS" ] && echo "    spec:       (disabled)" || echo "    spec:       $SPEC_ARGS"
fi
exec "$ROOT/.venv/bin/python" -m server.app
