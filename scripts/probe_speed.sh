#!/usr/bin/env bash
# Prefill/decode tok/s probe against a running llama-server on $PORT.
# Uses llama-server's `timings.prompt_per_second` and `timings.predicted_per_second`.
set -uo pipefail
PORT="${1:?port}"
TAG="${2:?tag}"
OUT="${OUT:-data/probes/$TAG.txt}"
mkdir -p "$(dirname "$OUT")"

# Wait for /health
for i in $(seq 1 120); do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

# Build a long prompt (~6000 tokens of filler)
REPS="${REPS:-800}"
LONG=$(python3 -c "import sys; print('The hybrid GDN plus MoE architecture caches KV on only a subset of layers. ' * int(sys.argv[1]))" "$REPS")

# Warmup
curl -fsS -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0}' >/dev/null

# Prefill probe: huge prompt, 1 generated token → prompt_per_second
P=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$LONG")
PRE=$(curl -fsS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":$P}],\"max_tokens\":1,\"temperature\":0,\"timings_per_token\":false}")

# Decode probe: tiny prompt, 200 generated tokens → predicted_per_second
DEC=$(curl -fsS -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a 200-word essay about clouds."}],"max_tokens":200,"temperature":0}')

python3 - "$TAG" "$PRE" "$DEC" <<'PY' | tee "$OUT"
import sys, json
tag, pre_s, dec_s = sys.argv[1], sys.argv[2], sys.argv[3]
def t(s):
    d = json.loads(s)
    return d.get("timings", {}), d.get("usage", {})
pt, pu = t(pre_s)
dt, du = t(dec_s)
print(f"=== {tag} ===")
print(f"prefill: prompt_n={pt.get('prompt_n')} prompt_ms={pt.get('prompt_ms'):.0f} -> {pt.get('prompt_per_second',0):.1f} t/s")
print(f"decode : predicted_n={dt.get('predicted_n')} predicted_ms={dt.get('predicted_ms'):.0f} -> {dt.get('predicted_per_second',0):.1f} t/s")
print(f"  (prompt usage: {pu}; decode usage: {du})")
draft = dt.get('draft_n')
if draft is not None:
    print(f"  draft tokens: {draft}, accepted: {dt.get('draft_n_accepted')}")
PY
