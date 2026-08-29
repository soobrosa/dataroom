#!/usr/bin/env bash
# Greedy-parity check (plan §0.4), SEQUENTIAL so the two ~21GB models never
# co-reside in 36GB RAM. Usage:
#   mlx_parity.sh capture <port> <outfile>   # query a running server, save answers
#   mlx_parity.sh compare <fileA> <fileB>    # diff two capture files
# Workflow: start llama.cpp -> capture 8080 -> stop -> start MLX -> capture 8081 -> stop -> compare.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BENCH="$ROOT/data/mlx-bench"; mkdir -p "$BENCH/logs"

PROMPTS=(
  "What is 17 * 23? Reply with only the number."
  "What is the chemical symbol for gold? Reply with only the symbol."
  "Complete with one word only: The powerhouse of the cell is the ___."
  "In one sentence, what does a cryptographic hash function do?"
  "List the eight planets in order from the Sun outward, comma-separated."
  "What year did Apollo 11 land humans on the Moon? Reply with only the year."
  "Translate 'good morning' into French. Reply with only the translation."
  "Boiling point of water at sea level in Celsius? Reply with only the number."
  "Past tense of the verb 'to go'? Reply with only the word."
  "How many sides does a hexagon have? Reply with only the number."
)

MAXTOK="${MAXTOK:-1024}"   # reasoning models need headroom to finish <think> before emitting content
ask(){ # port, json-encoded-content
  curl -fsS -m 300 "http://127.0.0.1:$1/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$2}],\"temperature\":0,\"max_tokens\":$MAXTOK}" \
  | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin); m=d['choices'][0]['message']; fr=d['choices'][0].get('finish_reason')
    c=(m.get('content') or '').strip().replace(chr(10),' ')
    if not c and fr=='length': print('<truncated: still thinking at max_tokens>')
    else: print(c if c else '<empty content>')
except Exception as e:
    print('<request failed: %s>'%e)
"
}

cmd="${1:-}"
case "$cmd" in
  capture)
    port="${2:?port}"; out="${3:?outfile}"
    : > "$out"
    i=0
    for p in "${PROMPTS[@]}"; do
      i=$((i+1))
      j=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$p")
      a=$(ask "$port" "$j")
      printf 'Q%02d\t%s\n' "$i" "$a" >> "$out"
      echo "Q$i (:$port): $a"
    done
    echo "saved -> $out"
    ;;
  compare)
    A="${2:?fileA}"; B="${3:?fileB}"
    python3 - "$A" "$B" <<'PY'
import sys
a=dict(l.rstrip("\n").split("\t",1) for l in open(sys.argv[1]) if "\t" in l)
b=dict(l.rstrip("\n").split("\t",1) for l in open(sys.argv[2]) if "\t" in l)
keys=sorted(set(a)|set(b))
exact=0
for k in keys:
    x=a.get(k,"<missing>"); y=b.get(k,"<missing>")
    same = x.strip().lower()==y.strip().lower()
    exact+=same
    print(f"{k}  {'MATCH' if same else 'DIFF '}")
    if not same:
        print(f"    llama: {x}")
        print(f"    mlx  : {y}")
print(f"\nexact (case-insensitive) matches: {exact}/{len(keys)} — judge DIFF rows for semantic equivalence")
PY
    ;;
  *)
    echo "usage: $0 capture <port> <outfile> | compare <fileA> <fileB>" >&2; exit 2;;
esac
