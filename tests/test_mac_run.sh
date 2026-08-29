#!/usr/bin/env bash
# GPU-free tests for scripts/mac-run.sh knob logic and error paths.
# Sources the script's functions where possible; where the script is monolithic, exercises it
# via subshell with a temp HOME/ROOT and stubs (no llama-server, no mlx-dspark, no model files).
# Run: bash tests/test_mac_run.sh   (exit 0 = all pass)
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PASS=0; FAIL=0

# Stub binaries on PATH so dependency checks pass without real installs.
STUBS="$(mktemp -d)"
trap 'rm -rf "$STUBS"' EXIT
for cmd in pi llama-server; do printf '#!/bin/sh\nexit 0\n' > "$STUBS/$cmd"; chmod +x "$STUBS"; done 2>/dev/null || true
printf '#!/bin/sh\nexit 0\n' > "$STUBS/pi"; chmod +x "$STUBS/pi"
printf '#!/bin/sh\nexit 0\n' > "$STUBS/llama-server"; chmod +x "$STUBS/llama-server"

run_mac() {
  ( export PATH="$STUBS:$PATH"; JINA_API_KEY=testkey "$@" bash scripts/mac-run.sh ) 2>&1
}

check() { # name, condition (0=pass)
  if [ "$2" -eq 0 ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); echo "FAIL: $1"; fi
}

# --- error paths -------------------------------------------------------------
out=$(run_mac env BACKEND=bogus 2>&1); check "rejects unknown BACKEND" $([ $? -eq 0 ] && echo 1 || echo 0) || true
echo "$out" | grep -q "BACKEND must be" && rc=0 || rc=1; check "error names the invalid backend" $rc

out=$(run_mac env BACKEND=mlx 2>&1)
echo "$out" | grep -q "retired; routing" && rc=0 || rc=1; check "BACKEND=mlx routes to dspark with a note" $rc

# --- validation paths (only meaningful when no live server holds :8080; the script's
# already-up shortcut skips backend checks by design) -----------------------------
if curl -fsS -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo "SKIP: live server on :8080 — backend-validation cases need the stack stopped"
else
  out=$(run_mac env BACKEND=dspark 2>&1)
  # dspark without the venv installed errors with the create command; that's the expected early exit.
  echo "$out" | grep -q "venv-dspark is missing mlx-dspark" && rc=0 || rc=1; check "dspark without venv errors with fix" $rc

  out=$(run_mac env BACKEND=llamacpp MODEL_FILE=nonexistent.gguf 2>&1)
  echo "$out" | grep -q "model not found" && rc=0 || rc=1; check "llamacpp errors on missing GGUF" $rc
fi

# --- WAIT_SECS knob exists ---------------------------------------------------
grep -q 'WAIT_SECS=' scripts/mac-run.sh && rc=0 || rc=1; check "WAIT_SECS knob present" $rc
grep -q 'SERVER_PID=$!' scripts/mac-run.sh && rc=0 || rc=1; check "server-death detection wired" $rc

# --- no-thinking logic is actually in the launch path ------------------------
grep -q -- '--no-thinking $DSPARK_EXTRA_ARGS\|--no-thinking' scripts/mac-run.sh && rc=0 || rc=1; check "--no-thinking in script" $rc
grep -q 'DSPARK_THINKING' scripts/mac-run.sh && rc=0 || rc=1; check "DSPARK_THINKING opt-out exists" $rc

echo
echo "mac-run.sh tests: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
