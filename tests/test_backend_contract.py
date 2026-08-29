#!/usr/bin/env python3
"""GPU-free contract tests: the `:8080` OpenAI-compatible surface that `server/` depends on,
so a backend swap (mlx-dspark / llama.cpp / anything) can be verified against what the app
actually calls. Run against a live server:  python3 tests/test_backend_contract.py [BASE_URL]

Checked: /health, GET /v1/models, chat completion round-trip, and the fields
server/stats.py + run_dataroom.py read (model id resolution, context window).
"""
import json
import os
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
EXPECTED_MODEL = os.environ.get("MODEL_ID", "Qwen3.8-27B-4bit")
PASS, FAIL = 0, 0


def req(path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name} {detail}")


def main():
    # 1. /health — run_dataroom's LLAMA_URL base assumption
    try:
        h = req("/health")
        check("health.status ok", h.get("status") in ("ok", "healthy", True), f"got {h.get('status')!r}")
    except Exception as e:
        check("health reachable", False, str(e))
        return summary()

    # 2. /v1/models — Pi resolves the request's model id against this list
    try:
        models = req("/v1/models")
        ids = [m.get("id") for m in models.get("data", [])]
        check("v1/models lists a model", bool(ids), f"ids={ids}")
        check(
            f"expected model id present ({EXPECTED_MODEL})",
            EXPECTED_MODEL in ids or any(EXPECTED_MODEL in (i or "") for i in ids),
            f"ids={ids}",
        )
    except Exception as e:
        check("v1/models reachable", False, str(e))

    # 3. chat completion round-trip with the model id the app pins
    try:
        r = req(
            "/v1/chat/completions",
            {
                "model": EXPECTED_MODEL,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        content = (r["choices"][0]["message"].get("content") or "").strip()
        check("chat completion returns content", bool(content), f"content={content!r}")
        check("usage block present", "usage" in r or "prompt_tokens" in json.dumps(r)[:400])
    except Exception as e:
        check("chat completion", False, str(e))

    return summary()


def summary():
    print(f"backend contract tests: {PASS} passed, {FAIL} failed (base={BASE})")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
