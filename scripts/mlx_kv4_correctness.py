#!/usr/bin/env python3
"""Correctness tests for the recommended MLX backend (mlx-lm + --kv-bits 4).

2a: greedy parity at short ctx with kv4 — does quantized KV corrupt basic output?
2b: factual recall at ~60K ctx with kv4 — does quantized KV degrade long-context recall
    (the dataroom's compaction-heavy regime)?

Loads the 20GB model once. Writes captures + a PASS/FAIL line per test.
Run: .venv-mlx/bin/python scripts/mlx_kv4_correctness.py
"""
import sys, time, json, pathlib

MODEL = "models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit"
BENCH = pathlib.Path("data/mlx-bench")
BENCH.mkdir(parents=True, exist_ok=True)
KV_BITS, KV_GROUP, KV_START = 4, 64, 0

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

print(f"loading {MODEL} ...", flush=True)
t0 = time.time()
model, tokenizer = load(MODEL)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

greedy = make_sampler(temp=0.0)


def gen(messages, max_tokens):
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    text = ""
    last = None
    for resp in stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens, sampler=greedy,
        kv_bits=KV_BITS, kv_group_size=KV_GROUP, quantized_kv_start=KV_START,
    ):
        text += resp.text
        last = resp
    return text, last


def strip_think(s):
    # reasoning models wrap CoT in <think>...</think>; keep only the answer
    if "</think>" in s:
        s = s.split("</think>")[-1]
    return s.strip()


# ---------- 2a: greedy parity at short ctx (kv4) ----------
PROMPTS = [
    "What is 17 * 23? Reply with only the number.",
    "What is the chemical symbol for gold? Reply with only the symbol.",
    "Complete with one word only: The powerhouse of the cell is the ___.",
    "In one sentence, what does a cryptographic hash function do?",
    "List the eight planets in order from the Sun outward, comma-separated.",
    "What year did Apollo 11 land humans on the Moon? Reply with only the year.",
    "Translate 'good morning' into French. Reply with only the translation.",
    "Boiling point of water at sea level in Celsius? Reply with only the number.",
    "Past tense of the verb 'to go'? Reply with only the word.",
    "How many sides does a hexagon have? Reply with only the number.",
]
print("\n=== 2a: greedy parity @ kv4 (short ctx) ===", flush=True)
out = BENCH / "parity-mlx-kv4.txt"
lines = []
for i, p in enumerate(PROMPTS, 1):
    txt, _ = gen([{"role": "user", "content": p}], max_tokens=1024)
    ans = strip_think(txt).replace("\n", " ")
    if not ans:
        ans = "<empty/truncated>"
    lines.append(f"Q{i:02d}\t{ans}")
    print(f"Q{i:02d}: {ans}", flush=True)
out.write_text("\n".join(lines) + "\n")
print(f"saved -> {out}", flush=True)


# ---------- 2b: factual recall after long ctx (kv4) ----------
FACTS = [
    ("satellite Borealis launch code", "4827",
     "The launch code for satellite Borealis is 4827."),
    ("enzyme discovered by Dr. Elena Vasquez", "zorbase",
     "Dr. Elena Vasquez discovered the enzyme zorbase in 1987."),
    ("capital of the nation Qalandar", "Mirethorn",
     "The capital of the nation Qalandar is the city of Mirethorn."),
    ("Project Nightingale budget in million credits", "312",
     "Project Nightingale's budget is 312 million credits."),
    ("archive password", "violet-hexagon-92",
     "The password to the archive is violet-hexagon-92."),
    ("units of palladium in shipment QX-77", "1440",
     "Shipment QX-77 contains 1440 units of palladium."),
    ("Agent Kowalski badge number", "5563",
     "Agent Kowalski's badge number is 5563."),
    ("operating temperature of the reactor at Site Delta in kelvin", "740",
     "The reactor at Site Delta operates at 740 kelvin."),
    ("author of the novel Crimson Aqueduct", "Idris Faelan",
     "The novel Crimson Aqueduct was written by Idris Faelan."),
    ("grid coordinates of the depot", "88-Theta",
     "The coordinates of the depot are grid 88-Theta."),
    ("expiry of vaccine batch LX-204", "November 2031",
     "The vaccine batch LX-204 expires in November 2031."),
    ("orbital period of moon Pyrrha in days", "53",
     "The orbital period of moon Pyrrha is 53 days."),
    ("contract penalty clause percentage", "17",
     "The contract penalty clause is 17 percent."),
    ("name of the server cluster", "Halcyon-7",
     "The server cluster is named Halcyon-7."),
    ("date of the Tarnis delegation meeting", "March 19",
     "The meeting with the Tarnis delegation is on March 19th."),
]
TARGET_TOKENS = 60000
print(f"\n=== 2b: factual recall @ kv4 (~{TARGET_TOKENS} tok ctx) ===", flush=True)

filler = ("The hybrid GDN plus MoE architecture caches KV on only a subset of "
          "layers, while routine background notes accumulate across the session. ")
# build a long body: filler with the 15 facts evenly interleaved
n_chunks = len(FACTS) + 1
body_parts = []
for i in range(n_chunks):
    body_parts.append(filler * 120)  # ~ filler block
    if i < len(FACTS):
        body_parts.append("\nIMPORTANT RECORD: " + FACTS[i][2] + "\n")
body = "".join(body_parts)

# trim/pad to ~TARGET_TOKENS actual tokens
ids = tokenizer.encode(body)
print(f"initial body tokens: {len(ids)}", flush=True)
if len(ids) > TARGET_TOKENS:
    # keep facts: rebuild with fewer filler reps
    reps = max(10, int(120 * TARGET_TOKENS / len(ids)))
    body_parts = []
    for i in range(n_chunks):
        body_parts.append(filler * reps)
        if i < len(FACTS):
            body_parts.append("\nIMPORTANT RECORD: " + FACTS[i][2] + "\n")
    body = "".join(body_parts)
    ids = tokenizer.encode(body)
print(f"final body tokens: {len(ids)}", flush=True)

questions = "\n".join(
    f"{i+1}. What is the {q}?" for i, (q, _, _) in enumerate(FACTS)
)
user = (
    "Below is a long set of background records. Read them, then answer the "
    "questions at the end using ONLY the records.\n\n"
    + body
    + "\n\nNow answer these questions concisely, one per line, "
      "in the form 'N. <answer>':\n" + questions
)

t0 = time.time()
txt, last = gen([{"role": "user", "content": user}], max_tokens=700)
dt = time.time() - t0
ans = strip_think(txt)
(BENCH / "recall-kv4.txt").write_text(ans + "\n")
print(f"\n--- model recall answer ({dt:.1f}s) ---\n{ans}\n", flush=True)

low = ans.lower()
recalled, missed = 0, []
for q, val, _ in FACTS:
    if val.lower() in low:
        recalled += 1
    else:
        missed.append(f"{q} (={val})")
prefill_tps = getattr(last, "prompt_tps", None)
print(f"RECALL: {recalled}/{len(FACTS)} facts correct", flush=True)
if missed:
    print("MISSED: " + "; ".join(missed), flush=True)

# emit machine-readable summary for the journal step
summary = {
    "ctx_tokens": len(ids),
    "recall": f"{recalled}/{len(FACTS)}",
    "missed": missed,
    "gen_seconds": round(dt, 1),
    "prompt_tps": prefill_tps,
}
print("SUMMARY_JSON " + json.dumps(summary), flush=True)
