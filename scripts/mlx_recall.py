#!/usr/bin/env python3
"""Factual-recall-after-long-ctx test for mlx-lm + --kv-bits 4 at a target ctx size.

Usage: .venv-mlx/bin/python scripts/mlx_recall.py [TARGET_TOKENS]   (default 85000)
Builds a long context with 15 unique facts interleaved in filler, sized to ~TARGET
tokens, then asks all 15 back. Writes the answer + a SUMMARY_JSON line.
"""
import sys, time, json, pathlib

MODEL = "models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit"
BENCH = pathlib.Path("data/mlx-bench"); BENCH.mkdir(parents=True, exist_ok=True)
KV_BITS, KV_GROUP, KV_START = 4, 64, 0
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 85000

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

print(f"loading {MODEL} ...", flush=True)
t0 = time.time()
model, tokenizer = load(MODEL)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
greedy = make_sampler(temp=0.0)

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
filler = ("The hybrid GDN plus MoE architecture caches KV on only a subset of "
          "layers, while routine background notes accumulate across the session. ")
n_chunks = len(FACTS) + 1


def build(reps):
    parts = []
    for i in range(n_chunks):
        parts.append(filler * reps)
        if i < len(FACTS):
            parts.append("\nIMPORTANT RECORD: " + FACTS[i][2] + "\n")
    return "".join(parts)


# converge reps so encoded body ~= TARGET tokens
reps = 200
for _ in range(6):
    body = build(reps)
    n = len(tokenizer.encode(body))
    print(f"reps={reps} -> {n} tokens", flush=True)
    if abs(n - TARGET) <= TARGET * 0.03 or n >= TARGET:
        break
    reps = max(1, int(reps * TARGET / max(n, 1)))
body = build(reps)
ctx_tokens = len(tokenizer.encode(body))
print(f"final ctx tokens: {ctx_tokens}", flush=True)

questions = "\n".join(f"{i+1}. What is the {q}?" for i, (q, _, _) in enumerate(FACTS))
user = ("Below is a long set of background records. Read them, then answer the "
        "questions at the end using ONLY the records.\n\n" + body +
        "\n\nNow answer these questions concisely, one per line, in the form "
        "'N. <answer>':\n" + questions)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)

t0 = time.time()
text, last = "", None
for resp in stream_generate(model, tokenizer, prompt, max_tokens=700, sampler=greedy,
                            kv_bits=KV_BITS, kv_group_size=KV_GROUP,
                            quantized_kv_start=KV_START):
    text += resp.text; last = resp
dt = time.time() - t0
ans = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
(BENCH / f"recall-kv4-{TARGET}.txt").write_text(ans + "\n")
print(f"\n--- recall answer ({dt:.1f}s) ---\n{ans}\n", flush=True)

low = ans.lower()
recalled, missed = 0, []
for q, val, _ in FACTS:
    if val.lower() in low:
        recalled += 1
    else:
        missed.append(f"{q} (={val})")
print(f"RECALL: {recalled}/{len(FACTS)} facts correct", flush=True)
if missed:
    print("MISSED: " + "; ".join(missed), flush=True)
print("SUMMARY_JSON " + json.dumps({
    "ctx_tokens": ctx_tokens, "recall": f"{recalled}/{len(FACTS)}", "missed": missed,
    "gen_seconds": round(dt, 1),
    "prompt_tps": getattr(last, "prompt_tps", None),
    "peak_memory": getattr(last, "peak_memory", None),
}), flush=True)
