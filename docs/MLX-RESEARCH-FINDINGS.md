# MLX Inference Research — Session Findings

**Date:** 2026-06-02 · **Repo:** `~/github/dataroom` · **Branch:** `main`

This captures the full picture of where the MLX inference research stands after the
autonomous dataroom run, including what completed, what crashed, and the open loose ends.

---

## 1. What we set out to do

Research **MLX-based inference backends for Qwen3.6-35B-A3B on Apple Silicon** and decide
which paths belong in `scripts/mac-run.sh` as selectable `LLAMA_URL` backends.

The research was run **self-referentially**: the local dataroom agent (driven by
`qwen3.6` on the local backend) was pointed at the topic of which local backend to use.
The research brief lives in [`docs/MLX-RESEARCH-PLAN.md`](./MLX-RESEARCH-PLAN.md)
(currently untracked — never committed).

---

## 2. The autonomous job `8f19b94d0b4b`

**Location:** `data/jobs/8f19b94d0b4b/`

### Outcome: crashed on backend timeout
- Ran ~9 orchestrator cycles (`orchestrator.log`: `files=0/30 jina=10` plateau after cycle 3).
- Tail of `pi.log` shows the local `qwen3.6` backend dying during a **CONSOLIDATION pass**:
  `"Request timed out"` → auto-retry (attempts 2, 3) → `"terminated"`.
- The backend stopped responding before the job could package a final deliverable —
  ironic, given the research is about that backend's context/timeout ceilings.
- **Backend is no longer running** — nothing on port `8080`, no `llama-server` / `pi`
  processes alive.

### What it produced before crashing
- `dataroom/reports/001-mlx-inference-comparison.md` (~45KB, internally marked
  **"Complete"**) — benchmarks 8 MLX backends + the llama.cpp baseline, with a
  recommendation matrix:

  | Priority | Recommended Backend |
  |---|---|
  | Max single-user speed (with MTP) | MTPLX or Rapid-MLX (~86-95 tok/s decode, ~2x llama.cpp) |
  | Multi-request concurrency | oMLX (continuous batching + tiered KV cache) |
  | Quick setup / minimal deps | mlx-openai-server |
  | Production server w/ SSD fallback | vllm-mlx (paged attention) |
  | Mature, well-known | llama.cpp (broadest community, but slower on Apple Silicon) |

- 4 sourced notes, incl. `dataroom/data/notes/004-mlplx-forks-pr990-independent-benchmarks.md`
- `OUTLINE.md`, `STATUS.md`, `CONTRACT.md`
- **No `dataroom.zip`** — the job never finished cleanly, so there is no packaged final
  deliverable (unlike the earlier job `8389c3036020`, which did produce a zip).

### STATUS.md
Still reads `IN_PROGRESS` with every checkbox unchecked (the agent does not reliably tick
them). Open questions remaining per `STATUS.md`:
- Per-backend benchmarks for mlx-lm, PR #990, MTPLX (Youssofal + rklosinski), oMLX,
  vLLM-MLX, mlx-openai-server, Rapid-MLX.
- The 86K context ceiling root cause (kernel bug vs memory OOM).
- K/V cache memory: fp16 vs quantized; all-40-layers vs 10-layer allocation.
- GDN recurrent-state handling across backends.
- Final synthesis: comparison table, `mac-run.sh` recommendation, 86K hard-blocker decision.

---

## 3. Already landed on `main` (committed)

- `ae3e49e feat: enable MTP speculative decoding on Metal (llama.cpp >= 9430)`
- `a7cede8 docs: Apple Silicon (Metal) native setup; add scripts/mac-run.sh`

Current verified baseline: **llama.cpp `9430`** with Metal offload (`-ngl 999`),
draft-MTP speculative decoding (`--spec-draft-n-max 2`) → **~37.5 tok/s decode,
~90 tok/s prefill**, stable past 65K context.

---

## 4. Key technical takeaways so far

- Qwen3.6-35B-A3B is a **hybrid GDN (Gated DeltaNet) + MoE**: only **10 of 40 layers**
  carry per-token KV; the rest use a small fixed recurrent state.
- KV is therefore tiny in llama.cpp (`q4_0`, ~0.65 GB even at 131K) — the ~22GB weights
  dominate memory, not the cache.
- ~~MLX backends store KV at fp16 (no `q4_0` KV equivalent)~~ **WRONG — corrected
  2026-06-03 (see §7).** Current mlx-lm (0.31.3) *does* quantize the KV cache via
  `--kv-bits` / `--kv-group-size` / `--quantized-kv-start`. At default fp16 KV it uses more
  memory per token, but with `--kv-bits 4` it matches llama.cpp's `q4_0` KV approach.
- ~~The **~86K MLX hang** is *likely a compute-kernel issue*~~ **WRONG — it is a
  memory-bound Metal OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), not a fixed
  kernel bug. On 36 GB at fp16 KV the OOM lands between ~78K and ~92K actual tokens
  (bracketing the 86K anecdote); `--kv-bits 4` clears 92K. Confirmed on local hardware.
- The dataroom agent's autonomous loop routinely exceeds 86K via repeated
  compaction (prefill-heavy) events — but since `--kv-bits 4` pushes the ceiling past 92K
  **and** MLX prefill is ~6x faster than llama.cpp, MLX is now a viable long-job backend,
  not a disqualified one.

---

## 5. Open loose ends

1. **`docs/MLX-RESEARCH-PLAN.md`** — untracked, never committed.
2. **Research job crashed mid-consolidation** — no final `dataroom.zip`; `STATUS.md`
   still `IN_PROGRESS`.
3. **Findings unverified** — `001-mlx-inference-comparison.md` benchmarks are sourced from
   web research by the agent, not reproduced on local hardware.

---

## 6. Next-step options

- **(a)** Restart the backend and re-run the job to completion (and produce a final zip).
- **(b)** Salvage/review `001-mlx-inference-comparison.md` and finalize it by hand.
- **(c)** Commit `MLX-RESEARCH-PLAN.md` (+ this findings doc) and move on.
- **(d)** Independently verify the headline benchmarks on local M3 Pro hardware before
  trusting the recommendation matrix.

---

## 7. Empirical verification on local hardware (2026-06-03, M3 Pro / 36 GB)

Option (d) was executed. mlx-lm 0.31.3 was run against a pre-converted MLX 4-bit model
(`unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`, in `models/mlx/`). Full journal:
`data/mlx-bench/RESULTS.md`. Plan + per-stage detail: `docs/MLX-EXECUTABLE-PLAN.md` §12.

**Gate — PASS (4/4):** loads correctly; KV layout confirmed **30 GDN + 10 full-attention
layers of 40** (MLX handles the hybrid arch, no all-40 over-allocation); clean 2048-token
generation; greedy parity 9/10 exact vs llama.cpp (sole diff `mitochondrion`/`mitochondria`).
Parity had to run **sequentially** — MLX (20.9 GB) + llama.cpp (22 GB) cannot co-reside in 36 GB.

**Speed:** MLX prefill **533 t/s (~5.9x** llama.cpp's ~90), decode **39.5 t/s** (≥ llama.cpp
MTP ~37.5) — *without* MTP. Confirms Han Xiao's "~5x prefill" claim.

**Ceiling (decisive):** at default **fp16 KV**, OOMs between ~78K (OK) and ~92K (FAIL)
actual tokens — a genuine `[METAL] kIOGPUCommandBufferCallbackErrorOutOfMemory`, i.e.
**memory-bound, not a kernel hang**. With **`--kv-bits 4`**, a ~92K-token prompt completes
(prefill 281 t/s, decode 17 t/s, peak 30.5 GB).

**Net correction to §4:** the "86K hard blocker" is a default-fp16-KV memory ceiling,
removable with `--kv-bits 4`. MLX is a viable long-context backend that is also ~6x faster
on prefill (the metric the compaction loop is bound by). MTPLX/MTP (next step) is about
stacking decode speed on top, not about fixing the ceiling.

**MTP via MTPLX (2026-06-03/04) — RESOLVED.** First attempt with **mtplx 0.3.7** +
verified model `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed` produced **incoherent
garbage** (both via mtplx and stock mlx-lm). Root cause = a pure **version mismatch**: the
model's `mtplx_runtime.json` declares `mtplx_version: 0.1.0-preview` (custom
`mlx_vector_paged` format), which **0.3.7 mis-reads** — *not* an MLX kernel issue. **Fix:**
`pip install mtplx==0.1.0rc3` (reports `0.1.0-preview.3`, same `mlx>=0.31,<0.32` pins) →
`mtplx run` then yields **correct, coherent output** in MTP mode. Measured MTP decode
speedup on the 27B (M3 Pro, performance-cold): **~2.07x** on a freeform prompt (16.62 vs
8.05 t/s, depth 3); on a coding prompt **1.47x/1.59x/1.39x** at depth 1/2/3 with **71/61/52%**
draft acceptance (depth 2 the sweet spot; acceptance ≈ hanxiao's ~72%). Caveat: this is the
**27B** (no verified 35B-A3B MTP build exists), so the *ratio*, not absolute t/s, is the result.

**35B-A3B MTP (2026-06-04):** a matching build exists — `samuelfaj/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed`
(`mtplx_version: 0.1.0rc3` = ours). It loads and emits correct text, but is **publisher-unverified**
(`gate=false`, needs `--unsafe-force-unverified`) and drafts poorly: acceptance **31.3/21.6/14.1%**
at depth 1/2/3 → speedup **1.25x / 1.10x / 0.93x** (net-negative by depth 3). AR baseline ~30–37 t/s
matches STAGE 1's mlx-lm decode, so the base path is fine; the unverified MTP head is the problem.
Contrast the verified 27B (71→52% acceptance, up to ~2x). **"Verified" matters** — MTP only pays
off with a properly calibrated head; it's a model-artifact issue, not an MLX defect.

**Correctness with `--kv-bits 4` (2026-06-04) — DONE.** `scripts/mlx_kv4_correctness.py` +
`scripts/mlx_recall.py` (load the 35B-A3B UD-MLX-4bit once): greedy parity @ kv4 = **10/10 identical
to fp16** (quantized KV is greedy-lossless), and **factual recall 15/15 at both 50,215 and 83,495-token
contexts** (prefill 346 / 249 t/s, peak 28.4 GB on 36 GB). So kv4 neither corrupts output nor degrades
long-context recall across the full 50K–85K compaction range — near the top of the ~92–113K kv4 ceiling
— removing the last correctness doubt about the recommended backend. Captures in `data/mlx-bench/`.

**Still open (future):** a *verified* 35B-A3B MTP build (the only matching one found is unverified);
whether MTPLX itself hits a ~86K hang (untested on the working setup) or OOMs like stock mlx-lm.

**Watch (2026-06-05):** Google shipped official Gemma 4 MTP drafters (May 2026, ~3x decode) now landing across the MLX stack (mlx-swift-lm, mlx-vlm) — signals MTP plumbing is maturing toward mlx-lm, but it's a different model and a decode optimization, so it doesn't help our prefill-bound loop or the kv4 *server* gap (mlx-lm PR #1353; issue #1043 tracks it); revisit only if a verified Qwen3.6-35B-A3B drafter appears.

**Bottom line:** adopt **mlx-lm + `--kv-bits 4`** as a fast MLX backend (auto-cap
`CTX_SIZE` ~85K on 36 GB); keep llama.cpp as the >128K stable default; MTP deferred.

---

## 8. Cross-check vs the June-2026 "state of local AI on Apple Silicon" brief + Ollama-MLX spike (2026-06-06)

A third-party June-2026 brief was reviewed against this work. It **confirms** our core findings
(MoE is the killer Mac arch; MLX ~15–40% / "~5x prefill"; KV stays in unified memory so the MLX
edge grows with context; MTP ≈ 1.5–2x from software) — our M3 Pro numbers (533 t/s prefill ≈ 5.9x,
greedy-lossless kv4, 15/15 recall at 83K) are stronger than the brief's generic framing, so the
brief adds **external validation, not new measurements**. Two genuinely new, actionable deltas:

### 8.1 Ollama 0.19+ MLX backend — capability spike (NOT a drop-in replacement)

The brief flags Ollama 0.19 (Mar 2026) rebuilt its Mac stack on MLX, raising the question of whether
it could replace the hand-rolled two-venv `mlx-lm` server (app `.venv` + `.venv-mlx` on `:8081`).
Spike done at the **feature-exposure** level (no 20 GB model pull — the blocker is capability, not speed):

- Installed: **Ollama 0.30.4**. MLX backend is opt-in via `OLLAMA_BACKEND=mlx` (needs 32 GB+;
  below that it auto-falls-back to llama.cpp Metal).
- **Decisive blocker — no MLX KV-quant knob.** `OLLAMA_KV_CACHE_TYPE` (default `f16`, accepts
  `q4_0`/`q8_0`) governs **only the ggml/llama.cpp path**. The MLX backend runs NVFP4/int4 weights +
  an opaque "improved cache" (checkpoints/eviction) and exposes **no `--kv-bits`-equivalent**. Our
  long-job survival depends on `--kv-bits 4` to clear the ~92K fp16-KV OOM ceiling (§7); Ollama-MLX
  has no equivalent, and falling back to `OLLAMA_KV_CACHE_TYPE` forces the slower non-MLX path.
- **Second blocker — no custom MLX import.** The MLX path is tuned to Ollama's own NVFP4
  `qwen3.5:35b-a3b` build; importing our `Qwen3.6-35B-A3B-UD-MLX-4bit` isn't supported yet
  ("we will introduce an easier way to import models").

**Verdict:** Ollama-MLX wins on ergonomics (single OpenAI-compatible daemon) but is **blocked** for
dataroom on (1) no MLX KV-quant and (2) no custom MLX model import — it does not clear the decisive
long-context survival requirement. Keep `mlx-lm + --kv-bits 4`. **Revisit when Ollama exposes
MLX-backend KV quantization + custom MLX model import.**

### 8.2 Native mlx-lm MTP for Qwen3.6 — recheck warranted

The brief reports MTP plumbing moving into `mlx-lm` proper (Qwen3.5: 15.3→23.3 t/s; official
Gemma 4 drafters landing across the MLX stack), not just the MTPLX fork. Our blocker (§7) was
specifically "no *verified* 35B-A3B MTP build via **MTPLX**" — a native mlx-lm MTP path for
Qwen3.6-35B-A3B would sidestep the unverified-MTPLX-head problem entirely. Not yet verified; added
to open items below.

### 8.3 QAT (quantization-aware training) — watch item + Gemma-4-QAT fallback

A follow-up brief (Gemma 4 QAT release, 2026-06-05) raised whether QAT helps here. QAT bakes
low-precision into training so a 4-bit model retains near-bf16 quality — which lands exactly on
this repo's weakest axis: the **no-correctness-axis** gap flagged in the plan evaluation and the
digit/fact-drift risk across the compaction loop. For a *factual* dataroom, a QAT build would be a
**near-free correctness upgrade at the same ~22 GB and same speed** (it improves quality-at-equal-
size; it does **not** shrink memory further — we already bank the vs-bf16 memory win by running Q4,
so QAT buys no extra context/ceiling headroom).

**But there is no official QAT Qwen3.6-35B-A3B today.** As of 2026-06-06 the model ships only as
base, **FP8**, NVIDIA **NVFP4**, AWQ (QuantTrio), and community GGUF (our Unsloth UD dynamic PTQ).
The QAT releases are Gemma 4/3, not Qwen. So:

- **No change now.** Our Unsloth `UD-Q4_K_XL` is the best-available *dynamic PTQ* for Qwen3.6; with
  no QAT alternative to switch to, and the "don't upgrade a 4-bit model to Q5/Q8" nuance still
  holding (small gains, costs memory/ceiling), Q4 stays correct.
- **Gemma-4-QAT fallback (documented, out of scope unless the Qwen Gate/ceiling/MTP path fails).**
  Gemma 4 **26B-A4B** (MoE, ~4B active — same architectural bet as our A3B) ships QAT **plus official
  assistant draft models for speculative decoding**, which would solve *two* open pains at once:
  4-bit correctness **and** the verified-MTP/draft-model gap we could not close on Qwen (the 35B-A3B
  MTPLX build was unverified and drafted poorly, §7). Keep it as the prime fallback candidate.

### 8.4 Head-to-head bench (2026-06-06, M3 Pro / 36 GB) — verdict

Same 6 prompts (temp 0: `391, Au, 5, planets-in-order, 9.9, bonjour`), 256-tok speed gen each.

| Run | Quality | Decode t/s | Peak mem | Note |
|---|---|---|---|---|
| Qwen3.6-35B-A3B — **MLX + kv4** | 6/6 | **40.2** | 20.9 GB | recommended fast path |
| Qwen3.6-35B-A3B — GGUF llama.cpp (no-spec) | 6/6 | 30.5 | ~22 GB | matches baseline (~37.5 w/ MTP) |
| Gemma-4-12B **QAT** — GGUF llama.cpp | 5/6 + 1 inconclusive | 15.2 | ~7 GB | reasoning model; trap prompt hit token cap |

**What sucks:** (1) Gemma-4-12B QAT is too slow for the loop — 15 t/s (dense, 12B active) vs Qwen's
40 (MoE, 3B active); great quality-per-GB, wrong tradeoff for a prefill-heavy hours-long job.
(2) No QAT exists for our model (§8.3). (3) Ollama-MLX confirmed unusable — empirically: `OLLAMA_BACKEND=mlx`
runs but exposes no `--kv-bits` (only ggml-path `OLLAMA_KV_CACHE_TYPE`) and `ollama create FROM ./models/mlx/...`
fails (`400 invalid model name`); its MLX path is fed by registry NVFP4 builds, not our safetensors (§8.1).
(4) 35B MTP still unsolved (§7). **What works:** MLX+kv4 — ~32% faster than llama.cpp, less memory,
6/6 identical output, recall verified to 86K.

**Decision:** ship **mlx-lm + `--kv-bits 4`** as the fast Mac backend; keep llama.cpp as the >128K
stable default. Drop Gemma-4-12B and Ollama-MLX (evaluated, rejected). Watch-only: an official QAT
**or** verified-MTP Qwen3.6-35B-A3B → adopt; else Gemma-4-QAT **26B-A4B** stays the documented
fallback if the Qwen Gate/ceiling ever fails.

**Status — already in flight:** the backend knob is **open as dataroom PR #4** (`feat(mac): add
BACKEND={llamacpp|mlx}`, branch `feat/mlx-backend-knob`). Caveat: it ships **fp16-KV** mlx
(`MLX_CTX_CAP=75000`), **not kv4** — stock `mlx_lm.server` has no `--kv-bits` flag yet. The upstream
dependency is **`ml-explore/mlx-lm` PR #1353** (`server: add KV cache quantization flags
--kv-bits/--kv-group-size/--quantized-kv-start`, our own branch `soobrosa:feat/server-kv-cache-quant`)
— currently a **draft** (issue #1043 is only the tracking request; a competing impl is PR #1309).
The kv4 ceiling-raise (~92-113K) lands only once #1353 is marked ready, merges upstream, and the
cap is raised; until then the fast path is shipped but capped conservatively.

### 8.5 Still open (future) — updated

- A **verified** 35B-A3B MTP build (the only matching MTPLX one found is unverified, §7); **plus**
  recheck whether **native mlx-lm** now ships MTP for Qwen3.6-35B-A3B (§8.2).
- Whether MTPLX itself hits a ~86K hang (untested on the working setup) or OOMs like stock mlx-lm.
- **Re-evaluate Ollama-MLX** once it exposes an MLX KV-quant knob + custom MLX model import (§8.1).
- **Watch for an official QAT Qwen3.6-35B-A3B** (GGUF or MLX) — adopt on release for a near-free
  correctness upgrade at equal memory/speed; else keep the Gemma-4-QAT 26B-A4B fallback (§8.3).
- **Land kv4 in `mlx_lm.server`**: mark upstream **PR #1353** ready for review (currently draft;
  issue #1043 is only the tracker), get it merged, then raise `MLX_CTX_CAP` 75K→~85K in dataroom PR #4.
