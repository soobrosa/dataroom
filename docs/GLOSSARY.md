# Glossary — Inference Backend Terms

Plain-language definitions for terms that come up in the MLX/llama.cpp backend research.

---

## Inference engine

The actual program that loads model weights and runs the math to generate tokens.
Several competing engines exist; they have different code, cache formats, and kernels,
and run on different hardware:

| Engine | Runs on | Used in this repo? |
|---|---|---|
| **llama.cpp** | CPU, Metal (Mac), CUDA (NVIDIA) | Yes — both `mac-run.sh` (Metal) and `docker-compose.yml` (`server-cuda`) |
| **mlx-lm** | Apple Silicon (Metal) only | Yes — the MLX research path |
| **vLLM** | NVIDIA GPUs (CUDA), mainly | No |

Because an engine's KV-cache format and attention kernels are internal to it, a feature
built into one engine (e.g. a custom KV quantizer in a **vLLM fork**) generally **cannot**
be lifted out and bolted onto another engine. Adopting such a feature means switching the
whole serving engine, not adding a flag.

> Analogy: an engine-specific upgrade (like a fuel-injection part) can't just be installed
> on a different engine block — you'd have to swap the entire engine first, and it may not
> even fit your hardware (e.g. CUDA parts don't run on Apple Silicon).

## vLLM

A high-throughput inference engine, primarily for **NVIDIA GPUs (CUDA)**. Not used in this
repo (the stack is llama.cpp on both Mac/Metal and Linux/CUDA, plus mlx-lm for research).
It does **not** run on Apple Silicon/Metal, so vLLM-based tools (e.g. KVarN) are non-starters
for the Mac path regardless of their merits.

## head_dim

A model architecture parameter — the **width of one attention head's vector** (typically
64, 96, or 128 numbers).

- Transformer attention is split into parallel **heads**, each viewing the text from a
  different angle.
- Each head works in a fixed-width vector space; `head_dim` is that width.
- Roughly: `hidden_size = num_heads × head_dim` (e.g. hidden 5120 / 40 heads → head_dim 128).

The **KV cache** stores, per token, a key vector and a value vector of size `head_dim` for
each head — that's what KV-cache quantizers operate on. Some implementations (e.g. KVarN)
hardcode `head_dim = 128` (128×128 tiles) and reject other sizes.

## KV cache

Per-token key/value vectors that attention saves so it doesn't recompute past tokens on
every step. Its memory grows with context length. For this repo's hybrid GDN+MoE model
(Qwen3.6-35B-A3B), only **10 of 40 layers** carry a per-token KV cache (the rest keep a
small fixed recurrent state), so KV is tiny (~0.65 GB at `q4_0`/131072) — the ~22 GB
**weights** dominate memory, not the KV cache.

## KV-cache quantization

Storing the KV cache at lower bit-width (e.g. 4-bit) to use less memory / fit longer
context. In this repo:
- **llama.cpp:** `--cache-type-k q4_0 --cache-type-v q4_0`
- **mlx-lm:** `--kv-bits 4 --kv-group-size 64 --quantized-kv-start 0`

Measured here to be **greedy-lossless** (10/10 parity vs fp16) with **15/15 long-context
recall** on the project model — see `MLX-RESEARCH-FINDINGS.md`.

## GDN (Gated DeltaNet)

A linear-attention layer type that keeps a small **fixed-size recurrent state** instead of
a growing per-token KV cache. Qwen3.6-35B-A3B is a **hybrid**: ~30 GDN layers + ~10
full-attention (KV-caching) layers out of 40. This is why its KV footprint stays tiny even
at very long context.

## MoE (Mixture of Experts)

A model where each token is routed to a small subset of "expert" sub-networks rather than
the whole network, so only a fraction of parameters activate per token (the "A3B" in
Qwen3.6-35B-A3B = ~3B active params). Cuts compute per token while keeping a large total
parameter count.

## MTP (Multi-Token Prediction) / speculative decoding

A technique to speed up **decode** by drafting several tokens ahead with a lightweight head
and verifying them in one pass; accepted drafts skip full forward passes. Lossless when
done right (verified output equals non-speculative greedy). In this repo:
- **llama.cpp:** `--spec-type draft-mtp --spec-draft-n-max 2`
- **MTPLX (MLX):** native MTP, version-matched (`mtplx==0.1.0rc3`); ~1.4–2.1x decode on the
  verified 27B at 51–71% draft acceptance.

**Draft acceptance rate** — the fraction of drafted tokens the model accepts; the stable
signal for whether MTP pays off (low acceptance can make MTP net-negative).

## Prefill vs decode

- **Prefill:** processing the input prompt to populate the KV cache (one big parallel pass).
  Measured in prompt tokens/sec. Dominates the dataroom's compaction-heavy loop.
- **Decode:** generating output tokens one at a time. Measured in generation tokens/sec.

## Compaction

When a session's context approaches the window limit, the agent summarizes/condenses
history to free space. It is a **prefill-heavy** event (the summary must be re-ingested) and
happens repeatedly in long autonomous runs — which is why prefill speed and the context
ceiling matter so much here.

## Context ceiling

The largest context (in tokens) the backend can handle before failing. For mlx-lm on a
36 GB Mac: fp16 KV OOMs ~78–92K; `--kv-bits 4` raises it to ~92–113K. The failure is a
genuine **memory-bound Metal OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), not a
fixed kernel hang — so it moves with KV precision/size.
